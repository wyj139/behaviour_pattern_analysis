"""
random_forest_classifier.py
Function: Train a Random Forest model to predict decision from embeddings,
          evaluate accuracy per window_id, and visualize feature importances.
          Also: find the best prediction window_id for trials ending at different windows.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, recall_score
from typing import Optional

from config import OUTPUT_BASE_DIR, VERBOSE, RANDOM_SEED


def load_and_clean_embeddings(output_base_dir: str = OUTPUT_BASE_DIR) -> pd.DataFrame:
    """
    加载并清理embeddings数据，将tensor(x)格式转换为数值
    Load and clean embeddings data, convert tensor(x) format to numeric values

    参数 Parameters:
        output_base_dir: 输出基础目录 (Output base directory)

    返回 Returns:
        清理后的DataFrame (Cleaned DataFrame)
    """
    embedding_path = os.path.join(output_base_dir, 'embeddings.csv')
    df = pd.read_csv(embedding_path)

    # 清理tensor(x)格式的列
    # Clean tensor(x) format columns
    for col in ['decision', 'window_id']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: int(re.search(r'\d+', str(x)).group())
                                    if 'tensor' in str(x) else int(x))

    if VERBOSE:
        print(f"数据加载完成: {len(df)} 条记录")
        print(f"Decision分布:\n{df['decision'].value_counts()}")
        print(f"Window ID范围: {df['window_id'].min()} - {df['window_id'].max()}")
        print(f"Embedding维度: {len([c for c in df.columns if c.startswith('emb_')])}")

    return df


def filter_window_ids_by_trial_duration(df: pd.DataFrame,
                                        trial_percentile: float = 90.0) -> pd.DataFrame:
    """
    基于trial时长分布截断window_id
    Filter window_ids based on trial duration distribution
    
    原理：计算每个trial(id)的最大window_id，取其第trial_percentile百分位数作为截断点，
    只保留 window_id <= 该截断点的数据。这样可以排除少数持续时间极长的异常trial产生的
    高window_id数据。
    
    Principle: Compute max window_id per trial(id), use the trial_percentile-th
    percentile as cutoff, keep only window_id <= cutoff.

    参数 Parameters:
        df: 原始DataFrame
        trial_percentile: trial时长的百分位数截断点（默认90%，即覆盖90%的trial）

    返回 Returns:
        过滤后的DataFrame
    """
    # 计算每个trial的最大window_id（代表trial时长）
    max_wids_per_trial = df.groupby('id')['window_id'].max()
    
    # 取百分位数作为截断点
    cutoff_wid = int(np.percentile(max_wids_per_trial.values, trial_percentile))
    
    df_filtered = df[df['window_id'] <= cutoff_wid].copy()
    
    if VERBOSE:
        print(f"\n基于trial时长的window_id过滤 (Trial duration based filtering):")
        print(f"  每个trial的最大window_id: median={max_wids_per_trial.median():.0f}, "
              f"mean={max_wids_per_trial.mean():.1f}, max={max_wids_per_trial.max()}")
        print(f"  第{trial_percentile:.0f}百分位截断点: window_id <= {cutoff_wid}")
        print(f"  过滤前: {df['window_id'].nunique()} 个window_id, {len(df)} 条记录")
        print(f"  过滤后: {df_filtered['window_id'].nunique()} 个window_id, {len(df_filtered)} 条记录")
        print(f"  保留样本比例: {len(df_filtered)/len(df)*100:.1f}%")
        print(f"  覆盖trial比例: {(max_wids_per_trial <= cutoff_wid).sum()}/{len(max_wids_per_trial)} "
              f"({(max_wids_per_trial <= cutoff_wid).sum()/len(max_wids_per_trial)*100:.1f}%)")

    return df_filtered


def train_random_forest(df: pd.DataFrame,
                        n_estimators: int = 200,
                        max_depth: Optional[int] = None,
                        random_seed: int = None,  # 允许None以获得不同结果；如果为None，使用系统时间
                        n_folds: int = 5,
                        output_dir: Optional[str] = None,
                        cv_method: str = 'stratified_kfold') -> dict:
    """
    训练随机森林模型，按window_id统计准确度，计算特征权重
    支持两种交叉验证方法：Stratified K-Fold 或 Leave-One-Pig-Out
    
    Train Random Forest model, evaluate accuracy per window_id, compute feature importances
    Support both Stratified K-Fold and Leave-One-Pig-Out cross-validation

    参数 Parameters:
        df: 包含window数据的DataFrame，可以包含embedding列或不包含（会自动加载）
        n_estimators: 随机森林中树的数量 (Number of trees)
        max_depth: 树的最大深度 (Max tree depth)
        random_seed: 随机种子 (Random seed)
        n_folds: K-Fold交叉验证折数 (Number of CV folds for K-Fold, ignored for LOPO)
        output_dir: 输出目录 (Output directory)
        cv_method: 交叉验证方法 ('stratified_kfold' 或 'leave_one_pig_out')

    返回 Returns:
        包含模型、准确度和特征重要性的字典 (Dict with model, accuracies and feature importances)
    """
    # 检查embedding列，如果没有则尝试加载
    # Check for embedding columns, load if missing
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    
    if len(emb_cols) == 0:
        # 需要加载embedding文件
        # Need to load embedding file
        if output_dir and 'window_id' in df.columns:
            embeddings_path = os.path.join(output_dir, 'embeddings.csv')
            if os.path.exists(embeddings_path):
                print(f"从{embeddings_path}加载embedding...")
                embeddings_df = pd.read_csv(embeddings_path)
                emb_cols = [c for c in embeddings_df.columns if c.startswith('emb_')]
                
                # 按window_id合并embedding
                print(f"按window_id合并embedding ({len(embeddings_df)} 条embedding 到 {len(df)} 条window数据)...")
                df = df.merge(embeddings_df[['window_id'] + emb_cols], on='window_id', how='left', suffixes=('', '_emb'))
                
                # 检查是否有缺失值
                missing_count = df[emb_cols].isnull().sum().sum()
                if missing_count > 0:
                    print(f"警告: 有 {missing_count} 个embedding缺失值，已删除")
                    df = df.dropna(subset=emb_cols)
                
                print(f"合并后: {len(df)} 条记录，{len(emb_cols)} 个embedding列")
            else:
                raise FileNotFoundError(f"未找到embedding文件: {embeddings_path}")
        else:
            raise ValueError(f"DataFrame中没有embedding列，也无法加载（output_dir={output_dir}）")

    # ========== 1. 全局模型训练（交叉验证）==========
    # ========== 1. Global model training (cross-validation) ==========
    print("\n" + "=" * 60)
    print(f"训练全局随机森林模型 (CV Method: {cv_method})")
    print("Training global Random Forest model")
    print("=" * 60)
    
    # 处理random_seed：如果为None，使用RANDOM_SEED作为默认值（参考）；如果需要不同结果可传入None
    if random_seed is None:
        random_seed = RANDOM_SEED  # 默认使用配置文件中的种子
    
    print(f"使用随机种子: {random_seed}")

    X_all = df[emb_cols].values
    y_all = df['decision'].values

    cv_scores = []
    cv_fold_info = []  # 存储每个fold的详细信息
    cv_predictions = np.full(len(df), -1, dtype=int)  # 存储所有样本的CV预测结果
    
    # 选择交叉验证方法
    if cv_method == 'leave_one_pig_out':
        # ===== Leave-One-Pig-Out 交叉验证 =====
        # 按猪进行分组，每次留出一只猪作为验证集
        # 猪编号在trial_id的中间位置（date-pig-trial格式）
        unique_pigs = sorted(df['pig'].unique())
        n_folds_actual = len(unique_pigs)
        
        print(f"\n使用 Leave-One-Pig-Out 交叉验证")
        print(f"总共 {n_folds_actual} 只猪，进行 {n_folds_actual} 轮验证")
        print(f"(每轮: 用其他猪的数据训练，用 1 只猪验证)\n")
        
        for fold, test_pig in enumerate(unique_pigs):
            # 验证集：test_pig 的数据
            val_mask = df['pig'] == test_pig
            val_idx = np.where(val_mask)[0]
            
            # 训练集：其他猪的数据
            train_mask = df['pig'] != test_pig
            train_idx = np.where(train_mask)[0]
            
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]
            
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_seed,
                n_jobs=-1,
                class_weight='balanced'
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            simple_acc = accuracy_score(y_val, y_pred)
            recall_nogo = recall_score(y_val, y_pred, pos_label=0)
            recall_go = recall_score(y_val, y_pred, pos_label=1)
            cv_scores.append(balanced_acc)
            
            # 保存这个fold的预测结果
            cv_predictions[val_idx] = y_pred
            
            cv_fold_info.append({
                'fold': fold + 1,
                'test_pig': test_pig,
                'n_test_samples': len(y_val),
                'accuracy': balanced_acc,
                'simple_accuracy': simple_acc,
                'recall_nogo': recall_nogo,
                'recall_go': recall_go
            })
            
            if VERBOSE and (fold + 1) % 2 == 0:
                print(f"  Fold {fold + 1}/{n_folds_actual} (pig {test_pig}): balanced_acc={balanced_acc:.4f} (recall_NOGO={recall_nogo:.4f}, recall_GO={recall_go:.4f})")
        
        if VERBOSE:
            print(f"  完成所有 {n_folds_actual} 轮验证")
    else:  # stratified_kfold（默认）
        # ===== 分层K折交叉验证 =====
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            X_train, X_val = X_all[train_idx], X_all[val_idx]
            y_train, y_val = y_all[train_idx], y_all[val_idx]

            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_seed,
                n_jobs=-1,
                class_weight='balanced'
            )
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)
            balanced_acc = balanced_accuracy_score(y_val, y_pred)
            simple_acc = accuracy_score(y_val, y_pred)
            recall_nogo = recall_score(y_val, y_pred, pos_label=0)
            recall_go = recall_score(y_val, y_pred, pos_label=1)
            cv_scores.append(balanced_acc)
            
            # 保存这个fold的预测结果
            cv_predictions[val_idx] = y_pred
            
            cv_fold_info.append({
                'fold': fold + 1,
                'n_test_samples': len(y_val),
                'accuracy': balanced_acc,
                'simple_accuracy': simple_acc,
                'recall_nogo': recall_nogo,
                'recall_go': recall_go
            })

            if VERBOSE:
                print(f"  Fold {fold + 1}/{n_folds}: balanced_acc={balanced_acc:.4f} (recall_NOGO={recall_nogo:.4f}, recall_GO={recall_go:.4f})")

    print(f"\n交叉验证平均准确度 (CV mean balanced accuracy): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"最小平衡准确度: {np.min(cv_scores):.4f}, 最大平衡准确度: {np.max(cv_scores):.4f}")
    
    # 计算平均recall
    avg_recall_nogo = np.mean([f['recall_nogo'] for f in cv_fold_info])
    avg_recall_go = np.mean([f['recall_go'] for f in cv_fold_info])
    print(f"\n平均Recall:")
    print(f"  NOGO (decision=0): {avg_recall_nogo:.4f}")
    print(f"  GO (decision=1): {avg_recall_go:.4f}")
    print(f"  平衡准确率 (Balanced): {(avg_recall_nogo + avg_recall_go)/2:.4f}")

    # 在全部数据上训练最终模型（用于获取特征重要性）
    # Train final model on all data (for feature importances)
    final_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed,
        n_jobs=-1,
        class_weight='balanced'
    )
    final_model.fit(X_all, y_all)

    # 特征重要性
    # Feature importances
    feature_importances = pd.DataFrame({
        'feature': emb_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 重要特征 (Top 10 important features):")
    print(feature_importances.head(10).to_string(index=False))

    # ========== 2. 按window_id分组评估准确度 ==========
    # ========== 2. Per-window_id accuracy evaluation ==========
    # 使用前面生成的CV预测结果来评估每个窗口的准确度
    # Use CV predictions from above to evaluate per-window accuracy
    print("\n" + "=" * 60)
    print("按window_id评估准确度 (Per-window_id accuracy evaluation)...")
    print("使用留一窗口评估 (Leave-one-window-out evaluation)")
    print("=" * 60)

    window_ids = sorted(df['window_id'].unique())
    window_accuracies = {}
    window_sample_counts = {}

    # 创建评估DataFrame，包含CV预测结果
    # Create evaluation DataFrame with CV predictions
    df_eval = df.copy()
    df_eval['cv_pred'] = cv_predictions

    # 检查cv_predictions是否有效
    if np.any(cv_predictions < 0):
        print(f"警告: 有 {np.sum(cv_predictions < 0)} 个样本没有获得预测结果")

    # 按window_id统计CV预测的准确度
    # Compute CV prediction accuracy per window_id
    for wid in window_ids:
        window_data = df_eval[df_eval['window_id'] == wid]

        # 跳过没有有效预测的窗口
        if np.any(window_data['cv_pred'] < 0):
            if VERBOSE:
                print(f"  跳过window {wid}: 有无效的预测结果")
            continue

        # 跳过只有一个类别的窗口
        # Skip windows with only one class
        if window_data['decision'].nunique() < 2:
            continue

        balanced_acc = balanced_accuracy_score(window_data['decision'], window_data['cv_pred'])
        window_accuracies[wid] = balanced_acc
        window_sample_counts[wid] = len(window_data)

    if VERBOSE:
        print(f"\n有效窗口数: {len(window_accuracies)} / {len(window_ids)}")
        if len(window_accuracies) > 0:
            accs = list(window_accuracies.values())
            print(f"窗口准确度: mean={np.mean(accs):.4f}, min={np.min(accs):.4f}, max={np.max(accs):.4f}")

    # 打印分类报告（基于CV预测）
    # Print classification report (based on CV predictions)
    print(f"\n全局分类报告 - 基于交叉验证 (Global classification report - CV based):")
    print(classification_report(y_all, cv_predictions, target_names=['decision=0', 'decision=1']))

    # ========== 3. 可视化 ==========
    # ========== 3. Visualization ==========
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # 3.1 柱状图：不同window_id的预测准确度
        # 3.1 Bar chart: prediction accuracy per window_id
        plot_window_accuracy(window_accuracies, window_sample_counts, output_dir)

        # 3.2 特征重要性图
        # 3.2 Feature importance plot
        plot_feature_importances(feature_importances, output_dir)

        # 3.3 按decision分类的准确度图
        # 3.3 Accuracy per window_id split by decision class
        plot_accuracy_by_decision(df_eval, output_dir)

        # 3.4 按housing分类的准确度图
        # 3.4 Accuracy per window_id split by housing
        plot_accuracy_by_housing(df_eval, output_dir)
        
        # 3.5 所有fold的准确率变化图
        # 3.5 Plot all fold accuracy scores
        plot_cv_fold_scores(cv_scores, cv_fold_info, output_dir, cv_method)

    return {
        'model': final_model,
        'cv_scores': cv_scores,
        'cv_fold_info': cv_fold_info,
        'feature_importances': feature_importances,
        'window_accuracies': window_accuracies,
        'window_sample_counts': window_sample_counts,
        'df_eval': df_eval
    }


def plot_window_accuracy(window_accuracies: dict,
                         window_sample_counts: dict,
                         output_dir: str):
    """
    绘制不同window_id的预测准确度柱状图
    Plot bar chart of prediction accuracy per window_id
    """
    wids = sorted(window_accuracies.keys())
    accs = [window_accuracies[w] for w in wids]
    counts = [window_sample_counts[w] for w in wids]

    fig, ax1 = plt.subplots(figsize=(max(14, len(wids) * 0.15), 6))

    # 准确度柱状图
    # Accuracy bar chart
    bars = ax1.bar(range(len(wids)), accs, color='steelblue', alpha=0.8, width=0.8)
    ax1.set_xlabel('Window ID', fontsize=12)
    ax1.set_ylabel('Balanced Accuracy', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    # 添加平均线
    # Add mean line
    mean_acc = np.mean(accs)
    ax1.axhline(y=mean_acc, color='red', linestyle='--', linewidth=1.5,
                label=f'Mean balanced accuracy = {mean_acc:.4f}')

    ax1.set_ylim(0, 1.05)

    # 简化x轴：只显示部分刻度
    # Simplify x-axis: show only some ticks
    if len(wids) > 50:
        step = max(1, len(wids) // 20)
        tick_positions = range(0, len(wids), step)
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels([str(wids[i]) for i in tick_positions], rotation=45, fontsize=8)
    else:
        ax1.set_xticks(range(len(wids)))
        ax1.set_xticklabels([str(w) for w in wids], rotation=45, fontsize=8)

    # 样本量副轴
    # Sample count secondary axis
    ax2 = ax1.twinx()
    ax2.plot(range(len(wids)), counts, color='orange', linewidth=1, alpha=0.6,
             label='Sample count')
    ax2.set_ylabel('Sample Count', fontsize=12, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

    plt.title('Random Forest Prediction Balanced Accuracy per Window ID', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'rf_accuracy_per_window_id.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"窗口准确度柱状图已保存到: {output_path}")


def plot_feature_importances(feature_importances: pd.DataFrame,
                             output_dir: str):
    """
    绘制随机森林特征重要性图
    Plot Random Forest feature importances
    """
    fi = feature_importances.copy()

    # 图1：所有特征重要性
    # Plot 1: All feature importances
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 左图：所有特征按重要性排序的柱状图
    # Left: All features sorted by importance
    ax1 = axes[0]
    ax1.barh(range(len(fi)), fi['importance'].values, color='teal', alpha=0.8)
    ax1.set_yticks(range(len(fi)))
    ax1.set_yticklabels(fi['feature'].values, fontsize=8)
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title('All Feature Importances (sorted)', fontsize=13)
    ax1.invert_yaxis()

    # 右图：Top 15 特征重要性
    # Right: Top 15 feature importances
    ax2 = axes[1]
    top_n = min(15, len(fi))
    top_fi = fi.head(top_n)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    bars = ax2.barh(range(top_n), top_fi['importance'].values, color=colors)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_fi['feature'].values, fontsize=10)
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title(f'Top {top_n} Feature Importances', fontsize=13)
    ax2.invert_yaxis()

    # 在柱上添加数值标注
    # Add value annotations on bars
    for i, (val, name) in enumerate(zip(top_fi['importance'].values, top_fi['feature'].values)):
        ax2.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'rf_feature_importances.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"特征重要性图已保存到: {output_path}")

    # 保存特征重要性到CSV
    # Save feature importances to CSV
    csv_path = os.path.join(output_dir, 'rf_feature_importances.csv')
    fi.to_csv(csv_path, index=False)
    print(f"特征重要性数据已保存到: {csv_path}")


def plot_accuracy_by_decision(df_eval: pd.DataFrame, output_dir: str):
    """
    按decision=0和decision=1分别计算每个window_id的预测准确率，绘制在同一张图上
    Plot per-window_id accuracy separately for decision=0 and decision=1
    """
    # 只保留同时有两个类别的window
    valid_wids = []
    for wid in sorted(df_eval['window_id'].unique()):
        w = df_eval[df_eval['window_id'] == wid]
        if w['decision'].nunique() >= 2:
            valid_wids.append(wid)

    acc_d0 = []
    acc_d1 = []
    wid_labels = []

    for wid in valid_wids:
        w = df_eval[df_eval['window_id'] == wid]

        # decision=0 (NOGO) 的recall
        w0 = w[w['decision'] == 0]
        if len(w0) > 0:
            a0 = recall_score(w0['decision'], w0['cv_pred'], pos_label=0)
        else:
            a0 = np.nan

        # decision=1 (GO) 的recall
        w1 = w[w['decision'] == 1]
        if len(w1) > 0:
            a1 = recall_score(w1['decision'], w1['cv_pred'], pos_label=1)
        else:
            a1 = np.nan

        acc_d0.append(a0)
        acc_d1.append(a1)
        wid_labels.append(wid)

    acc_d0 = np.array(acc_d0, dtype=float)
    acc_d1 = np.array(acc_d1, dtype=float)
    x = np.arange(len(wid_labels))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(max(14, len(wid_labels) * 0.2), 7))

    ax.bar(x - bar_width / 2, acc_d0, bar_width, color='#e74c3c', alpha=0.75, label='NOGO (decision=0) Recall')
    ax.bar(x + bar_width / 2, acc_d1, bar_width, color='#2ecc71', alpha=0.75, label='GO (decision=1) Recall')

    # 平均线
    mean_d0 = np.nanmean(acc_d0)
    mean_d1 = np.nanmean(acc_d1)
    balanced_mean = (mean_d0 + mean_d1) / 2
    ax.axhline(y=mean_d0, color='#e74c3c', linestyle='--', linewidth=1.2,
               label=f'Mean NOGO recall: {mean_d0:.4f}')
    ax.axhline(y=mean_d1, color='#2ecc71', linestyle='--', linewidth=1.2,
               label=f'Mean GO recall: {mean_d1:.4f}')

    ax.set_xlabel('Window ID', fontsize=12)
    ax.set_ylabel('Recall (Class-Specific)', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title(f'Per-Window Class-Specific Recall for NOGO vs GO (Balanced Acc: {balanced_mean:.4f})', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)

    # 简化x轴
    if len(wid_labels) > 50:
        step = max(1, len(wid_labels) // 20)
        tick_positions = list(range(0, len(wid_labels), step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(wid_labels[i]) for i in tick_positions], rotation=45, fontsize=8)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in wid_labels], rotation=45, fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rf_accuracy_by_decision.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"按Decision分类的准确度图已保存到: {output_path}")


def plot_accuracy_by_housing(df_eval: pd.DataFrame, output_dir: str):
    """
    按housing条件分类，绘制每个window_id的预测准确率
    Plot per-window_id accuracy split by housing condition
    """
    if 'Housing' not in df_eval.columns:
        print("警告: 数据中没有Housing列，跳过Housing准确度图")
        return

    housing_types = sorted(df_eval['Housing'].unique())
    colors = {'Baseline1': '#3498db', 'Baseline2': '#e74c3c',
              'Enriched': '#2ecc71', 'Barren': '#f39c12'}
    # 为未预设的housing类型分配颜色
    default_colors = ['#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    for i, h in enumerate(housing_types):
        if h not in colors:
            colors[h] = default_colors[i % len(default_colors)]

    # 只保留同时有两个decision类别的window
    valid_wids = []
    for wid in sorted(df_eval['window_id'].unique()):
        w = df_eval[df_eval['window_id'] == wid]
        if w['decision'].nunique() >= 2:
            valid_wids.append(wid)

    # 计算每个housing在每个window的平衡准确率
    housing_accs = {h: [] for h in housing_types}
    for wid in valid_wids:
        w = df_eval[df_eval['window_id'] == wid]
        for h in housing_types:
            wh = w[w['Housing'] == h]
            if len(wh) >= 2 and wh['decision'].nunique() >= 2:
                acc = balanced_accuracy_score(wh['decision'], wh['cv_pred'])
            else:
                acc = np.nan
            housing_accs[h].append(acc)

    x = np.arange(len(valid_wids))
    n_housing = len(housing_types)
    bar_width = 0.8 / n_housing

    fig, ax = plt.subplots(figsize=(max(14, len(valid_wids) * 0.2), 7))

    for i, h in enumerate(housing_types):
        accs = np.array(housing_accs[h], dtype=float)
        offset = (i - n_housing / 2 + 0.5) * bar_width
        ax.bar(x + offset, accs, bar_width, color=colors[h], alpha=0.75, label=h)
        mean_val = np.nanmean(accs)
        ax.axhline(y=mean_val, color=colors[h], linestyle='--', linewidth=1,
                   alpha=0.6, label=f'{h} mean: {mean_val:.4f}')

    ax.set_xlabel('Window ID', fontsize=12)
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title('Per-Window Balanced Accuracy by Housing Condition', fontsize=14)
    ax.legend(loc='lower left', fontsize=9, ncol=2)

    # 简化x轴
    if len(valid_wids) > 50:
        step = max(1, len(valid_wids) // 20)
        tick_positions = list(range(0, len(valid_wids), step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(valid_wids[i]) for i in tick_positions], rotation=45, fontsize=8)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in valid_wids], rotation=45, fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rf_accuracy_by_housing.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"按Housing分类的准确度图已保存到: {output_path}")


def plot_cv_fold_scores(cv_scores: list, cv_fold_info: list, 
                        output_dir: str, cv_method: str = 'stratified_kfold'):
    """
    绘制所有fold的准确率变化图
    - 柱状图展示每个fold的准确率
    - 显示平均值和标准差
    - 用不同颜色标记不同fold的变化趋势
    
    Plot all CV fold accuracy scores with visualization
    """
    fold_numbers = [f['fold'] for f in cv_fold_info]
    fold_accs = [f['accuracy'] for f in cv_fold_info]
    
    n_folds = len(cv_fold_info)
    mean_acc = np.mean(cv_scores)
    std_acc = np.std(cv_scores)
    
    # 图1: 柱状图 + 线图
    fig, ax = plt.subplots(figsize=(max(12, n_folds * 0.4), 7))
    
    # 根据准确率给柱子着色
    colors = ['#2ecc71' if acc >= mean_acc else '#e74c3c' for acc in fold_accs]
    bars = ax.bar(range(n_folds), fold_accs, color=colors, alpha=0.8, width=0.8, 
                   edgecolor='black', linewidth=1)
    
    # 添加准确率数值标注
    for i, (fold_num, acc) in enumerate(zip(fold_numbers, fold_accs)):
        ax.text(i, acc + 0.01, f'{acc:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 添加均值线
    ax.axhline(y=mean_acc, color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_acc:.4f}')
    
    # 添加±1标准差区域
    ax.fill_between(range(n_folds), mean_acc - std_acc, mean_acc + std_acc, 
                     alpha=0.2, color='blue', label=f'±1 Std: {std_acc:.4f}')
    
    # 设置x轴标签
    if cv_method == 'leave_one_pig_out':
        # 显示猪的ID
        if 'test_pig' in cv_fold_info[0]:
            labels = [f['test_pig'] for f in cv_fold_info]
            ax.set_xticks(range(n_folds))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.set_xlabel('Left-Out Pig ID', fontsize=12)
        else:
            ax.set_xticks(range(n_folds))
            ax.set_xticklabels(fold_numbers, rotation=45, fontsize=9)
            ax.set_xlabel('Fold', fontsize=12)
    else:
        ax.set_xticks(range(n_folds))
        ax.set_xticklabels(fold_numbers, rotation=45, fontsize=9)
        ax.set_xlabel('Fold', fontsize=12)
    
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Random Forest: CV Fold Balanced Accuracy ({cv_method})', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'rf_cv_fold_scores_{cv_method}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"CV Fold准确率图已保存到: {output_path}")
    
    # 图2: 线图 - 显示准确率变化趋势
    fig, ax = plt.subplots(figsize=(max(12, n_folds * 0.4), 7))
    
    ax.plot(range(n_folds), fold_accs, marker='o', markersize=8, 
            linewidth=2, color='steelblue', label='Fold accuracy')
    
    # 添加均值线
    ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_acc:.4f}')
    ax.fill_between(range(n_folds), mean_acc - std_acc, mean_acc + std_acc, 
                     alpha=0.2, color='red', label=f'±1 Std: {std_acc:.4f}')
    
    # 标注最高和最低准确率的fold
    max_idx = np.argmax(fold_accs)
    min_idx = np.argmin(fold_accs)
    ax.scatter([max_idx], [fold_accs[max_idx]], s=200, marker='*', 
              color='green', edgecolors='black', linewidth=2, 
              label=f'Best: {fold_accs[max_idx]:.4f}', zorder=5)
    ax.scatter([min_idx], [fold_accs[min_idx]], s=200, marker='v', 
              color='red', edgecolors='black', linewidth=2,
              label=f'Worst: {fold_accs[min_idx]:.4f}', zorder=5)
    
    if cv_method == 'leave_one_pig_out':
        if 'test_pig' in cv_fold_info[0]:
            labels = [f['test_pig'] for f in cv_fold_info]
            ax.set_xticks(range(n_folds))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.set_xlabel('Left-Out Pig ID', fontsize=12)
        else:
            ax.set_xticks(range(n_folds))
            ax.set_xticklabels(fold_numbers, rotation=45, fontsize=9)
            ax.set_xlabel('Fold', fontsize=12)
    else:
        ax.set_xticks(range(n_folds))
        ax.set_xticklabels(fold_numbers, rotation=45, fontsize=9)
        ax.set_xlabel('Fold', fontsize=12)
    
    ax.set_ylabel('Balanced Accuracy', fontsize=12)
    ax.set_ylim(min(fold_accs) - 0.05, max(fold_accs) + 0.05)
    ax.set_title(f'Random Forest: CV Fold Balanced Accuracy Trend ({cv_method})', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'rf_cv_fold_trend_{cv_method}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"CV Fold趋势图已保存到: {output_path}")
    
    # 保存详细信息到CSV
    fold_info_df = pd.DataFrame(cv_fold_info)
    csv_path = os.path.join(output_dir, f'rf_cv_fold_info_{cv_method}.csv')
    fold_info_df.to_csv(csv_path, index=False)
    print(f"CV Fold详细信息已保存到: {csv_path}")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"CV Fold 平衡准确率统计 ({cv_method})")
    print(f"{'='*60}")
    print(f"平均平衡准确率: {mean_acc:.4f}")
    print(f"标准差: {std_acc:.4f}")
    print(f"最高平衡准确率: {np.max(cv_scores):.4f} (Fold {fold_numbers[max_idx]})")
    print(f"最低平衡准确率: {np.min(cv_scores):.4f} (Fold {fold_numbers[min_idx]})")
    print(f"准确率范围: [{np.min(cv_scores):.4f}, {np.max(cv_scores):.4f}]")
    print(f"方差系数 (CV): {std_acc/mean_acc if mean_acc > 0 else 0:.4f}")


# ============================================================================
# 新功能：最优预测窗口分析
# New Feature: Best prediction window analysis
# ============================================================================

def analyze_best_prediction_window(df: pd.DataFrame,
                                   n_estimators: int = 200,
                                   max_depth: Optional[int] = None,
                                   random_seed: int = RANDOM_SEED,
                                   n_folds: int = 5,
                                   output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    分析最优预测窗口：对于在不同 window_id 结束的 trial，
    找出哪个 window_id 的 embedding 能给出最佳 decision 预测准确率。
    
    Best prediction window analysis: For trials ending at different window_ids,
    find which window_id's embedding gives the best prediction accuracy.
    
    思路 (Approach):
    1. 按 trial 的最大 window_id (max_wid) 分组
    2. 对于每个 max_wid 组，在每个 window_id <= max_wid 上训练/评估 RF
    3. 找出对该组 trial 预测最准确的 window_id
    4. 输出总体和按 decision 分别的最优窗口统计
    
    参数 Parameters:
        df: 包含 embedding、decision、window_id、id 的 DataFrame
        n_estimators: RF 树的数量
        max_depth: 树最大深度
        random_seed: 随机种子
        n_folds: 交叉验证折数
        output_dir: 输出目录
    
    返回 Returns:
        包含分析结果的 DataFrame
    """
    emb_cols = [c for c in df.columns if c.startswith('emb_')]
    
    print("\n" + "=" * 60)
    print("最优预测窗口分析 (Best Prediction Window Analysis)")
    print("=" * 60)
    
    # 计算每个 trial 的最大 window_id
    max_wid_per_trial = df.groupby('id')['window_id'].max().reset_index()
    max_wid_per_trial.columns = ['id', 'max_wid']
    
    # 合并回 df
    df_with_max = df.merge(max_wid_per_trial, on='id', how='left')
    
    # 选择有足够 trial 数量的 max_wid 进行分析
    max_wid_counts = max_wid_per_trial['max_wid'].value_counts().sort_index()
    # 只分析 trial 数 >= 10 的 max_wid 组
    valid_max_wids = max_wid_counts[max_wid_counts >= 10].index.tolist()
    valid_max_wids = sorted(valid_max_wids)
    
    print(f"有效的 max_wid 组数: {len(valid_max_wids)} (每组至少10个trial)")
    print(f"max_wid 范围: {valid_max_wids[0]} - {valid_max_wids[-1]}" if valid_max_wids else "无有效组")
    
    # 用全部数据训练一个全局 RF 模型（用 CV 产生预测）
    # Train a global RF model with CV to get predictions
    X_all = df[emb_cols].values
    y_all = df['decision'].values
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    cv_predictions = np.full(len(df), -1, dtype=int)
    cv_probas = np.full(len(df), np.nan, dtype=float)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
            n_jobs=-1,
            class_weight='balanced'
        )
        rf.fit(X_all[train_idx], y_all[train_idx])
        cv_predictions[val_idx] = rf.predict(X_all[val_idx])
        cv_probas[val_idx] = rf.predict_proba(X_all[val_idx])[:, 1]  # P(decision=1)
    
    df_with_max = df_with_max.copy()
    df_with_max['cv_pred'] = cv_predictions
    df_with_max['cv_proba'] = cv_probas
    
    # ========== 对每个 max_wid 组，分析每个 window_id 的预测准确率 ==========
    results = []
    
    for max_wid in valid_max_wids:
        # 获取这组 trial 的所有 id
        trial_ids = max_wid_per_trial[max_wid_per_trial['max_wid'] == max_wid]['id'].values
        group_data = df_with_max[df_with_max['id'].isin(trial_ids)]
        n_trials = len(trial_ids)
        
        # 对每个 window_id <= max_wid 计算准确率
        for wid in range(0, max_wid + 1):
            wid_data = group_data[group_data['window_id'] == wid]
            
            if len(wid_data) < 5:
                continue
            
            # 平衡准确率
            overall_acc = balanced_accuracy_score(wid_data['decision'], wid_data['cv_pred'])
            
            # 按 decision 分别计算 recall
            d0_data = wid_data[wid_data['decision'] == 0]
            d1_data = wid_data[wid_data['decision'] == 1]
            
            recall_d0 = recall_score(d0_data['decision'], d0_data['cv_pred'], pos_label=0) if len(d0_data) >= 2 else np.nan
            recall_d1 = recall_score(d1_data['decision'], d1_data['cv_pred'], pos_label=1) if len(d1_data) >= 2 else np.nan
            
            results.append({
                'max_wid': max_wid,
                'window_id': wid,
                'n_trials': n_trials,
                'n_samples': len(wid_data),
                'n_d0': len(d0_data),
                'n_d1': len(d1_data),
                'balanced_accuracy': overall_acc,
                'recall_d0': recall_d0,
                'recall_d1': recall_d1,
            })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("警告: 没有足够的数据进行最优窗口分析")
        return pd.DataFrame()
    
    # ========== 找出每个 max_wid 组的最优 window_id ==========
    best_windows = []
    for max_wid in valid_max_wids:
        group = results_df[results_df['max_wid'] == max_wid]
        if len(group) == 0:
            continue
        
        # 总体最优
        best_row = group.loc[group['balanced_accuracy'].idxmax()]
        
        # decision=0 最优
        d0_valid = group.dropna(subset=['recall_d0'])
        best_d0_wid = d0_valid.loc[d0_valid['recall_d0'].idxmax(), 'window_id'] if len(d0_valid) > 0 else np.nan
        best_d0_acc = d0_valid['recall_d0'].max() if len(d0_valid) > 0 else np.nan
        
        # decision=1 最优
        d1_valid = group.dropna(subset=['recall_d1'])
        best_d1_wid = d1_valid.loc[d1_valid['recall_d1'].idxmax(), 'window_id'] if len(d1_valid) > 0 else np.nan
        best_d1_acc = d1_valid['recall_d1'].max() if len(d1_valid) > 0 else np.nan
        
        best_windows.append({
            'max_wid': max_wid,
            'n_trials': int(best_row['n_trials']),
            'best_overall_wid': int(best_row['window_id']),
            'best_overall_acc': best_row['balanced_accuracy'],
            'best_d0_wid': int(best_d0_wid) if not np.isnan(best_d0_wid) else np.nan,
            'best_d0_recall': best_d0_acc,
            'best_d1_wid': int(best_d1_wid) if not np.isnan(best_d1_wid) else np.nan,
            'best_d1_recall': best_d1_acc,
        })
    
    best_df = pd.DataFrame(best_windows)
    
    print(f"\n最优预测窗口汇总 (Best Prediction Window Summary):")
    print(best_df.to_string(index=False))
    
    # ========== 可视化 ==========
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存 CSV
        results_df.to_csv(os.path.join(output_dir, 'rf_best_window_details.csv'), index=False)
        best_df.to_csv(os.path.join(output_dir, 'rf_best_window_summary.csv'), index=False)
        print(f"详细结果已保存到: rf_best_window_details.csv")
        print(f"汇总结果已保存到: rf_best_window_summary.csv")
        
        # 可视化1: 静态图 - 最优窗口 vs trial终止窗口
        _plot_best_window_static(best_df, results_df, output_dir)
        
        # 可视化2: 交互式 Plotly 热力图 - 每个 (max_wid, window_id) 的准确率
        _plot_best_window_heatmap_html(results_df, output_dir)
        
        # 可视化3: 交互式 Plotly 折线图 - 准确率随 window_id 的变化
        _plot_best_window_lines_html(results_df, valid_max_wids, output_dir)
    
    return results_df


def _plot_best_window_static(best_df: pd.DataFrame, results_df: pd.DataFrame,
                             output_dir: str):
    """
    静态图：最优预测窗口 vs trial终止窗口
    Static plot: Best prediction window vs trial end window
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 图1: 最优window_id vs max_wid（总体）
    ax = axes[0, 0]
    ax.scatter(best_df['max_wid'], best_df['best_overall_wid'], 
               c='steelblue', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.plot([0, best_df['max_wid'].max()], [0, best_df['max_wid'].max()], 
            'r--', alpha=0.5, label='y = x (use last window)')
    ax.set_xlabel('Trial End Window (max_wid)', fontsize=11)
    ax.set_ylabel('Best Prediction Window ID', fontsize=11)
    ax.set_title('Best Overall Prediction Window vs Trial Length', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图2: 最优窗口的准确率 vs max_wid
    ax = axes[0, 1]
    ax.bar(best_df['max_wid'], best_df['best_overall_acc'], 
           color='steelblue', alpha=0.7, width=0.8)
    mean_acc = best_df['best_overall_acc'].mean()
    ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=1.2,
               label=f'Mean = {mean_acc:.4f}')
    ax.set_xlabel('Trial End Window (max_wid)', fontsize=11)
    ax.set_ylabel('Best Accuracy', fontsize=11)
    ax.set_title('Best Achievable Accuracy per Trial Group', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图3: Decision=0 vs Decision=1 最优窗口位置对比
    ax = axes[1, 0]
    w = 0.35
    x_pos = np.arange(len(best_df))
    d0_wids = best_df['best_d0_wid'].values
    d1_wids = best_df['best_d1_wid'].values
    ax.bar(x_pos - w/2, d0_wids, w, color='#e74c3c', alpha=0.7, label='Decision=0')
    ax.bar(x_pos + w/2, d1_wids, w, color='#2ecc71', alpha=0.7, label='Decision=1')
    if len(best_df) <= 30:
        ax.set_xticks(x_pos)
        ax.set_xticklabels(best_df['max_wid'].values, rotation=45, fontsize=8)
    else:
        step = max(1, len(best_df) // 15)
        tick_pos = list(range(0, len(best_df), step))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([best_df.iloc[i]['max_wid'] for i in tick_pos], rotation=45, fontsize=8)
    ax.set_xlabel('Trial End Window (max_wid)', fontsize=11)
    ax.set_ylabel('Best Window ID', fontsize=11)
    ax.set_title('Best Window for Decision=0 vs Decision=1', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 图4: 最优窗口位置相对于trial长度的比例
    ax = axes[1, 1]
    ratio_overall = best_df['best_overall_wid'] / best_df['max_wid'].replace(0, np.nan)
    ax.hist(ratio_overall.dropna(), bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    mean_ratio = ratio_overall.dropna().mean()
    ax.axvline(x=mean_ratio, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean ratio = {mean_ratio:.2f}')
    ax.set_xlabel('Best Window Position / Trial Length', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Best Window Position (relative)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Best Prediction Window Analysis', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'rf_best_prediction_window.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"最优预测窗口分析图已保存到: {output_path}")


def _plot_best_window_heatmap_html(results_df: pd.DataFrame, output_dir: str):
    """
    交互式 Plotly 热力图：(max_wid, window_id) → accuracy
    Interactive Plotly heatmap: accuracy for each (max_wid, window_id) combination
    """
    # 构建 pivot table
    pivot = results_df.pivot_table(
        index='max_wid', columns='window_id', 
        values='balanced_accuracy', aggfunc='first'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale='RdYlGn',
        zmin=0.5,
        zmax=1.0,
        colorbar=dict(title='Accuracy'),
        hovertemplate='max_wid=%{y}<br>window_id=%{x}<br>accuracy=%{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Prediction Accuracy Heatmap: Trial End Window × Prediction Window',
        xaxis_title='Prediction Window ID',
        yaxis_title='Trial End Window (max_wid)',
        width=max(800, len(pivot.columns) * 30),
        height=max(600, len(pivot.index) * 25),
    )
    
    output_path = os.path.join(output_dir, 'rf_best_window_heatmap.html')
    fig.write_html(output_path)
    print(f"交互式热力图已保存到: {output_path}")


def _plot_best_window_lines_html(results_df: pd.DataFrame, 
                                  valid_max_wids: list,
                                  output_dir: str):
    """
    交互式 Plotly 折线图：对于每个 max_wid 组，显示准确率随 window_id 变化
    Interactive Plotly line chart: accuracy vs window_id for each max_wid group
    """
    # 选择有代表性的 max_wid 组（均匀采样 + 保证覆盖关键点）
    if len(valid_max_wids) > 15:
        # 采样一些代表性的组
        indices = np.linspace(0, len(valid_max_wids) - 1, 12, dtype=int)
        selected_max_wids = sorted(set([valid_max_wids[i] for i in indices]))
    else:
        selected_max_wids = valid_max_wids
    
    fig = go.Figure()
    
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    
    for i, max_wid in enumerate(selected_max_wids):
        group = results_df[results_df['max_wid'] == max_wid].sort_values('window_id')
        
        if len(group) < 2:
            continue
        
        color = colors[i % len(colors)]
        n_trials = group.iloc[0]['n_trials']
        
        # 总体准确率线
        fig.add_trace(go.Scatter(
            x=group['window_id'],
            y=group['balanced_accuracy'],
            mode='lines+markers',
            name=f'max_wid={max_wid} (n={n_trials})',
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=(
                f'max_wid={max_wid}<br>'
                'window_id=%{x}<br>'
                'accuracy=%{y:.4f}<br>'
                '<extra></extra>'
            )
        ))
        
        # 标记最优点
        best_idx = group['balanced_accuracy'].idxmax()
        best_row = group.loc[best_idx]
        fig.add_trace(go.Scatter(
            x=[best_row['window_id']],
            y=[best_row['balanced_accuracy']],
            mode='markers',
            marker=dict(color=color, size=14, symbol='star', 
                        line=dict(color='black', width=1)),
            showlegend=False,
            hovertemplate=(
                f'★ BEST for max_wid={max_wid}<br>'
                f'window_id={int(best_row["window_id"])}<br>'
                f'balanced_accuracy={best_row["balanced_accuracy"]:.4f}<br>'
                '<extra></extra>'
            )
        ))
    
    fig.update_layout(
        title='Prediction Accuracy vs Window ID for Different Trial Lengths<br>'
              '<sub>★ = Best prediction window for each group</sub>',
        xaxis_title='Prediction Window ID',
        yaxis_title='Overall Accuracy',
        yaxis=dict(range=[0.4, 1.05]),
        width=1100,
        height=700,
        legend=dict(title='Trial End Window', font=dict(size=10)),
        hovermode='closest'
    )
    
    output_path = os.path.join(output_dir, 'rf_best_window_lines.html')
    fig.write_html(output_path)
    print(f"交互式折线图已保存到: {output_path}")
    
    # ===== 额外：按 decision 分类的折线图 =====
    fig2 = go.Figure()
    
    for i, max_wid in enumerate(selected_max_wids):
        group = results_df[results_df['max_wid'] == max_wid].sort_values('window_id')
        if len(group) < 2:
            continue
        
        color = colors[i % len(colors)]
        n_trials = group.iloc[0]['n_trials']
        
        # Decision=0 recall（虚线）
        d0_valid = group.dropna(subset=['recall_d0'])
        if len(d0_valid) > 0:
            fig2.add_trace(go.Scatter(
                x=d0_valid['window_id'],
                y=d0_valid['recall_d0'],
                mode='lines+markers',
                name=f'max_wid={max_wid} D=0',
                line=dict(color=color, width=1.5, dash='dash'),
                marker=dict(size=4, symbol='circle'),
                legendgroup=f'mw{max_wid}',
                hovertemplate=f'max_wid={max_wid} (D=0)<br>wid=%{{x}}<br>recall=%{{y:.4f}}<extra></extra>'
            ))
        
        # Decision=1 recall（实线）
        d1_valid = group.dropna(subset=['recall_d1'])
        if len(d1_valid) > 0:
            fig2.add_trace(go.Scatter(
                x=d1_valid['window_id'],
                y=d1_valid['recall_d1'],
                mode='lines+markers',
                name=f'max_wid={max_wid} D=1',
                line=dict(color=color, width=2),
                marker=dict(size=5, symbol='diamond'),
                legendgroup=f'mw{max_wid}',
                hovertemplate=f'max_wid={max_wid} (D=1)<br>wid=%{{x}}<br>recall=%{{y:.4f}}<extra></extra>'
            ))
    
    fig2.update_layout(
        title='Per-Decision Recall vs Window ID for Different Trial Lengths<br>'
              '<sub>Solid=Decision=1, Dashed=Decision=0</sub>',
        xaxis_title='Prediction Window ID',
        yaxis_title='Recall',
        yaxis=dict(range=[0, 1.1]),
        width=1100,
        height=700,
        legend=dict(title='Trial Group × Decision', font=dict(size=9)),
        hovermode='closest'
    )
    
    output_path2 = os.path.join(output_dir, 'rf_best_window_by_decision.html')
    fig2.write_html(output_path2)
    print(f"按Decision分类的交互式折线图已保存到: {output_path2}")


def main():
    """主函数入口 (Main entry point)"""
    print("=" * 60)
    print("随机森林分类器 - 基于Embedding预测Decision")
    print("Random Forest Classifier - Predict Decision from Embeddings")
    print("=" * 60)

    # 加载数据
    df = load_and_clean_embeddings(OUTPUT_BASE_DIR)

    # 基于trial时长分布过滤window_id
    # 使用90百分位截断，覆盖90%的trial的完整时长范围
    df_filtered = filter_window_ids_by_trial_duration(df, trial_percentile=90.0)

    # 使用Leave-One-Pig-Out交叉验证进行训练和评估
    # 这提供了更真实的泛化性能评估
    print("\n" + "=" * 60)
    print("使用 Leave-One-Pig-Out 交叉验证进行训练...")
    print("=" * 60)
    results = train_random_forest(
        df_filtered,
        n_estimators=200,
        max_depth=None,
        n_folds=5,  # 此参数在LOPO中被忽略
        output_dir=OUTPUT_BASE_DIR,
        cv_method='leave_one_pig_out'  # 使用Leave-One-Pig-Out
    )

    # ===== 新功能：最优预测窗口分析 =====
    # 使用过滤后的数据，分析不同trial终点对应的最佳预测窗口
    print("\n" + "=" * 60)
    print("开始最优预测窗口分析...")
    print("=" * 60)
    best_window_results = analyze_best_prediction_window(
        df_filtered,
        n_estimators=200,
        max_depth=None,
        n_folds=5,
        output_dir=OUTPUT_BASE_DIR
    )

    print("\n" + "=" * 60)
    print("完成！(Done!)")
    print(f"交叉验证准确度: {np.mean(results['cv_scores']):.4f} ± {np.std(results['cv_scores']):.4f}")
    print(f"有效窗口数: {len(results['window_accuracies'])}")
    if len(best_window_results) > 0:
        print(f"最优窗口分析涵盖组数: {best_window_results['max_wid'].nunique()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
