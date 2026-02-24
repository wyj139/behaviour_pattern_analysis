#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window ID 独立信息量分析
Per-Window Decision Information Analysis

目标: 通过对每个window_id单独做LOPO-CV，量化该时间窗口含有多少决策信息。
核心思想: 只用该window的embedding训练一个RF，balanced_accuracy越高说明该窗口的
         运动模式越能区分GO/NOGO决策，即含有更多决策信息。

设计原则:
1. 每个window独立训练，互不干扰 → 避免模型学到"跨窗口整体模式"
2. LOPO-CV (Leave-One-Pig-Out) → 保证泛化性，评估模型能否在新个体上工作
3. 按window_id顺序分析 → 揭示决策信息随时间的演变
4. 内存高效：每次只处理一个window，及时释放内存

输出:
- window_info_results.csv  : 每个window的balanced accuracy及置信区间
- window_info_analysis.png : 综合可视化（含误差棒、rolling average、by-decision分析）
"""

import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score
import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_BASE_DIR

# ─── 配置 ────────────────────────────────────────────────────────────────────

# 只分析样本量≥MIN_SAMPLES的window（太少的样本LOPO无意义）
MIN_SAMPLES = 50
# 每个class至少需要MIN_CLASS_SAMPLES个样本
MIN_CLASS_SAMPLES = 5
# RF参数（轻量，每个window单独训练，不需要太深）
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 6
RF_RANDOM_STATE = 42

plt.style.use('seaborn-v0_8-darkgrid')


# ─── 数据加载（只加载必要的列，避免内存溢出）─────────────────────────────────

def _embeddings_csv_is_readable(path):
    """检查 embeddings.csv 是否为真实文件（非 OneDrive 占位符）"""
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'r') as f:
            first_line = f.readline()
        return len(first_line.strip()) > 0
    except Exception:
        return False


def _generate_embeddings_from_model(base):
    """
    用已训练的 autoencoder 做推理（纯前向传播，无需重新训练），
    生成 embeddings 并保存为 embeddings.csv，返回 DataFrame。
    """
    import torch
    from autoencoder import ConvAutoencoder, extract_embeddings
    from config import WINDOW_SIZE, EMBEDDING_DIM

    model_path = os.path.join(base, 'autoencoder_model.pth')
    print(f"  加载已训练模型: {model_path}")

    # 先读一行确认 n_bodyparts
    sample = pd.read_csv(os.path.join(base, 'window_data1.csv'), nrows=1)
    if 'n_bodyparts' in sample.columns:
        n_bodyparts = int(sample.iloc[0]['n_bodyparts'])
    else:
        import ast
        n_bodyparts = len(ast.literal_eval(sample.iloc[0]['all_window_rel_x']))
    print(f"  n_bodyparts = {n_bodyparts}")

    device = torch.device('cpu')
    model = ConvAutoencoder(window_size=WINDOW_SIZE, n_bodyparts=n_bodyparts, emb_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("  ✓ 模型加载完成，开始推理（前向传播，约1-3分钟）...")

    # 加载完整 window_data 用于推理（含轨迹坐标）
    window_df = pd.read_csv(os.path.join(base, 'window_data1.csv'))
    print(f"  推理数据: {len(window_df)} 个窗口")

    # window_data1.csv 的轨迹列是字符串格式：各 bodypart 用 ";" 分隔，每个 bodypart 内帧值用 "," 分隔
    # 需要先解析成 list of np.array，才能被 TrajDataset 正常处理
    def parse_traj_col(col):
        def parse_item(x):
            if isinstance(x, str):
                return [np.array(seg.split(','), dtype=float) for seg in x.split(';')]
            return x  # 已经是 list 则不处理
        return col.apply(parse_item)

    print("  解析轨迹字符串列...")
    window_df['all_window_rel_x'] = parse_traj_col(window_df['all_window_rel_x'])
    window_df['all_window_rel_y'] = parse_traj_col(window_df['all_window_rel_y'])

    emb_df = extract_embeddings(model, window_df, window_size=WINDOW_SIZE, output_dir=base)
    del model, window_df
    gc.collect()
    print("  ✓ embeddings 推理完成，已保存到 embeddings.csv")
    return emb_df


def load_data():
    """加载 window_id/decision/pig 元数据 + embeddings（仅 emb_* 列）。
    若 embeddings.csv 是 OneDrive 占位符，则自动用已训练模型推理生成。"""
    base = OUTPUT_BASE_DIR
    emb_path = os.path.join(base, 'embeddings.csv')

    print("加载元数据...")
    meta = pd.read_csv(
        os.path.join(base, 'window_data1.csv'),
        usecols=['window_id', 'decision', 'pig', 'Housing']
    )
    print(f"  元数据: {len(meta)} 行, pigs: {sorted(meta.pig.unique())}")

    if _embeddings_csv_is_readable(emb_path):
        print("加载 embeddings (仅 emb_* 列)...")
        emb_raw = pd.read_csv(emb_path)
    else:
        print("embeddings.csv 不可用（OneDrive 占位符），使用已训练模型推理生成...")
        emb_raw = _generate_embeddings_from_model(base)

    emb_cols = [c for c in emb_raw.columns if c.startswith('emb_')]

    # window_id 在不同 trial 中重复，不能用 merge —— 直接按行位置对齐
    assert len(emb_raw) == len(meta), \
        f"embeddings 行数 ({len(emb_raw)}) 与 window_data 行数 ({len(meta)}) 不匹配！"
    df = meta.copy().reset_index(drop=True)
    df[emb_cols] = emb_raw[emb_cols].values
    del emb_raw
    gc.collect()

    df = df.dropna(subset=emb_cols[:1])
    print(f"  合并后: {len(df)} 行, {len(emb_cols)} 个 embedding 维度")
    return df, emb_cols


# ─── 单window LOPO-CV ────────────────────────────────────────────────────────

def lopo_cv_single_window(window_df, emb_cols):
    """
    对单个window的数据做LOPO-CV（Leave-One-Pig-Out）
    
    返回:
        mean_balanced_acc: float  各fold的平均balanced accuracy
        std_balanced_acc : float  标准差（不确定性估计）
        recall_nogo      : float  NOGO类（少数类）的平均recall 
        recall_go        : float  GO类的平均recall
        n_folds_valid    : int    有效fold数
    """
    pigs = window_df['pig'].unique()
    X = window_df[emb_cols].values
    y = window_df['decision'].values
    pig_ids = window_df['pig'].values
    
    fold_results = []
    
    for pig in pigs:
        # 用该pig作为测试集，其余作为训练集
        test_mask = (pig_ids == pig)
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # 跳过测试集中类别不全的fold
        if len(np.unique(y_test)) < 2:
            continue
        # 跳过训练集中类别不全的fold
        if len(np.unique(y_train)) < 2:
            continue
        # 训练集中每类至少需要一定样本
        if min(np.bincount(y_train)) < 2:
            continue
        
        rf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RF_RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        ba = balanced_accuracy_score(y_test, y_pred)
        # recall per class
        classes_in_test = np.unique(y_test)
        rec_nogo = recall_score(y_test, y_pred, labels=[0], average=None, zero_division=0)
        rec_go   = recall_score(y_test, y_pred, labels=[1], average=None, zero_division=0)
        rec_nogo = rec_nogo[0] if len(rec_nogo) > 0 else np.nan
        rec_go   = rec_go[0]   if len(rec_go)   > 0 else np.nan
        
        fold_results.append({
            'balanced_acc': ba,
            'recall_nogo': rec_nogo,
            'recall_go': rec_go
        })
        
        del rf
    
    if len(fold_results) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    
    bas  = [r['balanced_acc'] for r in fold_results]
    rngs = [r['recall_nogo']  for r in fold_results]
    rgs  = [r['recall_go']    for r in fold_results]
    
    return (
        np.nanmean(bas),
        np.nanstd(bas),
        np.nanmean(rngs),
        np.nanmean(rgs),
        len(fold_results)
    )


# ─── 主分析函数 ──────────────────────────────────────────────────────────────

def run_analysis():
    print("\n" + "="*65)
    print("Window独立决策信息量分析 (LOPO-CV)")
    print("="*65)
    
    output_dir = os.path.join(OUTPUT_BASE_DIR, 'window_independent_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    df, emb_cols = load_data()
    
    # 确定要分析的windows（按样本量过滤）
    window_stats = df.groupby('window_id').agg(
        n_total   = ('decision', 'count'),
        n_nogo    = ('decision', lambda x: (x == 0).sum()),
        n_go      = ('decision', lambda x: (x == 1).sum()),
        n_pigs    = ('pig', 'nunique')
    ).reset_index()
    
    # 过滤条件：总样本≥MIN_SAMPLES，两类各≥MIN_CLASS_SAMPLES，pig数≥2
    valid_windows = window_stats[
        (window_stats.n_total >= MIN_SAMPLES) &
        (window_stats.n_nogo  >= MIN_CLASS_SAMPLES) &
        (window_stats.n_go    >= MIN_CLASS_SAMPLES) &
        (window_stats.n_pigs  >= 2)
    ]['window_id'].tolist()
    
    print(f"\n有效window数 (n_total≥{MIN_SAMPLES}, 每类≥{MIN_CLASS_SAMPLES}): {len(valid_windows)}")
    print(f"Window范围: {min(valid_windows)} ~ {max(valid_windows)}")
    
    # 逐window训练
    results = []
    total = len(valid_windows)
    
    for i, wid in enumerate(sorted(valid_windows)):
        wdf = df[df['window_id'] == wid].copy()
        n_nogo = (wdf.decision == 0).sum()
        n_go   = (wdf.decision == 1).sum()
        
        mean_ba, std_ba, rec_nogo, rec_go, n_folds = lopo_cv_single_window(wdf, emb_cols)
        
        results.append({
            'window_id':    wid,
            'n_total':      len(wdf),
            'n_nogo':       n_nogo,
            'n_go':         n_go,
            'n_pigs':       wdf.pig.nunique(),
            'n_folds':      n_folds,
            'balanced_acc': mean_ba,
            'ba_std':       std_ba,
            'recall_nogo':  rec_nogo,
            'recall_go':    rec_go,
        })
        
        # 进度
        if (i + 1) % 5 == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] w{wid}: BA={mean_ba:.3f}±{std_ba:.3f} "
                  f"(n={len(wdf)}, NOGO={n_nogo}, GO={n_go}, folds={n_folds})")
        
        del wdf
        gc.collect()
    
    results_df = pd.DataFrame(results)
    
    # 保存CSV
    csv_path = os.path.join(output_dir, 'window_info_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ 结果已保存: window_info_results.csv")
    
    # 绘图
    plot_results(results_df, output_dir)
    
    # 清理临时文件
    tmp = os.path.join(os.path.dirname(OUTPUT_BASE_DIR), '..', '_tmp_explore.py')
    if os.path.exists(tmp):
        os.remove(tmp)
    
    print("\n" + "="*65)
    print("✓ 分析完成！")
    print(f"输出目录: {output_dir}")
    print("="*65)
    
    return results_df


# ─── 可视化 ──────────────────────────────────────────────────────────────────

def plot_results(df, output_dir):
    print("\n绘制图表...")
    
    wids = df['window_id'].values
    ba   = df['balanced_acc'].values
    std  = df['ba_std'].values
    r_nogo = df['recall_nogo'].values
    r_go   = df['recall_go'].values
    
    # rolling average (窗口=5)
    roll_ba = pd.Series(ba).rolling(window=5, center=True, min_periods=1).mean().values
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle('Per-Window Decision Information (LOPO-CV Balanced Accuracy)',
                 fontsize=14, fontweight='bold', y=1.01)
    
    # ── 图1: Balanced Accuracy with error bars & rolling mean ──────────────
    ax = axes[0, 0]
    ax.fill_between(wids, ba - std, ba + std, alpha=0.2, color='steelblue', label='±1 SD')
    ax.plot(wids, ba, 'o', color='steelblue', alpha=0.5, markersize=4)
    ax.plot(wids, roll_ba, '-', color='navy', linewidth=2.5, label='Rolling mean (w=5)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Chance (0.5)')
    ax.set_xlabel('Window ID', fontsize=11)
    ax.set_ylabel('Balanced Accuracy (LOPO-CV)', fontsize=11)
    ax.set_title('Decision-Info by Window\n(higher = more decision info in this window)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim([0.3, 1.05])
    
    # ── 图2: Recall NOGO vs GO by window ───────────────────────────────────
    ax = axes[0, 1]
    ax.plot(wids, r_nogo, 'o-', color='coral',      linewidth=1.5, markersize=4,
            alpha=0.8, label='Recall NOGO (minority)')
    ax.plot(wids, r_go,   's-', color='steelblue',  linewidth=1.5, markersize=4,
            alpha=0.8, label='Recall GO (majority)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Window ID', fontsize=11)
    ax.set_ylabel('Recall', fontsize=11)
    ax.set_title('Per-Class Recall by Window\n(NOGO recall ↑ = model can detect GO→NOGO transitions)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim([0, 1.05])
    
    # ── 图3: 样本量分布（NOGO vs GO堆叠条形）──────────────────────────────
    ax = axes[1, 0]
    ax.bar(wids, df['n_nogo'], color='coral',     alpha=0.8, label='NOGO', width=0.8)
    ax.bar(wids, df['n_go'],   bottom=df['n_nogo'], color='steelblue', alpha=0.8,
           label='GO', width=0.8)
    ax.set_xlabel('Window ID', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('Sample Distribution (NOGO + GO per Window)', fontsize=11)
    ax.legend(fontsize=9)
    
    # ── 图4: BA vs log(n_total) 散点，颜色代表NOGO比例 ─────────────────────
    ax = axes[1, 1]
    nogo_ratio = df['n_nogo'] / df['n_total']
    sc = ax.scatter(np.log10(df['n_total']), ba,
                    c=nogo_ratio, cmap='RdYlGn', alpha=0.7, s=40,
                    vmin=0, vmax=0.5)
    plt.colorbar(sc, ax=ax, label='NOGO ratio')
    ax.set_xlabel('log₁₀(n_samples per window)', fontsize=11)
    ax.set_ylabel('Balanced Accuracy', fontsize=11)
    ax.set_title('BA vs Sample Size\n(color = NOGO fraction)', fontsize=11)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out = os.path.join(output_dir, 'window_info_analysis.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✓ window_info_analysis.png")
    
    # ── 补充图: 最佳/最差window的详细信息 ────────────────────────────────
    top_n = min(10, len(df))
    top_df = df.nlargest(top_n, 'balanced_acc')
    bot_df = df.nsmallest(top_n, 'balanced_acc')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Top / Bottom Windows by Decision Information', fontsize=13, fontweight='bold')
    
    for ax, sub, title, color in [
        (axes[0], top_df, f'Top {top_n} Windows (Most Decision Info)', 'forestgreen'),
        (axes[1], bot_df, f'Bottom {top_n} Windows (Least Decision Info)', 'tomato')
    ]:
        bars = ax.barh(sub['window_id'].astype(str), sub['balanced_acc'],
                       color=color, alpha=0.7, xerr=sub['ba_std'], capsize=4)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Chance')
        ax.set_xlabel('Balanced Accuracy', fontsize=11)
        ax.set_ylabel('Window ID', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xlim([0, 1.05])
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'n={int(row.n_total)}, NOGO={int(row.n_nogo)}',
                    va='center', fontsize=7)
    
    plt.tight_layout()
    out2 = os.path.join(output_dir, 'window_info_top_bottom.png')
    plt.savefig(out2, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  ✓ window_info_top_bottom.png")


if __name__ == '__main__':
    results = run_analysis()
    print("\n最高决策信息量的前10个windows:")
    print(results.nlargest(10, 'balanced_acc')[
        ['window_id', 'n_total', 'n_nogo', 'n_go', 'balanced_acc', 'ba_std', 'recall_nogo']
    ].to_string(index=False))
