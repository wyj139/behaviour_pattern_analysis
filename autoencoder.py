"""
autoencoder.py
"""

import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Tuple, Dict, List, Optional

from config import (
    EMBEDDING_DIM, CONV_CHANNELS, LEARNING_RATE, BATCH_SIZE,
    NUM_EPOCHS, TRAIN_TEST_SPLIT, RANDOM_SEED, VERBOSE, OUTPUT_BASE_DIR
)
import pandas as pd
from typing import List, Dict




class TrajDataset(Dataset):
    """trajectory dataset for multi-bodypart
    轨迹数据集 - 多bodypart通道输入"""
    
    def __init__(self, df: pd.DataFrame, window_size: int):
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # get all bodypart rel_x and rel_y
        # 获取所有bodypart的rel_x和rel_y
        # all_window_rel_x和all_window_rel_y是list of arrays
        all_rel_x = row['all_window_rel_x']  # list of n_bodyparts arrays
        all_rel_y = row['all_window_rel_y']  # list of n_bodyparts arrays
        
        # stack into tensor (2*n_bodyparts, window_size)
        # 堆叠成 (2*n_bodyparts, window_size)
        # order：bp0_x, bp0_y, bp1_x, bp1_y, ...
        channels = []
        for rel_x, rel_y in zip(all_rel_x, all_rel_y):
            channels.append(rel_x)
            channels.append(rel_y)
        
        traj = np.stack(channels, axis=0)  # shape: (2*n_bodyparts, window_size)
        traj = torch.tensor(traj, dtype=torch.float32)
        
        # metadata
        # 元数据
        meta = {
            'id': row['id'],
            'window_id': row['window_id'],
            'start_frame': row['start_frame'],
            'end_frame': row['end_frame']
        }
        
        for col in ['Housing', 'pig', 'cue', 'decision']:
            if col in row.index:
                meta[col] = row[col]
        
        # decision label as tensor for classification
        decision_label = torch.tensor(int(row['decision']), dtype=torch.float32) if 'decision' in row.index else torch.tensor(-1, dtype=torch.float32)
        
        return traj, meta, decision_label


class ConvAutoencoder(nn.Module):
    """
    分组卷积自动编码器 - Grouped Convolution Autoencoder
    
    架构设计思路 (Architecture design rationale):
    1. Stage 1 - 组内卷积 (Intra-bodypart convolution):
       每个 bodypart 有 (x, y) 两个通道，使用 groups=n_bodyparts 的分组卷积
       独立提取每个身体部位的时序运动模式。2→4 channels per group。
       Each bodypart has (x, y) channels. Grouped conv independently extracts
       temporal motion patterns per bodypart. 2→4 channels per group.
    
    2. Stage 2 - 跨组融合 (Cross-bodypart fusion):
       标准卷积混合所有 bodypart 的特征，学习身体部位间的协调关系。
       Standard conv mixes features across all bodyparts to learn coordination.
    
    3. Stage 3 - 深度特征提取 (Deep feature extraction):
       进一步提取跨 bodypart 的高级行为特征。
       Further extract high-level behavioral features across bodyparts.
    
    4. MaxPool → FC → Embedding
    
    解码器对称设计 (Decoder is symmetric).
    """
    
    def __init__(self, window_size: int, n_bodyparts: int, emb_dim: int = EMBEDDING_DIM, 
                 conv_channels: int = CONV_CHANNELS):
        super().__init__()
        self.window_size = window_size
        self.n_bodyparts = n_bodyparts
        self.in_channels = 2 * n_bodyparts  # 每个bodypart有rel_x和rel_y两个通道
        self.emb_dim = emb_dim
        self.conv_channels = conv_channels
        
        # 组内特征数：每个bodypart从2通道扩展到group_out_channels通道
        # Per-group feature count: each bodypart expands from 2 to group_out_channels
        self.group_out_channels = 4  # 每组输出4个特征通道
        self.fused_channels = self.group_out_channels * n_bodyparts  # 11*4 = 44
        
        # ===== 编码器 Encoder =====
        
        # Stage 1: 组内卷积 - 每个bodypart独立卷积 (Intra-bodypart grouped conv)
        # 输入 (2*n_bodyparts, T) → 输出 (group_out_channels*n_bodyparts, T)
        # 使用 groups=n_bodyparts，每组处理一个bodypart的(x,y)通道
        self.enc_intra = nn.Sequential(
            nn.Conv1d(self.in_channels, self.fused_channels, 
                      kernel_size=3, padding=1, groups=n_bodyparts),
            nn.ReLU(),
        )
        
        # Stage 2: 跨组融合卷积 (Cross-bodypart fusion conv)
        # 输入 (fused_channels=44, T) → 输出 (conv_channels=32, T)
        # 标准卷积，打破分组边界，学习bodypart间的关系
        self.enc_fusion = nn.Sequential(
            nn.Conv1d(self.fused_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Stage 3: 深度特征提取 (Deep feature extraction)
        # 输入 (conv_channels, T) → 输出 (conv_channels, T//2)
        self.enc_deep = nn.Sequential(
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 计算池化后的长度
        # Calculate length after pooling
        self.pooled_size = window_size // 2
        
        # 全连接层：展平为嵌入向量
        # FC layers: flatten to embedding vector
        self.fc_enc = nn.Linear(conv_channels * self.pooled_size, emb_dim)
        
        # ===== 解码器 Decoder =====
        self.fc_dec = nn.Linear(emb_dim, conv_channels * self.pooled_size)
        
        # Stage 3 逆向: 上采样 + 反卷积 (Reverse of Stage 3)
        self.dec_deep = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.ConvTranspose1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Stage 2 逆向: 融合层反卷积 (Reverse of Stage 2)
        self.dec_fusion = nn.Sequential(
            nn.ConvTranspose1d(conv_channels, self.fused_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Stage 1 逆向: 分组反卷积 (Reverse of Stage 1 - grouped deconv)
        self.dec_intra = nn.ConvTranspose1d(
            self.fused_channels, self.in_channels,
            kernel_size=3, padding=1, groups=n_bodyparts
        )
        
        # ===== 分类头 Classification Head =====
        # 从embedding预测decision（二分类），使用Sigmoid输出概率
        # Predict decision from embedding (binary classification) with Sigmoid
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch_size, 2*n_bodyparts, window_size)
        
        # ===== 编码 Encode =====
        h = self.enc_intra(x)     # (B, fused_channels=44, T)   - 组内特征
        h = self.enc_fusion(h)    # (B, conv_channels=32, T)    - 跨组融合
        h = self.enc_deep(h)      # (B, conv_channels=32, T//2) - 深度+池化
        
        h_flat = h.view(x.size(0), -1)       # (B, conv_channels * pooled_size)
        z = self.fc_enc(h_flat)               # (B, emb_dim)
        
        # ===== 分类 Classify =====
        decision_prob = self.classifier(z)    # (B, 1) - sigmoid概率
        
        # ===== 解码 Decode =====
        h_dec = self.fc_dec(z)                # (B, conv_channels * pooled_size)
        h_dec = h_dec.view(x.size(0), self.conv_channels, self.pooled_size)
        
        h_dec = self.dec_deep(h_dec)          # (B, conv_channels, T)   - 上采样+反卷积
        h_dec = self.dec_fusion(h_dec)        # (B, fused_channels, T)  - 反融合
        recon = self.dec_intra(h_dec)         # (B, in_channels, T)     - 分组反卷积
        
        return recon, z, decision_prob


def train_autoencoder(window_df: pd.DataFrame,
                     window_size: int,
                     emb_dim: int = EMBEDDING_DIM,
                     batch_size: int = BATCH_SIZE,
                     num_epochs: int = NUM_EPOCHS,
                     learning_rate: float = LEARNING_RATE,
                     test_split: float = TRAIN_TEST_SPLIT,
                     random_seed: int = RANDOM_SEED,
                     output_dir: Optional[str] = None) -> Tuple[nn.Module, List[float]]:
    """
    Train the autoencoder
    
    Parameters:
        window_df: DataFrame containing windowed data
        window_size: Size of the window
        emb_dim: Embedding dimension
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        test_split: Test set proportion
        random_seed: Random seed
        output_dir: Output directory
    
    Returns:
        Trained model and loss history
    """
    # Ensure all_window_rel_x and all_window_rel_y are in the correct format
    def convert_to_array_list(col):
        def convert_item(x):
            if isinstance(x, list):
                return [np.array(arr, dtype=float) if not isinstance(arr, np.ndarray) else arr for arr in x]
            return x
        return col.apply(convert_item)
    
    window_df = window_df.copy()
    window_df['all_window_rel_x'] = convert_to_array_list(window_df['all_window_rel_x'])
    window_df['all_window_rel_y'] = convert_to_array_list(window_df['all_window_rel_y'])
    
    # get number of bodyparts
    # 获取bodypart数量
    n_bodyparts = window_df.iloc[0]['n_bodyparts'] if 'n_bodyparts' in window_df.columns else len(window_df.iloc[0]['all_window_rel_x'])
    
    # split train and test sets
    # 划分训练集和测试集
    train_df, test_df = train_test_split(
        window_df, test_size=test_split, random_state=random_seed
    )
    
    if VERBOSE:
        print(f"训练集大小: {len(train_df)}")
        print(f"测试集大小: {len(test_df)}")
    
    # create datasets and dataloaders
    # 创建数据集和数据加载器
    train_dataset = TrajDataset(train_df, window_size)
    test_dataset = TrajDataset(test_df, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # create model
    # 创建模型
    model = ConvAutoencoder(window_size=window_size, n_bodyparts=n_bodyparts, emb_dim=emb_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    recon_criterion = nn.MSELoss()
    cls_criterion = nn.BCELoss()
    
    # 分类损失权重 (classification loss weight)
    cls_loss_weight = 0.5
    
    if VERBOSE:
        print(f"模型输入通道数: {2 * n_bodyparts} (n_bodyparts={n_bodyparts})")
        print(f"多任务训练: 重建损失 + {cls_loss_weight} * 分类损失 (BCE with Sigmoid)")
    
    # training loop
    # 训练
    loss_history = []
    recon_loss_history = []
    cls_loss_history = []
    cls_acc_history = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_cls_loss = 0
        total_correct = 0
        total_samples = 0
        n_batches = 0
        
        for traj, meta, decision_label in train_loader:
            optimizer.zero_grad()
            recon, z, decision_prob = model(traj)
            
            # 重建损失 (reconstruction loss)
            loss_recon = recon_criterion(recon, traj)
            
            # 分类损失 (classification loss)
            decision_label = decision_label.unsqueeze(1)  # (B, 1)
            loss_cls = cls_criterion(decision_prob, decision_label)
            
            # 总损失 (total loss)
            loss = loss_recon + cls_loss_weight * loss_cls
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_cls_loss += loss_cls.item()
            
            # 计算分类准确率
            preds = (decision_prob >= 0.5).float()
            total_correct += (preds == decision_label).sum().item()
            total_samples += decision_label.size(0)
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_recon = total_recon_loss / n_batches
        avg_cls = total_cls_loss / n_batches
        train_acc = total_correct / total_samples if total_samples > 0 else 0
        
        loss_history.append(avg_loss)
        recon_loss_history.append(avg_recon)
        cls_loss_history.append(avg_cls)
        cls_acc_history.append(train_acc)
        
        if VERBOSE and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Total: {avg_loss:.6f}, "
                  f"Recon: {avg_recon:.6f}, Cls: {avg_cls:.6f}, Acc: {train_acc:.4f}")
    
    # evaluate on test set
    # 评估测试集
    model.eval()
    test_loss = 0
    test_cls_loss = 0
    test_correct = 0
    test_total = 0
    n_test_batches = 0
    
    with torch.no_grad():
        for traj, meta, decision_label in test_loader:
            recon, z, decision_prob = model(traj)
            loss = recon_criterion(recon, traj)
            test_loss += loss.item()
            
            decision_label = decision_label.unsqueeze(1)
            loss_cls = cls_criterion(decision_prob, decision_label)
            test_cls_loss += loss_cls.item()
            
            preds = (decision_prob >= 0.5).float()
            test_correct += (preds == decision_label).sum().item()
            test_total += decision_label.size(0)
            n_test_batches += 1
    
    avg_test_loss = test_loss / n_test_batches if n_test_batches > 0 else 0
    avg_test_cls = test_cls_loss / n_test_batches if n_test_batches > 0 else 0
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    if VERBOSE:
        print(f"测试集重建损失: {avg_test_loss:.6f}")
        print(f"测试集分类损失: {avg_test_cls:.6f}")
        print(f"测试集分类准确率: {test_acc:.4f}")
    
    # 绘制损失曲线
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
        
        # 总损失 + 重建损失
        axes[0].plot(loss_history, label='Total Loss', color='blue')
        axes[0].plot(recon_loss_history, label='Recon Loss', color='green', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Reconstruction Loss Curve')
        axes[0].legend()
        axes[0].grid(True)
        
        # 分类损失
        axes[1].plot(cls_loss_history, label='Classification Loss (BCE)', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('BCE Loss')
        axes[1].set_title('Classification Loss Curve')
        axes[1].legend()
        axes[1].grid(True)
        
        # 分类准确率
        axes[2].plot(cls_acc_history, label='Train Accuracy', color='purple')
        axes[2].axhline(y=test_acc, color='orange', linestyle='--', label=f'Test Acc: {test_acc:.4f}')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Classification Accuracy Curve')
        axes[2].set_ylim(0, 1.05)
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        loss_plot_path = os.path.join(output_dir, 'training_loss.png')
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if VERBOSE:
            print(f"Save training loss curve to: {loss_plot_path}")
    
    return model, loss_history


def extract_embeddings(model: nn.Module,
                       window_df: pd.DataFrame,
                       window_size: int,
                       batch_size: int = BATCH_SIZE,
                       output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Extract embeddings using a trained model
    
    Parameters:
        model: Trained autoencoder
        window_df: DataFrame of windows
        window_size: Size of the window
        batch_size: Batch size
        output_dir: Output directory
    
    Returns:
        DataFrame containing embedding features
    """
    # Ensure format correctness
    # 确保数据格式正确
    def convert_to_array_list(col):
        def convert_item(x):
            if isinstance(x, list):
                return [np.array(arr, dtype=float) if not isinstance(arr, np.ndarray) else arr for arr in x]
            return x
        return col.apply(convert_item)
    
    window_df = window_df.copy()
    window_df['all_window_rel_x'] = convert_to_array_list(window_df['all_window_rel_x'])
    window_df['all_window_rel_y'] = convert_to_array_list(window_df['all_window_rel_y'])
    
    # create dataset and dataloader
    # 创建数据集
    dataset = TrajDataset(window_df, window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # extract embeddings
    # 提取嵌入
    emb_list = []
    model.eval()
    
    with torch.no_grad():
        for traj, meta, decision_label in dataloader:
            _, z, decision_prob = model(traj)  # z: (batch_size, emb_dim), decision_prob: (B, 1)
            
            for i in range(z.size(0)):
                row = {
                    'id': meta['id'][i],
                    'window_id': meta['window_id'][i].item(),
                    'start_frame': meta['start_frame'][i].item(),
                    'end_frame': meta['end_frame'][i].item()
                }
                
                # 添加可选列
                for col in ['Housing', 'pig', 'cue', 'decision']:
                    if col in meta:
                        row[col] = meta[col][i]
                
                # 添加分类器预测结果
                # Add classifier prediction
                row['ae_pred_prob'] = decision_prob[i, 0].item()
                row['ae_pred'] = int(decision_prob[i, 0].item() >= 0.5)
                
                # add embedding dimensions
                # 添加嵌入维度
                for j in range(z.size(1)):
                    row[f'emb_{j}'] = z[i, j].item()
                
                emb_list.append(row)
    
    embedding_df = pd.DataFrame(emb_list)
    
    if VERBOSE:
        print(f"Extraction of embeddings completed, total {len(embedding_df)} windows")
        if 'decision' in embedding_df.columns and 'ae_pred' in embedding_df.columns:
            acc = (embedding_df['decision'].astype(int) == embedding_df['ae_pred']).mean()
            print(f"Autoencoder classifier overall accuracy: {acc:.4f}")
    
    # save embeddings
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'embeddings.csv')
        embedding_df.to_csv(output_path, index=False)
        
        if VERBOSE:
            print(f"Save embeddings to: {output_path}")
    
    return embedding_df


# ============================================================================
# Autoencoder分类器评估与可视化
# Autoencoder Classifier Evaluation & Visualization
# ============================================================================

BODYPART_NAMES = [
    'base_of_the_left_ear', 'base_of_the_right_ear', 'base_of_the_tail',
    'left_flank', 'left_hip', 'left_shoulder', 'neck',
    'right_flank', 'right_hip', 'right_shoulder', 'tip_of_the_tail'
]


def load_and_clean_ae_embeddings(output_base_dir: str = OUTPUT_BASE_DIR) -> pd.DataFrame:
    """
    加载并清理embeddings数据（含ae_pred列）
    Load and clean embeddings data (with ae_pred column)
    """
    embedding_path = os.path.join(output_base_dir, 'embeddings.csv')
    df = pd.read_csv(embedding_path)

    for col in ['decision', 'window_id', 'ae_pred']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: int(re.search(r'\d+', str(x)).group())
                                    if 'tensor' in str(x) else int(x))
    
    if 'ae_pred_prob' in df.columns:
        df['ae_pred_prob'] = df['ae_pred_prob'].apply(
            lambda x: float(re.search(r'[\d.]+', str(x)).group())
            if 'tensor' in str(x) else float(x))

    if VERBOSE:
        print(f"数据加载完成: {len(df)} 条记录")
        print(f"Decision分布:\n{df['decision'].value_counts()}")
        print(f"Window ID范围: {df['window_id'].min()} - {df['window_id'].max()}")
        print(f"Embedding维度: {len([c for c in df.columns if c.startswith('emb_')])}")

    return df


def evaluate_ae_classifier(df: pd.DataFrame,
                           output_dir: Optional[str] = None) -> dict:
    """
    评估Autoencoder分类器的预测结果，按照随机森林模型的统计模式:
    - 整体准确率
    - 不同decision的准确率(recall)
    - 不同housing的准确率
    - 不同window_id下的准确率
    - 不同max window id下的准确率
    
    Evaluate AE classifier predictions following the RF evaluation pattern.
    """
    print("\n" + "=" * 60)
    print("Autoencoder 分类器评估 (AE Classifier Evaluation)")
    print("=" * 60)

    y_true = df['decision'].astype(int).values
    y_pred = df['ae_pred'].astype(int).values

    # ========== 1. 整体准确率 ==========
    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\n整体准确率 (Overall accuracy): {overall_acc:.4f}")
    print(f"\n全局分类报告 (Global classification report):")
    print(classification_report(y_true, y_pred, target_names=['decision=0', 'decision=1']))

    # ========== 2. 按window_id统计准确度 ==========
    print("\n" + "=" * 60)
    print("按window_id评估准确度 (Per-window_id accuracy)...")
    print("=" * 60)

    window_ids = sorted(df['window_id'].unique())
    window_accuracies = {}
    window_sample_counts = {}

    for wid in window_ids:
        window_data = df[df['window_id'] == wid]
        if window_data['decision'].nunique() < 2:
            continue
        acc = accuracy_score(window_data['decision'], window_data['ae_pred'])
        window_accuracies[wid] = acc
        window_sample_counts[wid] = len(window_data)

    if VERBOSE and len(window_accuracies) > 0:
        accs = list(window_accuracies.values())
        print(f"\n有效窗口数: {len(window_accuracies)} / {len(window_ids)}")
        print(f"窗口准确度: mean={np.mean(accs):.4f}, min={np.min(accs):.4f}, max={np.max(accs):.4f}")

    # ========== 3. 可视化 ==========
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # 3.1 窗口准确度柱状图
        _plot_ae_window_accuracy(window_accuracies, window_sample_counts, output_dir)

        # 3.2 按decision分类的准确度
        _plot_ae_accuracy_by_decision(df, output_dir)

        # 3.3 按housing分类的准确度
        _plot_ae_accuracy_by_housing(df, output_dir)

        # 3.4 最优预测窗口分析
        _analyze_ae_best_prediction_window(df, output_dir)

    return {
        'overall_accuracy': overall_acc,
        'window_accuracies': window_accuracies,
        'window_sample_counts': window_sample_counts,
    }


def _plot_ae_window_accuracy(window_accuracies: dict,
                             window_sample_counts: dict,
                             output_dir: str):
    """AE分类器: 不同window_id的预测准确度"""
    wids = sorted(window_accuracies.keys())
    accs = [window_accuracies[w] for w in wids]
    counts = [window_sample_counts[w] for w in wids]

    fig, ax1 = plt.subplots(figsize=(max(14, len(wids) * 0.15), 6))

    bars = ax1.bar(range(len(wids)), accs, color='steelblue', alpha=0.8, width=0.8)
    ax1.set_xlabel('Window ID', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    mean_acc = np.mean(accs)
    ax1.axhline(y=mean_acc, color='red', linestyle='--', linewidth=1.5,
                label=f'Mean accuracy = {mean_acc:.4f}')
    ax1.set_ylim(0, 1.05)

    if len(wids) > 50:
        step = max(1, len(wids) // 20)
        tick_positions = range(0, len(wids), step)
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels([str(wids[i]) for i in tick_positions], rotation=45, fontsize=8)
    else:
        ax1.set_xticks(range(len(wids)))
        ax1.set_xticklabels([str(w) for w in wids], rotation=45, fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(range(len(wids)), counts, color='orange', linewidth=1, alpha=0.6,
             label='Sample count')
    ax2.set_ylabel('Sample Count', fontsize=12, color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

    plt.title('AE Classifier Prediction Accuracy per Window ID', fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'ae_accuracy_per_window_id.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"AE窗口准确度柱状图已保存到: {output_path}")


def _plot_ae_accuracy_by_decision(df: pd.DataFrame, output_dir: str):
    """AE分类器: 按decision=0和decision=1分别计算每个window_id的准确率"""
    valid_wids = []
    for wid in sorted(df['window_id'].unique()):
        w = df[df['window_id'] == wid]
        if w['decision'].nunique() >= 2:
            valid_wids.append(wid)

    acc_d0 = []
    acc_d1 = []

    for wid in valid_wids:
        w = df[df['window_id'] == wid]

        w0 = w[w['decision'] == 0]
        a0 = accuracy_score(w0['decision'], w0['ae_pred']) if len(w0) > 0 else np.nan

        w1 = w[w['decision'] == 1]
        a1 = accuracy_score(w1['decision'], w1['ae_pred']) if len(w1) > 0 else np.nan

        acc_d0.append(a0)
        acc_d1.append(a1)

    acc_d0 = np.array(acc_d0, dtype=float)
    acc_d1 = np.array(acc_d1, dtype=float)
    x = np.arange(len(valid_wids))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(max(14, len(valid_wids) * 0.2), 7))

    ax.bar(x - bar_width / 2, acc_d0, bar_width, color='#e74c3c', alpha=0.75, label='Decision=0 (Recall)')
    ax.bar(x + bar_width / 2, acc_d1, bar_width, color='#2ecc71', alpha=0.75, label='Decision=1 (Recall)')

    mean_d0 = np.nanmean(acc_d0)
    mean_d1 = np.nanmean(acc_d1)
    ax.axhline(y=mean_d0, color='#e74c3c', linestyle='--', linewidth=1.2,
               label=f'Mean decision=0: {mean_d0:.4f}')
    ax.axhline(y=mean_d1, color='#2ecc71', linestyle='--', linewidth=1.2,
               label=f'Mean decision=1: {mean_d1:.4f}')

    ax.set_xlabel('Window ID', fontsize=12)
    ax.set_ylabel('Recall (per-class accuracy)', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title('AE Classifier: Per-Window Recall for Decision=0 vs Decision=1', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)

    if len(valid_wids) > 50:
        step = max(1, len(valid_wids) // 20)
        tick_positions = list(range(0, len(valid_wids), step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(valid_wids[i]) for i in tick_positions], rotation=45, fontsize=8)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in valid_wids], rotation=45, fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ae_accuracy_by_decision.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"AE按Decision分类的准确度图已保存到: {output_path}")


def _plot_ae_accuracy_by_housing(df: pd.DataFrame, output_dir: str):
    """AE分类器: 按housing条件分类，绘制每个window_id的预测准确率"""
    if 'Housing' not in df.columns:
        print("警告: 数据中没有Housing列，跳过Housing准确度图")
        return

    housing_types = sorted(df['Housing'].unique())
    colors = {'Baseline1': '#3498db', 'Baseline2': '#e74c3c',
              'Enriched': '#2ecc71', 'Barren': '#f39c12'}
    default_colors = ['#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    for i, h in enumerate(housing_types):
        if h not in colors:
            colors[h] = default_colors[i % len(default_colors)]

    valid_wids = []
    for wid in sorted(df['window_id'].unique()):
        w = df[df['window_id'] == wid]
        if w['decision'].nunique() >= 2:
            valid_wids.append(wid)

    housing_accs = {h: [] for h in housing_types}
    for wid in valid_wids:
        w = df[df['window_id'] == wid]
        for h in housing_types:
            wh = w[w['Housing'] == h]
            if len(wh) >= 2 and wh['decision'].nunique() >= 2:
                acc = accuracy_score(wh['decision'], wh['ae_pred'])
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
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title('AE Classifier: Per-Window Accuracy by Housing Condition', fontsize=14)
    ax.legend(loc='lower left', fontsize=9, ncol=2)

    if len(valid_wids) > 50:
        step = max(1, len(valid_wids) // 20)
        tick_positions = list(range(0, len(valid_wids), step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(valid_wids[i]) for i in tick_positions], rotation=45, fontsize=8)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([str(w) for w in valid_wids], rotation=45, fontsize=8)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ae_accuracy_by_housing.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"AE按Housing分类的准确度图已保存到: {output_path}")


def _analyze_ae_best_prediction_window(df: pd.DataFrame, output_dir: str):
    """
    AE分类器: 最优预测窗口分析
    对于在不同window_id结束的trial，找出哪个window_id的预测准确率最高
    """
    print("\n" + "=" * 60)
    print("AE分类器最优预测窗口分析 (AE Best Prediction Window)")
    print("=" * 60)

    max_wid_per_trial = df.groupby('id')['window_id'].max().reset_index()
    max_wid_per_trial.columns = ['id', 'max_wid']
    df_with_max = df.merge(max_wid_per_trial, on='id', how='left')

    max_wid_counts = max_wid_per_trial['max_wid'].value_counts().sort_index()
    valid_max_wids = max_wid_counts[max_wid_counts >= 10].index.tolist()
    valid_max_wids = sorted(valid_max_wids)

    if len(valid_max_wids) == 0:
        print("没有足够的数据进行最优窗口分析")
        return

    print(f"有效的 max_wid 组数: {len(valid_max_wids)}")

    results = []
    for max_wid in valid_max_wids:
        trial_ids = max_wid_per_trial[max_wid_per_trial['max_wid'] == max_wid]['id'].values
        group_data = df_with_max[df_with_max['id'].isin(trial_ids)]
        n_trials = len(trial_ids)

        for wid in range(0, max_wid + 1):
            wid_data = group_data[group_data['window_id'] == wid]
            if len(wid_data) < 5:
                continue

            overall_acc = accuracy_score(wid_data['decision'], wid_data['ae_pred'])

            d0_data = wid_data[wid_data['decision'] == 0]
            d1_data = wid_data[wid_data['decision'] == 1]
            recall_d0 = accuracy_score(d0_data['decision'], d0_data['ae_pred']) if len(d0_data) >= 2 else np.nan
            recall_d1 = accuracy_score(d1_data['decision'], d1_data['ae_pred']) if len(d1_data) >= 2 else np.nan

            results.append({
                'max_wid': max_wid,
                'window_id': wid,
                'n_trials': n_trials,
                'n_samples': len(wid_data),
                'overall_accuracy': overall_acc,
                'recall_d0': recall_d0,
                'recall_d1': recall_d1,
            })

    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        return

    # 找出每个max_wid组的最优window_id
    best_windows = []
    for max_wid in valid_max_wids:
        group = results_df[results_df['max_wid'] == max_wid]
        if len(group) == 0:
            continue
        best_row = group.loc[group['overall_accuracy'].idxmax()]
        best_windows.append({
            'max_wid': max_wid,
            'n_trials': int(best_row['n_trials']),
            'best_overall_wid': int(best_row['window_id']),
            'best_overall_acc': best_row['overall_accuracy'],
        })

    best_df = pd.DataFrame(best_windows)
    print(f"\n最优预测窗口汇总:")
    print(best_df.to_string(index=False))

    # 保存CSV
    results_df.to_csv(os.path.join(output_dir, 'ae_best_window_details.csv'), index=False)
    best_df.to_csv(os.path.join(output_dir, 'ae_best_window_summary.csv'), index=False)

    # 可视化: 热力图
    pivot = results_df.pivot_table(
        index='max_wid', columns='window_id',
        values='overall_accuracy', aggfunc='first'
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale='RdYlGn',
        zmin=0.3, zmax=1.0,
        colorbar=dict(title='Accuracy'),
        hovertemplate='max_wid=%{y}<br>window_id=%{x}<br>accuracy=%{z:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title='AE Classifier: Accuracy Heatmap (Trial End Window × Prediction Window)',
        xaxis_title='Prediction Window ID',
        yaxis_title='Trial End Window (max_wid)',
        width=max(800, len(pivot.columns) * 30),
        height=max(600, len(pivot.index) * 25),
    )
    fig.write_html(os.path.join(output_dir, 'ae_best_window_heatmap.html'))
    print(f"AE热力图已保存到: ae_best_window_heatmap.html")

    # 可视化: 折线图
    if len(valid_max_wids) > 15:
        indices = np.linspace(0, len(valid_max_wids) - 1, 12, dtype=int)
        selected_max_wids = sorted(set([valid_max_wids[i] for i in indices]))
    else:
        selected_max_wids = valid_max_wids

    colors_list = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    fig = go.Figure()
    for i, max_wid in enumerate(selected_max_wids):
        group = results_df[results_df['max_wid'] == max_wid].sort_values('window_id')
        if len(group) < 2:
            continue
        color = colors_list[i % len(colors_list)]
        n_trials = group.iloc[0]['n_trials']
        fig.add_trace(go.Scatter(
            x=group['window_id'], y=group['overall_accuracy'],
            mode='lines+markers',
            name=f'max_wid={max_wid} (n={n_trials})',
            line=dict(color=color, width=2), marker=dict(size=6),
        ))
        best_idx = group['overall_accuracy'].idxmax()
        best_row = group.loc[best_idx]
        fig.add_trace(go.Scatter(
            x=[best_row['window_id']], y=[best_row['overall_accuracy']],
            mode='markers',
            marker=dict(color=color, size=14, symbol='star',
                        line=dict(color='black', width=1)),
            showlegend=False,
        ))

    fig.update_layout(
        title='AE Classifier: Accuracy vs Window ID for Different Trial Lengths',
        xaxis_title='Prediction Window ID',
        yaxis_title='Overall Accuracy',
        yaxis=dict(range=[0.3, 1.05]),
        width=1100, height=700,
    )
    fig.write_html(os.path.join(output_dir, 'ae_best_window_lines.html'))
    print(f"AE折线图已保存到: ae_best_window_lines.html")

    # 按decision的折线图
    fig2 = go.Figure()
    for i, max_wid in enumerate(selected_max_wids):
        group = results_df[results_df['max_wid'] == max_wid].sort_values('window_id')
        if len(group) < 2:
            continue
        color = colors_list[i % len(colors_list)]

        d0_valid = group.dropna(subset=['recall_d0'])
        if len(d0_valid) > 0:
            fig2.add_trace(go.Scatter(
                x=d0_valid['window_id'], y=d0_valid['recall_d0'],
                mode='lines+markers', name=f'max_wid={max_wid} D=0',
                line=dict(color=color, width=1.5, dash='dash'),
                legendgroup=f'mw{max_wid}',
            ))
        d1_valid = group.dropna(subset=['recall_d1'])
        if len(d1_valid) > 0:
            fig2.add_trace(go.Scatter(
                x=d1_valid['window_id'], y=d1_valid['recall_d1'],
                mode='lines+markers', name=f'max_wid={max_wid} D=1',
                line=dict(color=color, width=2),
                legendgroup=f'mw{max_wid}',
            ))

    fig2.update_layout(
        title='AE Classifier: Per-Decision Recall vs Window ID',
        xaxis_title='Prediction Window ID',
        yaxis_title='Recall',
        yaxis=dict(range=[0, 1.1]),
        width=1100, height=700,
    )
    fig2.write_html(os.path.join(output_dir, 'ae_best_window_by_decision.html'))
    print(f"AE按Decision折线图已保存到: ae_best_window_by_decision.html")


# ============================================================================
# Autoencoder分类器权重分析
# Autoencoder Classifier Weight Analysis
# ============================================================================

def analyze_classifier_weights(model: nn.Module,
                               channel_names: list = None,
                               output_dir: Optional[str] = None):
    """
    分析Autoencoder分类头的参数权重:
    1. 分类头(classifier)各层的权重统计
    2. 通过梯度敏感度分析每个输入通道/bodypart对decision预测的贡献度
    3. 同时统计bodypart和bodypart_x, bodypart_y的重要性
    
    Analyze classifier head weights:
    1. Classifier layer weight statistics
    2. Gradient-based sensitivity for each input channel/bodypart → decision
    3. Report both bodypart and bodypart_x, bodypart_y importance
    """
    model.eval()
    n_channels = model.in_channels
    n_bodyparts = model.n_bodyparts
    window_size = model.window_size
    emb_dim = model.emb_dim

    if channel_names is None:
        channel_names = []
        for bp in BODYPART_NAMES[:n_bodyparts]:
            channel_names.append(f'{bp}_x')
            channel_names.append(f'{bp}_y')

    print("\n" + "=" * 60)
    print("Autoencoder 分类头权重分析 (Classifier Weight Analysis)")
    print("=" * 60)

    # ===== 1. 分类头参数统计 =====
    print("\n--- 分类头参数 (Classifier Head Parameters) ---")
    classifier_params = {}
    for name, param in model.classifier.named_parameters():
        p = param.detach().cpu().numpy()
        classifier_params[name] = p
        print(f"  {name}: shape={p.shape}, "
              f"mean={p.mean():.6f}, std={p.std():.6f}, "
              f"min={p.min():.6f}, max={p.max():.6f}")

    # ===== 2. Embedding维度对分类的贡献 =====
    # 分析classifier第一层的权重，看哪些embedding维度对分类贡献最大
    cls_w1 = classifier_params.get('0.weight', None)  # shape: (emb_dim//2, emb_dim)
    cls_w2 = classifier_params.get('2.weight', None)  # shape: (1, emb_dim//2)

    if cls_w1 is not None and cls_w2 is not None:
        # 通过权重链计算每个embedding维度对输出的影响
        # impact = |W2| @ |W1|  -> (1, emb_dim) 
        emb_impact = np.abs(cls_w2) @ np.abs(cls_w1)  # (1, emb_dim)
        emb_impact = emb_impact.flatten()
        emb_impact_norm = emb_impact / emb_impact.sum()

        emb_importance_df = pd.DataFrame({
            'embedding_dim': [f'emb_{i}' for i in range(len(emb_impact_norm))],
            'importance': emb_impact_norm
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 对分类最重要的Embedding维度:")
        print(emb_importance_df.head(10).to_string(index=False))
    else:
        emb_importance_df = pd.DataFrame()

    # ===== 3. 梯度敏感度分析: 输入通道 → decision预测 =====
    print("\n--- 输入通道对Decision预测的梯度敏感度 ---")
    n_samples = 500
    torch.manual_seed(42)
    x = torch.randn(n_samples, n_channels, window_size, requires_grad=True)

    _, z, decision_prob = model(x)  # decision_prob: (n_samples, 1)

    model.zero_grad()
    if x.grad is not None:
        x.grad.zero_()

    decision_prob.sum().backward()

    # 梯度绝对值在时间和样本维度上取平均 -> (n_channels,)
    grad_importance = x.grad.abs().mean(dim=(0, 2)).detach().numpy()
    grad_importance_norm = grad_importance / grad_importance.sum()

    # 通道级重要性
    channel_importance_df = pd.DataFrame({
        'channel': channel_names[:n_channels],
        'importance': grad_importance_norm
    }).sort_values('importance', ascending=False)

    print("\n通道重要性 (Channel importance for decision prediction):")
    print(channel_importance_df.to_string(index=False))

    # Bodypart级重要性 (x+y合并)
    bodypart_importance = np.zeros(n_bodyparts)
    bp_names = BODYPART_NAMES[:n_bodyparts]
    for i in range(n_bodyparts):
        bodypart_importance[i] = grad_importance_norm[2 * i] + grad_importance_norm[2 * i + 1]

    bodypart_importance_df = pd.DataFrame({
        'bodypart': bp_names,
        'importance': bodypart_importance
    }).sort_values('importance', ascending=False)

    print(f"\nBodypart重要性 (x+y combined):")
    print(bodypart_importance_df.to_string(index=False))

    # ===== 4. 可视化 =====
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # 图1: 通道重要性 + Bodypart重要性
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        ax1 = axes[0]
        ch_sorted = channel_importance_df
        colors = ['#e74c3c' if '_x' in c else '#3498db' for c in ch_sorted['channel']]
        ax1.barh(range(len(ch_sorted)), ch_sorted['importance'].values, color=colors)
        ax1.set_yticks(range(len(ch_sorted)))
        ax1.set_yticklabels(ch_sorted['channel'].values, fontsize=8)
        ax1.set_xlabel('Importance (gradient-based)', fontsize=11)
        ax1.set_title('AE Classifier: Channel Importance\n(red=x, blue=y)', fontsize=13)
        ax1.invert_yaxis()

        ax2 = axes[1]
        bp_sorted = bodypart_importance_df
        colors_bp = plt.cm.Set3(np.linspace(0, 1, len(bp_sorted)))
        ax2.barh(range(len(bp_sorted)), bp_sorted['importance'].values, color=colors_bp)
        ax2.set_yticks(range(len(bp_sorted)))
        ax2.set_yticklabels(bp_sorted['bodypart'].values, fontsize=10)
        ax2.set_xlabel('Importance (x + y combined)', fontsize=11)
        ax2.set_title('AE Classifier: Bodypart Importance', fontsize=13)
        ax2.invert_yaxis()
        for i, val in enumerate(bp_sorted['importance'].values):
            ax2.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        path1 = os.path.join(output_dir, 'ae_classifier_channel_importance.png')
        plt.savefig(path1, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\nAE分类器通道重要性图已保存到: {path1}")

        # 图2: Embedding维度对分类的贡献
        if len(emb_importance_df) > 0:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.bar(range(len(emb_importance_df)),
                   emb_importance_df['importance'].values,
                   color='teal', alpha=0.8)
            ax.set_xticks(range(len(emb_importance_df)))
            ax.set_xticklabels(emb_importance_df['embedding_dim'].values, rotation=90, fontsize=7)
            ax.set_xlabel('Embedding Dimension', fontsize=12)
            ax.set_ylabel('Importance', fontsize=12)
            ax.set_title('AE Classifier: Embedding Dimension Importance for Decision', fontsize=14)
            plt.tight_layout()
            path2 = os.path.join(output_dir, 'ae_classifier_embedding_importance.png')
            plt.savefig(path2, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"AE Embedding维度重要性图已保存到: {path2}")

        # 图3: 分类头权重热力图
        if cls_w1 is not None:
            fig, axes = plt.subplots(1, 2, figsize=(18, 6))
            
            # 第一层权重
            im1 = axes[0].imshow(np.abs(cls_w1), aspect='auto', cmap='YlOrRd')
            axes[0].set_xlabel('Input (Embedding Dim)', fontsize=11)
            axes[0].set_ylabel('Output (Hidden)', fontsize=11)
            axes[0].set_title('Classifier Layer 1: |W| Heatmap', fontsize=13)
            plt.colorbar(im1, ax=axes[0])

            if cls_w2 is not None:
                # 第二层权重
                im2 = axes[1].imshow(np.abs(cls_w2), aspect='auto', cmap='YlOrRd')
                axes[1].set_xlabel('Input (Hidden)', fontsize=11)
                axes[1].set_ylabel('Output (Decision)', fontsize=11)
                axes[1].set_title('Classifier Layer 2: |W| Heatmap', fontsize=13)
                plt.colorbar(im2, ax=axes[1])

            plt.tight_layout()
            path3 = os.path.join(output_dir, 'ae_classifier_weight_heatmap.png')
            plt.savefig(path3, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"AE分类头权重热力图已保存到: {path3}")

        # 保存CSV
        channel_importance_df.to_csv(
            os.path.join(output_dir, 'ae_classifier_channel_importance.csv'), index=False)
        bodypart_importance_df.to_csv(
            os.path.join(output_dir, 'ae_classifier_bodypart_importance.csv'), index=False)
        if len(emb_importance_df) > 0:
            emb_importance_df.to_csv(
                os.path.join(output_dir, 'ae_classifier_embedding_importance.csv'), index=False)
        print("AE分类器权重分析数据已保存到CSV文件")

    return {
        'channel_importance': channel_importance_df,
        'bodypart_importance': bodypart_importance_df,
        'embedding_importance': emb_importance_df,
    }


def run_ae_classifier_evaluation(output_base_dir: str = OUTPUT_BASE_DIR,
                                 model: nn.Module = None,
                                 trial_percentile: float = 90.0):
    """
    运行完整的AE分类器评估流程:
    1. 加载embeddings（含ae_pred）
    2. 基于trial时长过滤
    3. 评估准确率 + 可视化
    4. 分析分类器权重
    
    Run full AE classifier evaluation pipeline.
    """
    print("\n" + "=" * 80)
    print("Autoencoder 分类器完整评估流程")
    print("AE Classifier Full Evaluation Pipeline")
    print("=" * 80)

    # 1. 加载数据
    df = load_and_clean_ae_embeddings(output_base_dir)

    # 2. 基于trial时长过滤
    max_wids_per_trial = df.groupby('id')['window_id'].max()
    cutoff_wid = int(np.percentile(max_wids_per_trial.values, trial_percentile))
    df_filtered = df[df['window_id'] <= cutoff_wid].copy()

    if VERBOSE:
        print(f"\n基于trial时长过滤 (第{trial_percentile:.0f}百分位): "
              f"window_id <= {cutoff_wid}")
        print(f"过滤前: {len(df)} 条, 过滤后: {len(df_filtered)} 条")

    # 3. 评估准确率 + 可视化
    results = evaluate_ae_classifier(df_filtered, output_dir=output_base_dir)

    # 4. 分析分类器权重
    if model is not None:
        n_bodyparts = model.n_bodyparts
        channel_names = []
        for bp in BODYPART_NAMES[:n_bodyparts]:
            channel_names.append(f'{bp}_x')
            channel_names.append(f'{bp}_y')
        
        weight_results = analyze_classifier_weights(
            model, channel_names, output_dir=output_base_dir
        )
        results['weight_analysis'] = weight_results
    else:
        # 尝试从文件加载模型
        model_path = os.path.join(output_base_dir, 'autoencoder_model.pth')
        if os.path.exists(model_path):
            print(f"\n从文件加载模型: {model_path}")
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            
            # 推断模型参数
            n_bodyparts = len(BODYPART_NAMES)
            fc_enc_shape = state_dict['fc_enc.weight'].shape
            actual_emb_dim = fc_enc_shape[0]
            fc_input_size = fc_enc_shape[1]
            
            if 'enc_intra.0.weight' in state_dict:
                actual_conv_channels = state_dict['enc_fusion.0.weight'].shape[0]
                actual_pooled_size = fc_input_size // actual_conv_channels
                inferred_window_size = actual_pooled_size * 2
                
                loaded_model = ConvAutoencoder(
                    window_size=inferred_window_size,
                    n_bodyparts=n_bodyparts,
                    emb_dim=actual_emb_dim,
                    conv_channels=actual_conv_channels
                )
                loaded_model.load_state_dict(state_dict)
                
                channel_names = []
                for bp in BODYPART_NAMES[:n_bodyparts]:
                    channel_names.append(f'{bp}_x')
                    channel_names.append(f'{bp}_y')
                
                weight_results = analyze_classifier_weights(
                    loaded_model, channel_names, output_dir=output_base_dir
                )
                results['weight_analysis'] = weight_results
            else:
                print("警告: 模型不是分组卷积结构，跳过权重分析")
        else:
            print("警告: 未找到模型文件，跳过权重分析")

    print("\n" + "=" * 80)
    print("AE分类器评估完成!")
    print(f"整体准确率: {results['overall_accuracy']:.4f}")
    print(f"有效窗口数: {len(results['window_accuracies'])}")
    print("=" * 80)

    return results
