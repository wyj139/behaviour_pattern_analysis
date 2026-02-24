"""
analyze_autoencoder_weights.py
Function: Analyze the contribution of each input channel (bodypart x/y) to each embedding dimension
          by examining the trained autoencoder model weights.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from config import OUTPUT_BASE_DIR, WINDOW_SIZE, EMBEDDING_DIM, CONV_CHANNELS, VERBOSE
from autoencoder import ConvAutoencoder


# 输入通道名称（与TrajDataset中__getitem__的通道顺序一致）
# Input channel names (matching the channel order in TrajDataset.__getitem__)
BODYPART_NAMES = [
    'base_of_the_left_ear', 'base_of_the_right_ear', 'base_of_the_tail',
    'left_flank', 'left_hip', 'left_shoulder', 'neck',
    'right_flank', 'right_hip', 'right_shoulder', 'tip_of_the_tail'
]


def get_channel_names(bodypart_names):
    """生成通道名称列表：bp0_x, bp0_y, bp1_x, bp1_y, ..."""
    names = []
    for bp in bodypart_names:
        names.append(f'{bp}_x')
        names.append(f'{bp}_y')
    return names


def analyze_encoder_weights(model, channel_names: list,
                            output_dir: str = None):
    """
    通过梯度传播方式计算每个输入通道对每个embedding维度的贡献度
    Compute the contribution of each input channel to each embedding dimension
    via gradient-based sensitivity analysis.

    方法：对每个embedding维度，反向传播计算输入的梯度，
    取各通道梯度的绝对值平均作为该通道的贡献度。
    Method: For each embedding dim, backpropagate to get input gradients,
    average absolute gradient per channel as contribution.
    
    兼容所有模型结构（旧版、3层conv、分组卷积）。
    Compatible with all model structures (old, 3-conv, grouped conv).
    """
    model.eval()
    n_channels = model.in_channels
    window_size = model.window_size
    emb_dim = model.emb_dim

    # 用一批随机输入来估计梯度敏感度
    # Use random inputs to estimate gradient sensitivity
    n_samples = 500
    torch.manual_seed(42)
    x = torch.randn(n_samples, n_channels, window_size, requires_grad=True)

    # 通用前向传播：直接用model.forward获取embedding，兼容所有结构
    # Universal forward: use model.forward to get embedding, compatible with all structures
    output = model(x)
    if len(output) == 3:
        _, z, _ = output  # 新版模型返回 (recon, z, decision_prob)
    else:
        _, z = output  # 旧版模型返回 (recon, z)

    # 对每个embedding维度计算梯度
    channel_importance = np.zeros((emb_dim, n_channels))

    for emb_i in range(emb_dim):
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()

        # 对第emb_i维embedding求梯度
        z_i = z[:, emb_i].sum()
        z_i.backward(retain_graph=True)

        # 梯度的绝对值在时间维度上取平均，再在样本维度上取平均
        # |grad| averaged over time and samples -> (n_channels,)
        grad = x.grad.abs().mean(dim=(0, 2)).detach().numpy()  # (n_channels,)
        channel_importance[emb_i] = grad

    # 归一化：每个embedding维度的通道贡献之和为1
    row_sums = channel_importance.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    channel_importance_norm = channel_importance / row_sums

    # 全局通道重要性：各embedding维度取平均
    global_importance = channel_importance_norm.mean(axis=0)

    # 按bodypart聚合（每个bodypart的x和y加起来）
    n_bodyparts = len(BODYPART_NAMES)
    bodypart_importance = np.zeros(n_bodyparts)
    for i in range(n_bodyparts):
        bodypart_importance[i] = global_importance[2*i] + global_importance[2*i+1]

    # 构建结果DataFrame
    channel_df = pd.DataFrame({
        'channel': channel_names,
        'global_importance': global_importance
    }).sort_values('global_importance', ascending=False)

    bodypart_df = pd.DataFrame({
        'bodypart': BODYPART_NAMES,
        'importance': bodypart_importance
    }).sort_values('importance', ascending=False)

    # 打印结果
    print("\n" + "=" * 60)
    print("输入通道对Embedding的贡献度 (Channel -> Embedding Importance)")
    print("=" * 60)
    print("\n全局通道重要性 (Global channel importance):")
    print(channel_df.to_string(index=False))
    print(f"\nBodypart重要性 (Bodypart importance, x+y combined):")
    print(bodypart_df.to_string(index=False))

    # 可视化
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # 图1：全局通道重要性柱状图
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # 左图：每个通道的重要性
        ax1 = axes[0]
        ch_sorted = channel_df
        colors = ['#e74c3c' if '_x' in c else '#3498db' for c in ch_sorted['channel']]
        ax1.barh(range(len(ch_sorted)), ch_sorted['global_importance'].values, color=colors)
        ax1.set_yticks(range(len(ch_sorted)))
        ax1.set_yticklabels(ch_sorted['channel'].values, fontsize=8)
        ax1.set_xlabel('Importance (normalized)', fontsize=11)
        ax1.set_title('Channel Importance for Embedding\n(red=x, blue=y)', fontsize=13)
        ax1.invert_yaxis()

        # 右图：bodypart重要性
        ax2 = axes[1]
        bp_sorted = bodypart_df
        colors_bp = plt.cm.Set3(np.linspace(0, 1, len(bp_sorted)))
        ax2.barh(range(len(bp_sorted)), bp_sorted['importance'].values, color=colors_bp)
        ax2.set_yticks(range(len(bp_sorted)))
        ax2.set_yticklabels(bp_sorted['bodypart'].values, fontsize=10)
        ax2.set_xlabel('Importance (x + y combined)', fontsize=11)
        ax2.set_title('Bodypart Importance for Embedding', fontsize=13)
        ax2.invert_yaxis()
        for i, val in enumerate(bp_sorted['importance'].values):
            ax2.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        path1 = os.path.join(output_dir, 'autoencoder_channel_importance.png')
        plt.savefig(path1, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n通道重要性图已保存到: {path1}")

        # 图2：热力图 - 每个embedding维度对每个通道的贡献
        fig, ax = plt.subplots(figsize=(16, 10))
        im = ax.imshow(channel_importance_norm.T, aspect='auto', cmap='YlOrRd')
        ax.set_xlabel('Embedding Dimension', fontsize=12)
        ax.set_ylabel('Input Channel', fontsize=12)
        ax.set_xticks(range(emb_dim))
        ax.set_xticklabels([f'emb_{i}' for i in range(emb_dim)], rotation=90, fontsize=7)
        ax.set_yticks(range(n_channels))
        ax.set_yticklabels(channel_names, fontsize=8)
        ax.set_title('Channel Contribution to Each Embedding Dimension', fontsize=14)
        plt.colorbar(im, ax=ax, label='Normalized Importance')
        plt.tight_layout()

        path2 = os.path.join(output_dir, 'autoencoder_channel_embedding_heatmap.png')
        plt.savefig(path2, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"通道-Embedding热力图已保存到: {path2}")

        # 保存CSV
        channel_df.to_csv(os.path.join(output_dir, 'autoencoder_channel_importance.csv'), index=False)
        bodypart_df.to_csv(os.path.join(output_dir, 'autoencoder_bodypart_importance.csv'), index=False)

        # 保存完整矩阵
        heatmap_df = pd.DataFrame(channel_importance_norm.T,
                                  index=channel_names,
                                  columns=[f'emb_{i}' for i in range(emb_dim)])
        heatmap_df.to_csv(os.path.join(output_dir, 'autoencoder_channel_embedding_matrix.csv'))
        print(f"数据已保存到CSV文件")

    return channel_importance_norm, channel_df, bodypart_df


def main():
    print("=" * 60)
    print("Autoencoder 输入通道权重分析")
    print("Autoencoder Input Channel Weight Analysis")
    print("=" * 60)

    output_dir = OUTPUT_BASE_DIR
    n_bodyparts = len(BODYPART_NAMES)
    channel_names = get_channel_names(BODYPART_NAMES)

    print(f"\nBodyparts: {n_bodyparts}")
    print(f"Input channels: {len(channel_names)} (x,y per bodypart)")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Embedding dim: {EMBEDDING_DIM}")

    # 加载模型 - 自动检测模型结构
    model_path = os.path.join(output_dir, 'autoencoder_model.pth')
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

    # ========== 检测模型结构类型 ==========
    # Detect model structure type from state_dict keys
    has_grouped_conv = 'enc_intra.0.weight' in state_dict
    has_old_sequential = 'encoder.0.weight' in state_dict
    has_third_conv = 'encoder.4.weight' in state_dict if has_old_sequential else False

    # 从fc_enc的权重尺寸推断结构
    fc_enc_shape = state_dict['fc_enc.weight'].shape  # (emb_dim, conv_channels * pooled_size)
    actual_emb_dim = fc_enc_shape[0]
    fc_input_size = fc_enc_shape[1]

    if has_grouped_conv:
        # ===== 新版分组卷积模型 (Grouped Conv Model) =====
        print("\n检测到分组卷积模型结构 (Detected grouped convolution model)...")
        
        # 从 enc_fusion 层推断 conv_channels
        actual_conv_channels = state_dict['enc_fusion.0.weight'].shape[0]
        actual_pooled_size = fc_input_size // actual_conv_channels
        inferred_window_size = actual_pooled_size * 2  # MaxPool(2)
        
        print(f"  conv_channels: {actual_conv_channels}")
        print(f"  emb_dim: {actual_emb_dim}")
        print(f"  推断window_size: {inferred_window_size}")
        
        model = ConvAutoencoder(
            window_size=inferred_window_size,
            n_bodyparts=n_bodyparts,
            emb_dim=actual_emb_dim,
            conv_channels=actual_conv_channels
        )
    elif has_old_sequential:
        # 从encoder第一层推断conv_channels
        if 'encoder.2.weight' in state_dict:
            actual_conv_channels = state_dict['encoder.2.weight'].shape[0]
        else:
            actual_conv_channels = CONV_CHANNELS

        actual_pooled_size = fc_input_size // actual_conv_channels

        if has_third_conv:
            # 有MaxPool: pooled_size = window_size // 2
            inferred_window_size = actual_pooled_size * 2
        else:
            # 没有MaxPool: pooled_size = window_size
            inferred_window_size = actual_pooled_size

        print(f"\n从模型权重推断的结构:")
        print(f"  conv_channels: {actual_conv_channels}")
        print(f"  emb_dim: {actual_emb_dim}")
        print(f"  fc_input_size: {fc_input_size}")
        print(f"  有第3层conv: {has_third_conv}")
        print(f"  推断window_size: {inferred_window_size}")

        if not has_third_conv:
            # 旧版模型：2层conv, 无MaxPool
            print("\n检测到旧版模型结构（2层conv，无MaxPool），使用兼容加载...")

            class OldConvAutoencoder(nn.Module):
                def __init__(self, window_size, n_bodyparts, emb_dim, conv_channels):
                    super().__init__()
                    self.window_size = window_size
                    self.n_bodyparts = n_bodyparts
                    self.in_channels = 2 * n_bodyparts
                    self.emb_dim = emb_dim
                    self.conv_channels = conv_channels
                    self.encoder = nn.Sequential(
                        nn.Conv1d(self.in_channels, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv1d(16, conv_channels, kernel_size=3, padding=1),
                        nn.ReLU()
                    )
                    self.fc_enc = nn.Linear(conv_channels * window_size, emb_dim)
                    self.fc_dec = nn.Linear(emb_dim, conv_channels * window_size)
                    self.decoder = nn.Sequential(
                        nn.ConvTranspose1d(conv_channels, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose1d(16, self.in_channels, kernel_size=3, padding=1)
                    )

                def forward(self, x):
                    h = self.encoder(x)
                    h_flat = h.view(x.size(0), -1)
                    z = self.fc_enc(h_flat)
                    h_dec = self.fc_dec(z)
                    h_dec = h_dec.view(x.size(0), self.conv_channels, self.window_size)
                    recon = self.decoder(h_dec)
                    return recon, z

            model = OldConvAutoencoder(
                window_size=inferred_window_size,
                n_bodyparts=n_bodyparts,
                emb_dim=actual_emb_dim,
                conv_channels=actual_conv_channels
            )
        else:
            # 中间版本：3层conv + MaxPool（Sequential结构）
            print("\n检测到3层conv+MaxPool模型结构...")
            
            class MidConvAutoencoder(nn.Module):
                def __init__(self, window_size, n_bodyparts, emb_dim, conv_channels):
                    super().__init__()
                    self.window_size = window_size
                    self.n_bodyparts = n_bodyparts
                    self.in_channels = 2 * n_bodyparts
                    self.emb_dim = emb_dim
                    self.conv_channels = conv_channels
                    self.pooled_size = window_size // 2
                    self.encoder = nn.Sequential(
                        nn.Conv1d(self.in_channels, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv1d(16, conv_channels, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=2, stride=2)
                    )
                    self.fc_enc = nn.Linear(conv_channels * self.pooled_size, emb_dim)
                    self.fc_dec = nn.Linear(emb_dim, conv_channels * self.pooled_size)
                    self.decoder = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                        nn.ConvTranspose1d(conv_channels, conv_channels, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose1d(conv_channels, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.ConvTranspose1d(16, self.in_channels, kernel_size=3, padding=1)
                    )

                def forward(self, x):
                    h = self.encoder(x)
                    h_flat = h.view(x.size(0), -1)
                    z = self.fc_enc(h_flat)
                    h_dec = self.fc_dec(z)
                    h_dec = h_dec.view(x.size(0), self.conv_channels, self.pooled_size)
                    recon = self.decoder(h_dec)
                    return recon, z

            model = MidConvAutoencoder(
                window_size=inferred_window_size,
                n_bodyparts=n_bodyparts,
                emb_dim=actual_emb_dim,
                conv_channels=actual_conv_channels
            )
    else:
        raise ValueError(f"无法识别的模型结构，state_dict keys: {list(state_dict.keys())[:10]}")

    model.load_state_dict(state_dict)
    print(f"模型加载成功: {model_path}")

    # 分析权重
    analyze_encoder_weights(model, channel_names, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("分析完成！(Analysis complete!)")
    print("=" * 60)


if __name__ == '__main__':
    main()
