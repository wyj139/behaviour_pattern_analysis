"""
autoencoder.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional

from config import (
    EMBEDDING_DIM, CONV_CHANNELS, LEARNING_RATE, BATCH_SIZE,
    NUM_EPOCHS, TRAIN_TEST_SPLIT, RANDOM_SEED, VERBOSE
)

import numpy as np
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
        
        return traj, meta


class ConvAutoencoder(nn.Module):
    """卷积自动编码器 - 多bodypart通道"""
    
    def __init__(self, window_size: int, n_bodyparts: int, emb_dim: int = EMBEDDING_DIM, 
                 conv_channels: int = CONV_CHANNELS):
        super().__init__()
        self.window_size = window_size
        self.n_bodyparts = n_bodyparts
        self.in_channels = 2 * n_bodyparts  # 每个bodypart有rel_x和rel_y两个通道
        self.emb_dim = emb_dim
        self.conv_channels = conv_channels
        
        # 编码器
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(self.in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, conv_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # full connection layers: flatten to embedding vector
        # 全连接层：展平为嵌入向量
        self.fc_enc = nn.Linear(conv_channels * window_size, emb_dim)
        
        # Decoder
        # 解码器
        self.fc_dec = nn.Linear(emb_dim, conv_channels * window_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(conv_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, self.in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        # x: (batch_size, 2*n_bodyparts, window_size)
        h = self.encoder(x)  # (batch_size, conv_channels, window_size)
        
        h_flat = h.view(x.size(0), -1)  # (batch_size, conv_channels * window_size)
        z = self.fc_enc(h_flat)  # (batch_size, emb_dim)
        
        h_dec = self.fc_dec(z)  # (batch_size, conv_channels * window_size)
        h_dec = h_dec.view(x.size(0), self.conv_channels, self.window_size)
        recon = self.decoder(h_dec)  # (batch_size, 2*n_bodyparts, window_size)
        
        return recon, z


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
    criterion = nn.MSELoss()
    
    if VERBOSE:
        print(f"模型输入通道数: {2 * n_bodyparts} (n_bodyparts={n_bodyparts})")
    
    # training loop
    # 训练
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for traj, meta in train_loader:
            optimizer.zero_grad()
            recon, z = model(traj)
            loss = criterion(recon, traj)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        loss_history.append(avg_loss)
        
        if VERBOSE and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    # evaluate on test set
    # 评估测试集
    model.eval()
    test_loss = 0
    n_test_batches = 0
    
    with torch.no_grad():
        for traj, meta in test_loader:
            recon, z = model(traj)
            loss = criterion(recon, traj)
            test_loss += loss.item()
            n_test_batches += 1
    
    avg_test_loss = test_loss / n_test_batches if n_test_batches > 0 else 0
    
    if VERBOSE:
        print(f"测试集重建损失: {avg_test_loss:.6f}")
    
    # 绘制损失曲线
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        
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
        for traj, meta in dataloader:
            _, z = model(traj)  # z: (batch_size, emb_dim)
            
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
                
                # add embedding dimensions
                # 添加嵌入维度
                for j in range(z.size(1)):
                    row[f'emb_{j}'] = z[i, j].item()
                
                emb_list.append(row)
    
    embedding_df = pd.DataFrame(emb_list)
    
    if VERBOSE:
        print(f"Extraction of embeddings completed, total {len(embedding_df)} windows")
    
    # save embeddings
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'embeddings.csv')
        embedding_df.to_csv(output_path, index=False)
        
        if VERBOSE:
            print(f"Save embeddings to: {output_path}")
    
    return embedding_df
