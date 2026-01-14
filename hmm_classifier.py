"""
H Gaussian Hidden Markov Model Classification Module
Function: Use HMM to classify states based on embedding features
"""

import os
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from typing import Tuple, Optional

from config import N_STATES, HMM_COVARIANCE_TYPE, HMM_N_ITER, RANDOM_SEED, VERBOSE


def train_gaussian_hmm(embedding_df: pd.DataFrame,
                      n_states: int = N_STATES,
                      covariance_type: str = HMM_COVARIANCE_TYPE,
                      n_iter: int = HMM_N_ITER,
                      random_state: int = RANDOM_SEED,
                      output_dir: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray, GaussianHMM, pd.DataFrame]:
    """
    Train Gaussian Hidden Markov Model for state classification
    
    Parameters:
        embedding_df: DataFrame containing embedding features
        n_states: Number of HMM states
        covariance_type: Covariance type ('diag', 'full', 'tied', 'spherical')
        n_iter: Number of training iterations
        random_state: Random seed
        output_dir: Output directory
    
    Returns:
        
        - window_df_with_states: DataFrame with added state column
        - transition_matrix: Transition probability matrix
        - hmm_model: Trained HMM model
        - segments_df: DataFrame of state segments
    """
    # extract embedding columns
    # 提取嵌入列
    emb_cols = [c for c in embedding_df.columns if c.startswith('emb_')]
    
    if len(emb_cols) == 0:
        raise ValueError("No embedding columns (emb_*) found in DataFrame")
    
    if VERBOSE:
        print(f"Using {len(emb_cols)} embedding features for HMM training")
        print(f"HMM parameters: n_states={n_states}, covariance_type={covariance_type}, n_iter={n_iter}")
    
    #sort by id and start_frame
    #  按 id 和 start_frame 排序
    window_df_sorted = embedding_df.copy().sort_values(['id', 'start_frame']).reset_index(drop=True)
    
    # group by id and get lengths
    # 分组并获取每个序列的长度
    groups = [g for _, g in window_df_sorted.groupby('id')]
    lengths = [len(g) for g in groups]
    
    # concat all embeddings
    # 拼接所有嵌入特征
    X_concat = np.vstack([g[emb_cols].values for g in groups])
    
    if VERBOSE:
        print(f"Training data shape: {X_concat.shape}")
        print(f"Number of sequences: {len(lengths)}, Average length: {np.mean(lengths):.1f}")
    
    # train HMM
    # 训练HMM
    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state
    )
    
    model.fit(X_concat, lengths=lengths)
    
    # predict states
    # 预测状态
    states = model.predict(X_concat, lengths=lengths)
    window_df_sorted['state'] = states
    
    if VERBOSE:
        print(f"State classification completed with {n_states} states")
        print(f"State distribution: {np.bincount(states)}")
    
    # create state segments
    # 创建状态段（连续相同状态合并为一段）
    segments = []
    
    for pid, grp in window_df_sorted.groupby('id'):
        grp = grp.sort_values('start_frame').reset_index(drop=True)
        
        if len(grp) == 0:
            continue
        
        cur_state = grp.loc[0, 'state']
        windows = [int(grp.loc[0, 'window_id'])]
        seg_start_frame = int(grp.loc[0, 'start_frame'])
        seg_end_frame = int(grp.loc[0, 'end_frame'])
        
        for i in range(1, len(grp)):
            s = grp.loc[i, 'state']
            
            if s == cur_state:
                # 同一状态，扩展窗口
                # if the state is the same, extend the segment
                windows.append(int(grp.loc[i, 'window_id']))
                seg_end_frame = max(seg_end_frame, int(grp.loc[i, 'end_frame']))
            else:
                # 状态改变，保存当前段
                # if the state changes, save the current segment
                segments.append({
                    'id': pid,
                    'state': int(cur_state),
                    'n_windows': len(windows),
                    'start_frame': seg_start_frame,
                    'end_frame': seg_end_frame,
                    'duration': seg_end_frame - seg_start_frame + 1
                })
                
                # 开始新段
                # start a new segment
                cur_state = s
                windows = [int(grp.loc[i, 'window_id'])]
                seg_start_frame = int(grp.loc[i, 'start_frame'])
                seg_end_frame = int(grp.loc[i, 'end_frame'])
        
        # 保存最后一段
        # save the last segment
        segments.append({
            'id': pid,
            'state': int(cur_state),
            'n_windows': len(windows),
            'start_frame': seg_start_frame,
            'end_frame': seg_end_frame,
            'duration': seg_end_frame - seg_start_frame + 1
        })
    
    segments_df = pd.DataFrame(segments)
    
    if VERBOSE:
        print(f"generate {len(segments_df)} state segments")
    
    # save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存带状态的窗口数据
        # save windows with states
        window_output_path = os.path.join(output_dir, f'windows_with_states_n{n_states}.csv')
        window_df_sorted.to_csv(window_output_path, index=False)
        
        # 保存状态段
        # save state segments
        segments_output_path = os.path.join(output_dir, f'state_segments_n{n_states}.csv')
        segments_df.to_csv(segments_output_path, index=False)
        
        # 保存转移矩阵
        # save transition matrix
        transition_output_path = os.path.join(output_dir, f'transition_matrix_n{n_states}.csv')
        trans_df = pd.DataFrame(
            model.transmat_,
            columns=[f'to_state_{i}' for i in range(n_states)],
            index=[f'from_state_{i}' for i in range(n_states)]
        )
        trans_df.to_csv(transition_output_path)
        
        if VERBOSE:
            print(f"save windows with states to: {window_output_path}")
            print(f"save state segments to: {segments_output_path}")
            print(f"save transition matrix to: {transition_output_path}")
    
    return window_df_sorted, model.transmat_, model, segments_df


def analyze_state_transitions(transition_matrix: np.ndarray,
                              n_states: int,
                              output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    分析状态转移概率
    analze state transition probabilities
    
    参数:
    parameters:
        transition_matrix: 转移概率矩阵
        transistion_matrix: Transition probability matrix
        n_states: 状态数
        n_states: Number of states
        output_dir: 输出目录
        output_dir: Output directory
    
    返回:
    returns:
        状态转移统计数据框
        DataFrame of state transition statistics
    """
    # 计算每个状态的主要转移目标
    transitions = []
    
    for i in range(n_states):
        row = transition_matrix[i]
        
        # 找到最可能的下一个状态
        top_next_states = np.argsort(row)[::-1][:3]  # 前3个
        
        transitions.append({
            'state': i,
            'self_transition_prob': row[i],
            'top_1_next_state': int(top_next_states[0]),
            'top_1_prob': row[top_next_states[0]],
            'top_2_next_state': int(top_next_states[1]),
            'top_2_prob': row[top_next_states[1]],
            'top_3_next_state': int(top_next_states[2]),
            'top_3_prob': row[top_next_states[2]]
        })
    
    trans_df = pd.DataFrame(transitions)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'state_transition_analysis_n{n_states}.csv')
        trans_df.to_csv(output_path, index=False)
        
        if VERBOSE:
            print(f"save state transition analysis to: {output_path}")
    
    return trans_df


def get_state_statistics(window_df_with_states: pd.DataFrame,
                         segments_df: pd.DataFrame,
                         output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    计算状态统计信息
    calculate state statistics
    参数:
    parameters:
        window_df_with_states: 带状态的窗口数据框
        window_df_with_states: DataFrame with window states
        segments_df: 状态段数据框
        segments_df: DataFrame of state segments
        output_dir: 输出目录
        output_dir: Output directory
    
    返回:
    returns:
        状态统计数据框
        DataFrame of state statistics
    """
    stats = []
    
    for state in sorted(window_df_with_states['state'].unique()):
        state_windows = window_df_with_states[window_df_with_states['state'] == state]
        state_segments = segments_df[segments_df['state'] == state]
        
        stats.append({
            'state': state,
            'n_windows': len(state_windows),
            'n_segments': len(state_segments),
            'avg_segment_duration': state_segments['duration'].mean() if len(state_segments) > 0 else 0,
            'median_segment_duration': state_segments['duration'].median() if len(state_segments) > 0 else 0,
            'total_frames': state_segments['duration'].sum() if len(state_segments) > 0 else 0
        })
    
    stats_df = pd.DataFrame(stats)
    
    if VERBOSE:
        print("\nState statistics:")
        print(stats_df.to_string(index=False))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        n_states = len(stats)
        output_path = os.path.join(output_dir, f'state_statistics_n{n_states}.csv')
        stats_df.to_csv(output_path, index=False)
        
        if VERBOSE:
            print(f"save state statistics to: {output_path}")
    
    return stats_df
