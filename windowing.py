"""
windowing.py
sliding window
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional

from config import WINDOW_SIZE, WINDOW_STRIDE, VERBOSE


def sliding_window(df: pd.DataFrame, 
                   window_size: int = WINDOW_SIZE,
                   stride: int = WINDOW_STRIDE,
                   id_col: str = 'id',
                   frame_col: str = 'frame',
                   bodypart_col: str = 'bodypart') -> pd.DataFrame:
    """
    group by id and bodypart, then create sliding windows for rel_x and rel_y
    parameters:
        df: input dataframe
        window_size: size of the window
        stride: sliding step
        id_col: column name for unique identifier
        frame_col: column name for frame index
        bodypart_col: column name for bodypart
    returns:
        DataFrame containing sliding windows
    """
    results = []
    
    # 获取需要处理的列
    required_cols = [id_col, frame_col, 'rel_x', 'rel_y']
    optional_cols = ['Housing', 'pig', 'cue', 'decision', 'test', 'trial']
    
    cols_to_keep = [c for c in required_cols if c in df.columns]
    for col in optional_cols:
        if col in df.columns:
            cols_to_keep.append(col)
    
    df_window = df[cols_to_keep].copy()
    df_window = df_window.sort_values([id_col, frame_col])
    
    for id_val, group in df_window.groupby(id_col):
        group = group.reset_index(drop=True)
        
        rel_x = group['rel_x'].values
        rel_y = group['rel_y'].values
        frames = group[frame_col].values
        
        # 插值处理 NaN 值
        sx = pd.Series(rel_x).interpolate(limit_direction='both')
        sx = sx.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        rel_x = sx.values.astype(float)
        
        sy = pd.Series(rel_y).interpolate(limit_direction='both')
        sy = sy.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
        rel_y = sy.values.astype(float)
        
        # 如果数据不足一个窗口，用最后一帧填充
        if len(rel_x) < window_size:
            last_x = rel_x[-1] if len(rel_x) > 0 else 0.0
            last_y = rel_y[-1] if len(rel_y) > 0 else 0.0
            last_frame = frames[-1] if len(frames) > 0 else 0
            padding_length = window_size - len(rel_x)
            
            rel_x = np.concatenate([rel_x, np.full(padding_length, last_x)])
            rel_y = np.concatenate([rel_y, np.full(padding_length, last_y)])
            frames = np.concatenate([frames, np.full(padding_length, last_frame)])
        
        # 获取元数据（每个id只取一次）
        meta_data = {}
        for col in optional_cols:
            if col in group.columns and len(group[col]) > 0:
                meta_data[col] = group[col].iloc[0]
        
        # 滑动窗口
        window_id = 0
        for start in range(0, len(rel_x) - window_size + 1, stride):
            end = start + window_size
            
            window_data = {
                id_col: id_val,
                'window_id': window_id,
                'start_frame': int(frames[start]),
                'end_frame': int(frames[end - 1]),
                'window_rel_x': rel_x[start:end],
                'window_rel_y': rel_y[start:end],
            }
            
            # 添加元数据
            window_data.update(meta_data)
            
            results.append(window_data)
            window_id += 1
    
    result_df = pd.DataFrame(results)
    
    if VERBOSE:
        print(f"滑动窗口划分完成")
        print(f"  窗口大小: {window_size}, 步长: {stride}")
        print(f"  生成窗口数: {len(result_df)}")
        print(f"  唯一ID数: {result_df[id_col].nunique()}")
    
    return result_df


def create_windows_by_bodypart(df: pd.DataFrame,
                               window_size: int = WINDOW_SIZE,
                               stride: int = WINDOW_STRIDE,
                               output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    create windows for the dataframe, each window contains data for all bodyparts (multi-channel)
    
    Parameters:
        df: input dataframe
        window_size: size of the window
        stride: sliding step
        output_dir: output directory
    
    Returns:
        DataFrame containing windows for all bodyparts, each window includes all bodyparts
    """
    # 检查有多少个不同的bodypart
    bodyparts = sorted(df['bodypart'].unique())
    n_bodyparts = len(bodyparts)
    
    if VERBOSE:
        print(f"检测到 {n_bodyparts} 个bodypart: {bodyparts}")
    
    # 为每个bodypart分别创建窗口
    bodypart_windows = {}
    for bodypart in bodyparts:
        df_bp = df[df['bodypart'] == bodypart].copy()
        windows = sliding_window(df_bp, window_size=window_size, stride=stride)
        bodypart_windows[bodypart] = windows
    
    # merge all bodypart windows into one dataframe
    # 合并所有bodypart的窗口：每个(id, window_id)组合成一个样本
    # first get all unique (id, window_id) combinations
    # 首先获取所有唯一的(id, window_id)组合
    first_bp_windows = bodypart_windows[bodyparts[0]]
    combined_windows = []
    
    for _, row in first_bp_windows.iterrows():
        id_val = row['id']
        window_id = row['window_id']
        
        # collect data for all bodyparts for this (id, window_id)
        # 收集该window下所有bodypart的rel_x和rel_y
        all_rel_x = []
        all_rel_y = []
        
        for bp in bodyparts:
            bp_windows = bodypart_windows[bp]
            bp_row = bp_windows[(bp_windows['id'] == id_val) & 
                               (bp_windows['window_id'] == window_id)]
            
            if len(bp_row) > 0:
                all_rel_x.append(bp_row.iloc[0]['window_rel_x'])
                all_rel_y.append(bp_row.iloc[0]['window_rel_y'])
            else:
                # if this bodypart is missing, fill with zeros
                # 如果某个bodypart缺失，用零填充
                all_rel_x.append(np.zeros(window_size))
                all_rel_y.append(np.zeros(window_size))
        
        # create merged window record
        # 创建合并后的窗口记录
        merged_row = {
            'id': id_val,
            'window_id': window_id,
            'start_frame': row['start_frame'],
            'end_frame': row['end_frame'],
            'all_window_rel_x': all_rel_x,  # list of arrays, one per bodypart
            'all_window_rel_y': all_rel_y,  # list of arrays, one per bodypart
            'bodyparts': bodyparts,  # 保存bodypart顺序
            'n_bodyparts': n_bodyparts
        }
        
        # add metadata
        # 添加元数据
        for col in ['Housing', 'pig', 'cue', 'decision', 'test', 'trial']:
            if col in row.index:
                merged_row[col] = row[col]
        
        combined_windows.append(merged_row)
    
    combined_windows = pd.DataFrame(combined_windows)
    
    # save window data
    # 保存窗口数据
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'window_data1.csv')
        
        # 保存时需要特殊处理数组列
        save_df = combined_windows.copy()
        # 将list of arrays转换为字符串
        save_df['all_window_rel_x'] = save_df['all_window_rel_x'].apply(
            lambda x: ';'.join([','.join(map(str, arr)) for arr in x])
        )
        save_df['all_window_rel_y'] = save_df['all_window_rel_y'].apply(
            lambda x: ';'.join([','.join(map(str, arr)) for arr in x])
        )
        save_df['bodyparts'] = save_df['bodyparts'].apply(lambda x: ','.join(x))
        save_df.to_csv(output_path, index=False)
        
        if VERBOSE:
            print(f"保存窗口数据到: {output_path}")
    
    if VERBOSE:
        print(f"窗口划分完成，每个窗口包含 {n_bodyparts} 个bodypart")
    
    return combined_windows


def load_window_data(file_path: str) -> pd.DataFrame:
    """
    load window data from CSV file
    Parameters:
        file_path: path to the CSV file
    Returns:
        DataFrame containing window data
    """
    df = pd.read_csv(file_path)
    
    # 将字符串转回list of arrays
    df['all_window_rel_x'] = df['all_window_rel_x'].apply(
        lambda x: [np.array([float(v) for v in arr.split(',')]) 
                   for arr in x.split(';')] if isinstance(x, str) else x
    )
    df['all_window_rel_y'] = df['all_window_rel_y'].apply(
        lambda x: [np.array([float(v) for v in arr.split(',')]) 
                   for arr in x.split(';')] if isinstance(x, str) else x
    )
    df['bodyparts'] = df['bodyparts'].apply(
        lambda x: x.split(',') if isinstance(x, str) else x
    )
    
    if VERBOSE:
        print(f"加载窗口数据: {file_path}")
        print(f"窗口数: {len(df)}")
    
    return df
