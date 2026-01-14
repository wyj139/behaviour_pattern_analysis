"""
visualization.py
generate UMAP plots

"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap.umap_ as umap
from typing import List, Optional

from config import UMAP_N_COMPONENTS, UMAP_RANDOM_STATE, VERBOSE


def create_umap_from_coordinates(df: pd.DataFrame,
                                 color_by: str = 'Housing',
                                 hover_cols: Optional[List[str]] = None,
                                 output_dir: Optional[str] = None,
                                 title_suffix: str = '') -> go.Figure:
    """
    基于相对坐标创建UMAP图 - 读取每个id的全部坐标（所有bodypart的所有帧）
    Build UMAP using all coordinates for each id (all bodyparts, all frames)
    
    参数 Parameters:
        df: 预处理后的数据框，包含 id, bodypart, frame, rel_x, rel_y 列
            Preprocessed dataframe with id, bodypart, frame, rel_x, rel_y columns
        color_by: 用于染色的列名 (Column name for coloring)
        hover_cols: 悬停时显示的列 (Columns to show on hover)
        output_dir: 输出目录 (Output directory)
        title_suffix: 标题后缀 (Title suffix)
    
    返回 Returns:
        Plotly图形对象 (Plotly figure object)
    """
    # 按 id 分组，提取每个 id 的所有坐标作为特征
    # Group by id and extract all coordinates as features
    features_list = []
    id_metadata = []
    
    for id_val, group in df.groupby('id'):
        group = group.sort_values(['bodypart', 'frame'])
        
        # 收集该 id 下所有 bodypart 的所有 rel_x 和 rel_y
        # Collect all rel_x and rel_y for all bodyparts under this id
        all_coords = []
        for bodypart, bp_group in group.groupby('bodypart'):
            bp_group = bp_group.sort_values('frame')
            all_coords.extend(bp_group['rel_x'].values)
            all_coords.extend(bp_group['rel_y'].values)
        
        features_list.append(all_coords)
        
        # 保存元数据（每个id一条）
        # Save metadata (one entry per id)
        meta_row = {'id': id_val}
        for col in ['Housing', 'pig', 'cue', 'decision', 'test', 'trial']:
            if col in group.columns:
                meta_row[col] = group[col].iloc[0]
        id_metadata.append(meta_row)
    
    # 将所有特征填充到相同长度（用0填充）
    # Pad all features to the same length (pad with 0)
    max_len = max(len(f) for f in features_list)
    X = np.array([f + [0] * (max_len - len(f)) for f in features_list])
    
    # 执行UMAP降维 (Perform UMAP dimensionality reduction)
    reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, random_state=UMAP_RANDOM_STATE)
    Z = reducer.fit_transform(X)
    
    # 创建用于绘图的数据框 (Create dataframe for plotting)
    df_plot = pd.DataFrame(id_metadata)
    df_plot['umap_1'] = Z[:, 0]
    df_plot['umap_2'] = Z[:, 1]
    
    # 设置悬停列 (Set hover columns)
    if hover_cols is None:
        hover_cols = ['id']
    hover_cols = [c for c in hover_cols if c in df_plot.columns]
    
    # 确保染色列为字符串类型 (Ensure color_by column is string type)
    if color_by in df_plot.columns:
        df_plot[color_by] = df_plot[color_by].astype(str)
    
    # create figure
    # 创建图形
    fig = px.scatter(
        df_plot, 
        x='umap_1', 
        y='umap_2', 
        color=color_by,
        hover_data=hover_cols,
        title=f'UMAP (coordinates) - colored by {color_by}{title_suffix}'
    )
    
    fig.update_layout(
        width=800,
        height=800,
        dragmode='zoom'
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    # save figure
    # 保存图形
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'umap_coords_{color_by}{title_suffix}.html')
        fig.write_html(output_path)
        
        if VERBOSE:
            print(f"保存UMAP图到: {output_path}")
    
    return fig


def create_umap_from_embeddings(embedding_df: pd.DataFrame,
                                color_by: str = 'Housing',
                                hover_cols: Optional[List[str]] = None,
                                output_dir: Optional[str] = None,
                                title_suffix: str = '') -> go.Figure:
    """
    基于嵌入特征创建UMAP图
    build UMAP using embedding features
    
    参数:
    parameters:
        embedding_df: dataframe containing emb_* columns
        color_by: column name to color by
        hover_cols: columns to show on hover
        output_dir: output directory
        title_suffix: suffix for the title
        embedding_df: 包含 emb_* 列的数据框
        color_by: 用于染色的列名
        hover_cols: 悬停时显示的列
        output_dir: 输出目录
        title_suffix: 标题后缀
    
    返回:
    return:
        plotly.graph_objects.
        Plotly图形对象
    """
    # extract embedding columns
    # 提取嵌入列
    emb_cols = [c for c in embedding_df.columns if c.startswith('emb_')]
    
    if len(emb_cols) == 0:
        raise ValueError("数据框中没有找到嵌入列（emb_*）")
    
    X = embedding_df[emb_cols].values
    
    # perform UMAP dimensionality reduction
    # 执行UMAP降维
    reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, random_state=UMAP_RANDOM_STATE)
    Z = reducer.fit_transform(X)
    
    # add UMAP coordinates
    # 添加UMAP坐标
    df_plot = embedding_df.copy()
    df_plot['umap_1'] = Z[:, 0]
    df_plot['umap_2'] = Z[:, 1]
    
    # set hover columns
    # 设置悬停列
    if hover_cols is None:
        hover_cols = ['id', 'window_id']
    hover_cols = [c for c in hover_cols if c in df_plot.columns]
    
    # ensure color_by column is string type
    # 确保染色列为字符串类型（对于分类变量）
    if color_by in df_plot.columns:
        if color_by in ['state', 'pig', 'cue', 'Housing', 'decision']:
            df_plot[color_by] = df_plot[color_by].astype(str)
    
    # create figure
    # 创建图形
    fig = px.scatter(
        df_plot,
        x='umap_1',
        y='umap_2',
        color=color_by,
        hover_data=hover_cols,
        title=f'UMAP (embeddings) - colored by {color_by}{title_suffix}',
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )
    
    fig.update_layout(
        width=800,
        height=800,
        dragmode='zoom'
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    # save figure
    # 保存图形
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'umap_emb_{color_by}{title_suffix}.html')
        fig.write_html(output_path)
        
        if VERBOSE:
            print(f"保存UMAP图到: {output_path}")
    
    return fig


def create_multiple_umaps(df: pd.DataFrame,
                         color_by_list: List[str],
                         from_embeddings: bool = True,
                         output_dir: Optional[str] = None,
                         title_suffix: str = '') -> List[go.Figure]:
    """
    创建多个UMAP图，使用不同的染色条件
    
    参数:
        df: 数据框
        color_by_list: 染色列名列表
        from_embeddings: 是否使用嵌入特征（否则使用坐标）
        output_dir: 输出目录
        title_suffix: 标题后缀
    
    返回:
        图形对象列表
    """
    figures = []
    
    for color_by in color_by_list:
        if color_by not in df.columns:
            if VERBOSE:
                print(f"警告: 列 '{color_by}' 不存在，跳过")
            continue
        
        try:
            if from_embeddings:
                fig = create_umap_from_embeddings(
                    df, 
                    color_by=color_by, 
                    output_dir=output_dir,
                    title_suffix=title_suffix
                )
            else:
                fig = create_umap_from_coordinates(
                    df,
                    color_by=color_by,
                    output_dir=output_dir,
                    title_suffix=title_suffix
                )
            
            figures.append(fig)
            
        except Exception as e:
            if VERBOSE:
                print(f"创建 {color_by} 的UMAP图时出错: {e}")
    
    return figures


def create_3d_trajectory_plot(traj_df: pd.DataFrame,
                              segments_df: pd.DataFrame,
                              target_id: str,
                              bodypart: Optional[str] = None,
                              output_dir: Optional[str] = None) -> go.Figure:
    """
    创建3D轨迹图，按状态染色，支持选择特定bodypart
    Create 3D trajectory plot colored by state, supports selecting specific bodypart
    
    参数 Parameters:
        traj_df: 轨迹数据框（包含 id, bodypart, frame, rel_x, rel_y）
                 Trajectory dataframe (containing id, bodypart, frame, rel_x, rel_y)
        segments_df: 状态段数据框 (State segments dataframe)
        target_id: 目标ID (Target ID)
        bodypart: 要绘制的bodypart名称，None表示绘制所有bodypart
                  Bodypart name to plot, None means plot all bodyparts
        output_dir: 输出目录 (Output directory)
    
    返回 Returns:
        Plotly 3D图形对象 (Plotly 3D figure object)
    """
    # 过滤指定ID (Filter by target ID)
    segs_id = segments_df[segments_df['id'].astype(str) == str(target_id)].copy()
    traj_id = traj_df[traj_df['id'].astype(str) == str(target_id)].copy()
    
    # 如果指定了bodypart，进一步过滤 (Further filter by bodypart if specified)
    if bodypart is not None:
        traj_id = traj_id[traj_id['bodypart'] == bodypart].copy()
        if len(traj_id) == 0:
            raise ValueError(f"未找到ID为 {target_id}、bodypart为 {bodypart} 的轨迹数据")
    else:
        if len(traj_id) == 0:
            raise ValueError(f"未找到ID为 {target_id} 的轨迹数据")
    
    traj_id = traj_id.sort_values('frame')
    
    # 颜色映射 (Color mapping)
    palette = px.colors.qualitative.Plotly
    states = sorted(segs_id['state'].unique()) if len(segs_id) > 0 else []
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(states)}
    
    fig = go.Figure()
    
    # 如果没有指定bodypart，按bodypart分别绘制
    # If no bodypart specified, plot each bodypart separately
    if bodypart is None:
        bodyparts = traj_id['bodypart'].unique()
        for bp in bodyparts:
            bp_traj = traj_id[traj_id['bodypart'] == bp].sort_values('frame')
            
            # 绘制该bodypart的完整轨迹（灰色背景）
            # Plot full trajectory for this bodypart (gray background)
            fig.add_trace(go.Scatter3d(
                x=bp_traj['rel_x'],
                y=bp_traj['rel_y'],
                z=bp_traj['frame'],
                mode='lines',
                line=dict(color='lightgray', width=1),
                name=f'{bp} (full)',
                showlegend=True
            ))
            
            # 按状态段绘制该bodypart
            # Plot this bodypart by state segments
            for _, row in segs_id.iterrows():
                start_frame, end_frame = int(row['start_frame']), int(row['end_frame'])
                state = int(row['state'])
                
                seg_traj = bp_traj[(bp_traj['frame'] >= start_frame) & 
                                  (bp_traj['frame'] <= end_frame)]
                
                if len(seg_traj) == 0:
                    continue
                
                fig.add_trace(go.Scatter3d(
                    x=seg_traj['rel_x'],
                    y=seg_traj['rel_y'],
                    z=seg_traj['frame'],
                    mode='lines+markers',
                    line=dict(color=color_map[state], width=3),
                    marker=dict(size=2, color=color_map[state]),
                    name=f'{bp} - state {state}',
                    showlegend=True
                ))
    else:
        # 指定了bodypart，只绘制该bodypart
        # Bodypart specified, only plot that bodypart
        # 绘制完整轨迹（灰色背景）(Plot full trajectory in gray)
        fig.add_trace(go.Scatter3d(
            x=traj_id['rel_x'],
            y=traj_id['rel_y'],
            z=traj_id['frame'],
            mode='lines',
            line=dict(color='lightgray', width=2),
            name=f'{target_id} {bodypart} full'
        ))
        
        # 按状态段绘制 (Plot by state segments)
        for _, row in segs_id.iterrows():
            start_frame, end_frame = int(row['start_frame']), int(row['end_frame'])
            state = int(row['state'])
            
            seg_traj = traj_id[(traj_id['frame'] >= start_frame) & 
                              (traj_id['frame'] <= end_frame)]
            
            if len(seg_traj) == 0:
                continue
            
            fig.add_trace(go.Scatter3d(
                x=seg_traj['rel_x'],
                y=seg_traj['rel_y'],
                z=seg_traj['frame'],
                mode='lines+markers',
                line=dict(color=color_map[state], width=4),
                marker=dict(size=2, color=color_map[state]),
                name=f'state {state}'
            ))
    
    # 更新布局 (Update layout)
    bp_suffix = f'_{bodypart}' if bodypart else '_all_bodyparts'
    fig.update_layout(
        scene=dict(
            xaxis_title='rel_x',
            yaxis_title='rel_y',
            zaxis_title='frame'
        ),
        title=f'{target_id}{bp_suffix} - 3D trajectory by HMM state',
        height=800
    )
    
    # 保存图形 (Save figure)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'3d_trajectory_{target_id}{bp_suffix}.html')
        fig.write_html(output_path)
        
        if VERBOSE:
            print(f"保存3D轨迹图到: {output_path}")
    
    return fig


def create_expanded_window_umap(window_df: pd.DataFrame,
                                embedding_df: Optional[pd.DataFrame] = None,
                                window_size: int = 50,
                                window_id_filter: int = 0,
                                color_by: str = 'Housing',
                                use_embeddings: bool = True,
                                output_dir: Optional[str] = None) -> go.Figure:
    """
    创建第一个窗口的UMAP图（支持使用embedding或coordinate）
    Create UMAP for the first window (supports using embeddings or coordinates)
    
    参数 Parameters:
        window_df: 窗口数据框 (Window dataframe)
        embedding_df: 嵌入数据框，如果use_embeddings=True则必需 (Embedding dataframe, required if use_embeddings=True)
        window_size: 窗口大小 (Window size)
        window_id_filter: 窗口ID过滤器，默认0（第一个窗口）(Window ID filter, default 0 for first window)
        color_by: 染色列 (Column for coloring)
        use_embeddings: True=使用embedding，False=使用coordinate (True=use embeddings, False=use coordinates)
        output_dir: 输出目录 (Output directory)
    
    返回 Returns:
        Plotly图形对象 (Plotly figure object)
    """
    if use_embeddings:
        # 模式1：使用embedding值绘制UMAP
        # Mode 1: Use embedding values for UMAP
        if embedding_df is None:
            raise ValueError("use_embeddings=True requires embedding_df to be provided")
        
        # 过滤第一个窗口
        # Filter first window
        first_window = embedding_df[embedding_df['window_id'] == window_id_filter].copy()
        
        # 提取embedding列
        # Extract embedding columns
        emb_cols = [c for c in first_window.columns if c.startswith('emb_')]
        X = first_window[emb_cols].values
        
        df_plot = first_window.copy()
        title_prefix = 'Embeddings'
        file_suffix = 'emb'
        
    else:
        # 模式2：使用所有bodypart的coordinate值绘制UMAP
        # Mode 2: Use all bodypart coordinates for UMAP
        first_window = window_df[window_df['window_id'] == window_id_filter].copy()
        
        # 展开所有bodypart的coordinate为特征向量
        # Expand all bodypart coordinates as feature vector
        features_list = []
        for _, row in first_window.iterrows():
            all_rel_x = row['all_window_rel_x']
            all_rel_y = row['all_window_rel_y']
            
            # 展平所有bodypart的坐标
            # Flatten coordinates of all bodyparts
            features = []
            for rel_x, rel_y in zip(all_rel_x, all_rel_y):
                features.extend(rel_x)
                features.extend(rel_y)
            features_list.append(features)
        
        X = np.array(features_list)
        df_plot = first_window.copy()
        title_prefix = 'Coordinates'
        file_suffix = 'coords'
    
    # UMAP降维 (UMAP dimensionality reduction)
    reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS, random_state=UMAP_RANDOM_STATE)
    Z = reducer.fit_transform(X)
    
    df_plot['umap_1'] = Z[:, 0]
    df_plot['umap_2'] = Z[:, 1]
    
    # 确保染色列为字符串 (Ensure color_by is string)
    if color_by in df_plot.columns:
        df_plot[color_by] = df_plot[color_by].astype(str)
    
    # 创建图形 (Create figure)
    fig = px.scatter(
        df_plot,
        x='umap_1',
        y='umap_2',
        color=color_by,
        hover_data=['id', 'window_id'],
        title=f'UMAP ({title_prefix}, Window {window_id_filter}) - colored by {color_by}'
    )
    
    fig.update_layout(width=800, height=800, dragmode='zoom')
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    # 保存图形 (Save figure)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'umap_window{window_id_filter}_{file_suffix}_{color_by}.html')
        fig.write_html(output_path)
        
        if VERBOSE:
            print(f"保存窗口UMAP图到: {output_path}")
    
    return fig
