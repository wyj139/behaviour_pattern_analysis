"""
主流程文件
功能：整合所有模块，完成完整的数据分析流程
"""
import os
import argparse
import pandas as pd
import torch

from config import (
    OUTPUT_BASE_DIR, WINDOW_SIZE, WINDOW_STRIDE,
    EMBEDDING_DIM, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    N_STATES, UMAP_COLOR_OPTIONS, VERBOSE,
    SKIP_VIS_EMBEDDING_UMAP, SKIP_VIS_COORDINATE_UMAP,
    SKIP_VIS_WINDOW_EMBEDDING_UMAP, SKIP_VIS_WINDOW_COORDINATE_UMAP
)

#from data_prep import prepare_meta_and_timeseries
from windowing import create_windows_by_bodypart
from autoencoder import train_autoencoder, extract_embeddings
from hmm_classifier import train_gaussian_hmm, analyze_state_transitions, get_state_statistics
from visualization import (
    create_multiple_umaps, create_umap_from_coordinates,
    create_umap_from_embeddings, create_expanded_window_umap
)


def run_pipeline(
                output_base_dir: str = OUTPUT_BASE_DIR,
                reference_bodyparts: list = None,
                window_size: int = WINDOW_SIZE,
                window_stride: int = WINDOW_STRIDE,
                emb_dim: int = EMBEDDING_DIM,
                batch_size: int = BATCH_SIZE,
                num_epochs: int = NUM_EPOCHS,
                learning_rate: float = LEARNING_RATE,
                n_states: int = N_STATES,
                color_options: list = None,
                skip_preprocessing: bool = False,
                skip_windowing: bool = True,
                skip_training: bool = True,
                skip_hmm: bool = True,
                skip_visualization: bool = False,
                skip_vis_embedding_umap: bool = SKIP_VIS_EMBEDDING_UMAP,
                skip_vis_coordinate_umap: bool = SKIP_VIS_COORDINATE_UMAP,
                skip_vis_window_embedding_umap: bool = SKIP_VIS_WINDOW_EMBEDDING_UMAP,
                skip_vis_window_coordinate_umap: bool = SKIP_VIS_WINDOW_COORDINATE_UMAP):
    """
    完整的分析流程
    
    参数:
        input_file: 输入数据文件路径
        output_base_dir: 输出基础目录
        reference_bodyparts: 用于计算中心的参考点
        window_size: 窗口大小
        window_stride: 窗口步长
        emb_dim: 嵌入维度
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
        n_states: HMM状态数
        color_options: UMAP染色选项
        skip_preprocessing: 跳过预处理
        skip_windowing: 跳过窗口划分
        skip_training: 跳过模型训练
        skip_hmm: 跳过HMM分类
        skip_visualization: 跳过可视化
    """
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    if VERBOSE:
        print("=" * 80)
        print("开始行为模式分析流程")
        print("=" * 80)
        print(f"输出目录: {output_base_dir}")
        print(f"窗口大小: {window_size}, 步长: {window_stride}")
        print(f"嵌入维度: {emb_dim}, 训练轮数: {num_epochs}")
        print(f"HMM状态数: {n_states}")
        print("=" * 80)
    
    # ========== 步骤1: 数据预处理 ==========
        print("loading preprocessed data...")
        preprocessed_path = os.path.join(output_base_dir, 'preprocessed_data.csv')
        df = pd.read_csv(preprocessed_path)
    
    # ========== 步骤2: 窗口划分 ==========
    if not skip_windowing:
        print("process windowing data...")
        window_df = create_windows_by_bodypart(
            df,
            window_size=window_size,
            stride=window_stride,
            output_dir=output_base_dir
        )
    else:
        print("loading windowed data...")
        window_path = os.path.join(output_base_dir, 'window_data.csv')
        from windowing import load_window_data
        window_df = load_window_data(window_path)
    
    # ========== 步骤3: 训练自动编码器 ==========
    if not skip_training:
        print("training autoencoder model...")
        
        # 获取bodypart数量来确定输入数据的形状
        n_bodyparts = window_df.iloc[0]['n_bodyparts'] if 'n_bodyparts' in window_df.columns else 1
        
        if VERBOSE:
            print(f"检测到 {n_bodyparts} 个bodypart用于autoencoder训练")
            print(f"模型输入形状: (batch_size, {2*n_bodyparts}, {window_size})")
        
        # 为每个bodypart单独训练或合并训练（这里简化为合并）
        model, loss_history = train_autoencoder(
            window_df,
            window_size=window_size,
            emb_dim=emb_dim,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            output_dir=output_base_dir
        )
        
        # 保存模型
        model_path = os.path.join(output_base_dir, 'autoencoder_model.pth')
        torch.save(model.state_dict(), model_path)
        if VERBOSE:
            print(f"save model in: {model_path}")
        
        # 提取嵌入
        print("extracting embeddings...")
        embedding_df = extract_embeddings(
            model,
            window_df,
            window_size=window_size,
            batch_size=batch_size,
            output_dir=output_base_dir
        )
    else:
        print("loading existing embeddings...")
        embedding_path = os.path.join(output_base_dir, 'embeddings.csv')
        embedding_df = pd.read_csv(embedding_path)
    
    # ========== 步骤4: HMM状态分类 ==========
    if not skip_hmm:
        print(f"\n[4/5] HMM state classification (n_states={n_states})...")
        
        window_with_states, trans_matrix, hmm_model, segments_df = train_gaussian_hmm(
            embedding_df,
            n_states=n_states,
            output_dir=output_base_dir
        )
        
        # 分析转移矩阵
        print("analyzing state transitions...")
        analyze_state_transitions(trans_matrix, n_states, output_dir=output_base_dir)
        
        # 计算状态统计
        print("calculating state statistics...")
        get_state_statistics(window_with_states, segments_df, output_dir=output_base_dir)
    else:
        print("\n[4/5] skipping HMM classification...")
        window_with_states_path = os.path.join(output_base_dir, f'windows_with_states_n{n_states}.csv')
        if os.path.exists(window_with_states_path):
            window_with_states = pd.read_csv(window_with_states_path)
        else:
            window_with_states = embedding_df
    
    # ========== 步骤5: 可视化visualization==========
    if not skip_visualization:
        print("\n[5/5] generating visualizations...")
        
        # set up color options
        # 设置染色选项
        if color_options is None:
            color_options = UMAP_COLOR_OPTIONS
        
        # get available colors based on window_with_states in data
        # 从window_with_states中获取可用的染色列
        available_colors = [c for c in color_options if c in window_with_states.columns]
        
        if VERBOSE:
            print(f"available color options: {available_colors}")
        
        #  create UMAPs from embeddings
        #  创建多个UMAP图（基于嵌入）
        if not skip_vis_embedding_umap:
            print("creating multiple UMAPs from embeddings...")
            create_multiple_umaps(
                window_with_states,
                color_by_list=available_colors,
                from_embeddings=True,
                output_dir=output_base_dir,
                title_suffix=f'_n{n_states}'
            )
        else:
            print("skipping embedding UMAP visualization...")
        
        #  create UMAPs from original coordinates (if rel_x, rel_y exist in data)
        #  创建基于原始坐标的UMAP图（如果数据中有rel_x, rel_y）
        if not skip_vis_coordinate_umap:
            if 'rel_x' in df.columns and 'rel_y' in df.columns:
                print("creating UMAPs from original coordinates...")
                coord_colors = [c for c in available_colors if c in df.columns]
                for color_by in coord_colors:
                    try:
                        create_umap_from_coordinates(
                            df,
                            color_by=color_by,
                            hover_cols=['id', 'bodyparts', 'frame'],
                            output_dir=output_base_dir,
                            title_suffix=f"({color_by})"
                        )
                    except Exception as e:
                        if VERBOSE:
                            print(f"Error creating coordinate UMAP ({color_by}): {e}")
        else:
            print("skipping coordinate UMAP visualization...")
        
        # 创建扩展窗口UMAP（基于embedding）
        # Create expanded window UMAP (embedding-based)
        if not skip_vis_window_embedding_umap:
            print("creating expanded window UMAP (embedding-based)...")
            coord_colors = [c for c in available_colors if c in df.columns]
            for color_by in coord_colors:
                try:
                    create_expanded_window_umap(
                        window_df,
                        embedding_df=embedding_df,
                        window_size=window_size,
                        window_id_filter=0,
                        color_by=color_by,
                        use_embeddings=True,
                        output_dir=output_base_dir
                    )
                except Exception as e:
                    if VERBOSE:
                        print(f"Error creating expanded window UMAP (embedding): {e}")
        else:
            print("skipping window embedding UMAP visualization...")
            
        # 创建扩展窗口UMAP（基于坐标）
        # Create expanded window UMAP (coordinate-based)
        if not skip_vis_window_coordinate_umap:
            print("creating expanded window UMAP (coordinate-based)...")
            coord_colors = [c for c in available_colors if c in df.columns]
            for color_by in coord_colors:
                try:
                    create_expanded_window_umap(
                        window_df,
                        embedding_df=None,
                        window_size=window_size,
                        window_id_filter=0,
                        color_by=color_by,
                        use_embeddings=False,
                        output_dir=output_base_dir
                    )
                except Exception as e:
                    if VERBOSE:
                        print(f"Error creating expanded window UMAP (coordinate): {e}")
        else:
            print("skipping window coordinate UMAP visualization...")
    else:
        print("\n[5/5] skipping visualization...")
    
    print("\n" + "=" * 80)
    print("Analysis pipeline completed!")
    print(f"All outputs saved to: {output_base_dir}")
    print("=" * 80)
    
    return {
        'preprocessed_data': df if not skip_preprocessing else None,
        'window_data': window_df if not skip_windowing else None,
        'embeddings': embedding_df if not skip_training else None,
        'windows_with_states': window_with_states if not skip_hmm else None
    }


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='行为模式分析流程')
    
    # # 必需参数
    # parser.add_argument('input_file', type=str, help='输入CSV文件路径')
    
    # 可选参数
    parser.add_argument('--output-dir', type=str, default=OUTPUT_BASE_DIR,
                       help=f'输出目录 (默认: {OUTPUT_BASE_DIR})')
    parser.add_argument('--window-size', type=int, default=WINDOW_SIZE,
                       help=f'窗口大小 (默认: {WINDOW_SIZE})')
    parser.add_argument('--window-stride', type=int, default=WINDOW_STRIDE,
                       help=f'窗口步长 (默认: {WINDOW_STRIDE})')
    parser.add_argument('--emb-dim', type=int, default=EMBEDDING_DIM,
                       help=f'嵌入维度 (默认: {EMBEDDING_DIM})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'批次大小 (默认: {BATCH_SIZE})')
    parser.add_argument('--num-epochs', type=int, default=NUM_EPOCHS,
                       help=f'训练轮数 (默认: {NUM_EPOCHS})')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                       help=f'学习率 (默认: {LEARNING_RATE})')
    parser.add_argument('--n-states', type=int, default=N_STATES,
                       help=f'HMM状态数 (默认: {N_STATES})')
    parser.add_argument('--reference-bodyparts', type=str, nargs='+',
                       help='用于计算中心的参考bodypart列表')
    
    # 跳过选项
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='跳过预处理步骤')
    parser.add_argument('--skip-windowing', action='store_true',
                       help='跳过窗口划分步骤')
    parser.add_argument('--skip-training', action='store_true',
                       help='跳过模型训练步骤')
    parser.add_argument('--skip-hmm', action='store_true',
                       help='跳过HMM分类步骤')
    parser.add_argument('--skip-visualization', action='store_true',
                       help='跳过可视化步骤')
    
    args = parser.parse_args()
    
    # 运行流程
    run_pipeline(
        
        output_base_dir=args.output_dir,
        reference_bodyparts=args.reference_bodyparts,
        window_size=args.window_size,
        window_stride=args.window_stride,
        emb_dim=args.emb_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        n_states=args.n_states,
        skip_preprocessing=args.skip_preprocessing,
        skip_windowing=True,
        skip_training=True,
        skip_hmm=True,
        skip_visualization=args.skip_visualization
    )


if __name__ == '__main__':
    main()
