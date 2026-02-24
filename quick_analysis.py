#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速Window和MaxWID分组分析
Quick Window and MaxWID Group Analysis
基于已生成的自信度数据进行简单统计分析
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config import OUTPUT_BASE_DIR

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def analyze_from_confidence_data():
    """
    从已有的自信度数据进行window分析
    """
    print("\n" + "="*60)
    print("快速Window和MaxWID分析")
    print("="*60)
    
    output_dir = OUTPUT_BASE_DIR
    
    # 加载自信度预测数据
    confidence_pred_path = os.path.join(output_dir, 'confidence_analysis', 'confidence_predictions.csv')
    window_data_path = os.path.join(output_dir, 'window_data1.csv')
    
    print("\n加载数据...")
    conf_df = pd.read_csv(confidence_pred_path)
    window_df = pd.read_csv(window_data_path)
    
    # 获取每个window_id对应的最大window_id
    window_maxwid = window_df[['window_id', 'max_window_id']].drop_duplicates()
    
    # 合并
    df = conf_df.merge(window_maxwid, on='window_id', how='left')
    
    print(f"✓ 加载数据: {len(df)} 样本")
    print(f"✓ Window范围: {df['window_id'].min()} - {df['window_id'].max()}")
    print(f"✓ MaxWID范围: {df['max_window_id'].min()} - {df['max_window_id'].max()}")
    
    # ===== Window独立分析 =====
    print("\n生成Window独立分析...")
    
    window_output_dir = os.path.join(output_dir, 'window_independent_analysis')
    os.makedirs(window_output_dir, exist_ok=True)
    
    # 按window统计
    window_stats = []
    for wid in sorted(df['window_id'].unique()):
        wdata = df[df['window_id'] == wid]
        window_stats.append({
            'window_id': wid,
            'n_samples': len(wdata),
            'mean_confidence': wdata['confidence'].mean(),
            'std_confidence': wdata['confidence'].std(),
            'min_confidence': wdata['confidence'].min(),
            'max_confidence': wdata['confidence'].max(),
            'nogo_samples': len(wdata[wdata['decision'] == 0]),
            'go_samples': len(wdata[wdata['decision'] == 1])
        })
    
    window_summary = pd.DataFrame(window_stats)
    
    # 保存window统计
    window_summary.to_csv(os.path.join(window_output_dir, 'window_analysis_summary.csv'), index=False)
    print(f"✓ 已保存: window_analysis_summary.csv ({len(window_summary)} windows)")
    
    # 绘制window分析图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 按window的平均自信度
    ax = axes[0, 0]
    ax.plot(window_summary['window_id'], window_summary['mean_confidence'], 'o-', 
            linewidth=2, markersize=8, color='steelblue')
    ax.fill_between(window_summary['window_id'], 
                     window_summary['mean_confidence'] - window_summary['std_confidence'],
                     window_summary['mean_confidence'] + window_summary['std_confidence'],
                     alpha=0.2)
    ax.set_xlabel('Window ID', fontsize=11)
    ax.set_ylabel('Mean Confidence', fontsize=11)
    ax.set_title('Confidence vs Window ID', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.05])
    
    # 图2: 样本分布
    ax = axes[0, 1]
    ax.bar(window_summary['window_id'], window_summary['n_samples'], 
           color='skyblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Window ID', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('Sample Count by Window', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图3: NOGO vs GO分布
    ax = axes[1, 0]
    width = 0.35
    x = window_summary['window_id']
    ax.bar(x - width/2, window_summary['nogo_samples'], width, label='NOGO', color='coral')
    ax.bar(x + width/2, window_summary['go_samples'], width, label='GO', color='lightgreen')
    ax.set_xlabel('Window ID', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('NOGO vs GO Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图4: 自信度范围
    ax = axes[1, 1]
    ax.fill_between(window_summary['window_id'], 
                     window_summary['min_confidence'],
                     window_summary['max_confidence'],
                     alpha=0.3, color='steelblue')
    ax.plot(window_summary['window_id'], window_summary['mean_confidence'], 'o-',
            linewidth=2, markersize=6, color='darkblue', label='Mean')
    ax.set_xlabel('Window ID', fontsize=11)
    ax.set_ylabel('Confidence', fontsize=11)
    ax.set_title('Confidence Range by Window', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(os.path.join(window_output_dir, 'window_analysis_overview.png'), 
                dpi=100, bbox_inches='tight')
    print(f"✓ 已保存: window_analysis_overview.png")
    plt.close()
    
    # ===== MaxWID分组分析 =====
    print("\n生成MaxWID分组分析...")
    
    maxwid_output_dir = os.path.join(output_dir, 'maxwid_group_analysis')
    os.makedirs(maxwid_output_dir, exist_ok=True)
    
    # 创建maxwid分组（每50或100个间隔）
    df['maxwid_group'] = pd.cut(df['max_window_id'], bins=5, labels=False)
    
    # 按maxwid分组统计
    group_stats = []
    for gid in sorted(df['maxwid_group'].unique()):
        gdata = df[df['maxwid_group'] == gid]
        maxwid_range = gdata['max_window_id'].min(), gdata['max_window_id'].max()
        group_stats.append({
            'group_id': int(gid),
            'maxwid_range': f"{int(maxwid_range[0])}-{int(maxwid_range[1])}",
            'n_samples': len(gdata),
            'mean_confidence': gdata['confidence'].mean(),
            'std_confidence': gdata['confidence'].std(),
            'nogo_samples': len(gdata[gdata['decision'] == 0]),
            'go_samples': len(gdata[gdata['decision'] == 1])
        })
    
    group_summary = pd.DataFrame(group_stats)
    
    # 保存分组统计
    group_summary.to_csv(os.path.join(maxwid_output_dir, 'maxwid_group_summary.csv'), index=False)
    print(f"✓ 已保存: maxwid_group_summary.csv ({len(group_summary)} groups)")
    
    # 绘制分组分析图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 按分组的平均自信度
    ax = axes[0, 0]
    ax.bar(group_summary['group_id'], group_summary['mean_confidence'], 
           color='steelblue', alpha=0.7, edgecolor='navy')
    ax.set_xlabel('MaxWID Group', fontsize=11)
    ax.set_ylabel('Mean Confidence', fontsize=11)
    ax.set_title('Confidence by MaxWID Group', fontsize=12, fontweight='bold')
    ax.set_xticklabels(group_summary['maxwid_range'], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # 图2: 样本分布
    ax = axes[0, 1]
    ax.bar(group_summary['group_id'], group_summary['n_samples'], 
           color='skyblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('MaxWID Group', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('Sample Count by Group', fontsize=12, fontweight='bold')
    ax.set_xticklabels(group_summary['maxwid_range'], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图3: NOGO vs GO分布
    ax = axes[1, 0]
    width = 0.35
    x = group_summary['group_id']
    ax.bar(x - width/2, group_summary['nogo_samples'], width, label='NOGO', color='coral')
    ax.bar(x + width/2, group_summary['go_samples'], width, label='GO', color='lightgreen')
    ax.set_xlabel('MaxWID Group', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('NOGO vs GO Distribution', fontsize=12, fontweight='bold')
    ax.set_xticklabels(group_summary['maxwid_range'], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 图4: 每个分组中window数分布
    ax = axes[1, 1]
    windows_per_group = df.groupby('maxwid_group')['window_id'].nunique()
    ax.bar(group_summary['group_id'], windows_per_group.values, 
           color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax.set_xlabel('MaxWID Group', fontsize=11)
    ax.set_ylabel('Number of Unique Windows', fontsize=11)
    ax.set_title('Window Diversity by Group', fontsize=12, fontweight='bold')
    ax.set_xticklabels(group_summary['maxwid_range'], rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(maxwid_output_dir, 'maxwid_group_overview.png'), 
                dpi=100, bbox_inches='tight')
    print(f"✓ 已保存: maxwid_group_overview.png")
    plt.close()
    
    print("\n" + "="*60)
    print("✓ 分析完成！")
    print("="*60)
    print(f"\nWindow分析结果: {window_output_dir}")
    print(f"MaxWID分组分析结果: {maxwid_output_dir}")


if __name__ == '__main__':
    analyze_from_confidence_data()
