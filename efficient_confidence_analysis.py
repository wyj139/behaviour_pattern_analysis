#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测自信度分析 - Prediction Confidence Analysis

使用全局 RF 模型计算每个样本的预测自信度（max predicted probability），
分析自信度随 window_id、decision、Housing 的变化规律。

数据过滤: window_id <= 各 trial 最大 window_id 的 95th percentile

输出: OUTPUT_BASE_DIR/confidence_analysis/
  confidence_by_window.png    - 随 window_id 变化趋势（按 decision / housing）
  confidence_distributions.png - 整体分布（直方图 + 箱线图）
  confidence_by_window.csv    - 每个 window_id 统计摘要（按真实 window_id 分组）
  confidence_predictions.csv  - 每个样本预测结果
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
from config import OUTPUT_BASE_DIR

plt.style.use("seaborn-v0_8-darkgrid")

# ---------- 工具函数 -----------------------------------------------------------

def _is_readable(path):
    """检查文件是否为真实内容（非 OneDrive 占位符）"""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as f:
            return len(f.readline().strip()) > 0
    except Exception:
        return False


def _load_embeddings(base):
    """加载 embeddings；若文件不可读则用已训练模型重新推理"""
    emb_path = os.path.join(base, "embeddings.csv")
    if _is_readable(emb_path):
        return pd.read_csv(emb_path)

    print("  embeddings.csv 不可用，从已训练模型推理...")
    import torch
    from autoencoder import ConvAutoencoder, extract_embeddings
    from config import WINDOW_SIZE, EMBEDDING_DIM

    window_df = pd.read_csv(os.path.join(base, "window_data1.csv"))
    sample = window_df.iloc[0]
    n_bodyparts = int(sample["n_bodyparts"]) if "n_bodyparts" in window_df.columns \
        else len(str(sample["all_window_rel_x"]).split(";"))

    model = ConvAutoencoder(window_size=WINDOW_SIZE, n_bodyparts=n_bodyparts, emb_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load(
        os.path.join(base, "autoencoder_model.pth"),
        map_location="cpu", weights_only=True
    ))
    model.eval()

    def parse_traj(col):
        return col.apply(lambda x: [np.array(s.split(","), dtype=float)
                                     for s in x.split(";")]
                         if isinstance(x, str) else x)
    window_df["all_window_rel_x"] = parse_traj(window_df["all_window_rel_x"])
    window_df["all_window_rel_y"] = parse_traj(window_df["all_window_rel_y"])

    emb_df = extract_embeddings(model, window_df, window_size=WINDOW_SIZE, output_dir=base)
    del model, window_df
    gc.collect()
    return emb_df


def load_data(trial_pct=0.95):
    """
    加载 window 元数据 + embeddings，过滤大 window_id。

    trial_pct=0.95  =>  截断 window_id 到各 trial 最大 window_id 的 95th percentile
    返回: df（含 window_id 列）、emb_cols 列表、max_wid 截断值
    """
    base = OUTPUT_BASE_DIR
    meta_path = os.path.join(base, "window_data1.csv")

    full_meta = pd.read_csv(meta_path,
                             usecols=["window_id", "decision", "Housing", "pig", "trial"])

    trial_max = full_meta.groupby(["pig", "trial"])["window_id"].max()
    max_wid = int(trial_max.quantile(trial_pct))
    if max_wid < 5:          # 极端情况退为中位数
        max_wid = int(trial_max.quantile(0.5))
    print(f"  截断阈值 (trial_pct={trial_pct:.0%}): window_id <= {max_wid}")

    emb_raw = _load_embeddings(base)
    emb_cols = [c for c in emb_raw.columns if c.startswith("emb_")]

    assert len(emb_raw) == len(full_meta), \
        f"embeddings ({len(emb_raw)}) 与 window_data ({len(full_meta)}) 行数不匹配"

    full_meta = full_meta.drop(columns="trial")
    full_meta[emb_cols] = emb_raw[emb_cols].values     # 行对齐赋值
    df = full_meta[full_meta["window_id"] <= max_wid].copy().reset_index(drop=True)
    df = df.dropna(subset=[emb_cols[0]])

    del emb_raw, full_meta
    gc.collect()
    print(f"  过滤后: {len(df)} 样本, window_id 0-{max_wid}, {len(emb_cols)} 维 embedding")
    return df, emb_cols, max_wid


# ---------- 绘图 --------------------------------------------------------------

def plot_window_trends(df, output_dir):
    """图1：自信度随 window_id 变化——按 decision 和 Housing 各一子图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Prediction Confidence vs Window ID", fontsize=13, fontweight="bold")

    # 子图1: 按 decision 分组
    ax = axes[0]
    colors = {0: "coral", 1: "steelblue"}
    labels = {0: "NOGO", 1: "GO"}
    for dec in [0, 1]:
        grp = df[df["decision"] == dec].groupby("window_id")["confidence"]
        m, s = grp.mean(), grp.std()
        ax.plot(m.index, m.values, "o-", color=colors[dec], linewidth=2,
                markersize=5, label=labels[dec])
        ax.fill_between(m.index, m - s, m + s, color=colors[dec], alpha=0.15)
    ax.axhline(df["confidence"].mean(), color="gray", linestyle="--",
               linewidth=1.2, alpha=0.7, label="Overall mean")
    ax.set(xlabel="Window ID", ylabel="Mean Confidence (+/- 1 SD)", title="By Decision")
    ax.legend()
    ax.set_ylim([0.4, 1.02])

    # 子图2: 按 Housing 分组
    ax = axes[1]
    palette = sns.color_palette("husl", n_colors=df["Housing"].nunique())
    for i, housing in enumerate(sorted(df["Housing"].unique())):
        grp = df[df["Housing"] == housing].groupby("window_id")["confidence"].mean()
        ax.plot(grp.index, grp.values, "o-", color=palette[i], linewidth=1.8,
                markersize=4, label=housing)
    ax.axhline(df["confidence"].mean(), color="gray", linestyle="--",
               linewidth=1.2, alpha=0.7)
    ax.set(xlabel="Window ID", ylabel="Mean Confidence", title="By Housing Type")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim([0.4, 1.02])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_by_window.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ confidence_by_window.png")


def plot_distributions(df, output_dir):
    """图2：自信度分布——直方图 + 按 decision 和 Housing 的箱线图"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Prediction Confidence Distributions", fontsize=13, fontweight="bold")

    # 子图1: 直方图 (GO vs NOGO)
    ax = axes[0]
    ax.hist(df[df["decision"] == 1]["confidence"], bins=40, alpha=0.6,
            color="steelblue", label="GO", density=True)
    ax.hist(df[df["decision"] == 0]["confidence"], bins=40, alpha=0.6,
            color="coral", label="NOGO", density=True)
    ax.set(xlabel="Confidence", ylabel="Density", title="Distribution: GO vs NOGO")
    ax.legend()

    # 子图2: 按 Decision 箱线图
    ax = axes[1]
    df_p = df.copy()
    df_p["label"] = df_p["decision"].map({0: "NOGO", 1: "GO"})
    sns.boxplot(data=df_p, x="label", y="confidence", palette=["coral", "steelblue"],
                order=["NOGO", "GO"], ax=ax, width=0.5)
    ax.set(xlabel="Decision", ylabel="Confidence", title="Confidence by Decision")
    ax.set_ylim([0.4, 1.02])

    # 子图3: 按 Housing 箱线图
    ax = axes[2]
    order = sorted(df["Housing"].unique())
    sns.boxplot(data=df, x="Housing", y="confidence", order=order,
                palette="husl", ax=ax, width=0.6)
    ax.set(xlabel="Housing Type", ylabel="Confidence", title="Confidence by Housing Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    ax.set_ylim([0.4, 1.02])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distributions.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  ✓ confidence_distributions.png")


# ---------- 主函数 -----------------------------------------------------------

def run_confidence_analysis(trial_pct=0.95):
    print("\n" + "="*60)
    print("预测自信度分析 (Prediction Confidence Analysis)")
    print("="*60)

    output_dir = os.path.join(OUTPUT_BASE_DIR, "confidence_analysis")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载数据（含截断过滤）
    print("\n[1] 加载数据...")
    df, emb_cols, max_wid = load_data(trial_pct=trial_pct)

    # 2. 训练全局 RF 模型
    print("[2] 训练 RF 模型...")
    X = df[emb_cols].values
    y = df["decision"].values
    rf = RandomForestClassifier(n_estimators=150, max_depth=10,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X, y)
    print("  ✓ 训练完成")

    # 3. 计算自信度
    print("[3] 计算自信度...")
    proba = rf.predict_proba(X)        # shape (N, 2)
    df["pred_proba_nogo"] = proba[:, 0]
    df["pred_proba_go"]   = proba[:, 1]
    df["confidence"]      = proba.max(axis=1)
    del X, y, proba, rf
    gc.collect()
    print(f"  均值: {df['confidence'].mean():.4f}  "
          f"范围: [{df['confidence'].min():.4f}, {df['confidence'].max():.4f}]")

    # 4. 绘图
    print("[4] 绘制图表...")
    plot_window_trends(df, output_dir)
    plot_distributions(df, output_dir)

    # 5. 保存结果
    print("[5] 保存结果...")

    # 每个样本的预测结果
    (df[["window_id", "decision", "Housing", "pig",
         "confidence", "pred_proba_nogo", "pred_proba_go"]]
     .to_csv(os.path.join(output_dir, "confidence_predictions.csv"), index=False))
    print("  ✓ confidence_predictions.csv")

    # 每个 window_id 的统计摘要（用 'window_id' 列分组，不用行索引）
    wstats = (df.groupby("window_id")["confidence"]
              .agg(n_samples="count", mean="mean", std="std", min="min", max="max")
              .round(4))
    wstats.to_csv(os.path.join(output_dir, "confidence_by_window.csv"))
    print("  ✓ confidence_by_window.csv")

    print(f"\n✓ 分析完成！输出目录: {output_dir}")
    print(f"  window_id 范围: 0-{max_wid}（{trial_pct:.0%} trial 时长分位数）")


if __name__ == "__main__":
    run_confidence_analysis(trial_pct=0.95)
