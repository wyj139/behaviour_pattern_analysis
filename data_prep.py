# 数据准备模块：读取元表、生成 uid、按多条件筛选并保存、初步平滑与补全
from typing import List, Dict, Any, Optional, Sequence
import os
import re
import errno
import pandas as pd
import numpy as np

from config import (
    RAW_META_CSV, OUTPUT_BASE_DIR,
    UID_COLS, UID_COL_NAME, UID_SEP,
    SMOOTH_MAX_SPEED, INTERPOLATE_LIMIT_DIRECTION,
    FILLNA_METHODS, PAD_VALUE_IF_ALL_NA, VERBOSE,
    LIKELIHOOD_THRESHOLD,
    PREPROCESSED_DATA_DIR
)

#I/O helpers
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False)
    if VERBOSE:
        print(f"Saved {len(df)} rows -> {path}")

def generate_uid_column(df: pd.DataFrame,
                        uid_cols: Sequence[str] = UID_COLS,
                        uid_col_name: str = UID_COL_NAME,
                        sep: str = UID_SEP) -> pd.DataFrame:
    """
    为每一行生成唯一 uid（字符串拼接），保留所有原始列。
    缺失会被转换为空字符串后拼接。
    """
    df = df.copy()
    missing = [c for c in uid_cols if c not in df.columns]
    if missing:
        raise KeyError(f"UID 列缺失: {missing}")
    parts = []
    for c in uid_cols:
        part = df[c].astype(str).fillna("").str.strip()
        parts.append(part)
    df[uid_col_name] = parts[0]
    for p in parts[1:]:
        df[uid_col_name] = df[uid_col_name].astype(str) + sep + p.astype(str)
    # 清理多余空白和重复 sep
    df[uid_col_name] = df[uid_col_name].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def preprocess_dataframe_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    预处理表格：标准化关键列的数据类型
    - decision -> int
    - pig, Housing, id, bodypart, test -> string
    保留其他列不变，返回处理后的副本
    """
    df = df.copy()
    
    # decision 转为 int（处理可能的浮点或字符串）
    if 'decision' in df.columns:
        df['decision'] = df['decision'].astype(int)
    
    string_cols = ['pig', 'Housing', 'id', 'bodypart', 'test']
    for col in string_cols:
        if col in df.columns:
            old_dtype = df[col].dtype
            df[col] = df[col].astype(str)  # 或 "string"
            new_dtype = df[col].dtype
    
    return df

def filter_by_decision_cue_bodypart(
    df: pd.DataFrame,
    decision: int,
    bodypart: str,
    pig: Optional[str] = None,
    cue_type: Optional[Sequence[str]] = None,
    base_outdir: str = OUTPUT_BASE_DIR,
) -> pd.DataFrame:
    """
    按 decision、bodypart（以及可选的 cue_type 列表）筛选 df，并将结果保存到 base_outdir 下的子文件夹中。
    - df: 包含至少列 ['decision','bodypart']，若提供 cue_type 则需包含 'cue' 列
    - decision: 要匹配的 decision 值（以字符串比较以兼容数值/字符串）
    - bodypart: 要匹配的 bodypart 字符串（精确匹配）
    - cue_type: 可选的 cue 列值序列（如 ['NP','NN']）；若为 None 则不按 cue 筛选
    - base_outdir: 输出基础目录，函数会在其中创建子文件夹并保存 filtered_*.csv
    返回:
      - filtered_df: 筛选后的 DataFrame（副本）
    """
    df = df.copy()
    # 生成 mask（兼容数值/字符串）
    mask = df["decision"].astype(float) == float(decision)

    mask = mask & (df["bodypart"].astype(str).isin([str(b) for b in bodypart]))
    
    if pig is not None:
        mask = mask & (df["pig"].astype(str).isin([str(p) for p in pig]))

    if cue_type is not None:
        mask = mask & (df["cue"].astype(str).isin([str(c) for c in cue_type]))

    filtered = df[mask].copy()
    print(f"[step] filtered_inside rows={len(filtered)}, unique ids={filtered[UID_COL_NAME].nunique()}")


    # 生成安全的子文件夹名
    def _safe_label(s: str) -> str:
        return re.sub(r"[^\w\-]+", "_", str(s))

    label_parts = [f"{decision}", f"{_safe_label(bodypart)}"]
    if pig is not None:
        label_parts.append("+".join([_safe_label(p) for p in pig]))
    if cue_type:
        label_parts.append("+".join([_safe_label(c) for c in cue_type]))
    folder_label = "+".join(label_parts)

    out_dir = os.path.join(base_outdir, folder_label)
    ensure_dir(out_dir)
    

    return filtered,folder_label

# Smoothing and imputation
def _smooth_array(values: np.ndarray, max_speed: float = SMOOTH_MAX_SPEED) -> np.ndarray:
    if len(values) == 0:
        return values
    s = pd.Series(values.astype(float))
    s = s.interpolate(limit_direction=INTERPOLATE_LIMIT_DIRECTION)
    for m in FILLNA_METHODS:
        s = s.fillna(method=m)
    s = s.fillna(PAD_VALUE_IF_ALL_NA)
    arr = s.values.copy()
    # 限制速度：逐步修正
    for i in range(1, len(arr)):
        delta = arr[i] - arr[i-1]
        if np.isfinite(delta) and abs(delta) > max_speed:
            arr[i] = arr[i-1] + np.sign(delta) * max_speed
    return arr

def calculate_relative_positions(df: pd.DataFrame,
                                 reference_bodyparts: Sequence[str] = None,
                                 id_col: str = UID_COL_NAME,
                                 bodypart_col: str = "bodypart",
                                 frame_col: str = "frame",
                                 x_col: str = "x",
                                 y_col: str = "y") -> pd.DataFrame:
    """
    计算相对位移（优化版本，减少逐行计算）：以指定 bodypart 群的中心作为参考框
    """
    df = df.copy()
    
    if reference_bodyparts is None:
        reference_bodyparts = df[bodypart_col].unique()
    
    print(f"[step] calculating relative positions using reference bodyparts: {reference_bodyparts}")
    
    # 筛选参考点（一次性筛选，避免重复）
    ref_mask = df[bodypart_col].isin(reference_bodyparts)
    ref_df = df[ref_mask]
    
    # 计算参考点在每个 id-frame 组合下的中心点（使用 groupby 批量计算）
    center_stats = ref_df.groupby([id_col, frame_col]).agg({
        x_col: 'mean',
        y_col: 'mean'
    }).rename(columns={x_col: 'center_x', y_col: 'center_y'}).reset_index()
    
    # 对没有参考点的帧，使用该帧所有点的中心（备用计算）
    all_stats = df.groupby([id_col, frame_col]).agg({
        x_col: 'mean',
        y_col: 'mean'
    }).rename(columns={x_col: 'center_x_fallback', y_col: 'center_y_fallback'}).reset_index()
    
    # 合并中心点信息到原始数据（left join 以保留所有原始行）
    df = df.merge(center_stats, on=[id_col, frame_col], how='left')
    df = df.merge(all_stats, on=[id_col, frame_col], how='left')
    
    # 对没有参考中心的行使用备用中心点（避免 NaN）
    df['center_x'] = df['center_x'].fillna(df['center_x_fallback'])
    df['center_y'] = df['center_y'].fillna(df['center_y_fallback'])
    
    # 向量化计算相对位移（一次性计算，不用循环）
    df['rel_x'] = df[x_col] - df['center_x']
    df['rel_y'] = df[y_col] - df['center_y']
    
    # 删除临时列（可选）
    df = df.drop(['center_x_fallback', 'center_y_fallback'], axis=1)
    
    print(f"[step] relative positions calculated, result rows={len(df)}")
    
    return df

def smooth_and_impute_all(df: pd.DataFrame,
                          x_col: str = "x",
                          y_col: str = "y",
                          frame_col: str = "frame",
                          max_speed: float = SMOOTH_MAX_SPEED,
                          calculate_relative: bool = True,
                          reference_bodyparts: Sequence[str] = None) -> pd.DataFrame:
    """
    优化版本：先批量计算相对位移，再分组平滑
    """
    df = df.copy()
    
    # 如果需要相对位移，先计算（在分组平滑之前，避免重复计算中心点）
    if calculate_relative:
        df = calculate_relative_positions(df, reference_bodyparts=reference_bodyparts)
        # 添加需要平滑的列
        cols_to_smooth = [x_col, y_col, 'rel_x', 'rel_y']
    else:
        cols_to_smooth = [x_col, y_col]
    
    # 分组平滑（减少 groupby 次数）
    processed = []
    for (id_val, bodypart_val), group in df.groupby([UID_COL_NAME, 'bodypart']):
        # 按帧排序
        group = group.sort_values(frame_col).copy()
        
        # 对所有需要平滑的列进行平滑
        for col in cols_to_smooth:
            if col in group.columns:
                group[col] = _smooth_array(group[col].values, max_speed=max_speed)
        
        processed.append(group)
    
    out = pd.concat(processed, ignore_index=True) if processed else df.copy()
    print(f"[step] smoothing completed, result rows={len(out)}")
    
    return out

# ---- 高层流程示例 ----
def prepare_meta_and_timeseries(meta: pd.DataFrame,
                                decision: int ,
                                bodypart: str,
                                cue_type: Optional[Sequence[str]] = None,
                                pig: Optional[Sequence[str]] = None,
                                uid_cols: Sequence[str] = UID_COLS,
                                uid_col_name: str = UID_COL_NAME,
                                out_base: str = OUTPUT_BASE_DIR,
                                calculate_relative: bool = True,
                                reference_bodyparts: Sequence[str] = None) -> Dict[str, str]:
    """
    高层包装：
      1) 读取 meta CSV,生成 uid(保留所有列）
      2) 如果提供 filters,则对元表进行筛选并将筛选结果按 filter 保存到子文件夹
      3) 若提供 timeseries_path,会读取并把 filtered ids 对应的时序行合并保存（同一子文件夹）
      4) 对保存的时序数据进行初步平滑与补全（并覆盖保存）
    返回一个 mapping,key 为子文件夹 label,value 为最终保存的时序文件路径（或 meta 路径）
    """
    meta = meta.copy()
    meta = generate_uid_column(meta, uid_cols=uid_cols, uid_col_name=uid_col_name, sep=UID_SEP)
    meta = preprocess_dataframe_types(meta)
    print(meta.dtypes)
    print(f"[step] meta initial rows={len(meta)}, unique ids={meta[uid_col_name].nunique()}")
    results = {}
    thr = float(LIKELIHOOD_THRESHOLD)
    meta = meta[meta["likelihood"].astype(float) >= thr].copy()
    print(f"[step] meta after likelihood filtering rows={len(meta)}, unique ids={meta[uid_col_name].nunique()}")
    # 有 filters，则生成对应子文件夹并存储
    
    mapping,label = filter_by_decision_cue_bodypart(meta, decision, bodypart,cue_type,pig,base_outdir=out_base)
    print(f"[step] filtered_meta rows={len(mapping)}, unique ids={mapping[uid_col_name].nunique()}, folder_label={label}")

    out_dir = os.path.join(out_base, label)
    ensure_dir(out_dir)
    out_ts = os.path.join(out_dir, f"preprocessed.csv")
    
    if os.path.exists(out_ts): 
        print(f"Smoothed file exists, skipping smoothing: {out_ts}")
        results[f"{label}_timeseries"] = out_ts
    else:
        # 设定参考 bodyparts（如果未指定则使用当前筛选的 bodyparts）
        if reference_bodyparts is None and isinstance(bodypart, (list, tuple)):
            reference_bodyparts = bodypart
        elif reference_bodyparts is None:
            reference_bodyparts = [bodypart] if isinstance(bodypart, str) else None
            
        # 平滑/补全（包括相对位移计算）
        merged_smoothed = smooth_and_impute_all(mapping, 
                                                calculate_relative=calculate_relative,
                                                reference_bodyparts=reference_bodyparts)
        save_csv(merged_smoothed, out_ts)
        results[f"{label}_timeseries"] = out_ts
        print(f"[step] smoothed rows={len(merged_smoothed)}, unique ids={(merged_smoothed[uid_col_name].nunique() if uid_col_name in merged_smoothed.columns else 'N/A')}")
    return results

# ---- 若作为脚本运行的示例入口 ----
if __name__ == "__main__":
    # 简单示例：按 decision==1 且 cue in ['NP','NN'] 过滤并处理
    meta = pd.read_csv(RAW_META_CSV)
    decision = 1
    bodypart = ["base_of_the_tail","tip_of_the_tail","right_hip","left_hip","right_shoulder",
                "left_shoulder","neck","base_of_the_right_ear","base_of_the_left_ear","left_flank",
                "right_flank"]
    #cue_type = ["NP", "NN"]
    #print(f"Loaded meta rows={len(meta)}, unique ids={meta[UID_COL_NAME].nunique()}")
    #filter_by_decision_cue_bodypart(meta, decision, bodypart, cue_type=None, pig=None)
    res = prepare_meta_and_timeseries(meta,
                                      decision=decision,
                                      bodypart=bodypart,
                                      cue_type=None,
                                      uid_cols=UID_COLS,
                                      uid_col_name=UID_COL_NAME,
                                      out_base=PREPROCESSED_DATA_DIR)
    print("Complete. outputs:", res)