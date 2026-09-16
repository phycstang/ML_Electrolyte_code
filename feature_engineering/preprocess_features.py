#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
preprocess_features.py

功能：
1) 读取 CSV 数据集
2) 自动识别数值/分类型特征与可选 ID 列
3) 缺失值处理：数值(中位数)、分类型(众数)，并可丢弃高缺失列
4) 去除常量/准常量特征
5) 去除高相关特征（皮尔逊相关；可按与目标列相关性优先保留）
6) 数值特征标准化（StandardScaler）
7) 导出多份结果与工件以便复现

用法示例：
python preprocess_features.py --input /path/to/extra_features.csv --target id_score \
    --corr-thresh 0.95 --max-missing 0.4 --min-variance 1e-12 --outdir ./preproc_out
"""

import argparse
import json
import os
import sys
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump


def infer_id_columns(df: pd.DataFrame) -> List[str]:
    """根据常见命名启发式识别 ID 列（仅作为默认），用户也可通过 --id-cols 覆盖。"""
    id_like = []
    candidates = [
        "id", "name", "filename", "file", "cif_file", "cif_path",
        "id_cif", "id_score", "score_name", "sample", "uid"
    ]
    # 也包含前缀匹配
    id_prefixes = ["id_", "idx_", "uid_", "meta_"]
    for col in df.columns:
        lc = col.lower()
        if lc in candidates or any(lc.startswith(p) for p in id_prefixes):
            id_like.append(col)
    # 保守：若列数据全唯一或接近唯一，也可能是 ID
    for col in df.columns:
        if col not in id_like:
            try:
                nunique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)
                if nunique_ratio > 0.95 and df[col].dtype == object:
                    id_like.append(col)
            except Exception:
                pass
    # 目标列不应被当作 ID
    return list(dict.fromkeys(id_like))


def split_feature_types(df: pd.DataFrame, exclude: List[str]) -> Tuple[List[str], List[str]]:
    """区分数值/分类型列（排除 exclude 列）。"""
    num_cols, cat_cols = [], []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return num_cols, cat_cols


def drop_high_missing_columns(
    df: pd.DataFrame, cols: List[str], max_missing: float
) -> Tuple[pd.DataFrame, List[str]]:
    """删除缺失率高于阈值的列，返回新的 df 与删除列表。"""
    to_drop = []
    n = len(df)
    for c in cols:
        miss = df[c].isna().sum() / n if n > 0 else 1.0
        if miss > max_missing:
            to_drop.append(c)
    return df.drop(columns=to_drop, errors="ignore"), to_drop


def remove_high_correlation(
    df_num: pd.DataFrame,
    corr_thresh: float,
    target: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    基于绝对皮尔逊相关系数去冗余。
    - 若提供 target，则在一组高相关特征中保留与 target 相关更强者；
      如两者与 target 相近，则保留列名较小者。
    - 若未提供 target，按列名排序保留第一个。
    返回去冗余后的数值特征 df 与被删除的列名列表。
    """
    if df_num.shape[1] <= 1:
        return df_num, []

    corr = df_num.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drops = set()

    if target is not None and len(target) == len(df_num):
        # 计算每个特征与目标的相关度（用皮尔逊；若目标非数值，会失败）
        try:
            tcorr = df_num.apply(lambda s: s.corr(target, method="pearson"))
            tcorr = tcorr.abs().fillna(0.0)
        except Exception:
            tcorr = pd.Series(0.0, index=df_num.columns)
    else:
        tcorr = pd.Series(0.0, index=df_num.columns)

    for col in upper.columns:
        if col in drops:
            continue
        high_peers = [row for row in upper.index if (upper.loc[row, col] is not None
                                                     and upper.loc[row, col] >= corr_thresh)]
        for row in high_peers:
            if row in drops or row == col:
                continue
            # 决策：保留与目标更相关者；若相等则按列名字典序
            keep, remove = col, row
            if tcorr[col] < tcorr[row]:
                keep, remove = row, col
            elif tcorr[col] == tcorr[row]:
                keep, remove = sorted([col, row])[0], sorted([col, row])[1]
            # 标记移除
            if remove != keep:
                drops.add(remove)

    pruned = df_num.drop(columns=list(drops), errors="ignore")
    return pruned, sorted(list(drops))


def main():
    parser = argparse.ArgumentParser(description="Feature preprocessing: impute, scale, deduplicate by correlation.")
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--outdir", default="preproc_out", help="输出目录（默认 preproc_out）")
    parser.add_argument("--target", default=None, help="目标列名（可选；若提供则相关性筛除会优先保留与目标更相关者）")
    parser.add_argument("--id-cols", nargs="*", default=None, help="指定 ID/标识列（可选，多列）")
    parser.add_argument("--max-missing", type=float, default=0.4, help="允许的最大缺失率，超出将丢弃该列（默认 0.4）")
    parser.add_argument("--min-variance", type=float, default=1e-12, help="准常量特征的方差阈值（默认 1e-12）")
    parser.add_argument("--corr-thresh", type=float, default=0.95, help="去冗余的相关系数阈值（默认 0.95）")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) 读取
    df = pd.read_csv(args.input)
    original_columns = df.columns.tolist()

    # 2) 识别 ID/目标列
    id_cols = args.id_cols if args.id_cols is not None else infer_id_columns(df)
    id_cols = [c for c in id_cols if c in df.columns]
    target_col = args.target if (args.target in df.columns) else None

    # 记录将被排除参与数值处理的列
    exclude_cols = set(id_cols + ([target_col] if target_col else []))

    # 3) 缺失率报告与删除高缺失列
    miss_summary = df.isna().mean().sort_values(ascending=False).rename("missing_rate").to_frame()
    miss_summary.to_csv(os.path.join(args.outdir, "missingness_summary.csv"))
    df, dropped_high_missing = drop_high_missing_columns(
        df, [c for c in df.columns if c not in exclude_cols], args.max_missing
    )

    # 4) 按类型划分
    num_cols, cat_cols = split_feature_types(df, exclude=list(exclude_cols))

    # 5) 去除常量/准常量（仅对数值列）
    dropped_low_var: List[str] = []
    if num_cols:
        vt = VarianceThreshold(threshold=args.min_variance)
        # 需要先填充缺失，否则方差计算会报 NaN；这里仅为筛选做一个临时填充（不会写回数据）
        tmp = df[num_cols].copy()
        tmp_imputer = SimpleImputer(strategy="median")
        tmp = pd.DataFrame(tmp_imputer.fit_transform(tmp), columns=num_cols, index=df.index)
        vt.fit(tmp)
        kept_mask = vt.get_support()
        kept_num_cols = [c for c, keep in zip(num_cols, kept_mask) if keep]
        dropped_low_var = [c for c in num_cols if c not in kept_num_cols]
        num_cols = kept_num_cols

    # 6) 构建正式的缺失填充与缩放流程
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    # 注意：缩放器只作用于数值列
    scaler = StandardScaler(with_mean=True, with_std=True)

    # 先进行填充
    df_num_imputed = pd.DataFrame(index=df.index)
    if num_cols:
        df_num_imputed = pd.DataFrame(num_imputer.fit_transform(df[num_cols]), columns=num_cols, index=df.index)
    df_cat_imputed = pd.DataFrame(index=df.index)
    if cat_cols:
        df_cat_imputed = pd.DataFrame(cat_imputer.fit_transform(df[cat_cols]), columns=cat_cols, index=df.index)

    # 7) 相关性去冗余（在数值列上）
    removed_by_corr: List[str] = []
    if num_cols:
        # 用填充后的数值列做相关性
        if target_col is not None and pd.api.types.is_numeric_dtype(df[target_col]):
            target_series = df[target_col]
        else:
            target_series = None

        df_num_pruned, removed_by_corr = remove_high_correlation(
            df_num_imputed, corr_thresh=args.corr_thresh, target=target_series
        )
        # 导出相关矩阵（去冗余前，为参考）
        corr_mat = df_num_imputed.corr(method="pearson")
        corr_mat.to_csv(os.path.join(args.outdir, "correlation_matrix.csv"))
    else:
        df_num_pruned = df_num_imputed.copy()

    # 8) 数值缩放（对去冗余后的列）
    scaled_num = pd.DataFrame(index=df.index)
    if df_num_pruned.shape[1] > 0:
        scaled_num = pd.DataFrame(
            scaler.fit_transform(df_num_pruned),
            columns=df_num_pruned.columns, index=df.index
        )

    # 9) 组装“未缩放版本”与“缩放版本”
    # 未缩放版本：数值列用填充值（去冗余后的列），分类型用填充值，ID/目标原样合并
    unscaled_df_parts = []
    if id_cols:
        unscaled_df_parts.append(df[id_cols])
    if target_col:
        unscaled_df_parts.append(df[[target_col]])
    if df_num_pruned.shape[1] > 0:
        unscaled_df_parts.append(df_num_pruned)
    if df_cat_imputed.shape[1] > 0:
        unscaled_df_parts.append(df_cat_imputed)

    processed_unscaled = pd.concat(unscaled_df_parts, axis=1)

    # 缩放版本：在未缩放版本基础上，仅替换数值特征为 scaled_num
    scaled_df_parts = []
    if id_cols:
        scaled_df_parts.append(df[id_cols])
    if target_col:
        scaled_df_parts.append(df[[target_col]])
    if scaled_num.shape[1] > 0:
        scaled_df_parts.append(scaled_num)
    if df_cat_imputed.shape[1] > 0:
        scaled_df_parts.append(df_cat_imputed)

    processed_scaled = pd.concat(scaled_df_parts, axis=1)

    # 10) 保存结果与工件
    processed_unscaled.to_csv(os.path.join(args.outdir, "processed_data.csv"), index=False)
    processed_scaled.to_csv(os.path.join(args.outdir, "processed_data_scaled.csv"), index=False)

    dump(scaler, os.path.join(args.outdir, "scaler.joblib"))
    if num_cols:
        dump(num_imputer, os.path.join(args.outdir, "num_imputer.joblib"))
    if cat_cols:
        dump(cat_imputer, os.path.join(args.outdir, "cat_imputer.joblib"))

    dropped: Dict[str, Dict] = {}
    for c in dropped_low_var:
        dropped[c] = {"reason": "low_variance", "threshold": args.min_variance}
    for c in dropped_high_missing:
        dropped[c] = {"reason": "high_missing", "max_missing": args.max_missing}
    for c in removed_by_corr:
        dropped[c] = {"reason": "high_correlation", "corr_threshold": args.corr_thresh}

    with open(os.path.join(args.outdir, "dropped_columns.json"), "w", encoding="utf-8") as f:
        json.dump(dropped, f, ensure_ascii=False, indent=2)

    # 简要日志
    print("=== Summary ===")
    print(f"Input file           : {args.input}")
    print(f"Output dir           : {args.outdir}")
    print(f"Target column        : {target_col}")
    print(f"ID columns           : {id_cols}")
    print(f"Numeric features kept: {df_num_pruned.shape[1]}")
    print(f"Categorical kept     : {df_cat_imputed.shape[1]}")
    print(f"Dropped (missing)    : {len(dropped_high_missing)}")
    print(f"Dropped (low var)    : {len(dropped_low_var)}")
    print(f"Dropped (correlated) : {len(removed_by_corr)}")
    print("Files written:")
    print(" - processed_data.csv")
    print(" - processed_data_scaled.csv")
    print(" - missingness_summary.csv")
    print(" - correlation_matrix.csv")
    print(" - dropped_columns.json")
    if num_cols:
        print(" - num_imputer.joblib")
    if cat_cols:
        print(" - cat_imputer.joblib")
    print(" - scaler.joblib")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
