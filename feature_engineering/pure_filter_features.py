#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pure_filter_features.py

纯过滤法（与目标无关）的特征精简脚本：
1) 删除高缺失列（缺失率 > 阈值）
2) 删除近常量列（方差≈0 或者某个取值占比 ≥ 阈值）
3) 删除强相关冗余（|Pearson ρ| > 阈值，按方差保留“信息量”更大的那一列）
4) 保留 ID-like 列（可显式指定，或自动识别“字符串且几乎都不重复”的列）

使用示例：
    python pure_filter_features.py \
        --input feat.csv \
        --output-data filtered_features.csv \
        --output-report feature_filter_report.csv \
        --missing-frac 0.3 \
        --majority-ratio 0.98 \
        --corr-thresh 0.95 \
        --id-cols id_cif_path,id_cif

仅依赖：pandas, numpy, scikit-learn（用于简单插补）。

作者：ChatGPT
"""
import argparse
import json
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def detect_id_like_columns(df: pd.DataFrame, exclude: List[str], unique_ratio: float = 0.9) -> List[str]:
    """
    自动识别 ID-like 列：非数值列，且唯一值个数接近样本数（默认 >= 0.9*n）
    """
    id_like = []
    n = len(df)
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        nun = df[c].nunique(dropna=True)
        if n > 0 and nun >= unique_ratio * n:
            id_like.append(c)
    return id_like


def near_constant(series: pd.Series, majority_ratio_max: float) -> Tuple[bool, float, float]:
    """
    判断是否近常量：
      - 方差≈0（在浮点误差下）
      - 或者某个取值占比 >= majority_ratio_max
    返回：(is_near_constant, variance, majority_ratio)
    """
    s = series.dropna()
    var = float(s.var(ddof=0)) if len(s) > 0 else 0.0
    if not np.isfinite(var) or abs(var) < 1e-12:
        # 没有有效方差
        maj_ratio = 1.0 if len(s) <= 1 else float(s.value_counts(dropna=True).iloc[0]) / float(len(s))
        return True, var, maj_ratio
    # 计算主值占比（对浮点做轻微 round 抑制噪声）
    s_round = s.round(12) if pd.api.types.is_float_dtype(s) else s
    counts = s_round.value_counts(dropna=True)
    maj_ratio = float(counts.iloc[0]) / float(len(s_round)) if len(s_round) > 0 else 1.0
    if maj_ratio >= majority_ratio_max:
        return True, var, maj_ratio
    return False, var, maj_ratio


def correlation_prune(df_num: pd.DataFrame, thresh: float) -> Tuple[List[str], List[dict]]:
    """
    基于皮尔逊相关的冗余裁剪（无监督）：
      - 计算绝对相关矩阵（对缺失做中位数插补）
      - 对 |ρ| > thresh 的成对特征，保留“方差更大”的那一列（信息量更大），删除另一列
    返回：保留列名列表、报告条目列表
    """
    report_rows = []
    if df_num.shape[1] <= 1:
        return list(df_num.columns), report_rows

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(df_num), columns=df_num.columns)
    corr = X.corr().abs()
    vars_ = df_num.var(ddof=0).astype(float)

    to_drop = set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        if cols[i] in to_drop:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_drop:
                continue
            if corr.iloc[i, j] > thresh:
                ci, cj = cols[i], cols[j]
                # 方差大的保留（作为“信息量”替代），删除方差小的
                vi, vj = float(vars_[ci]), float(vars_[cj])
                drop_c = cj if vi >= vj else ci
                to_drop.add(drop_c)
                report_rows.append({
                    "feature": drop_c,
                    "action": "drop",
                    "reason": "high_corr",
                    "note": f"|rho|>{thresh} with {ci if drop_c==cj else cj}; kept higher-variance twin"
                })

    kept = [c for c in cols if c not in to_drop]
    return kept, report_rows


def main():
    parser = argparse.ArgumentParser(description="Pure Filter Feature Slimming (missing/near-constant/correlation).")
    parser.add_argument("--input", required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output-data", required=True, help="输出过滤后数据 CSV")
    parser.add_argument("--output-report", required=True, help="输出特征过滤报告 CSV")
    parser.add_argument("--missing-frac", type=float, default=0.30, help="缺失率阈值，超过则删除（默认 0.30）")
    parser.add_argument("--majority-ratio", type=float, default=0.98, help="主值占比阈值，>= 则认为近常量（默认 0.98）")
    parser.add_argument("--corr-thresh", type=float, default=0.95, help="相关性裁剪阈值 |ρ|（默认 0.95）")
    parser.add_argument("--id-cols", type=str, default="", help="逗号分隔，显式指定需要保留的 ID 列名")
    parser.add_argument("--auto-id-like", action="store_true", help="自动识别并保留字符串且几乎唯一的列（ID-like）")
    parser.add_argument("--auto-id-unique-ratio", type=float, default=0.9, help="ID-like 唯一率阈值（默认 0.9）")

    args = parser.parse_args()

    # 读取
    df = pd.read_csv(args.input)
    n_rows, n_cols = df.shape

    # 解析 ID 列
    explicit_ids = [c for c in args.id_cols.split(",") if c.strip()] if args.id_cols else []
    explicit_ids = [c for c in explicit_ids if c in df.columns]
    auto_ids = detect_id_like_columns(df, exclude=explicit_ids, unique_ratio=args.auto_id_unique_ratio) if args.auto_id_like else []

    # 候选特征：数值列（不包含显式/自动ID列）
    numeric_cols = [c for c in df.columns if c not in explicit_ids + auto_ids and pd.api.types.is_numeric_dtype(df[c])]

    report_rows = []

    # 1) 高缺失
    miss_frac = df[numeric_cols].isna().mean() if numeric_cols else pd.Series(dtype=float)
    drop_missing = miss_frac[miss_frac > args.missing_frac].index.tolist()
    for c in drop_missing:
        report_rows.append({
            "feature": c, "action": "drop", "reason": "high_missing",
            "missing_frac": float(miss_frac[c]), "variance": None, "maj_ratio": None, "note": f">{args.missing_frac}"
        })
    keep = [c for c in numeric_cols if c not in drop_missing]

    # 2) 近常量
    kept_after_nc = []
    for c in keep:
        is_nc, var, maj = near_constant(df[c], args.majority_ratio)
        if is_nc:
            report_rows.append({
                "feature": c, "action": "drop", "reason": "near_constant",
                "missing_frac": float(df[c].isna().mean()), "variance": float(var), "maj_ratio": float(maj), "note": f"var≈0 or maj≥{args.majority_ratio}"
            })
        else:
            kept_after_nc.append(c)

    # 3) 相关性裁剪
    kept_corr, corr_report = correlation_prune(df[kept_after_nc], args.corr_thresh) if kept_after_nc else ([], [])
    report_rows.extend(corr_report)

    # 将保留特征补充“通过过滤”的记录
    recorded = set([r["feature"] for r in report_rows])
    for c in kept_corr:
        if c not in recorded:
            report_rows.append({
                "feature": c, "action": "keep", "reason": "passed_filters",
                "missing_frac": float(df[c].isna().mean()), "variance": float(df[c].var(ddof=0)),
                "maj_ratio": None, "note": ""
            })

    # 输出数据：ID + 保留特征
    out_cols = explicit_ids + auto_ids + kept_corr
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(args.output_data, index=False)

    # 输出报告
    pd.DataFrame(report_rows).sort_values(["action", "reason", "feature"]).to_csv(args.output_report, index=False)

    # 控制台摘要
    summary = {
        "input_shape": [int(n_rows), int(n_cols)],
        "explicit_id_cols": explicit_ids,
        "auto_id_cols": auto_ids,
        "kept_features": len(kept_corr),
        "output_data": args.output_data,
        "output_report": args.output_report,
        "params": {
            "missing_frac": args.missing_frac,
            "majority_ratio": args.majority_ratio,
            "corr_thresh": args.corr_thresh
        }
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
