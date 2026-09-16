#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_and_report.py
-------------------
数据清洗脚本：
1) 删除全空列；
2) 删除高缺失列（缺失比例 > 阈值）；
3) 删除常数列（所有样本取值相同）；
4) （可选）删除近零方差列（唯一值很少且众数占比极高）；
5) （可选）删除完全重复列（数值相同，仅列名不同）；
6) 在数值列上去除共线特征（|corr| >= 阈值，保留缺失更少、方差更大的列）；
7) 输出清洗后的 CSV 与 Markdown 报告。

示例：
python clean_and_report.py \
  --input data_imputed.csv \
  --output data.csv \
  --report data_cleaning_report.md \
  --missing-thresh 0.10 \
  --corr-thresh 0.97 \
  --corr-method pearson \
  --nzv-enable 0 \
  --nzv-unique-frac 0.01 \
  --nzv-freq-ratio 95 \
  --drop-duplicate-cols 0

作者：ChatGPT
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="清洗数据并输出报告")
    p.add_argument("--input", required=True, help="输入 CSV 文件路径")
    p.add_argument("--output", required=True, help="清洗后 CSV 输出路径")
    p.add_argument("--report", required=True, help="Markdown 报告输出路径")
    p.add_argument("--missing-thresh", type=float, default=0.40,
                   help="高缺失列删除阈值（缺失比例 > 该值即删除），默认 0.40")
    p.add_argument("--corr-thresh", type=float, default=0.97,
                   help="共线性相关系数阈值（|r| >= 该值删除其一），默认 0.97")
    p.add_argument("--corr-method", choices=["pearson", "spearman"], default="pearson",
                   help="相关系数计算方法，默认 pearson")
    p.add_argument("--id-cols", type=str, default="",
                   help="以逗号分隔的列名清单，这些列将从共线性剔除中豁免（但仍参与缺失检查）")
    p.add_argument("--preview", type=int, default=100,
                   help="报告中保留列的预览个数，默认 100")

    # 新增：近零方差与重复列处理
    p.add_argument("--nzv-enable", type=int, default=0,
                   help="是否启用近零方差删除（0/1），默认 0（先与我讨论再开）")
    p.add_argument("--nzv-unique-frac", type=float, default=0.01,
                   help="NZV 判定：唯一值占比上限（默认 0.01）")
    p.add_argument("--nzv-freq-ratio", type=float, default=95.0,
                   help="NZV 判定：众数/次众数 频数比阈值（默认 95）")
    p.add_argument("--drop-duplicate-cols", type=int, default=0,
                   help="是否删除完全重复列（0/1），默认 0")

    return p.parse_args()


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # 标准化空白为 NaN，并把无限值转 NaN，便于统一缺失处理
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def drop_all_empty_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    is_all_na = df.isna().all()
    to_drop = list(is_all_na[is_all_na].index)
    return df.drop(columns=to_drop), to_drop


def drop_high_missing_columns(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str], pd.Series]:
    na_frac = df.isna().mean()
    to_drop = list(na_frac[na_frac > threshold].index)
    return df.drop(columns=to_drop), to_drop, na_frac


def drop_constant_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    # 常数列（所有非缺失值相同或整列 NaN 已在上一步处理）
    nunique = df.nunique(dropna=True)
    to_drop = nunique.index[nunique <= 1].tolist()
    return df.drop(columns=to_drop), to_drop


def drop_near_zero_variance(
    df: pd.DataFrame,
    unique_frac_thresh: float = 0.01,
    freq_ratio_thresh: float = 95.0
) -> Tuple[pd.DataFrame, List[str]]:
    """
    caret::nearZeroVar 思路（简化版）：
    - 唯一值占比很低（<= unique_frac_thresh）
    - 且 众数/次众数 频数比 >= freq_ratio_thresh
    满足者判为 NZV
    """
    to_drop = []
    for col in df.columns:
        s = df[col].dropna()
        if s.empty:
            continue
        # 离散化视角：对数值列先尝试不做 binning，以严格识别“几乎不变”
        counts = s.value_counts()
        unique_frac = counts.size / (len(s) if len(s) > 0 else 1)
        if counts.size == 1:
            # 纯常数列已由 drop_constant_columns 处理，这里跳过
            continue
        # 计算众数/次众数频数比
        top = counts.iloc[0]
        second = counts.iloc[1] if counts.size > 1 else 1
        freq_ratio = (top / second) if second != 0 else np.inf
        if (unique_frac <= unique_frac_thresh) and (freq_ratio >= freq_ratio_thresh):
            to_drop.append(col)
    return df.drop(columns=to_drop), to_drop


def drop_duplicate_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    删除完全重复列：通过对每列的值（含 NaN）做 tuple 序列哈希来判等。
    """
    seen = {}
    dup_cols = []
    for c in df.columns:
        # 使用 pandas 的等价序列（含 NaN）作为键
        key = tuple(pd.util.hash_pandas_object(df[c], index=False, categorize=False).values)
        if key in seen:
            dup_cols.append(c)
        else:
            seen[key] = c
    return df.drop(columns=dup_cols), dup_cols


def choose_keep_col(col_a: str, col_b: str,
                    df: pd.DataFrame,
                    na_frac: pd.Series,
                    variances: Dict[str, float]) -> str:
    # 优先保留缺失比例低者；再比方差；再按列名字典序稳定决策
    a_na = na_frac.get(col_a, df[col_a].isna().mean())
    b_na = na_frac.get(col_b, df[col_b].isna().mean())
    if a_na != b_na:
        return col_a if a_na < b_na else col_b
    a_var = variances.get(col_a, np.nanvar(pd.to_numeric(df[col_a], errors="coerce"), ddof=1))
    b_var = variances.get(col_b, np.nanvar(pd.to_numeric(df[col_b], errors="coerce"), ddof=1))
    if not np.isfinite(a_var):
        a_var = -np.inf
    if not np.isfinite(b_var):
        b_var = -np.inf
    if a_var != b_var:
        return col_a if a_var > b_var else col_b
    return min(col_a, col_b)


def remove_collinear_features(df: pd.DataFrame,
                              threshold: float,
                              method: str,
                              na_frac: pd.Series,
                              exempt: List[str]) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    # 数值化副本
    num_df = df.apply(pd.to_numeric, errors="ignore")
    num_df = num_df.select_dtypes(include=[np.number])
    cols = [c for c in num_df.columns if c not in exempt]

    if len(cols) <= 1:
        return df, [], {}

    # 预计算方差
    variances = {c: np.nanvar(num_df[c].astype(float), ddof=1) for c in cols}

    # 相关矩阵（按方法）
    corr = num_df[cols].corr(method=method, min_periods=2)

    to_remove = set()
    kept_for: Dict[str, str] = {}

    # 贪心：按矩阵上三角遍历
    for i, c1 in enumerate(cols):
        if c1 in to_remove:
            continue
        for c2 in cols[i + 1:]:
            if c2 in to_remove:
                continue
            r = corr.at[c1, c2]
            if pd.notna(r) and abs(r) >= threshold:
                keep = choose_keep_col(c1, c2, df, na_frac, variances)
                drop = c2 if keep == c1 else c1
                # 豁免列不删除
                if drop in exempt and keep not in exempt:
                    drop, keep = keep, drop
                if drop in exempt:
                    # 两个都是豁免：都不删
                    continue
                to_remove.add(drop)
                kept_for[drop] = keep

    cleaned = df.drop(columns=list(to_remove), errors="ignore")
    removed_list = sorted(list(to_remove))
    return cleaned, removed_list, kept_for


def build_report(input_path: str,
                 output_path: str,
                 original_shape: Tuple[int, int],
                 df1: pd.DataFrame,
                 df2: pd.DataFrame,
                 df3: pd.DataFrame,
                 df4: pd.DataFrame,
                 df5: pd.DataFrame,
                 df_final: pd.DataFrame,
                 dropped_all_empty: List[str],
                 dropped_high_missing: List[str],
                 dropped_constant: List[str],
                 dropped_nzv: List[str],
                 dropped_duplicates: List[str],
                 na_frac: pd.Series,
                 dropped_collinear: List[str],
                 kept_mapping: Dict[str, str],
                 method: str,
                 missing_thresh: float,
                 corr_thresh: float,
                 preview_n: int,
                 nzv_on: bool,
                 nzv_unique_frac: float,
                 nzv_freq_ratio: float,
                 dup_on: bool) -> str:
    lines: List[str] = []
    lines.append(f"# 数据清洗报告")
    lines.append("")
    lines.append(f"- 执行时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 输入文件：`{os.path.basename(input_path)}`")
    lines.append(f"- 输出文件：`{os.path.basename(output_path)}`")
    lines.append(f"- 原始形状：{original_shape[0]} 行 × {original_shape[1]} 列")
    lines.append("")
    lines.append(f"## 规则与阈值")
    lines.append(f"- 删除全空列：列内所有元素均为缺失（NaN/空字符串/±inf→NaN）")
    lines.append(f"- 删除高缺失列：缺失比例 > **{missing_thresh:.0%}**")
    lines.append(f"- 删除常数列：唯一值计数 ≤ 1（零方差）")
    if nzv_on:
        lines.append(f"- 删除近零方差列（NZV）：唯一值占比 ≤ **{nzv_unique_frac:.2%}** 且 众数/次众数 ≥ **{nzv_freq_ratio:.1f}**")
    if dup_on:
        lines.append(f"- 删除完全重复列：数值序列完全一致但列名不同")
    lines.append(f"- 去除共线特征：仅在**数值型**列上计算 `{method}` 相关；若 |r| ≥ **{corr_thresh:.2f}**，保留缺失更少、方差更大者")
    lines.append("")
    lines.append(f"## 结果概览")
    lines.append(f"- 删除全空列数量：**{len(dropped_all_empty)}**")
    lines.append(f"- 删除高缺失列数量：**{len(dropped_high_missing)}**")
    lines.append(f"- 删除常数列数量：**{len(dropped_constant)}**")
    if nzv_on:
        lines.append(f"- 删除近零方差列数量：**{len(dropped_nzv)}**")
    if dup_on:
        lines.append(f"- 删除完全重复列数量：**{len(dropped_duplicates)}**")
    lines.append(f"- 删除共线特征数量：**{len(dropped_collinear)}**")
    lines.append(f"- 清洗后形状：{df_final.shape[0]} 行 × {df_final.shape[1]} 列")
    lines.append("")

    def section_list(title: str, items: List[str]):
        lines.append(f"## {title}")
        if items:
            lines.append("| 列名 |")
            lines.append("|---|")
            for c in items:
                lines.append(f"| `{c}` |")
        else:
            lines.append("（无）")
        lines.append("")

    section_list("全空列清单", dropped_all_empty)

    lines.append("## 高缺失列清单（缺失比例 > 阈值）")
    if dropped_high_missing:
        lines.append("| 列名 | 缺失比例 |")
        lines.append("|---|---:|")
        for c in dropped_high_missing:
            frac = na_frac.get(c, np.nan)
            frac_str = f"{frac:.2%}" if pd.notna(frac) else "NA"
            lines.append(f"| `{c}` | {frac_str} |")
    else:
        lines.append("（无）")
    lines.append("")

    section_list("常数列清单（零方差）", dropped_constant)
    if nzv_on:
        section_list("近零方差列清单（NZV）", dropped_nzv)
    if dup_on:
        section_list("完全重复列清单", dropped_duplicates)

    lines.append("## 共线性剔除（|r| ≥ 阈值）")
    if dropped_collinear:
        lines.append("| 被删除列 | 保留列 |")
        lines.append("|---|---|")
        for drop_col in dropped_collinear:
            keep_col = kept_mapping.get(drop_col, "")
            lines.append(f"| `{drop_col}` | `{keep_col}` |")
    else:
        lines.append("（无）")
    lines.append("")

    lines.append("## 保留的列（预览）")
    survivors = df_final.columns.tolist()
    preview = survivors[:preview_n]
    lines.append(", ".join(f"`{c}`" for c in preview) + (" ..." if len(survivors) > preview_n else ""))
    lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    id_exempt = [c for c in (args.id_cols.split(",") if args.id_cols else []) if c]

    df = load_dataframe(args.input)
    original_shape = df.shape

    # 1) 全空列
    df1, dropped_all_empty = drop_all_empty_columns(df)

    # 2) 高缺失列
    df2, dropped_high_missing, na_frac = drop_high_missing_columns(df1, args.missing_thresh)

    # 3) 常数列（零方差）
    df3, dropped_constant = drop_constant_columns(df2)

    # 4) 近零方差（可选）
    if args.nzv_enable:
        df4, dropped_nzv = drop_near_zero_variance(df3, args.nzv_unique_frac, args.nzv_freq_ratio)
    else:
        df4, dropped_nzv = df3, []

    # 5) 完全重复列（可选）
    if args.drop_duplicate_cols:
        df5, dropped_duplicates = drop_duplicate_columns(df4)
    else:
        df5, dropped_duplicates = df4, []

    # 6) 共线性（数值列）
    df_final, dropped_collinear, kept_mapping = remove_collinear_features(
        df5, args.corr_thresh, args.corr_method, na_frac, id_exempt
    )

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    df_final.to_csv(args.output, index=False)

    # 报告
    report_md = build_report(
        input_path=args.input,
        output_path=args.output,
        original_shape=original_shape,
        df1=df1, df2=df2, df3=df3, df4=df4, df5=df5,
        df_final=df_final,
        dropped_all_empty=dropped_all_empty,
        dropped_high_missing=dropped_high_missing,
        dropped_constant=dropped_constant,
        dropped_nzv=dropped_nzv,
        dropped_duplicates=dropped_duplicates,
        na_frac=na_frac,
        dropped_collinear=dropped_collinear,
        kept_mapping=kept_mapping,
        method=args.corr_method,
        missing_thresh=args.missing_thresh,
        corr_thresh=args.corr_thresh,
        preview_n=args.preview,
        nzv_on=bool(args.nzv_enable),
        nzv_unique_frac=args.nzv_unique_frac,
        nzv_freq_ratio=args.nzv_freq_ratio,
        dup_on=bool(args.drop_duplicate_cols),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(report_md)

    # 控制台摘要
    print("=== 清洗完成 ===")
    print(f"输入：{args.input}")
    print(f"输出：{args.output}")
    print(f"报告：{args.report}")
    print(f"原始列数：{original_shape[1]}  -> 清洗后列数：{df_final.shape[1]}")
    print(f"删除全空列：{len(dropped_all_empty)}")
    print(f"删除高缺失列：{len(dropped_high_missing)}")
    print(f"删除常数列：{len(dropped_constant)}")
    if args.nzv_enable:
        print(f"删除近零方差列：{len(dropped_nzv)}")
    if args.drop_duplicate_cols:
        print(f"删除完全重复列：{len(dropped_duplicates)}")
    print(f"删除共线列：{len(dropped_collinear)}")


if __name__ == "__main__":
    main()
