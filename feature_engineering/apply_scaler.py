#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Apply saved scaler to the intersection of column names (robust).")
    p.add_argument("--scaler", required=True, help="Path to scaler_stats.npz (must contain mean_, scale_; feature_names optional)")
    p.add_argument("--in", dest="input_csv", required=True, help="Input CSV (new dataset)")
    p.add_argument("--out", dest="output_csv", required=True, help="Output CSV (merged: preserved + scaled-intersection + untouched extra)")
    p.add_argument("--id-cols", nargs="*", default=[], help="Columns to preserve unscaled (IDs etc.)")
    p.add_argument("--target-col", default=None, help="Optional target column to preserve unscaled")
    return p.parse_args()

def main():
    args = parse_args()

    stats = np.load(args.scaler, allow_pickle=True)
    if not all(k in stats for k in ["mean_", "scale_"]):
        print("❌ scaler_stats.npz 必须包含 'mean_' 和 'scale_'。", file=sys.stderr)
        sys.exit(2)

    mean_ = stats["mean_"].astype(float)
    scale_ = stats["scale_"].astype(float)
    feat_names = list(stats["feature_names"].tolist()) if "feature_names" in stats else None

    df = pd.read_csv(args.input_csv)

    preserve = []
    for c in args.id_cols:
        if c in df.columns:
            preserve.append(c)
    if args.target_col and args.target_col in df.columns and args.target_col not in preserve:
        preserve.append(args.target_col)

    if feat_names is None:
        # 没有列名时只能强行按顺序匹配，且列数必须一致（除去保留列）
        numeric_cols = [c for c in df.columns if c not in preserve]
        if len(numeric_cols) != len(mean_):
            print("❌ 此 npz 不含 feature_names，且与新数据列数不一致，无法对齐。请补写 feature_names。", file=sys.stderr)
            sys.exit(3)
        X = df[numeric_cols].astype(float).values
        scale_safe = scale_.copy()
        scale_safe[scale_safe == 0.0] = 1.0
        Xs = (X - mean_) / scale_safe
        out_df = pd.concat([df[preserve], pd.DataFrame(Xs, columns=numeric_cols, index=df.index)], axis=1)
    else:
        # 关键修复：只用前 K 个列名与 mean_/scale_ 对齐
        K = len(mean_)
        if len(feat_names) < K:
            raise ValueError(f"feature_names 长度({len(feat_names)}) 小于 scaler 参数长度({K})")
        feat_names_effective = feat_names[:K]
        name_to_idx = {n: i for i, n in enumerate(feat_names_effective)}

        intersection = [c for c in df.columns if c in name_to_idx and c not in preserve]
        idxs = np.array([name_to_idx[c] for c in intersection], dtype=int)

        scale_sel = scale_[idxs].copy()
        scale_sel[scale_sel == 0.0] = 1.0

        df_scaled_part = df[intersection].astype(float)
        df_scaled_part = (df_scaled_part.values - mean_[idxs]) / scale_sel
        df_scaled_part = pd.DataFrame(df_scaled_part, columns=intersection, index=df.index)

        extra_cols = [c for c in df.columns if c not in preserve and c not in intersection]
        parts = []
        if preserve:
            parts.append(df[preserve])
        if extra_cols:
            parts.append(df[extra_cols])  # 原样保留未缩放列
        parts.append(df_scaled_part)
        out_df = pd.concat(parts, axis=1)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"✅ 已保存到：{args.output_csv}")
    print(f"ℹ️ 缩放列数：{len([c for c in out_df.columns if c in (feat_names[:len(mean_)] if feat_names else [])])}")

if __name__ == "__main__":
    main()
