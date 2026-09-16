#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Analysis Utilities
=======================

Quick, publication-ready plots for your CSV.
Rules honored:
- matplotlib only
- each chart is a separate figure
- no explicit color choices

Examples:
  python plot_analysis.py --csv /mnt/data/extra_features_max.csv --target score --outdir plots_out
"""
import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="plots_out1")
    ap.add_argument("--target", type=str, default="id_score")
    ap.add_argument("--id_cols", type=str, nargs="*", default=["id_dim","id_connect","id_cif","id_cif_path"])
    ap.add_argument("--bins", type=int, default=30)
    return ap.parse_args()

def pick_numeric_features(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in nums if c not in set(exclude)]

def fig_save(fig, outpath_no_ext: Path):
    png = outpath_no_ext.with_suffix(".png")
    svg = outpath_no_ext.with_suffix(".svg")
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_hist(ax, data: np.ndarray, title: str, bins=30, logy=False, annotate_counts=False):
    counts, edges, _ = ax.hist(data, bins=bins, log=logy)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count (log)" if logy else "count")
    if annotate_counts:
        for i, c in enumerate(counts):
            if c <= 0:
                continue
            x = 0.5*(edges[i] + edges[i+1])
            ax.text(x, c, f"{int(c)}", ha="center", va="bottom", fontsize=8)

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # Score distribution (log y + count labels) if target exists
    if args.target in df.columns:
        y = pd.to_numeric(df[args.target], errors="coerce").values
        fig, ax = plt.subplots(figsize=(6,4))
        plot_hist(ax, y[~np.isnan(y)], f"Distribution of {args.target}", bins=args.bins, logy=True, annotate_counts=True)
        fig_save(fig, outdir/"score_hist_logy")

    # Missingness bar (top 30)
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss>0]
    if len(miss) > 0:
        top = miss.head(30)
        fig, ax = plt.subplots(figsize=(7, max(3, 0.28*len(top))))
        ax.barh(np.arange(len(top)), top.values)
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels(top.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_title("Top-30 missingness by column")
        ax.set_xlabel("# missing")
        fig_save(fig, outdir/"missingness_top30")

    # Numeric correlation heatmap (block-wise if wide)
    exclude = [args.target] + args.id_cols
    feat_cols = pick_numeric_features(df, exclude=exclude)
    if len(feat_cols) >= 2:
        corr = df[feat_cols].corr(method="pearson")
        max_cols = 40
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]
        for i, sub in enumerate(chunks(feat_cols, max_cols)):
            fig, ax = plt.subplots(figsize=(max(6, len(sub)*0.25), max(6, len(sub)*0.25)))
            im = ax.imshow(corr.loc[sub, sub].values, vmin=-1.0, vmax=1.0, aspect="auto")
            ax.set_title(f"Pearson corr (block {i+1})")
            ax.set_xticks(np.arange(len(sub))); ax.set_yticks(np.arange(len(sub)))
            ax.set_xticklabels(sub, rotation=90, fontsize=6); ax.set_yticklabels(sub, fontsize=6)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("corr", rotation=90)
            fig_save(fig, outdir/f"corr_block_{i+1}")

    print(f"[plot_analysis] Figures saved to: {outdir}")

if __name__ == "__main__":
    main()
