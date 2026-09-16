#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_and_analyze_features.py

Analyze linear relationships among features and preprocess them to produce clean CSVs for NN training.
- Detect ID/target columns (override-able)
- Handle missing values (numeric: median; categorical: most_frequent)
- Remove zero-variance features
- Drop one side of highly linearly correlated pairs (|r| >= threshold)
- One-Hot encode categorical features
- Standardize numeric features (switchable)
- Save: processed_data.csv (no scaling), processed_data_scaled.csv (scaled), correlations, heatmap, metadata, scaler stats

Usage:
  python preprocess_and_analyze_features.py --in feats.csv --out-dir out_preprocessed --corr-threshold 0.95 --topk 200

Optional:
  --target-col id_score        # override target detection
  --id-cols name cif_file      # override/add ID columns
  --no-scale                   # skip standardization (still saves raw processed_data.csv)
  --heatmap                    # force draw heatmap (default auto if >=2 numeric cols)

Notes:
- Designed to be compatible with older scikit-learn: OneHotEncoder(sparse=True/False).
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import List, Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold


def compute_top_correlations(df_num: pd.DataFrame, top_k: int = 150, thresh_abs: float = 0.95) -> Tuple[pd.DataFrame, List[tuple]]:
    """Return top-K correlation pairs and pairs exceeding |r| >= thresh_abs."""
    if df_num.shape[1] <= 1:
        return pd.DataFrame(columns=["f1", "f2", "r"]), []
    corr = df_num.corr(method="pearson")
    pairs = []
    cols = df_num.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            pairs.append((cols[i], cols[j], r))
    pairs_sorted = sorted(pairs, key=lambda x: -abs(x[2]))
    top = pd.DataFrame(pairs_sorted[:top_k], columns=["f1", "f2", "r"])
    to_drop_pairs = [(a, b) for a, b, r in pairs_sorted if abs(r) >= thresh_abs]
    return top, to_drop_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input_path", required=True, help="Input CSV (features)")
    ap.add_argument("--out-dir", dest="out_dir", required=True, help="Output directory")
    ap.add_argument("--corr-threshold", type=float, default=0.95, help="|r| threshold to drop one of correlated features")
    ap.add_argument("--topk", type=int, default=200, help="Top-K correlation pairs to save")
    ap.add_argument("--target-col", type=str, default=None, help="Target column name override")
    ap.add_argument("--id-cols", nargs="*", default=None, help="ID columns (space separated). If provided, extend/override detection")
    ap.add_argument("--no-scale", action="store_true", help="Do not standardize numeric features")
    ap.add_argument("--heatmap", action="store_true", help="Force make heatmap")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    df = pd.read_csv(args.input_path)

    # Detect ID / Target
    detected_id_candidates = ["id", "name", "cif_file", "id_cif", "cif_path", "id_cif_path"]
    id_cols_present = [c for c in detected_id_candidates if c in df.columns]
    if args.id_cols:
        # include any provided ones that exist
        extra = [c for c in args.id_cols if c in df.columns and c not in id_cols_present]
        id_cols_present += extra

    if args.target_col and args.target_col in df.columns:
        target_col = args.target_col
    else:
        possible_target_cols = ["id_score", "score", "target", "y"]
        target_col = next((c for c in possible_target_cols if c in df.columns), None)

    y = df[target_col].copy() if target_col else None

    # Separate features
    feature_df = df.drop(columns=id_cols_present + ([target_col] if target_col else []), errors="ignore").copy()

    # Types
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    # Correlations
    top_corr_df, high_corr_pairs = compute_top_correlations(feature_df[numeric_cols], top_k=args.topk, thresh_abs=args.corr_threshold)
    top_corr_csv = os.path.join(args.out_dir, "correlations_top.csv")
    top_corr_df.to_csv(top_corr_csv, index=False)

    # Heatmap
    heatmap_path = None
    if args.heatmap or len(numeric_cols) >= 2:
        cor = feature_df[numeric_cols].corr().values if len(numeric_cols) >= 2 else np.zeros((1, 1))
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(cor, aspect="auto", interpolation="nearest")
        plt.colorbar()
        plt.title("Pearson correlation heatmap (numeric features)")
        plt.xlabel("features")
        plt.ylabel("features")
        heatmap_path = os.path.join(args.out_dir, "correlation_heatmap.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=200)
        plt.close(fig)

    # Drop one side of highly correlated pairs
    to_drop = set()
    for a, b in high_corr_pairs:
        if b not in to_drop and a not in to_drop:
            to_drop.add(b)
    numeric_cols_kept = [c for c in numeric_cols if c not in to_drop]

    # Pipelines
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
        *([] if args.no_scale else [("scaler", StandardScaler(with_mean=True, with_std=True))]),
    ])
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_cols_kept),
            ("cat", cat_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X = feature_df[numeric_cols_kept + categorical_cols].copy()
    X_proc = preprocessor.fit_transform(X)

    # Feature names after transform
    # Numeric after variance threshold
    if args.no_scale:
        var_step = preprocessor.named_transformers_["num"].named_steps["var"]
    else:
        var_step = preprocessor.named_transformers_["num"].named_steps["var"]

    feature_names_num_mask = var_step.get_support()
    num_kept_after_var = [f for f, m in zip(numeric_cols_kept, feature_names_num_mask) if m]

    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(categorical_cols) if len(categorical_cols) > 0 else np.array([])
    final_feature_names = list(num_kept_after_var) + list(cat_feature_names)

    # Save scaled (if scaling enabled)
    if not args.no_scale:
        scaled_csv = os.path.join(args.out_dir, "processed_data_scaled.csv")
        X_proc_df = pd.DataFrame(X_proc, columns=final_feature_names)
        if id_cols_present:
            X_proc_df = pd.concat([df[id_cols_present].reset_index(drop=True), X_proc_df], axis=1)
        if target_col:
            X_proc_df[target_col] = y.values
        X_proc_df.to_csv(scaled_csv, index=False)
    else:
        scaled_csv = None

    # Save raw/imputed/encoded (no scaling)
    # Redo without scaler to ensure truly "raw processed"
    num_transformer_raw = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(threshold=0.0)),
    ])
    cat_transformer_raw = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor_raw = ColumnTransformer(
        transformers=[
            ("num", num_transformer_raw, numeric_cols_kept),
            ("cat", cat_transformer_raw, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    X_raw = preprocessor_raw.fit_transform(X)
    feature_names_num_mask_raw = preprocessor_raw.named_transformers_["num"].named_steps["var"].get_support()
    num_kept_after_var_raw = [f for f, m in zip(numeric_cols_kept, feature_names_num_mask_raw) if m]
    ohe_raw = preprocessor_raw.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names_raw = ohe_raw.get_feature_names_out(categorical_cols) if len(categorical_cols) > 0 else np.array([])
    final_feature_names_raw = list(num_kept_after_var_raw) + list(cat_feature_names_raw)

    raw_csv = os.path.join(args.out_dir, "processed_data.csv")
    X_raw_df = pd.DataFrame(X_raw, columns=final_feature_names_raw)
    if id_cols_present:
        X_raw_df = pd.concat([df[id_cols_present].reset_index(drop=True), X_raw_df], axis=1)
    if target_col:
        X_raw_df[target_col] = y.values
    X_raw_df.to_csv(raw_csv, index=False)

    # Save scaler stats for reproducibility (only if scaled)
    scaler_stats_path = None
    if not args.no_scale:
        scaler = preprocessor.named_transformers_["num"].named_steps.get("scaler", None)
        if scaler is not None and hasattr(scaler, "mean_"):
            scaler_stats = {
                "mean_": scaler.mean_.tolist(),
                "scale_": scaler.scale_.tolist(),
                "numeric_feature_order_before_var": numeric_cols_kept,
                "numeric_features_after_var": num_kept_after_var,
                "categorical_feature_names": categorical_cols,
                "ohe_feature_names": final_feature_names[len(num_kept_after_var):],
            }
            scaler_stats_path = os.path.join(args.out_dir, "scaler_stats.npz")
            np.savez(scaler_stats_path, **scaler_stats)

    # Metadata
    metadata = {
        "input_csv": args.input_path,
        "id_columns_detected": id_cols_present,
        "target_column_detected": target_col,
        "numeric_columns_initial": numeric_cols,
        "categorical_columns_initial": categorical_cols,
        "high_corr_threshold": args.corr_threshold,
        "high_corr_dropped_features": sorted(list(set([b for (a,b) in high_corr_pairs]))),
        "n_high_corr_pairs": len(high_corr_pairs),
        "top_corr_csv": top_corr_csv,
        "correlation_heatmap_png": heatmap_path,
        "processed_scaled_csv": scaled_csv,
        "processed_raw_csv": raw_csv,
        "scaler_stats_npz": scaler_stats_path,
    }
    with open(os.path.join(args.out_dir, "columns_dropped.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("DONE.")
    print(f"- Processed (raw): {raw_csv}")
    if scaled_csv:
        print(f"- Processed (scaled): {scaled_csv}")
    print(f"- Correlations: {top_corr_csv}")
    if heatmap_path:
        print(f"- Heatmap: {heatmap_path}")
    if scaler_stats_path:
        print(f"- Scaler stats: {scaler_stats_path}")
    print(f"- Metadata: {os.path.join(args.out_dir, 'columns_dropped.json')}")


if __name__ == "__main__":
    main()
