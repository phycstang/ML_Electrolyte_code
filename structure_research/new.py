#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# generate_and_score_deterministic.py
#
# Deterministically enumerate M–X halides with charge balance using mendeleev oxidation states,
# clone structure tuples (dim, st1, st2, st3) from prior dataset grouped by comp_x_over_m,
# compute features via halide_minifeats.py, align/scale features like training,
# and score with the trained MLP regressor (best_model.pt + best_config.json).
#
# Requirements:
# - mendeleev, pandas, numpy, torch, tqdm
# - halide_minifeats.py in the same directory or importable
#
# IMPORTANT on scaling:
#   You MUST provide either:
#   (A) scaler_stats.npz (with arrays 'mean' and 'scale' for the exact training feature order), OR
#   (B) the UNscaled training CSV used to fit the scaler (so we can refit a matching StandardScaler).
# If neither is available, the script will exit to avoid misleading scores.
#
# Usage example:
#   python generate_and_score_deterministic.py \
#     --metals Al Fe Ti \
#     --halogens F Cl Br I \
#     --struct-csv /mnt/data/feats.csv \
#     --train-csv /mnt/data/processed_data.csv \
#     --train-csv-scaled /mnt/data/processed_data_scaled.csv \
#     --best-model /mnt/data/best_model.pt \
#     --best-config /mnt/data/best_config.json \
#     --scaler /mnt/data/scaler_stats.npz \
#     --outdir /mnt/data/generated_runs/run_det1
#
import os
import json
import math
import argparse
import subprocess
import tempfile
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from mendeleev import element as md_element

import torch
import torch.nn as nn

# --------------------- Utility ---------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def round_ratio(x, nd=3):
    try:
        return float(np.round(float(x), nd))
    except Exception:
        return np.nan

# ----------------- Model definition ----------------

class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], activation="relu", dropout=0.1, batchnorm=True):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.1),
            "elu": nn.ELU,
            "tanh": nn.Tanh,
        }
        Act = acts[activation.lower()]
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---------------- Scaling helpers ------------------

def load_scaler_npz(npz_path: str):
    d = np.load(npz_path)
    return d["mean"], d["scale"]

def fit_scaler_from_unscaled(train_csv_unscaled: str, feature_cols: List[str]):
    df = pd.read_csv(train_csv_unscaled)
    X = df[feature_cols].values.astype(np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

def apply_standard_scale(df_feats: pd.DataFrame, feature_cols: List[str], mean: np.ndarray, scale: np.ndarray):
    X = df_feats[feature_cols].values.astype(np.float64)
    Xs = (X - mean) / scale
    df_scaled = df_feats.copy()
    df_scaled[feature_cols] = Xs
    return df_scaled

# --------------- Structure mapping -----------------

def build_struct_map(struct_csv: str, tol_decimals: int = 3):
    """
    Build mapping from comp_x_over_m (rounded) -> list of unique (dim, st1, st2, st3) tuples, with counts.
    """
    df = pd.read_csv(struct_csv)
    required = ["comp_x_over_m", "dim", "st1", "st2", "st3"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{struct_csv} missing columns: {miss}")
    df = df.dropna(subset=["comp_x_over_m"])
    df["_ratio_key"] = df["comp_x_over_m"].apply(lambda x: round_ratio(x, tol_decimals))
    gb = df.groupby(["_ratio_key", "dim", "st1", "st2", "st3"]).size().reset_index(name="count")
    # collect as map of lists sorted by frequency
    out = defaultdict(list)
    for _, row in gb.sort_values(["_ratio_key", "count"], ascending=[True, False]).iterrows():
        key = float(row["_ratio_key"])
        tpl = (int(row["dim"]), int(row["st1"]), int(row["st2"]), int(row["st3"]), int(row["count"]))
        out[key].append(tpl)
    return out

# --------------- Charge-balanced enum --------------

def positive_oxidations(symbol: str) -> List[int]:
    try:
        el = md_element(symbol)
        ox = [int(o) for o in el.oxidation_states if o > 0]
        return sorted(set(ox))
    except Exception:
        return []

def enumerate_charge_balanced(metals: List[str], halogens: List[str], struct_map: Dict[float, List[Tuple[int,int,int,int,int]]],
                              ratio_round_decimals: int = 3):
    """
    Yield dicts: {metal, halogen, ox, x_count, y_count, ratio_key, struct_tuple_without_count}
    Where ratio_key = comp_x_over_m rounded (y/x).
    Only yield if ratio_key exists in struct_map (i.e., seen in training data).
    """
    for M in metals:
        v_list = positive_oxidations(M)
        if not v_list:
            continue
        for X in halogens:
            for v in v_list:
                x = 1      # metal count
                y = v      # halogen count to balance charge (-1 each)
                ratio = y / x
                key = round_ratio(ratio, ratio_round_decimals)
                if key in struct_map and len(struct_map[key]) > 0:
                    for (dim, st1, st2, st3, cnt) in struct_map[key]:
                        yield {
                            "metal": M, "halogen": X, "ox": v,
                            "x_count": x, "y_count": y,
                            "ratio_key": key,
                            "dim": dim, "st1": st1, "st2": st2, "st3": st3,
                        }

# --------------- Feature generation ----------------

def run_halide_minifeats(formulas: List[str], halide_minifeats_py: str, out_csv: str) -> pd.DataFrame:
    # write temp input
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        in_path = f.name
        pd.DataFrame({"formula": formulas}).to_csv(in_path, index=False)
    # call CLI
    cmd = ["python", halide_minifeats_py, "--csv", in_path, "--out_csv", out_csv]
    subprocess.run(cmd, check=True)
    df = pd.read_csv(out_csv)
    # clean temp in
    try: os.remove(in_path)
    except: pass
    return df

def formula_str(metal: str, halogen: str, x: int, y: int) -> str:
    if x == 1 and y == 1:
        return f"{metal}{halogen}"
    elif x == 1:
        return f"{metal}{halogen}{y}"
    elif y == 1:
        return f"{metal}{x}{halogen}"
    else:
        return f"{metal}{x}{halogen}{y}"

# ----------------- Main routine --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metals", nargs="*", default=["Al","Fe","Ti","Mg","Ca","Zn","Cu","Na","K","Li"], help="Metal symbols to enumerate")
    ap.add_argument("--halogens", nargs="*", default=["F","Cl","Br","I"])
    ap.add_argument("--struct-csv", type=str, default="/data/home/tmy/ML_Electrolyte1/make_features/feats.csv", help="CSV containing comp_x_over_m, dim, st1, st2, st3")
    ap.add_argument("--train-csv-scaled", type=str, default="/data/home/tmy/ML_Electrolyte1/make_features/out_preprocessed2/processed_data_scaled.csv", help="Scaled training CSV (for feature order)")
    ap.add_argument("--train-csv", type=str, default=None, help="UNSCALED training CSV (to fit scaler if no scaler npz provided)")
    ap.add_argument("--scaler", type=str, default="/data/home/tmy/ML_Electrolyte1/make_features/out_preprocessed2/scaler_stats.npz", help="npz file with 'mean' and 'scale' arrays")
    ap.add_argument("--best-model", type=str, default="/data/home/tmy/ML_Electrolyte1/MLP/runs_reg_20250917-150855/best_model.pt")
    ap.add_argument("--best-config", type=str, default="/data/home/tmy/ML_Electrolyte1/MLP/runs_reg_20250917-150855/best_config.json")
    ap.add_argument("--halide-minifeats", type=str, default="/data/home/tmy/ML_Electrolyte1/make_features/halide_minifeats.py")
    ap.add_argument("--outdir", type=str, default="/generated_runs")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # 1) Build structure mapping by comp_x_over_m
    struct_map = build_struct_map(args.struct_csv, tol_decimals=3)
    # 2) Enumerate charge-balanced combos
    combos = list(enumerate_charge_balanced(args.metals, args.halogens, struct_map))
    if len(combos) == 0:
        raise SystemExit("No charge-balanced combos matched any comp_x_over_m seen in struct-csv.")
    # 3) Make formulas
    formulas = [formula_str(c["metal"], c["halogen"], c["x_count"], c["y_count"]) for c in combos]

    # 4) Generate features via halide_minifeats
    feats_path = os.path.join(args.outdir, "raw_minifeats.csv")
    df_feats = run_halide_minifeats(formulas, args.halide_minifeats, feats_path)

    # 5) Attach structure columns and meta
    df_meta = pd.DataFrame(combos)
    df_merged = pd.concat([df_meta.reset_index(drop=True), df_feats.reset_index(drop=True)], axis=1)

    # 6) Align to training feature order
    df_train_scaled = pd.read_csv(args.train_csv_scaled)
    id_col = df_train_scaled.columns[0]
    target_col = df_train_scaled.columns[-1]
    feature_cols = df_train_scaled.columns[1:-1].tolist()

    # Ensure presence of required columns
    missing = [c for c in feature_cols if c not in df_merged.columns]
    if missing:
        # create missing columns with zeros (or reasonable defaults)
        for c in missing:
            df_merged[c] = 0.0
    # Drop extras not used by the model
    df_model = df_merged[[c for c in ([id_col] + feature_cols) if c in df_merged.columns]].copy()
    # For id column, if not present, synthesize
    if id_col not in df_model.columns:
        df_model[id_col] = [f"{m}_{x}{h}{y}" for m,h,x,y in zip(df_meta["metal"], df_meta["halogen"], df_meta["x_count"], df_meta["y_count"])]
        df_model = df_model[[id_col] + feature_cols]

    # 7) Scaling
    scaler_used = None
    if args.scaler and os.path.isfile(args.scaler):
        mean, scale = load_scaler_npz(args.scaler)
        scaler_used = ("npz", args.scaler)
    elif args.train_csv is not None and os.path.isfile(args.train_csv):
        mean, scale = fit_scaler_from_unscaled(args.train_csv, feature_cols)
        scaler_used = ("fit_from_unscaled", args.train_csv)
    else:
        raise SystemExit("No scaler npz provided AND no unscaled training CSV available. Aborting to avoid inconsistent scaling.")

    # apply scale
    df_scaled = apply_standard_scale(df_model, feature_cols, mean, scale)
    np.savez(os.path.join(args.outdir, "scaler_used.npz"), mean=mean, scale=scale)

    # 8) Load model
    cfg = load_json(args.best_config)
    model = MLPRegressor(
        in_dim=len(feature_cols),
        hidden_dims=cfg["hidden_dims"],
        activation=cfg["activation"],
        dropout=cfg["dropout"],
        batchnorm=cfg["batchnorm"],
    )
    state = torch.load(args.best_model, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # 9) Predict
    X = df_scaled[feature_cols].values.astype(np.float32)
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X)).squeeze(1).cpu().numpy()

    # 10) Save outputs
    out = df_merged.copy()
    out["pred_score"] = y_pred
    out["ratio_key"] = out["ratio_key"].astype(float)
    out["scaler_source"] = scaler_used[0] if scaler_used else "unknown"
    out.to_csv(os.path.join(args.outdir, "generated_candidates.csv"), index=False, encoding="utf-8")

    # Also dump a small top-K summary
    topk = out.sort_values("pred_score", ascending=False).head(50)
    topk[[id_col, "metal", "halogen", "x_count", "y_count", "dim", "st1", "st2", "st3", "pred_score"]]\
        .to_csv(os.path.join(args.outdir, "top50.csv"), index=False, encoding="utf-8")

    print(f"[Done] Generated {len(out)} candidates. Saved to {args.outdir}")

if __name__ == "__main__":
    main()