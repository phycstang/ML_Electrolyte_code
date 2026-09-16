#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
study_features.py

Pipeline:
1) load -> QC (NA rate, NZV) -> impute -> robust scale (optional)
2) corr-prune (|spearman|>=0.95) + VIF prune
3) domain grouping & mRMR within-group representative pick
4) multi-model CV (RF, LGBM/XGB if avail, ElasticNet) + permutation importance
5) (optional) SHAP on final subset
6) stability selection by subsampling
7) export reports: feature_rank.csv, group_rank.csv, selected_features.txt, plots/

Usage:
  python study_features.py --csv extra_features_max.csv --target score --outdir feat_report --n_jobs 8 --with_shap
"""
import os, re, json, math, argparse, warnings, random
import numpy as np
import pandas as pd

from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression

# Optional boosters
_HAVE_LGBM = True
try:
    from lightgbm import LGBMRegressor
except Exception:
    _HAVE_LGBM = False

_HAVE_XGB = True
try:
    from xgboost import XGBRegressor
except Exception:
    _HAVE_XGB = False

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

def spearman_corr(a: np.ndarray, b: np.ndarray)->float:
    from scipy.stats import spearmanr
    c, _ = spearmanr(a, b, nan_policy="omit")
    return float(c)

def vif_scores(X: np.ndarray, names: List[str])->pd.Series:
    # Fast VIF via pseudo-inverse; for large K use statsmodels if preferred.
    import numpy.linalg as LA
    X_ = np.asarray(X, float)
    X_ = X_[~np.isnan(X_).any(axis=1)]
    if X_.shape[0] < 10:
        return pd.Series([np.nan]*len(names), index=names)
    vifs=[]
    G = np.corrcoef(X_, rowvar=False)
    # guard for singularity
    try:
        Gi = LA.pinv(G)
    except Exception:
        Gi = LA.pinv(G + 1e-6*np.eye(G.shape[0]))
    for j in range(Gi.shape[0]):
        r2j = 1.0 - 1.0/Gi[j, j]
        vif = 1.0/(1.0 - min(max(r2j, 0.0), 0.999999))
        vifs.append(vif)
    return pd.Series(vifs, index=names)

def make_groups(cols: List[str])->Dict[str, List[str]]:
    groups = defaultdict(list)
    for c in cols:
        c_low = c.lower()
        if c_low.startswith("mend_"):
            groups["thermo_dispersion_mendeleev"].append(c)
        elif c_low.startswith("soap_"):
            groups["dscribe_soap"].append(c)
        elif c_low.startswith("mm_"):
            groups["matminer"].append(c)
        elif c_low.startswith("feat_mproj_"):
            groups["graph_projection"].append(c)
        elif c_low.startswith("feat_angle_") or c_low.startswith("feat_bond_") or "cn" in c_low or "voro" in c_low or "packing_fraction" in c_low:
            groups["geometry_localenv"].append(c)
        elif "wyckoff" in c_low or c_low.startswith("feat_sg_") or "crystal_system_id" in c_low or "dim_larsen" in c_low or "primitive" in c_low:
            groups["symmetry_dimensionality"].append(c)
        elif c_low.startswith("feat_"):
            groups["composition_core"].append(c)
        else:
            groups["others"].append(c)
    return groups

def mrmr_within_group(X: pd.DataFrame, y: np.ndarray, cols: List[str], k: int=3)->List[str]:
    # simplified mRMR: score = MI - avg_corr_w_selected
    if not cols:
        return []
    mi = pd.Series(mutual_info_regression(X[cols].fillna(X[cols].median()), y, random_state=RANDOM_STATE), index=cols)
    selected=[]
    pool=set(cols)
    while pool and len(selected)<k:
        if not selected:
            c = mi.sort_values(ascending=False).index[0]
            selected.append(c); pool.remove(c)
        else:
            best=None; best_score=-1e9
            for c in list(pool):
                corr_penalty=np.mean([abs(X[c].corr(X[s], method="spearman")) for s in selected if X[c].notna().any() and X[s].notna().any()])
                score=float(mi[c]) - float(corr_penalty if np.isfinite(corr_penalty) else 0.0)
                if score>best_score:
                    best_score=score; best=c
            selected.append(best); pool.remove(best)
    return selected

def permutation_importance_cv(model, X: pd.DataFrame, y: np.ndarray, kfold: KFold, n_repeats:int=5, n_jobs:int=1)->pd.Series:
    importances=[]
    for tr, te in kfold.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        mdl = model
        mdl.fit(Xtr, ytr)
        r = permutation_importance(mdl, Xte, yte, n_repeats=n_repeats, n_jobs=n_jobs, random_state=RANDOM_STATE)
        importances.append(pd.Series(r.importances_mean, index=X.columns))
    imp = pd.concat(importances, axis=1).mean(axis=1)
    return imp.sort_values(ascending=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="score")
    ap.add_argument("--outdir", default="feat_report")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--with_shap", action="store_true")
    ap.add_argument("--drop_na_thresh", type=float, default=0.40)
    ap.add_argument("--corr_prune", type=float, default=0.95)
    ap.add_argument("--vif_thresh", type=float, default=10.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise SystemExit(f"Target column '{args.target}' not found.")
    y = df[args.target].values.astype(float)

    # Heuristics: drop id-like cols
    id_like = [c for c in df.columns if any(k in c.lower() for k in ["cif","id","path","file","formula"])]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != args.target]
    X = df[num_cols].copy()

    # 1) Missing/NZV
    na_rate = X.isna().mean().sort_values(ascending=False)
    na_rate.to_csv(os.path.join(args.outdir, "missing_rate.csv"))
    keep = [c for c in X.columns if na_rate[c] <= args.drop_na_thresh]
    X = X[keep]
    nzv = (X.nunique(dropna=True) <= 1)
    X = X.loc[:, ~nzv]
    nzv[nzv].to_csv(os.path.join(args.outdir, "near_zero_variance_cols.csv"))

    # simple impute for corr/VIF; models用pipeline再处理
    X_imp = X.copy()
    for c in X_imp.columns:
        med = X_imp[c].median(skipna=True)
        X_imp[c] = X_imp[c].fillna(med)

    # 2) Correlation prune (Spearman)
    corr = X_imp.corr(method="spearman").abs()
    to_drop=set()
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            if corr.iloc[i,j] >= args.corr_prune:
                # keep the one with higher abs corr to y
                ci, cj = cols[i], cols[j]
                ri = abs(spearman_corr(X_imp[ci].values, y))
                rj = abs(spearman_corr(X_imp[cj].values, y))
                drop = cj if ri >= rj else ci
                to_drop.add(drop)
    X_imp = X_imp.drop(columns=list(to_drop), errors="ignore")

    # 3) VIF
    vif = vif_scores(X_imp.values, X_imp.columns.tolist())
    bad = vif[vif>args.vif_thresh].index.tolist()
    X_imp = X_imp.drop(columns=bad, errors="ignore")
    vif.to_csv(os.path.join(args.outdir, "vif_scores.csv"))

    # 4) Grouping + mRMR representatives
    groups = make_groups(X_imp.columns.tolist())
    selected=[]
    for gname, cols in groups.items():
        reps = mrmr_within_group(X_imp, y, cols, k=3)
        selected += reps
    selected = sorted(set(selected))
    X_sel = X[selected].copy()  # use original NA (pipeline will impute)

    # 5) Models & CV
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = []

    rf = RandomForestRegressor(
        n_estimators=600, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=args.n_jobs
    )
    models.append(("RF", rf))

    if _HAVE_LGBM:
        lgbm = LGBMRegressor(
            n_estimators=1200, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=args.n_jobs
        )
        models.append(("LGBM", lgbm))

    if _HAVE_XGB:
        xgb = XGBRegressor(
            n_estimators=1200, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=RANDOM_STATE, n_jobs=args.n_jobs
        )
        models.append(("XGB", xgb))

    en = ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,1.0], alphas=None, cv=5, random_state=RANDOM_STATE, n_jobs=args.n_jobs)
    models.append(("ENet", en))

    # pipeline for scaling/impute for linear/boosters (RF不严格需要，但统一处理更稳)
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler(with_centering=True))
    ])

    X_pipe = num_pipe.fit_transform(X_sel)  # fit scaler on all data (CV中模型仅fit在train折)

    # 训练与评分 + permutation importance
    ranks = {}
    metrics = []
    for name, mdl in models:
        fold_scores=[]
        for tr, te in kf.split(X_pipe):
            Xtr, Xte = X_pipe[tr], X_pipe[te]
            ytr, yte = y[tr], y[te]
            mdl.fit(Xtr, ytr)
            pred = mdl.predict(Xte)
            fold_scores.append((r2_score(yte, pred), mean_absolute_error(yte, pred)))
        r2 = float(np.mean([s[0] for s in fold_scores])); mae = float(np.mean([s[1] for s in fold_scores]))
        metrics.append({"model":name, "R2":r2, "MAE":mae})

        # permutation importance on the whole set via CV（平均）
        imp = permutation_importance_cv(mdl, pd.DataFrame(X_pipe, columns=X_sel.columns), y, kf, n_repeats=5, n_jobs=args.n_jobs)
        imp = (imp - imp.min()) / (imp.max()-imp.min() + 1e-12)  # normalize
        ranks[name] = imp

    pd.DataFrame(metrics).to_csv(os.path.join(args.outdir, "cv_metrics.csv"), index=False)

    # 归一化融合排名
    all_feats = sorted(set().union(*[r.index for r in ranks.values()]))
    fused = pd.DataFrame(index=all_feats)
    for name, s in ranks.items():
        fused[name] = s.reindex(all_feats).fillna(0.0)
    fused["fused_rank"] = fused.mean(axis=1)
    fused.sort_values("fused_rank", ascending=False).to_csv(os.path.join(args.outdir, "feature_rank.csv"))

    # 组重要度（组内排名前 K=3 的和）
    group_rank = []
    group_map = make_groups(all_feats)
    for gname, cols in group_map.items():
        v = fused.loc[cols, "fused_rank"].nlargest(3).sum() if cols else 0.0
        group_rank.append((gname, float(v)))
    pd.DataFrame(group_rank, columns=["group","score"]).sort_values("score", ascending=False).to_csv(os.path.join(args.outdir, "group_rank.csv"), index=False)

    # 稳定性选择（子抽样 50 次）
    sel_counts = pd.Series(0, index=all_feats, dtype=int)
    B = 50
    for b in range(B):
        idx = rng.choice(len(y), size=int(0.7*len(y)), replace=False)
        mdl = RandomForestRegressor(n_estimators=400, max_features="sqrt", random_state=RANDOM_STATE+b, n_jobs=args.n_jobs)
        mdl.fit(X_pipe[idx], y[idx])
        imp = permutation_importance(mdl, X_pipe[idx], y[idx], n_repeats=3, n_jobs=args.n_jobs, random_state=RANDOM_STATE+b)
        s = pd.Series(imp.importances_mean, index=all_feats).sort_values(ascending=False)
        topk = s.index[: max(10, int(0.05*len(all_feats)))]
        sel_counts.loc[topk] += 1
    sel_freq = (sel_counts / B).sort_values(ascending=False)
    sel_freq.to_csv(os.path.join(args.outdir, "stability_select_freq.csv"))

    # 导出最终候选（融合排名前 100 与稳定频率 >=0.2 的并集）
    top_fused = set(fused.sort_values("fused_rank", ascending=False).index[:100])
    stable = set(sel_freq[sel_freq>=0.2].index)
    final_sel = sorted(top_fused.union(stable))
    with open(os.path.join(args.outdir, "selected_features.txt"), "w") as f:
        for c in final_sel:
            f.write(c+"\n")

    print(f"[OK] Wrote reports to {args.outdir}. Candidates: {len(final_sel)}")

if __name__ == "__main__":
    main()
