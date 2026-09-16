#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature selection full pipeline (for small-sample, high-dim materials data)

Capabilities:
- Load features CSV + target CSV (or combined)
- Manual 'priority' features (manual20): keep/weight/test-without
- Name-level fuzzy deduplication and numeric equivalence detection
- Correlation clustering, cluster representatives or PC1 replacement
- Filter: remove constant/low-variance/high-missing
- Embedded importance: ElasticNet, RandomForest, LightGBM/XGBoost
- Model-agnostic: permutation importance
- Stability Selection (subsampling + Lasso)
- Boruta wrapper (RF shadow features)
- mRMR (if pymrmr available) or mutual_info fallback
- Voting/ensemble aggregation of selection methods
- SHAP explanations (TreeSHAP)
- Ablation curves and final Panel-10/20/50 outputs
- Outputs many CSV and PNG artifacts for reporting

References (for methods):
- Stability selection: Meinshausen & Bühlmann (2010).
- Boruta: Kursa & Rudnicki (2010).
- mRMR: Ding & Peng (2005).
- SHAP: Lundberg & Lee (2017).
"""

import argparse, json, os, sys, warnings, difflib
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import defaultdict, Counter

# sklearn / ml
from sklearn.model_selection import KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# optional libs
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import shap
except Exception:
    shap = None
try:
    from boruta import BorutaPy
except Exception:
    try:
        from boruta_py import BorutaPy
    except Exception:
        BorutaPy = None
try:
    import pymrmr
except Exception:
    pymrmr = None
try:
    from pyHSICLasso import HSICLasso
except Exception:
    HSICLasso = None
try:
    from stability_selection import StabilitySelection
    from sklearn.linear_model import Lasso as Lasso_s
except Exception:
    StabilitySelection = None

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- utility funcs ----------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }

def numeric_cols(df, exclude=[]):
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

# ---------------- dedup / fuzzy name match ----------------
def fuzzy_name_matches(listA, listB, thresh=0.9):
    """Return pairs (a,b,score) where a in A and b in B and similarity >= thresh"""
    matches=[]
    for a in listA:
        for b in listB:
            s = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            if s >= thresh:
                matches.append((a,b,float(s)))
    return matches

def numeric_equivalence_table(X, listA, listB, corr_thresh=0.995, diff_tol=1e-8):
    """Check numeric equivalence between columns in listA and listB."""
    rows=[]
    for a in listA:
        if a not in X.columns: continue
        for b in listB:
            if b not in X.columns: continue
            xa = X[a].replace([np.inf,-np.inf], np.nan)
            xb = X[b].replace([np.inf,-np.inf], np.nan)
            # align non-nan index
            mask = (~xa.isna()) & (~xb.isna())
            if mask.sum() < max(10, int(0.05*len(X))):
                continue
            rho = xa[mask].corr(xb[mask])
            medabs = float(np.median(np.abs(xa[mask]-xb[mask])))
            if rho is not None and abs(rho) >= corr_thresh and medabs <= diff_tol:
                rows.append({"a":a,"b":b,"pearson":float(rho),"median_abs_diff":medabs,"n":int(mask.sum())})
    return pd.DataFrame(rows)

# ---------------- correlation clustering ----------------
def correlation_clusters(X, method='average', threshold=0.90, metric='pearson'):
    """Hierarchical clustering on abs(correlation), return cluster labels (1..k)"""
    corr = X.corr().abs()
    # convert to distance
    dist = 1 - corr
    # convert to condensed distance matrix for linkage? We'll use Agglomerative on distance matrix features
    # Use features' pairwise distance by MDS? Simple approximate: use condensed vector from dist
    # Instead: use linkage on condensed form via scipy
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    # condensed distance matrix:
    # ensure diagonal zero
    from scipy.spatial.distance import squareform
    mat = dist.values
    np.fill_diagonal(mat, 0.0)
    condensed = squareform(mat, checks=False)
    Z = linkage(condensed, method=method)
    # threshold t in distance space (1 - rho_thresh)
    t = 1 - threshold
    clusters = fcluster(Z, t=t, criterion='distance')
    # map feature -> cluster
    lab = dict(zip(X.columns, clusters))
    return lab, corr

# ---------------- selection building blocks ----------------
def run_permutation_importance(model, X, y, n_repeats=30, random_state=42):
    res = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    df = pd.DataFrame({
        "feature": X.columns,
        "perm_mean": res.importances_mean,
        "perm_std": res.importances_std
    }).sort_values("perm_mean", ascending=False)
    return df

def run_mrmr_selection(X, y, K=50, outdir=None):
    if pymrmr is not None:
        # pymrmr expects categorical target as first col maybe; produce quantile discretized
        df = X.copy()
        # discretize target to 10 bins
        ydisc = pd.qcut(y, 10, labels=False, duplicates='drop')
        df_insert = df.copy()
        df_insert.insert(0, 'target_disc', ydisc)
        try:
            feat_list = pymrmr.mRMR(df_insert, 'MIQ', min(K, X.shape[1]))
            df_out = pd.DataFrame({"feature": feat_list, "rank_mrmr": list(range(1, len(feat_list)+1))})
            if outdir is not None:
                df_out.to_csv(Path(outdir)/"mrmr_results.csv", index=False)
            return df_out
        except Exception as e:
            print("pymrmr failed:", e)
    # fallback: mutual_info ranking
    try:
        mi = mutual_info_regression(X.fillna(0.0), y, random_state=0)
        dfmi = pd.DataFrame({"feature": X.columns, "mi": mi}).sort_values("mi", ascending=False)
        dfmi['rank_mrmr'] = range(1, len(dfmi)+1)
        if outdir is not None:
            dfmi.to_csv(Path(outdir)/"mrmr_fallback.csv", index=False)
        return dfmi[['feature','rank_mrmr']]
    except Exception as e:
        print("mrmr fallback failed:", e)
        return None

def run_boruta_selection(X, y, seed=42, max_iter=200, outdir=None):
    if BorutaPy is None:
        print("Boruta not available")
        return None
    # Boruta expects numpy and an sklearn estimator
    rf = RandomForestRegressor(n_jobs=-1, random_state=seed)
    bor = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=seed, max_iter=max_iter)
    bor.fit(X.values, y.values)
    df = pd.DataFrame({"feature": X.columns, "boruta_support": bor.support_, "boruta_rank": bor.ranking_})
    if outdir is not None:
        df.to_csv(Path(outdir)/"boruta_results.csv", index=False)
    return df.sort_values("boruta_rank")

def run_hsic_lasso_selection(X, y, k=50, outdir=None):
    if HSICLasso is None:
        print("HSICLasso not available")
        return None
    try:
        hsic = HSICLasso()
        hsic.input(X.values, y.values.reshape(-1,1), X.columns.tolist())
        hsic.classification = False
        hsic.numFeat = min(k, X.shape[1])
        hsic.lambda_ = 0.1
        hsic.run()
        feats = hsic.getFeatures()
        df = pd.DataFrame({"feature": feats, "hsic_rank": list(range(1, len(feats)+1))})
        if outdir is not None:
            df.to_csv(Path(outdir)/"hsic_results.csv", index=False)
        return df
    except Exception as e:
        print("HSICLasso failed:", e)
        return None

def run_stability_selection(X, y, outdir=None, seed=42):
    if StabilitySelection is None:
        print("stability_selection package not available")
        return None
    try:
        base = Lasso_s(random_state=seed, max_iter=20000)
        ss = StabilitySelection(base_estimator=base, lambda_name='alpha',
                                lambda_grid=np.logspace(-3,1,20),
                                n_bootstrap_iterations=100, threshold=0.6, random_state=seed)
        ss.fit(X.values, y.values)
        try:
            stability_scores = ss.stability_scores_
            df = pd.DataFrame({"feature": X.columns, "stability_score": stability_scores})
            df['stability_keep'] = df['stability_score'] >= 0.6
            if outdir is not None:
                df.to_csv(Path(outdir)/"stability_selection.csv", index=False)
            return df.sort_values("stability_score", ascending=False)
        except Exception:
            # fallback: use get_support
            mask = ss.get_support()
            df = pd.DataFrame({"feature": X.columns, "stability_keep": mask})
            if outdir is not None:
                df.to_csv(Path(outdir)/"stability_selection.csv", index=False)
            return df
    except Exception as e:
        print("StabilitySelection failed:", e)
        return None

# ---------------- main pipeline ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-csv", required=True)
    ap.add_argument("--target-csv", default=None)
    ap.add_argument("--merge-key", default=None)
    ap.add_argument("--target-col", required=True)
    ap.add_argument("--manual-features", default=None, help="JSON list or comma-separated list of manual priority features")
    ap.add_argument("--outdir", default="./feature_select_out")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fuzzy-thresh", type=float, default=0.9)
    ap.add_argument("--corr-cluster-thresh", type=float, default=0.90)
    ap.add_argument("--panel-sizes", type=int, nargs="+", default=[10,20,50])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    ensure_dir(outdir/"00_audit")
    ensure_dir(outdir/"01_models")
    ensure_dir(outdir/"02_importance")
    ensure_dir(outdir/"03_selection")
    ensure_dir(outdir/"04_panels")
    ensure_dir(outdir/"05_ablation")
    seed = args.seed
    np.random.seed(seed)

    # Load
    feats = pd.read_csv(args.feature_csv)
    if args.target_csv:
        targ = pd.read_csv(args.target_csv)
        assert args.merge_key is not None, "--merge-key required when target-csv provided"
        df = feats.merge(targ, on=args.merge_key, how="inner")
    else:
        df = feats.copy()
    assert args.target_col in df.columns, f"{args.target_col} not found"

    y = df[args.target_col].astype(float)
    # drop rows with NaN target
    mask_ok = ~y.isna()
    df = df.loc[mask_ok].copy()
    y = df[args.target_col].astype(float)

    # prepare manual features list
    if args.manual_features:
        try:
            manual_list = json.loads(args.manual_features)
            if not isinstance(manual_list, list):
                raise ValueError
        except Exception:
            manual_list = [s.strip() for s in args.manual_features.split(",") if s.strip()]
    else:
        manual_list = []

    # meta cols
    meta_cols = [args.target_col]
    if args.merge_key and args.merge_key in df.columns:
        meta_cols.append(args.merge_key)

    # numeric features only
    X_all = df.drop(columns=meta_cols)
    numeric = [c for c in X_all.columns if pd.api.types.is_numeric_dtype(X_all[c])]
    X_all = X_all[numeric].copy()
    # replace inf
    X_all = X_all.replace([np.inf, -np.inf], np.nan)

    # 00 audit: missingness and variance
    missing = X_all.isna().mean().sort_values(ascending=False)
    missing.to_csv(outdir/"00_audit"/"missing_rate.csv")
    var = X_all.var(axis=0, skipna=True).sort_values()
    var.to_csv(outdir/"00_audit"/"variance.csv")

    # simple filter
    high_missing_thresh = 0.7
    low_var_thresh = 1e-12
    drop_missing = missing[missing > high_missing_thresh].index.tolist()
    drop_lowvar = var[var <= low_var_thresh].index.tolist()
    pd.Series(drop_missing).to_csv(outdir/"00_audit"/"drop_high_missing.csv", index=False)
    pd.Series(drop_lowvar).to_csv(outdir/"00_audit"/"drop_low_variance.csv", index=False)

    X_filtered = X_all.drop(columns = list(set(drop_missing + drop_lowvar)), errors='ignore').copy()
    print(f"Features before: {X_all.shape[1]}, after basic filter: {X_filtered.shape[1]}")

    # 1) name-level fuzzy dedupe against manual list
    fuzzy_matches = fuzzy_name_matches(manual_list, X_filtered.columns.tolist(), thresh=args.fuzzy_thresh)
    pd.DataFrame(fuzzy_matches, columns=["manual","candidate","score"]).to_csv(outdir/"00_audit"/"fuzzy_name_matches.csv", index=False)

    # 2) numeric equivalence detection (manual vs candidates)
    if len(manual_list) > 0:
        numeq = numeric_equivalence_table(X_filtered, manual_list, X_filtered.columns.tolist(), corr_thresh=0.995, diff_tol=1e-9)
        if not numeq.empty:
            numeq.to_csv(outdir/"00_audit"/"numeric_equivalents_manual_vs_candidates.csv", index=False)

    # 3) correlation clustering
    cluster_map, corrmat = correlation_clusters(X_filtered, threshold=args.corr_cluster_thresh)
    pd.Series(cluster_map).to_frame("cluster").to_csv(outdir/"00_audit"/"feature_clusters.csv")
    corrmat.to_csv(outdir/"00_audit"/"corr_abs.csv")

    # 4) build cluster representatives (choose feature with max var or max MI with target)
    clusters = defaultdict(list)
    for feat, c in cluster_map.items():
        clusters[c].append(feat)
    cluster_reps = {}
    for c, feats_list in clusters.items():
        if len(feats_list) == 1:
            cluster_reps[c] = feats_list[0]
            continue
        # pick the feature in cluster with highest mutual_info with target (fallback to variance)
        try:
            mi = mutual_info_regression(X_filtered[feats_list].fillna(0.0), y, random_state=0)
            idx = int(np.argmax(mi))
            cluster_reps[c] = feats_list[idx]
        except Exception:
            vs = X_filtered[feats_list].var().fillna(0.0)
            cluster_reps[c] = vs.idxmax()
    pd.DataFrame([{"cluster":c, "rep":cluster_reps[c], "size":len(clusters[c])} for c in clusters]).to_csv(outdir/"00_audit"/"cluster_representatives.csv", index=False)

    # 5) create a reduced candidate set: cluster reps + manual list (ensuring manual features included)
    rep_features = sorted(set(cluster_reps.values()))
    manual_present = [f for f in manual_list if f in X_filtered.columns]
    candidate_pool = list(dict.fromkeys(manual_present + rep_features))  # manual first
    print(f"Candidate pool size after clustering & manual merge: {len(candidate_pool)}")

    # 6) baseline models on manual20 alone (if provided)
    results = {}
    Xc = X_filtered[candidate_pool].copy()
    # define simple models
    models = {}
    # ElasticNet pipeline with median imputer
    enet_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("enet", ElasticNet(random_state=seed, max_iter=20000))])
    models['enet'] = enet_pipe
    models['rf'] = Pipeline([("imputer", SimpleImputer(strategy="median")), ("rf", RandomForestRegressor(n_estimators=600, n_jobs=-1, random_state=seed))])
    if lgb is not None:
        models['lgbm'] = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.05, num_leaves=63, min_data_in_leaf=5, feature_pre_filter=False,
                                          subsample=0.8, colsample_bytree=0.8, verbose=-1, force_row_wise=True)
    if xgb is not None:
        models['xgb'] = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.05, random_state=seed, n_jobs=-1, objective='reg:squarederror')

    # cross-validate each on the manual/rep feature pool
    cv = KFold(n_splits=args.kfold, shuffle=True, random_state=seed)
    for name, mdl in models.items():
        try:
            print("CV eval model", name)
            # cross_val_score for R2 and MAE via scoring
            r2 = np.mean(cross_val_score(mdl, Xc.fillna(0.0), y, cv=cv, scoring="r2", n_jobs=-1))
            mae = -np.mean(cross_val_score(mdl, Xc.fillna(0.0), y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1))
            results[name] = {"r2": float(r2), "mae": float(mae)}
        except Exception as e:
            print("cv fail for", name, e)
    with open(outdir/"01_models"/"baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 7) fit a tree model on full candidate pool for importance/SHAP if available
    # pick best available tree
    tree_model = None
    if 'lgbm' in models:
        tree_model = models['lgbm']
    elif 'xgb' in models:
        tree_model = models['xgb']
    elif 'rf' in models:
        tree_model = models['rf']

    if tree_model is not None:
        try:
            tree_model.fit(Xc.fillna(0.0), y)
            # permutation importance
            perm_df = run_permutation_importance(tree_model, Xc.fillna(0.0), y, n_repeats=30, random_state=seed)
            perm_df.to_csv(outdir/"02_importance"/"permutation_importance.csv", index=False)
            # SHAP
            if shap is not None and (hasattr(tree_model, "booster") or hasattr(tree_model, "predict")):
                try:
                    explainer = shap.Explainer(tree_model)
                    shap_values = explainer(Xc.fillna(0.0))
                    # summary plots
                    plt.figure(figsize=(8,6))
                    shap.summary_plot(shap_values, Xc, show=False)
                    plt.tight_layout()
                    plt.savefig(outdir/"02_importance"/"shap_summary_dot.png", dpi=200)
                    plt.close()
                    # mean abs
                    mean_abs = np.abs(shap_values.values).mean(axis=0)
                    pd.DataFrame({"feature": Xc.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).to_csv(outdir/"02_importance"/"shap_mean_abs.csv", index=False)
                except Exception as e:
                    print("SHAP failed:", e)
        except Exception as e:
            print("Tree fit failed:", e)

    # 8) run selection methods (Boruta, mRMR, HSIC, Stability)
    sel_out = outdir/"03_selection"
    ensure_dir(sel_out)

    boruta_df = run_boruta_selection(Xc, y, seed=seed, max_iter=200, outdir=sel_out)
    mrmr_df = run_mrmr_selection(Xc, y, K=min(100, Xc.shape[1]), outdir=sel_out)
    hsic_df = run_hsic_lasso_selection(Xc, y, k=min(100, Xc.shape[1]), outdir=sel_out)
    stab_df = run_stability_selection(Xc, y, outdir=sel_out, seed=seed)

    # 9) rank aggregation & voting
    rank_frames = []
    if 'perm_df' in locals() and perm_df is not None:
        tmp = perm_df[['feature']].copy(); tmp['rank_perm'] = range(1, len(tmp)+1); rank_frames.append(tmp)
    if boruta_df is not None:
        tmp = boruta_df[['feature','boruta_rank']].copy().rename(columns={'boruta_rank':'rank_boruta'}); rank_frames.append(tmp)
    if mrmr_df is not None:
        tmp = mrmr_df[['feature','rank_mrmr']].copy(); rank_frames.append(tmp)
    if hsic_df is not None:
        tmp = hsic_df[['feature','hsic_rank']].copy(); rank_frames.append(tmp)
    if stab_df is not None and 'stability_score' in stab_df.columns:
        tmp = stab_df[['feature','stability_score']].copy().rename(columns={'stability_score':'rank_stability'}) 
        # invert to rank-like
        tmp['rank_stability'] = tmp['rank_stability'].rank(ascending=False)
        rank_frames.append(tmp)

    if len(rank_frames) == 0:
        print("No selection results to aggregate")
    else:
        from functools import reduce
        rank_tbl = reduce(lambda a,b: pd.merge(a,b, on='feature', how='outer'), rank_frames)
        rank_cols = [c for c in rank_tbl.columns if c.startswith('rank_')]
        # replace NaN with large penalty
        big = 1e6
        for c in rank_cols:
            rank_tbl[c] = pd.to_numeric(rank_tbl[c], errors='coerce').fillna(big)
        rank_tbl['borda'] = rank_tbl[rank_cols].sum(axis=1)
        rank_tbl['votes'] = (rank_tbl[rank_cols] < big).sum(axis=1)
        rank_tbl = rank_tbl.sort_values(['votes','borda'], ascending=[False, True])
        rank_tbl.to_csv(outdir/"04_panels"/"rank_aggregation.csv", index=False)

        # Panels
        panels = {}
        for k in args.panel_sizes:
            topk = rank_tbl.head(min(k, len(rank_tbl)))['feature'].tolist()
            panels[f"Panel-{k}"] = topk
            pd.Series(topk, name='feature').to_csv(outdir/"04_panels"/f"panel_{k}.csv", index=False)
        # save
        with open(outdir/"04_panels"/"final_panels.json", "w") as f:
            json.dump(panels, f, indent=2)

    # 10) ablation using permutation ordering (if available)
    if 'perm_df' in locals() and perm_df is not None:
        order = perm_df['feature'].tolist()
        ks = [5,10,20,30,50,100]
        ks = [k for k in ks if k <= len(order)]
        ablation = []
        for k in ks:
            feats = order[:k]
            # pick a simple model to evaluate
            eval_model = enet_pipe
            try:
                r2 = np.mean(cross_val_score(eval_model, Xc[feats].fillna(0.0), y, cv=cv, scoring="r2", n_jobs=-1))
                mae = -np.mean(cross_val_score(eval_model, Xc[feats].fillna(0.0), y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1))
                ablation.append({"k":k,"r2":float(r2),"mae":float(mae)})
            except Exception as e:
                print("ablation CV fail", e)
        abldf = pd.DataFrame(ablation)
        abldf.to_csv(outdir/"05_ablation"/"ablation_curve.csv", index=False)
        # plot
        if not abldf.empty:
            plt.figure()
            plt.plot(abldf['k'], abldf['r2'], marker='o', label='R2')
            plt.xlabel("top-k features")
            plt.ylabel("R2")
            plt.tight_layout()
            plt.savefig(outdir/"05_ablation"/"ablation_r2.png", dpi=200)
            plt.close()

    print("Done. Outputs in", outdir)

if __name__ == "__main__":
    main()
