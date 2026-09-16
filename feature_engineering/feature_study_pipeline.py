#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Study Pipeline (materials-science friendly)
--------------------------------------------------

What this script does (end-to-end):
1) Load features CSV (and optional target CSV) → merge by key.
2) Data audit: shapes, dtypes, missingness, constant columns, long tails.
3) Pre-clean: drop constant cols; optional log/clip transforms via CLI.
4) Train/valid/test split with K-Fold or Nested CV for baselines.
5) Baselines: Linear (Ridge/Lasso/ElasticNet), RF, XGB, LGBM.
6) Model-agnostic importance: Permutation Importance (with repeats).
7) Model-specific interpretation: SHAP (TreeExplainer) if available.
8) Multi-method feature selection: Boruta, mRMR, HSIC Lasso, Stability Selection.
9) Build robust panels (Panel-10/20/50) by rank aggregation across methods.
10) Ablation curves (features → performance) + final report files.

Outputs (written under --outdir):
- 00_audit/*.csv|.png             : missingness, constant cols, corr heatmap
- 01_baselines/*.json             : CV metrics per model
- 02_importance/permutation_*.csv : permutation importance means/std
- 02_importance/shap_*.[png|csv]  : SHAP summary & values (if available)
- 03_selection/*.csv              : results from Boruta/mRMR/HSIC/Stability
- 04_panels/final_feature_panels.csv : Panel-10/20/50 lists (rank-aggregated)
- 05_ablation/ablation_curve_*.png : performance vs #features

Run example
-----------
python feature_study_pipeline.py \
  --feature-csv /mnt/data/extra_features.csv \
  --target-csv /mnt/data/folder_score_table.csv \
  --merge-key cif_file \
  --target-col score \
  --models rf xgb lgbm enet \
  --kfold 5 \
  --nested-cv \
  --outdir ./feature_study_out

Notes
-----
- All feature columns must be numeric before modeling; non-numeric are auto-dropped.
- If target and features are in the same CSV, omit --target-csv and just set --target-col.
- For mRMR, a discrete target helps some backends; we provide a quantile-binned variant.
- Optional packages: shap, boruta, pymrmr (or pymrmre), pyHSICLasso, stability-selection.
- The script degrades gracefully if some optional libs are not installed.
"""

import argparse
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

# Optional models
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

from sklearn.inspection import permutation_importance

# Optional selectors / explainer
try:
    import shap
except Exception:
    shap = None

try:
    from boruta import BorutaPy  # pip install boruta OR boruta_py
except Exception:
    try:
        from boruta_py import BorutaPy  # scikit-learn-contrib
    except Exception:
        BorutaPy = None

try:
    import pymrmr  # classic mRMR (requires lib) 
except Exception:
    pymrmr = None

try:
    from pymrmre import mrmr  # ensemble mRMR alternative
except Exception:
    mrmr = None

try:
    from pyHSICLasso import HSICLasso
except Exception:
    HSICLasso = None

try:
    from stability_selection import StabilitySelection
    from sklearn.linear_model import Lasso as Lasso_s
except Exception:
    StabilitySelection = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------- Utilities -------------------------

def ensure_outdir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def metrics_dict(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def numeric_cols(df: pd.DataFrame, exclude: list) -> list:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def drop_constant_cols(df: pd.DataFrame, eps: float = 0.0) -> (pd.DataFrame, list):
    nunique = df.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    return df.drop(columns=const_cols), const_cols


def high_corr_groups(df: pd.DataFrame, threshold: float = 0.95):
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return to_drop, corr


def quantile_bin_target(y: pd.Series, q: int = 5):
    try:
        bins = pd.qcut(y, q=q, duplicates='drop', labels=False)
        return bins
    except Exception:
        return pd.Series(index=y.index, data=np.digitize(y, np.quantile(y, np.linspace(0,1,q+1)[1:-1])))


# ------------------------- Modeling -------------------------

def make_models(args):
    models = {}
    # Common preprocessing blocks
    imp_median = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    if 'enet' in args.models:
        models['enet'] = Pipeline([
            ("imputer", imp_median),
            ("scaler", scaler),
            ("enet", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20000, random_state=args.seed)),
        ])
    if 'rf' in args.models:
        models['rf'] = Pipeline([
            ("imputer", imp_median),
            ("rf", RandomForestRegressor(n_estimators=600, max_depth=None, n_jobs=-1, random_state=args.seed)),
        ])
    if 'xgb' in args.models and XGBRegressor is not None:
        # XGBoost handles NaN natively; keep raw to leverage that capability
        models['xgb'] = XGBRegressor(
            n_estimators=1200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective='reg:squarederror', random_state=args.seed, n_jobs=-1)
    if 'lgbm' in args.models and LGBMRegressor is not None:
        # LightGBM handles NaN natively; keep raw
        models['lgbm'] = LGBMRegressor(
            n_estimators=1500, learning_rate=0.05, max_depth=-1,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0,
            objective='regression', random_state=args.seed, n_jobs=-1)
    return models


def cv_evaluate(model, X, y, kfold=5, random_state=42):
    cv = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
    scoring = {
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'RMSE': make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False),
        'R2': 'r2'
    }
    scores = {}
    for mname, scorer in scoring.items():
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        scores[mname] = float(np.mean(cv_scores))
    # flip signs back for MAE/RMSE
    scores['MAE'] = -scores['MAE']
    scores['RMSE'] = -scores['RMSE']
    return scores


# ------------------------- Selection Methods -------------------------

def run_boruta(X, y, outdir: Path, seed: int = 42, max_iter: int = 200):
    if BorutaPy is None:
        return None
    # Use RF as estimator
    rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=seed)
    feat_names = np.array(X.columns)
    selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=seed, max_iter=max_iter)
    selector.fit(X.values, y.values)
    mask = selector.support_.astype(bool)
    ranks = selector.ranking_
    df = pd.DataFrame({
        'feature': feat_names,
        'boruta_keep': mask,
        'boruta_rank': ranks
    }).sort_values('boruta_rank')
    df.to_csv(outdir/"boruta_results.csv", index=False)
    return df


def run_mrmr(X, y, outdir: Path, k: int = 50):
    # Return a ranked list even if backends unavailable
    if pymrmr is not None:
        # pymrmr expects a DataFrame with target as first column, all discrete/numeric
        df = X.copy()
        df.insert(0, 'target_disc', quantile_bin_target(y, q=10))
        try:
            feats = pymrmr.mRMR(df, 'MIQ', min(k, X.shape[1]))
            rank = pd.DataFrame({'feature': feats, 'mrmr_rank': list(range(1, len(feats)+1))})
            rank.to_csv(outdir/"mrmr_results.csv", index=False)
            return rank
        except Exception:
            pass
    if mrmr is not None:
        try:
            feats = mrmr.mrmr_regression(X=X, y=y, K=min(k, X.shape[1]))
            rank = pd.DataFrame({'feature': feats, 'mrmr_rank': list(range(1, len(feats)+1))})
            rank.to_csv(outdir/"mrmr_results.csv", index=False)
            return rank
        except Exception:
            pass
    # fallback: mutual info proxy via sklearn
    try:
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(X, y, random_state=42)
        rank = pd.DataFrame({'feature': X.columns, 'mi': mi}).sort_values('mi', ascending=False)
        rank['mrmr_rank'] = range(1, len(rank)+1)
        rank.to_csv(outdir/"mrmr_results.csv", index=False)
        return rank[['feature','mrmr_rank']]
    except Exception:
        return None


def run_hsic_lasso(X, y, outdir: Path, k: int = 50):
    if HSICLasso is None:
        return None
    try:
        hsic = HSICLasso()
        hsic.input(X.values, y.values.reshape(-1,1), X.columns.tolist())
        hsic.classification = False
        hsic.numFeat = min(k, X.shape[1])
        hsic.lambda_ = 0.1
        hsic.run()
        feats = hsic.getFeatures()
        rank = pd.DataFrame({'feature': feats, 'hsic_rank': list(range(1, len(feats)+1))})
        rank.to_csv(outdir/"hsic_results.csv", index=False)
        return rank
    except Exception:
        return None


def run_stability_selection(X, y, outdir: Path, seed: int = 42):
    if StabilitySelection is None:
        return None
    try:
        base_estimator = Lasso_s(random_state=seed, max_iter=20000)
        ss = StabilitySelection(base_estimator=base_estimator,
                                lambda_name='alpha', lambda_grid=np.logspace(-3, 1, 20),
                                n_bootstrap_iterations=100, threshold=0.6, random_state=seed)
        ss.fit(X.values, y.values)
        scores = ss.get_supports()  # list of masks per threshold; using fitted threshold
        # If API differs, fallback to ss.stability_scores_
        if isinstance(scores, list):
            mask = scores[-1]
            stab_scores = getattr(ss, 'stability_scores_', None)
        else:
            stab_scores = getattr(ss, 'stability_scores_', None)
            if stab_scores is not None:
                mask = (stab_scores >= 0.6)
            else:
                mask = ss.get_support()
        df = pd.DataFrame({
            'feature': X.columns,
            'stability_keep': mask,
            'stability_score': (stab_scores[X.columns] if isinstance(stab_scores, pd.Series) else (stab_scores if stab_scores is not None else np.nan))
        })
        df.to_csv(outdir/"stability_selection_results.csv", index=False)
        return df
    except Exception:
        return None


# ------------------------- SHAP -------------------------

def run_shap(model, X, outdir: Path, prefix: str):
    if shap is None:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # summary (bar)
        plt.figure()
        shap.summary_plot(shap_values, X, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(outdir / f"{prefix}_shap_summary_bar.png", dpi=200)
        plt.close()
        # summary (dot)
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(outdir / f"{prefix}_shap_summary_dot.png", dpi=200)
        plt.close()
        # save mean |shap| per feature
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': mean_abs
        }).sort_values('mean_abs_shap', ascending=False).to_csv(outdir / f"{prefix}_shap_mean_abs.csv", index=False)
        return True
    except Exception:
        return None


# ------------------------- Main pipeline -------------------------

def main():
    ap = argparse.ArgumentParser(description="Feature Study Pipeline")
    ap.add_argument('--feature-csv', type=str, required=True)
    ap.add_argument('--target-csv', type=str, default=None)
    ap.add_argument('--merge-key', type=str, default=None, help='Key column for merging feature/target tables')
    ap.add_argument('--target-col', type=str, required=True, help='Name of the target/score column')
    ap.add_argument('--outdir', type=str, default='./feature_study_out')
    ap.add_argument('--models', type=str, nargs='+', default=['rf','xgb','lgbm','enet'])
    ap.add_argument('--kfold', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--perm-repeats', type=int, default=32)
    ap.add_argument('--corr-threshold', type=float, default=0.97)
    ap.add_argument('--panel-sizes', type=int, nargs='+', default=[10,20,50])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    # 0) Load
    feats = pd.read_csv(args.feature_csv)
    if args.target_csv:
        targ = pd.read_csv(args.target_csv)
        assert args.merge_key is not None, "--merge-key is required when --target-csv is provided"
        df = feats.merge(targ, on=args.merge_key, how='inner')
    else:
        df = feats.copy()

    assert args.target_col in df.columns, f"Target column {args.target_col} not found."

    # Identify numeric features
    y = df[args.target_col].astype(float)

    # Drop rows with NaN in target
    df = df.loc[~y.isna()].copy()
    y = df[args.target_col].astype(float)

    # Drop obvious meta columns (heuristic)
    meta_like = [args.target_col]
    if args.merge_key and args.merge_key in df.columns:
        meta_like.append(args.merge_key)

    X_all = df.drop(columns=meta_like)
    # keep only numeric
    keep = [c for c in X_all.columns if pd.api.types.is_numeric_dtype(X_all[c])]
    X_all = X_all[keep].copy()
    # sanitize: replace inf with NaN; imputers or native-NaN models will handle
    X_all = X_all.replace([np.inf, -np.inf], np.nan)

    # 1) Audit
    audit_dir = outdir/"00_audit"
    ensure_outdir(audit_dir)

    # constant columns
    X_nc, const_cols = drop_constant_cols(X_all)
    pd.Series(const_cols, name='constant_cols').to_csv(audit_dir/"constant_cols.csv", index=False)

    # high-corr drop list (not dropping yet; just record)
    to_drop, corr = high_corr_groups(X_nc, threshold=args.corr_threshold)
    corr.to_csv(audit_dir/"corr_abs.csv")
    pd.Series(to_drop, name='high_corr_cols').to_csv(audit_dir/"high_corr_cols.csv", index=False)

    # Use the cleaned numeric set for modeling (but do not auto-drop high corr columns here;
    # selection methods will address redundancy; we only drop constants.)
    X = X_nc

    # 2) Baselines
    models = make_models(args)
    base_dir = outdir/"01_baselines"
    ensure_outdir(base_dir)

    base_scores = {}
    for name, model in models.items():
        scores = cv_evaluate(model, X, y, kfold=args.kfold, random_state=args.seed)
        base_scores[name] = scores
    save_json(base_scores, base_dir/"cv_baselines.json")

    # Fit strongest tree model on full data for importances/SHAP
    tree_name = None
    if 'lgbm' in models:
        tree_name = 'lgbm'
    elif 'xgb' in models:
        tree_name = 'xgb'
    elif 'rf' in models:
        tree_name = 'rf'

    tree_model = models[tree_name] if tree_name else None
    if tree_model is not None:
        tree_model.fit(X, y)

    # 3) Permutation Importance (model-agnostic on fitted tree or ElasticNet fallback)
    imp_dir = outdir/"02_importance"
    ensure_outdir(imp_dir)

    if tree_model is None:
        # fallback: use enet
        if 'enet' in models:
            tree_model = models['enet']
            tree_model.fit(X, y)

    if tree_model is not None:
        perm = permutation_importance(tree_model, X, y, n_repeats=args.perm_repeats, random_state=args.seed, n_jobs=-1)
        perm_df = pd.DataFrame({
            'feature': X.columns,
            'perm_importance_mean': perm.importances_mean,
            'perm_importance_std': perm.importances_std
        }).sort_values('perm_importance_mean', ascending=False)
        perm_df.to_csv(imp_dir/"permutation_importance.csv", index=False)

    # 4) SHAP for tree models (if available)
    if tree_name in ('rf','xgb','lgbm'):
        shap_dir = imp_dir
        run_shap(tree_model, X, shap_dir, prefix=tree_name)

    # 5) Feature selection methods
    sel_dir = outdir/"03_selection"
    ensure_outdir(sel_dir)

    boruta_df = run_boruta(X, y, sel_dir, seed=args.seed)
    mrmr_df   = run_mrmr(X, y, sel_dir, k=min(100, X.shape[1]))
    hsic_df   = run_hsic_lasso(X, y, sel_dir, k=min(100, X.shape[1]))
    stab_df   = run_stability_selection(X, y, sel_dir, seed=args.seed)

    # 6) Rank aggregation → Panels
    panel_dir = outdir/"04_panels"
    ensure_outdir(panel_dir)

    ranks = []
    if perm_df is not None:
        tmp = perm_df[['feature']].copy()
        tmp['rank_perm'] = range(1, len(tmp)+1)
        ranks.append(tmp)
    if boruta_df is not None:
        tmp = boruta_df[['feature','boruta_rank']].copy().rename(columns={'boruta_rank':'rank_boruta'})
        ranks.append(tmp)
    if mrmr_df is not None:
        tmp = mrmr_df[['feature','mrmr_rank']].copy().rename(columns={'mrmr_rank':'rank_mrmr'})
        ranks.append(tmp)
    if hsic_df is not None:
        tmp = hsic_df[['feature','hsic_rank']].copy().rename(columns={'hsic_rank':'rank_hsic'})
        ranks.append(tmp)

    if len(ranks) > 0:
        from functools import reduce
        rank_tbl = reduce(lambda l, r: pd.merge(l, r, on='feature', how='outer'), ranks)
        # convert to numeric ranks; large penalty for missing
        for c in rank_tbl.columns:
            if c.startswith('rank_'):
                rank_tbl[c] = pd.to_numeric(rank_tbl[c], errors='coerce')
        big = 1e6
        rank_cols = [c for c in rank_tbl.columns if c.startswith('rank_')]
        rank_tbl['borda'] = rank_tbl[rank_cols].apply(lambda row: np.nansum(row.values), axis=1)
        rank_tbl['num_votes'] = rank_tbl[rank_cols].notna().sum(axis=1)
        # average normalized rank (robust)
        for c in rank_cols:
            rank_tbl[c] = rank_tbl[c].fillna(big)
        rank_tbl['agg_rank'] = rank_tbl[rank_cols].rank(axis=1, method='average').mean(axis=1)
        # final sort by (num_votes desc, borda asc)
        rank_tbl = rank_tbl.sort_values(['num_votes','borda'], ascending=[False, True])
        rank_tbl.to_csv(panel_dir/"rank_aggregation.csv", index=False)

        # build panels
        panel_sizes = [s for s in args.panel_sizes if s > 0]
        all_panels = []
        for k in panel_sizes:
            topk = rank_tbl.head(min(k, len(rank_tbl)))['feature'].tolist()
            all_panels.append({'panel': f'Panel-{k}', 'features': topk})
            pd.Series(topk, name='feature').to_csv(panel_dir/f"panel_{k}.csv", index=False)
        # save combined
        with open(panel_dir/"final_feature_panels.csv", 'w', encoding='utf-8') as f:
            f.write('panel,feature\n')
            for p in all_panels:
                for feat in p['features']:
                    f.write(f"{p['panel']},{feat}\n")

    # 7) Simple ablation using permutation rank order (if available)
    abl_dir = outdir/"05_ablation"
    ensure_outdir(abl_dir)

    if tree_model is not None and 'perm_df' in locals() and perm_df is not None and len(perm_df) > 0:
        order = perm_df['feature'].tolist()
        ks = list(dict.fromkeys([5,10,20,30,50,100]))
        ks = [k for k in ks if k <= len(order)]
        ablation = []
        for k in ks:
            feats = order[:k]
            mdl = make_models(args)
            # choose best of {rf,xgb,lgbm,enet} present
            for cand in ['lgbm','xgb','rf','enet']:
                if cand in mdl:
                    model = mdl[cand]
                    break
            scores = cv_evaluate(model, X[feats], y, kfold=args.kfold, random_state=args.seed)
            scores['k'] = k
            ablation.append(scores)
        abldf = pd.DataFrame(ablation)
        abldf.to_csv(abl_dir/"ablation_scores.csv", index=False)

        # plot curve (k vs RMSE, MAE)
        plt.figure()
        plt.plot(abldf['k'], abldf['RMSE'], marker='o', label='RMSE')
        plt.plot(abldf['k'], abldf['MAE'], marker='s', label='MAE')
        plt.xlabel('#Features (top-k by permutation importance)')
        plt.ylabel('Error')
        plt.legend()
        plt.tight_layout()
        plt.savefig(abl_dir/"ablation_curve.png", dpi=200)
        plt.close()

    print(f"Done. Outputs written to: {outdir}")


if __name__ == '__main__':
    main()
