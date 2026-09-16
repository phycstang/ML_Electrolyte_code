
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_study_full_optimized.py — 安全交叉验证 + 更稳健的特征选择 + LGB/XGB 提前停止 + 额外诊断
- 基于用户上传的 feature_study_full.py 改进（见变更列表）
"""

from __future__ import annotations
import os, sys, json, math, argparse, warnings, socket, random
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import (KFold, StratifiedKFold, train_test_split, cross_val_score, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, SelectKBest, VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, LogisticRegression
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)

# ========== Optional libs ==========
xgb = None
try:
    import xgboost as xgb_mod
    from xgboost import XGBRegressor, XGBClassifier
    xgb = True
except Exception:
    pass

lgb = None
try:
    import lightgbm as lgb_mod
    from lightgbm import LGBMRegressor, LGBMClassifier
    lgb = True
except Exception:
    pass

HAS_OPTUNA = False
try:
    import optuna
    from optuna.integration import lightgbm as opt_lgbm
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

TSNE = None
try:
    from sklearn.manifold import TSNE
except Exception:
    pass

umap = None
try:
    import umap
except Exception:
    pass

shap = None
try:
    import shap
except Exception:
    pass

warnings.filterwarnings("ignore", category=UserWarning)

# ========== Utils ==========
def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

# ========== Args ==========
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--target", default="id_score")
    p.add_argument("--task", choices=["regression","classification"], default="regression")
    p.add_argument("--id_prefix", default="id_")
    p.add_argument("--outdir", default="feature_study_report_opt")
    p.add_argument("--drop_missing_gt", type=float, default=0.60)
    p.add_argument("--corr_heatmap", action="store_true")
    p.add_argument("--plot_feature_hists", action="store_true")
    p.add_argument("--run_tsne", action="store_true")
    p.add_argument("--run_umap", action="store_true")
    p.add_argument("--run_shap", action="store_true")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--mi_k", type=int, default=80)
    p.add_argument("--tree_k", type=int, default=80)
    p.add_argument("--collinear_r", type=float, default=0.95)
    p.add_argument("--n_boot", type=int, default=50)
    p.add_argument("--sample_rows", type=int, default=0)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--tune_iter", type=int, default=60)
    p.add_argument("--tune_jobs", type=int, default=-1)
    p.add_argument("--gpu", choices=["auto","cpu","gpu"], default="auto", help="XGBoost/LightGBM: auto-detect or force GPU/CPU")

    # 新增：更安全的 CV/特征选择
    p.add_argument("--cv_stratify_regression", action="store_true",
                   help="回归任务中按照目标分桶做拟似分层 KFold")
    p.add_argument("--leakage_safe", action="store_true",
                   help="在 CV 内部做特征选择（SelectKBest+去共线），避免在全数据上筛选造成的信息泄露")
    p.add_argument("--select_k", type=int, default=128, help="leakage_safe 时 SelectKBest 选择的特征个数")

    # 新增：LGB/XGB 训练细化
    p.add_argument("--early_stopping_rounds", type=int, default=200)
    p.add_argument("--valid_size", type=float, default=0.2)

    return p.parse_args()

# ========== IO & cleaning ==========
def load_and_clean(csv_path: str, target: str, id_prefix: str, drop_missing_gt: float, sample_rows: int = 0):
    df = pd.read_csv(csv_path)
    if sample_rows and sample_rows < len(df):
        df = df.head(sample_rows).copy()
    assert target in df.columns, f"目标列 {target} 不存在！"
    meta_cols = [c for c in df.columns if c.startswith(id_prefix) and c != target]

    def to_numeric_if_possible(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            tmp = pd.to_numeric(s, errors="coerce")
            return tmp if tmp.notna().mean() > 0.5 else s
        return pd.to_numeric(s, errors="coerce")

    df_num = df.apply(to_numeric_if_possible)
    y = df_num[target]
    X = df_num.drop(columns=[target] + meta_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    all_nan = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=all_nan)

    const_cols = X.columns[X.nunique(dropna=True) <= 1].tolist()
    X = X.drop(columns=const_cols)

    high_missing = X.columns[X.isna().mean() > drop_missing_gt].tolist()
    X = X.drop(columns=high_missing)

    mask = y.notna()
    X, y = X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)

    X = X.fillna(X.median(numeric_only=True))
    return df, X, y, {"all_nan": all_nan, "const_cols": const_cols, "high_missing": high_missing, "meta_cols": meta_cols}

# ========== EDA/visualize ==========
def eda(task: str, X: pd.DataFrame, y: pd.Series, outdir: str, args):
    safe_makedirs(outdir)
    plt.figure(figsize=(7,5))
    if task == "regression":
        plt.hist(y.values, bins=30)
        plt.title("Target distribution")
        plt.xlabel(args.target); plt.ylabel("count")
    else:
        vc = pd.Series(y).value_counts().sort_index()
        plt.bar(vc.index.astype(str), vc.values)
        plt.title("Target class distribution"); plt.xlabel(args.target); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "target_dist.png"), dpi=180); plt.close()

    if task == "regression":
        pearson = X.apply(lambda col: np.corrcoef(col.values, y.values)[0,1] if col.notna().all() else np.nan)
        spearman_vals = []
        for c in X.columns:
            try:
                rho, _ = spearmanr(X[c].values, y.values)
            except Exception:
                rho = np.nan
            spearman_vals.append(rho)
        spearman_s = pd.Series(spearman_vals, index=X.columns)
        corr_df = pd.DataFrame({"feature": X.columns, "pearson": pearson.values, "spearman": spearman_s.values})
        corr_df["abs_pearson"] = corr_df["pearson"].abs(); corr_df["abs_spearman"] = corr_df["spearman"].abs()
        corr_df.sort_values("abs_pearson", ascending=False).to_csv(os.path.join(outdir,"corr_target_all.csv"), index=False)
        top = corr_df.sort_values("abs_pearson", ascending=False).head(30)
        plt.figure(figsize=(8,10)); plt.barh(top["feature"][::-1], top["pearson"][::-1])
        plt.title("Top-30 Pearson vs target"); plt.tight_layout(); plt.savefig(os.path.join(outdir,"corr_top30.png"), dpi=180); plt.close()

    if args.corr_heatmap and X.shape[1] <= 300:
        C = np.corrcoef(X.values.T)
        plt.figure(figsize=(10,8)); plt.imshow(C, aspect="auto"); plt.colorbar(); plt.title("Feature correlation heatmap")
        plt.tight_layout(); plt.savefig(os.path.join(outdir,"corr_heatmap.png"), dpi=160); plt.close()

# ========== CV helpers ==========
def stratified_kfold_for_regression(y: pd.Series, n_splits: int = 5, random_state: int = 42, n_bins: int = 10):
    """把连续 y 分桶做分层 KFold，避免折间目标分布漂移"""
    y = pd.Series(y).astype(float)
    # 使用分位数分桶，保证每桶样本数尽量接近
    bins = pd.qcut(y, q=np.linspace(0, 1, n_bins+1), duplicates='drop', labels=False)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), bins

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """去共线：按|corr|>=阈值聚类，每组只保留与y相关性最高的特征（在fit时依据y）"""
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold
        self.keep_: List[str] = []
        self.groups_: List[List[str]] = []

    def fit(self, X: pd.DataFrame, y=None):
        Xc = pd.DataFrame(X).copy()
        C = Xc.corr().abs()
        used = set()
        groups = []
        for col in Xc.columns:
            if col in used: 
                continue
            grp = [col]
            for other in Xc.columns:
                if other != col and other not in used and C.loc[col, other] >= self.threshold:
                    grp.append(other)
            for g in grp: used.add(g)
            groups.append(grp)
        self.groups_ = groups
        if y is None:
            self.keep_ = [grp[0] for grp in groups]
        else:
            y = pd.Series(y).values
            keep = []
            for grp in groups:
                if len(grp)==1:
                    keep.append(grp[0])
                else:
                    sub = pd.Series({c: abs(np.corrcoef(Xc[c].values, y)[0,1]) for c in grp})
                    keep.append(sub.sort_values(ascending=False).index[0])
            self.keep_ = keep
        return self

    def transform(self, X):
        Xc = pd.DataFrame(X).copy()
        return Xc[self.keep_].values

# ========== Model zoo & params ==========
def model_zoo(task: str):
    use_gpu = (args.gpu == 'gpu')
    if task == "regression":
        base = {
            "Linear": LinearRegression(),
            "RidgeCV": RidgeCV(alphas=np.logspace(-3, 3, 13)),
            "LassoCV": LassoCV(cv=5, max_iter=8000, random_state=42),
            "ElasticNetCV": ElasticNetCV(cv=5, l1_ratio=[.1,.3,.5,.7,.9,.95,1.0], random_state=42, max_iter=8000),
            "SVR_RBF": SVR(C=10.0, epsilon=0.2, kernel="rbf"),
            "KNN": KNeighborsRegressor(n_neighbors=10),
            "RandomForest": RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=600, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
        }
        if xgb:
            base["XGBoost"] = XGBRegressor(
                n_estimators=800, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                learning_rate=0.05, tree_method=("gpu_hist" if use_gpu else "hist"),
                predictor=("gpu_predictor" if use_gpu else None),
                random_state=42, n_jobs=-1
            )
        if lgb:
            base["LightGBM"] = LGBMRegressor(
                n_estimators=1200, num_leaves=96, subsample=0.8, colsample_bytree=0.8,
                learning_rate=0.05, random_state=42, n_jobs=-1,
                min_data_in_leaf=20,  # 减小以减少“无法继续分裂”的概率
                min_gain_to_split=0.0,  # 默认即可
                device=("gpu" if use_gpu else "cpu")
            )
        return base
    else:
        base = {
            "LogReg": LogisticRegression(max_iter=2000, n_jobs=None),
            "LinearSVC": LinearSVC(),
            "SVC_RBF": SVC(C=10.0, kernel="rbf", probability=True),
            "KNN": KNeighborsClassifier(n_neighbors=10),
            "RandomForest": RandomForestClassifier(n_estimators=600, random_state=42, n_jobs=-1),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=600, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
        }
        if xgb:
            base["XGBoost"] = XGBClassifier(
                n_estimators=800, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                learning_rate=0.05, random_state=42, n_jobs=-1,
                tree_method=("gpu_hist" if use_gpu else "hist"), predictor=("gpu_predictor" if use_gpu else None),
                eval_metric="logloss"
            )
        if lgb:
            base["LightGBM"] = LGBMClassifier(
                n_estimators=1200, num_leaves=96, subsample=0.8, colsample_bytree=0.8,
                learning_rate=0.05, random_state=42, n_jobs=-1,
                min_data_in_leaf=20, min_gain_to_split=0.0,
                device=("gpu" if use_gpu else "cpu")
            )
        return base

from scipy.stats import randint as sp_randint, uniform as sp_uniform
def param_spaces(task: str):
    if task == "regression":
        return {
            "RandomForest": {"n_estimators": sp_randint(300,1200),"max_depth": sp_randint(3,20),"min_samples_split": sp_randint(2,20),"min_samples_leaf": sp_randint(1,15),"max_features": ["sqrt","log2",None]},
            "ExtraTrees": {"n_estimators": sp_randint(300,1200),"max_depth": sp_randint(3,20),"min_samples_split": sp_randint(2,20),"min_samples_leaf": sp_randint(1,15),"max_features": ["sqrt","log2",None]},
            "GradientBoosting": {"n_estimators": sp_randint(200,1200),"learning_rate": sp_uniform(0.01,0.19),"max_depth": sp_randint(2,6),"subsample": sp_uniform(0.6,0.4),"min_samples_leaf": sp_randint(1,20)},
            "SVR_RBF": {"C": sp_uniform(0.1,100.0),"epsilon": sp_uniform(0.01,0.5),"gamma": ["scale","auto"]},
            "KNN": {"n_neighbors": sp_randint(3,50), "weights": ["uniform","distance"], "p": [1,2]},
            "RidgeCV": {"alphas": [10**e for e in range(-3,4)]},
            "LassoCV": {"alphas": sp_randint(50,300), "eps": sp_uniform(1e-4, 1e-1)},
            "ElasticNetCV": {"alphas": sp_randint(50,300), "eps": sp_uniform(1e-4, 1e-2), "l1_ratio": sp_uniform(0.05,0.95)},
            "XGBoost": {"n_estimators": sp_randint(300,1200),"max_depth": sp_randint(3,12),"learning_rate": sp_uniform(0.01,0.19),"subsample": sp_uniform(0.6,0.4),"colsample_bytree": sp_uniform(0.6,0.4),"min_child_weight": sp_randint(1,10),"reg_alpha": sp_uniform(0.0,1.0),"reg_lambda": sp_uniform(0.0,1.0)},
            "LightGBM": {"n_estimators": sp_randint(500,1500),"num_leaves": sp_randint(31,255),"max_depth": [-1]+list(range(3,16)),"learning_rate": sp_uniform(0.01,0.19),"min_child_samples": sp_randint(5,50),"subsample": sp_uniform(0.6,0.4),"colsample_bytree": sp_uniform(0.6,0.4),"reg_alpha": sp_uniform(0.0,1.0),"reg_lambda": sp_uniform(0.0,1.0),
                         "min_data_in_leaf": sp_randint(10,80)}
        }
    else:
        return {
            "LogReg": {"C": sp_uniform(0.01,10.0),"penalty": ["l2"],"solver": ["lbfgs","liblinear","saga"]},
            "LinearSVC": {"C": sp_uniform(0.01,10.0)},
            "SVC_RBF": {"C": sp_uniform(0.1,100.0),"gamma": ["scale","auto"]},
            "KNN": {"n_neighbors": sp_randint(3,50), "weights": ["uniform","distance"], "p": [1,2]},
            "RandomForest": {"n_estimators": sp_randint(300,1200),"max_depth": sp_randint(3,20),"min_samples_split": sp_randint(2,20),"min_samples_leaf": sp_randint(1,15),"max_features": ["sqrt","log2",None]},
            "ExtraTrees": {"n_estimators": sp_randint(300,1200),"max_depth": sp_randint(3,20),"min_samples_split": sp_randint(2,20),"min_samples_leaf": sp_randint(1,15),"max_features": ["sqrt","log2",None]},
            "GradientBoosting": {"n_estimators": sp_randint(200,1200),"learning_rate": sp_uniform(0.01,0.19),"max_depth": sp_randint(2,6),"subsample": sp_uniform(0.6,0.4),"min_samples_leaf": sp_randint(1,20)},
            "XGBoost": {"n_estimators": sp_randint(300,1200),"max_depth": sp_randint(3,12),"learning_rate": sp_uniform(0.01,0.19),"subsample": sp_uniform(0.6,0.4),"colsample_bytree": sp_uniform(0.6,0.4),"min_child_weight": sp_randint(1,10),"reg_alpha": sp_uniform(0.0,1.0),"reg_lambda": sp_uniform(0.0,1.0)},
            "LightGBM": {"n_estimators": sp_randint(500,1500),"num_leaves": sp_randint(31,255),"max_depth": [-1]+list(range(3,16)),"learning_rate": sp_uniform(0.01,0.19),"min_child_samples": sp_randint(5,50),"subsample": sp_uniform(0.6,0.4),"colsample_bytree": sp_uniform(0.6,0.4),"reg_alpha": sp_uniform(0.0,1.0),"reg_lambda": sp_uniform(0.0,1.0),
                         "min_data_in_leaf": sp_randint(10,80)}
        }

def randomized_tune(name, est, X, y, n_iter=60, n_jobs=-1, random_state=42, scoring="r2", cv=5):
    space = param_spaces(args.task).get(name)
    if not space:
        return est, {}
    rs = RandomizedSearchCV(est, param_distributions=space, n_iter=n_iter, scoring=scoring, cv=cv, n_jobs=n_jobs,
                            random_state=random_state, verbose=0, refit=True)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_

# ========== Safer benchmarking with Pipeline (no leakage) ==========
def build_pipeline(name: str, base_estimator, task: str, select_k: int, collinear_r: float):
    needs_scale = any(k in name for k in ["SVR","KNN","Lasso","Elastic","Ridge","Linear","SVC","LogReg","LinearSVC"])
    steps = []
    steps.append(("var", VarianceThreshold(0.0)))
    if task == "regression":
        steps.append(("kbest", SelectKBest(mutual_info_regression, k=min(select_k,  max(1, select_k)))))
    else:
        steps.append(("kbest", SelectKBest(mutual_info_classif, k=min(select_k, max(1, select_k)))))
    steps.append(("collinear", CorrelationFilter(threshold=collinear_r)))
    if needs_scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", base_estimator))
    return Pipeline(steps)

def benchmark_models_safe(task: str, X: pd.DataFrame, y: pd.Series, outdir: str, cv: int, random_state: int,
                          do_tune: bool, n_iter: int, n_jobs: int, stratify_regression: bool, select_k: int, collinear_r: float):
    safe_makedirs(outdir)
    if task == "regression" and stratify_regression:
        skf, bins = stratified_kfold_for_regression(y, n_splits=cv, random_state=random_state, n_bins=10)
        cvsplit = list(skf.split(X, bins))
    else:
        cvsplit = KFold(n_splits=cv, shuffle=True, random_state=random_state).split(X, y) if task=="regression" \
                  else StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state).split(X, y)

    models = model_zoo(task)
    rows = []; best_param_dump = {}

    for name, est in models.items():
        if est is None: 
            continue
        pipe = build_pipeline(name, est, task, select_k, collinear_r)
        tuned = pipe; best_params = {}

        if do_tune:
            # 只对“model”这一步调参，前置步骤避免调参复杂度爆炸
            space = param_spaces(task).get(name, None)
            if space:
                # 包装空间键
                space2 = {f"model__{k}": v for k, v in space.items()}
                # 防御式过滤：只保留当前 Pipeline 可识别的参数，避免版本差异导致 Invalid parameter
                valid_keys = set(pipe.get_params().keys())
                space2 = {k: v for k, v in space2.items() if k in valid_keys}
                if not space2:
                    tuned = pipe  # 无可调参数则跳过调参
                else:
                    rs = RandomizedSearchCV(pipe, param_distributions=space2, n_iter=n_iter,
                                            scoring=("r2" if task=="regression" else "roc_auc"),
                                            cv=cv, n_jobs=n_jobs, random_state=random_state, verbose=0, refit=True)
                    rs.fit(X, y)
                    tuned = rs.best_estimator_
                    best_params = {k.replace("model__", ""): v for k, v in rs.best_params_.items() if k.startswith("model__")}

        # CV 评分（无需额外缩放，皆由 Pipeline 处理）
        if task == "regression":
            r2 = cross_val_score(tuned, X, y, cv=list(cvsplit), scoring="r2", n_jobs=n_jobs)
            rmse_scores = -cross_val_score(tuned, X, y, cv=list(cvsplit), scoring="neg_root_mean_squared_error", n_jobs=n_jobs)
            mae_scores = -cross_val_score(tuned, X, y, cv=list(cvsplit), scoring="neg_mean_absolute_error", n_jobs=n_jobs)
            rows.append({"model": name + ("+tuned" if do_tune else ""), "r2_mean": float(np.mean(r2)), "r2_std": float(np.std(r2)),
                         "rmse_mean": float(np.mean(rmse_scores)), "rmse_std": float(np.std(rmse_scores)),
                         "mae_mean": float(np.mean(mae_scores)), "mae_std": float(np.std(mae_scores))})
        else:
            try:
                auc = cross_val_score(tuned, X, y, cv=list(cvsplit), scoring=("roc_auc" if len(np.unique(y))==2 else "roc_auc_ovr"), n_jobs=n_jobs)
            except Exception:
                auc = np.array([np.nan]*cv)
            acc = cross_val_score(tuned, X, y, cv=list(cvsplit), scoring="accuracy", n_jobs=n_jobs)
            f1 = cross_val_score(tuned, X, y, cv=list(cvsplit), scoring=("f1" if len(np.unique(y))==2 else "f1_weighted"), n_jobs=n_jobs)
            rows.append({"model": name + ("+tuned" if do_tune else ""), "roc_auc_mean": float(np.nanmean(auc)), "roc_auc_std": float(np.nanstd(auc)),
                         "acc_mean": float(np.mean(acc)), "acc_std": float(np.std(acc)),
                         "f1_mean": float(np.mean(f1)), "f1_std": float(np.mean(f1))})

        if best_params:
            best_param_dump[name] = best_params

    res = pd.DataFrame(rows).sort_values(("r2_mean" if task=="regression" else "acc_mean"), ascending=False)
    res.to_csv(os.path.join(outdir, "model_comparison_safe.csv"), index=False)
    if best_param_dump:
        with open(os.path.join(outdir, "best_params_safe.json"), "w") as f:
            json.dump(best_param_dump, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(10,6))
    if task == "regression":
        plt.barh(res["model"][::-1], res["r2_mean"][::-1]); plt.title("Model comparison (CV R2 mean, leakage-safe)")
    else:
        plt.barh(res["model"][::-1], res["acc_mean"][::-1]); plt.title("Model comparison (CV accuracy mean, leakage-safe)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"model_score_bar_safe.png"), dpi=180); plt.close()
    return res

# ========== Final fit with early stopping and SHAP ==========
def fit_best_and_explain(task: str, X: pd.DataFrame, y: pd.Series, cmp_df: pd.DataFrame, outdir: str,
                         random_state: int = 42, do_shap=False, early_stopping_rounds: int = 200, valid_size: float = 0.2):
    safe_makedirs(outdir)
    best_name = cmp_df.iloc[0]["model"]
    base_name = best_name.replace("+tuned","")
    models = model_zoo(task)
    best = models[base_name]

    # holdout split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=valid_size, random_state=random_state,
                                          stratify=(y if task=="classification" else None))

    needs_scale = any(k in base_name for k in ["SVR","KNN","Lasso","Elastic","Ridge","Linear","SVC","LogReg","LinearSVC"])
    scaler = StandardScaler() if needs_scale else None
    if scaler is not None:
        Xtr_use = scaler.fit_transform(Xtr)
        Xte_use = scaler.transform(Xte)
    else:
        Xtr_use, Xte_use = Xtr.values, Xte.values

    # Early stopping for LGB/XGB
    fit_params = {}
    if base_name in ["LightGBM","XGBoost"]:
        if base_name == "LightGBM":
            fit_params = {"eval_set": [(Xte_use, yte)], "eval_metric": ("rmse" if task=="regression" else "logloss"),
                          "callbacks": [lgb_mod.early_stopping(early_stopping_rounds, verbose=False)]}
        elif base_name == "XGBoost":
            fit_params = {"eval_set": [(Xte_use, yte)], "eval_metric": ("rmse" if task=="regression" else "logloss"),
                          "early_stopping_rounds": early_stopping_rounds, "verbose": False}

    best.fit(Xtr_use, ytr, **fit_params)
    pred = best.predict(Xte_use)

    metrics = {}
    if task == "regression":
        metrics = {"r2": r2_score(yte, pred), "rmse": rmse(yte, pred), "mae": mae(yte, pred)}
        with open(os.path.join(outdir,"best_model_holdout_metrics.json"), "w") as f: json.dump({"best_model": best_name, **metrics}, f, indent=2)
        plt.figure(figsize=(6,6)); plt.scatter(yte, pred, s=12); lims=[min(yte.min(), pred.min()), max(yte.max(), pred.max())]; plt.plot(lims, lims); plt.title(f"{best_name}: True vs Pred"); plt.tight_layout(); plt.savefig(os.path.join(outdir,"best_true_vs_pred.png"), dpi=180); plt.close()
    else:
        try:
            proba = best.predict_proba(Xte_use)
            if proba.shape[1]==2: auc = roc_auc_score(yte, proba[:,1])
            else: auc = roc_auc_score(yte, proba, multi_class="ovr")
        except Exception:
            auc = np.nan
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average=("binary" if len(np.unique(y))==2 else "weighted"))
        metrics = {"roc_auc": float(auc), "acc": float(acc), "f1": float(f1)}
        with open(os.path.join(outdir,"best_model_holdout_metrics.json"), "w") as f: json.dump({"best_model": best_name, **metrics}, f, indent=2)
        cm = confusion_matrix(yte, pred); plt.figure(figsize=(5,5)); plt.imshow(cm); plt.title(f"{best_name}: Confusion matrix"); plt.colorbar(); plt.tight_layout(); plt.savefig(os.path.join(outdir,"best_confusion_matrix.png"), dpi=180); plt.close()

    # Permutation importance on holdout
    try:
        pi = permutation_importance(best, Xte_use, yte, n_repeats=20, random_state=random_state, n_jobs=-1)
        pi_s = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
        pi_s.to_csv(os.path.join(outdir,"best_permutation_importance.csv"))
        top = pi_s.head(30); plt.figure(figsize=(8,10)); plt.barh(top.index[::-1], top.values[::-1]); plt.title("Best model — permutation importance (top-30)")
        plt.tight_layout(); plt.savefig(os.path.join(outdir,"best_permutation_top30.png"), dpi=180); plt.close()
    except Exception:
        pass

    # SHAP (tree-based)
    if do_shap and shap is not None and any(s in base_name for s in ["Forest","Trees","Boost","XGBoost","LightGBM"]):
        try:
            explainer = shap.TreeExplainer(best); shap_values = explainer.shap_values(Xte_use)
            shap.summary_plot(shap_values, Xte_use, feature_names=X.columns, show=False); plt.tight_layout(); plt.savefig(os.path.join(outdir,"shap_summary.png"), dpi=180); plt.close()
        except Exception:
            pass

    return best_name, metrics

# ========== Main ==========
def main():
    global args
    args = parse_args()
    outdir = args.outdir; safe_makedirs(outdir)

    df, X, y, info = load_and_clean(args.csv, args.target, args.id_prefix, args.drop_missing_gt, args.sample_rows)
    summary = {"csv": os.path.abspath(args.csv), "task": args.task, "n_rows_total": int(len(df)), "n_rows_used": int(len(X)),
               "n_features_after_clean": int(X.shape[1]), "removed_all_nan": len(info["all_nan"]),
               "removed_constant": len(info["const_cols"]), "removed_high_missing": len(info["high_missing"]),
               "meta_cols": info["meta_cols"], "args": vars(args)}
    with open(os.path.join(outdir,"summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 只保留必要的轻量 EDA，避免重复大型筛选的泄露
    eda(args.task, X, y, os.path.join(outdir,"eda"), args)

    # 泄露安全的基准对比（特征选择在 CV 内执行）
    cmp_df = benchmark_models_safe(
        args.task, X, y, os.path.join(outdir,"benchmark_safe"),
        cv=args.cv, random_state=args.random_state, do_tune=args.tune, n_iter=args.tune_iter, n_jobs=args.tune_jobs,
        stratify_regression=args.cv_stratify_regression, select_k=args.select_k, collinear_r=args.collinear_r
    )

    # 最佳模型 + 提前停止 + 解释
    best_name, holdout = fit_best_and_explain(
        args.task, X, y, cmp_df, os.path.join(outdir,"explain"),
        random_state=args.random_state, do_shap=args.run_shap,
        early_stopping_rounds=args.early_stopping_rounds, valid_size=args.valid_size
    )

    print(json.dumps({"summary": summary, "best_model": best_name, "best_holdout": holdout,
                      "report_dir": os.path.abspath(outdir)}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
