#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Optimizer
=================
从 extra_features_max.csv 中自动完成“特征清洗 + 去冗余 + 模型择优”的流水线，
输出精简后的特征子集、精简后的数据集、筛除原因表，以及若干可视化。

主要步骤：
1) 读入数据，锁定数值特征（排除 id_cols 与 target）。
2) 清洗：
   - 删除零方差与“近零方差”特征（频率占比阈值）。
   - 删除高缺失率特征（阈值可配）。
   - 中位数插补（仅用于相关性与训练阶段，不改动原 CSV）。
3) 去冗余：
   - Spearman 相关性逐步剔除（阈值可配，保留更“有信息”的一列）。
   - 可选 VIF（若安装 statsmodels）。
4) 模型择优：
   - 训练 RandomForest 或 XGBoost（若可用），5 折交叉验证聚合重要性；
   - 也可使用 Permutation Importance（可选）；
   - 依据“Top-N”或“累计重要性”两种准则选特征。
5) 导出：
   - selected_features.txt
   - dropped_features.csv（含“删除原因”）
   - X_selected.csv（仅含 id_cols + target + 精简后的特征）
   - 重要性条形图、相关性热图（matplotlib，无自定义颜色）
"""

import argparse, json, socket, time, math, warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# 可选依赖
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import statsmodels.api as sm
except Exception:
    sm = None


# ========== 基础工具 ==========
def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def fig_save(fig, out_no_ext: Path):
    fig.tight_layout()
    fig.savefig(out_no_ext.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_no_ext.with_suffix(".svg"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def pick_numeric_features(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in nums if c not in set(exclude)]

def zero_and_near_zero_variance(df: pd.DataFrame, cols: List[str],
                                near_zero_freq_thresh: float = 0.95) -> Tuple[List[str], List[str]]:
    """返回(零方差, 近零方差)列名列表。近零方差：某一取值频率 >= near_zero_freq_thresh。"""
    zero_var, near_zero = [], []
    for c in cols:
        v = df[c].dropna().values
        if v.size == 0:
            zero_var.append(c)
            continue
        if np.nanstd(v) == 0:
            zero_var.append(c)
        # 近零方差
        vals, cnts = np.unique(v, return_counts=True)
        top_frac = cnts.max() / v.size
        if top_frac >= near_zero_freq_thresh and c not in zero_var:
            near_zero.append(c)
    return zero_var, near_zero

def drop_by_missing_rate(df: pd.DataFrame, cols: List[str], max_missing: float) -> List[str]:
    drop = []
    n = len(df)
    for c in cols:
        miss = df[c].isna().sum() / n
        if miss > max_missing:
            drop.append(c)
    return drop

def greedy_corr_prune(df_imp: pd.DataFrame, cols: List[str], method="spearman",
                      threshold: float = 0.95) -> Tuple[List[str], Dict[str, str]]:
    """
    相关性贪心剔除：绝对相关系数>=threshold 时，仅保留信息量更高的一列。
    信息量指标用标准差（也可换成 MI/方差等）。
    返回 (保留列, {被剔除列: 保留的那列})
    """
    corr = df_imp[cols].corr(method=method).abs()
    remaining = set(cols)
    dropped_map: Dict[str, str] = {}

    # 预先计算每列的“信息量”（std）
    stds = df_imp[cols].std().to_dict()

    # 按照列的标准差从大到小遍历，优先保留“信息量大”的列
    sorted_cols = sorted(cols, key=lambda c: stds.get(c, 0.0), reverse=True)
    for i, c in enumerate(sorted_cols):
        if c not in remaining:
            continue
        # 找到与 c 强相关的其他列
        high_corr_partners = [k for k in remaining if k != c and corr.loc[c, k] >= threshold]
        for k in high_corr_partners:
            remaining.discard(k)
            dropped_map[k] = c
    kept = list(remaining)
    return kept, dropped_map

def compute_vif(df_imp: pd.DataFrame, cols: List[str], vif_thresh: float = 20.0) -> Tuple[List[str], Dict[str, float]]:
    """
    可选 VIF 过滤（需要 statsmodels）。高于阈值的列按 VIF 从高到低剔除。
    返回 (保留列, {列名: VIF})
    """
    if sm is None:
        return cols, {}
    cur = cols[:]
    changed = True
    vif_scores = {}
    while changed and len(cur) > 2:
        changed = False
        X = df_imp[cur].values
        X = np.nan_to_num(X, copy=False)
        X = np.column_stack([np.ones(X.shape[0]), X])  # 加截距
        vifs = []
        for i in range(1, X.shape[1]):  # 跳过截距
            try:
                v = sm.stats.outliers_influence.variance_inflation_factor(X, i)
            except Exception:
                v = np.nan
            vifs.append(v)
        vif_scores = {c: v for c, v in zip(cur, vifs)}
        worst = max(vif_scores.items(), key=lambda kv: (0 if np.isnan(kv[1]) else kv[1]))
        if not np.isnan(worst[1]) and worst[1] > vif_thresh:
            cur.remove(worst[0])
            changed = True
    return cur, vif_scores


# ========== 训练与重要性 ==========
def kfold_model_importance(X: np.ndarray, y: np.ndarray, feat_names: List[str],
                           model_name: str = "rf",
                           use_perm: bool = False,
                           n_splits: int = 5,
                           seed: int = 42) -> pd.DataFrame:
    """
    以 KFold 聚合特征重要性（可选置换重要性）。
    返回列：feature, importance, fold, model
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, va) in enumerate(kf.split(X), 1):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        if model_name == "xgb" and xgb is not None:
            model = xgb.XGBRegressor(
                n_estimators=1500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                tree_method="hist", n_jobs=8, random_state=seed
            )
        else:
            model = RandomForestRegressor(
                n_estimators=600, random_state=seed, n_jobs=8, max_features="auto"
            )
        model.fit(Xtr, ytr)

        if use_perm:
            pi = permutation_importance(model, Xva, yva, n_repeats=10,
                                        random_state=seed, n_jobs=8)
            imp = pi.importances_mean
        else:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
            else:
                # XGB 若失败，fallback 到置换
                pi = permutation_importance(model, Xva, yva, n_repeats=10,
                                            random_state=seed, n_jobs=8)
                imp = pi.importances_mean

        for f, v in zip(feat_names, imp):
            rows.append({"feature": f, "importance": float(v), "fold": fold,
                         "model": "XGBoost" if (model_name == "xgb" and xgb is not None) else "RandomForest"})
    df_imp = pd.DataFrame(rows)
    df_imp = df_imp.groupby(["feature", "model"], as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    return df_imp


# ========== 主流程 ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to features CSV (e.g., extra_features_max.csv)")
    ap.add_argument("--outdir", type=str, default="feature_opt_out")
    ap.add_argument("--target", type=str, default="score")
    ap.add_argument("--id_cols", type=str, nargs="*", default=["name","formula","mpid","cif_file"])
    # 清洗与去冗余阈值
    ap.add_argument("--max-missing", type=float, default=0.50, help="删除缺失率高于此阈值的特征")
    ap.add_argument("--near-zero-freq", type=float, default=0.95, help="近零方差频率阈值")
    ap.add_argument("--corr-threshold", type=float, default=0.95, help="Spearman 相关阈值")
    ap.add_argument("--vif-threshold", type=float, default=0.0, help=">0 启用 VIF 并作为剔除阈值，如 20")
    # 重要性与选择
    ap.add_argument("--model", type=str, choices=["rf","xgb"], default="rf")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--perm", action="store_true", help="使用置换重要性 (较慢)")
    ap.add_argument("--topn", type=int, default=128, help="最终最多保留的特征数")
    ap.add_argument("--cum-imp", type=float, default=0.0, help="累计重要性阈值(0-1)，例如 0.95；0 表示关闭")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # 元信息
    meta = {
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "csv": args.csv,
        "target": args.target,
        "id_cols": args.id_cols,
        "params": vars(args),
    }
    save_json(meta, out/"run_meta.json")

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise SystemExit(f"Target '{args.target}' not in CSV.")

    exclude = list(set(args.id_cols + [args.target]))
    feat_all = pick_numeric_features(df, exclude)
    pd.DataFrame({"feature": feat_all}).to_csv(out/"feature_all_numeric.csv", index=False)

    # -------- 清洗 --------
    drop_missing = drop_by_missing_rate(df, feat_all, args.max_missing)
    keep1 = [c for c in feat_all if c not in drop_missing]

    zero_var, near_zero = zero_and_near_zero_variance(df, keep1, near_zero_freq_thresh=args.near_zero_freq)
    keep2 = [c for c in keep1 if c not in set(zero_var) | set(near_zero)]

    # 插补（下游使用，不修改原 df）
    imp = SimpleImputer(strategy="median")
    X_imp_stage = imp.fit_transform(df[keep2].values)
    X_imp_df = pd.DataFrame(X_imp_stage, columns=keep2)

    # -------- 去冗余（相关）--------
    kept_corr, corr_dropped_map = greedy_corr_prune(X_imp_df, keep2, method="spearman", threshold=args.corr_threshold)

    # -------- VIF（可选）--------
    if args.vif_threshold and args.vif_threshold > 0:
        kept_vif, vif_scores = compute_vif(X_imp_df, kept_corr, vif_thresh=args.vif_threshold)
    else:
        kept_vif, vif_scores = kept_corr, {}

    # 统计删除原因
    drop_records = []
    for c in drop_missing:
        drop_records.append((c, "high_missing"))
    for c in zero_var:
        drop_records.append((c, "zero_variance"))
    for c in near_zero:
        drop_records.append((c, "near_zero_variance"))
    for c, kept in corr_dropped_map.items():
        drop_records.append((c, f"high_corr_with:{kept}"))
    if args.vif_threshold and args.vif_threshold > 0:
        dropped_by_vif = set(kept_corr) - set(kept_vif)
        for c in dropped_by_vif:
            drop_records.append((c, f"high_VIF:{vif_scores.get(c, float('nan')):.3f}"))

    pd.DataFrame(drop_records, columns=["feature","reason"]).to_csv(out/"dropped_features.csv", index=False)

    # -------- 模型重要性 & 最终选择 --------
    y = pd.to_numeric(df[args.target], errors="coerce").values
    y[np.isnan(y)] = np.nanmedian(y)

    # 只用 kept_vif 的列进入模型排序
    cols_for_rank = kept_vif[:]
    X_imp_final = imp.fit_transform(df[cols_for_rank].values)

    # 可选标准化（树模型通常不需要，这里不做）
    imp_df = kfold_model_importance(
        X_imp_final, y, cols_for_rank, model_name=args.model,
        use_perm=args.perm, n_splits=args.cv, seed=args.seed
    )
    imp_df.to_csv(out/"feature_importance_cv.csv", index=False)

    # 依据累计重要性 or Top-N 选特征
    if args.cum_imp and 0 < args.cum_imp <= 1.0:
        s = imp_df["importance"].clip(lower=0)
        s = s / (s.sum() + 1e-12)
        cum = s.cumsum()
        cutoff = cum.searchsorted(args.cum_imp)
        keep_by_imp = imp_df.iloc[:(cutoff+1)]["feature"].tolist()
    else:
        keep_by_imp = imp_df["feature"].head(args.topn).tolist()

    # 最终特征（同时保证出现在 cols_for_rank 中）
    final_feats = [f for f in keep_by_imp if f in cols_for_rank]
    pd.Series(final_feats, name="feature").to_csv(out/"selected_features.txt", index=False)

    # 导出精简后的数据集（保留 id_cols + target + 最终特征）
    cols_export = args.id_cols + [args.target] + final_feats
    df_out = df[cols_export].copy()
    df_out.to_csv(out/"X_selected.csv", index=False)

    # -------- 可视化 --------
    # 1) 重要性条形图
    top_plot = imp_df.head(min(50, len(imp_df)))
    fig, ax = plt.subplots(figsize=(6, max(4, 0.28*len(top_plot))))
    idx = np.arange(len(top_plot))
    ax.barh(idx, top_plot["importance"].values)
    ax.set_yticks(idx); ax.set_yticklabels(top_plot["feature"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Top Features by CV Importance")
    ax.set_xlabel("importance")
    fig_save(fig, out/"importance_top")

    # 2) 最终特征的相关性热图（Spearman）
    if len(final_feats) >= 2:
        c = pd.DataFrame(imp.fit_transform(df[final_feats]), columns=final_feats)
        corr = c.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(max(6, len(final_feats)*0.25), max(6, len(final_feats)*0.25)))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(corr.shape[1])); ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(corr.shape[0])); ax.set_yticklabels(corr.index, fontsize=6)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.set_ylabel("corr", rotation=90)
        ax.set_title("Spearman Corr of Selected Features")
        fig_save(fig, out/"corr_selected")

    print(f"[feature_optimize] Done. Outputs in: {out}")
    

if __name__ == "__main__":
    main()
