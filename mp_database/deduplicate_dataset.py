#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deduplicate_dataset.py

去重与一致性检查脚本：
- 删除完全重复行
- 按 ID 列去重（默认保留首个；若目标列存在且同 ID 的目标不一致，会输出冲突报告）
- 按数值特征向量去重（近似相等，保留首个）
- 删除完全相同的数值列（保留首个）

用法：
  python deduplicate_dataset.py \
    --input filtered_features.csv \
    --output filtered_features_dedup.csv \
    --id-cols id_cif_path,id_cif \
    --target id_score

输出：
  filtered_features_dedup.csv
  dedup_report_summary.json
  duplicate_rows_index.csv
  duplicate_id_rows.csv
  duplicate_feature_rows.csv
  duplicate_columns_map.csv
  id_conflicts.csv
"""
import argparse, json, os, numpy as np, pandas as pd

def parse_list(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--id-cols", type=str, default="")
    ap.add_argument("--target", type=str, default="")
    ap.add_argument("--round", type=int, default=12, help="数值近似比较的小数位")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    id_cols = [c for c in parse_list(args.id_cols) if c in df.columns]
    target_col = args.target if args.target in df.columns else None
    num_cols = [c for c in df.columns if c not in id_cols + ([target_col] if target_col else []) and pd.api.types.is_numeric_dtype(df[c])]

    # 1) 完全重复行
    dup_all_mask = df.duplicated(keep="first")
    pd.Series(np.where(dup_all_mask)[0]).to_csv("duplicate_rows_index.csv", index=False, header=["row_index"])
    clean = df.drop_duplicates(keep="first")

    # 2) ID 去重与冲突报告
    id_conflicts = pd.DataFrame()
    if id_cols:
        # rows with duplicated IDs (for report)
        key = clean[id_cols].astype(str).agg("||".join, axis=1)
        dup_id_mask = key.duplicated(keep=False)
        clean.loc[dup_id_mask, id_cols + ([target_col] if target_col else [])].to_csv("duplicate_id_rows.csv", index=False)

        if target_col:
            nun = clean.groupby(id_cols, dropna=False)[target_col].nunique().reset_index(name="y_nunique")
            conflicts = nun[nun["y_nunique"] > 1][id_cols]
            if not conflicts.empty:
                id_conflicts = clean.merge(conflicts, on=id_cols, how="inner")
                id_conflicts.to_csv("id_conflicts.csv", index=False)
        clean = clean.drop_duplicates(subset=id_cols, keep="first").reset_index(drop=True)

    # 3) 数值特征向量去重（近似）
    if num_cols:
        key_feat = clean[num_cols].round(args.round).apply(lambda r: tuple(r.values.tolist()), axis=1)
        dup_feat_mask = key_feat.duplicated(keep="first")
        clean.loc[key_feat.duplicated(keep=False), id_cols + num_cols + ([target_col] if target_col else [])].to_csv("duplicate_feature_rows.csv", index=False)
        clean = clean.loc[~dup_feat_mask].reset_index(drop=True)

    # 4) 完全重复的数值列
    dup_cols_map = []
    if len(num_cols) > 1:
        Xsig = clean[num_cols].round(args.round).astype(object)
        sigs = {}
        for c in num_cols:
            sig = tuple(Xsig[c].values.tolist())
            if sig in sigs:
                dup_cols_map.append((c, sigs[sig]))
            else:
                sigs[sig] = c
    if dup_cols_map:
        pd.DataFrame(dup_cols_map, columns=["duplicate_col","kept_col"]).to_csv("duplicate_columns_map.csv", index=False)
        clean = clean.drop(columns=[c for c,_ in dup_cols_map], errors="ignore")

    # 保存结果与摘要
    clean.to_csv(args.output, index=False)
    summary = dict(
        rows_in=int(df.shape[0]), cols_in=int(df.shape[1]),
        rows_out=int(clean.shape[0]), cols_out=int(clean.shape[1]),
        id_cols=id_cols, target_col=target_col,
        dropped_duplicate_columns=[c for c,_ in dup_cols_map],
        id_conflicts_rows=int(id_conflicts.shape[0]) if not id_conflicts.empty else 0
    )
    with open("dedup_report_summary.json","w",encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
