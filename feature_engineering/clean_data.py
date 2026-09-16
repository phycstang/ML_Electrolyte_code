#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse

def main(infile: str, outfile: str):
    df = pd.read_csv(infile)
    if df.empty:
        raise ValueError("输入CSV为空")

    # 目标列自动识别
    if "score" in df.columns:
        target = "score"
    elif "id_score" in df.columns:
        target = "id_score"
    else:
        raise ValueError("找不到目标列，请确保存在 'score' 或 'id_score'。")

    # 多ID列支持：这里把 cif_file 和 cif_path 都当成 ID 列（你也可以加别名）
    candidate_ids = ["id", "cif_file", "cif_path"]
    id_cols = [c for c in candidate_ids if c in df.columns]
    if not id_cols:
        raise ValueError("未检测到 ID 列。请确保至少包含 'cif_file' 或 'cif_path'。")

    # 判定“完全重复”的键：所有列 - ID列
    key_cols = [c for c in df.columns if c not in id_cols]

    # —— 1) 完全重复（特征+目标一致）去重（保留组内第一条的各ID列）
    dedup_simple = df.drop_duplicates(subset=key_cols, keep="first").copy()

    # —— 2) 相同特征但目标冲突（同一特征键下 target 多个值）
    feature_only_cols = [c for c in key_cols if c != target]
    if len(feature_only_cols) == 0:
        # 极端情况：除了目标无其他特征
        for c in id_cols:
            dedup_simple[f"ids_merged_{c}"] = dedup_simple[c].astype(str)
        out = dedup_simple
    else:
        grp = df.groupby(feature_only_cols, dropna=False)[target]
        conflict_keys = grp.nunique()
        has_conflict = conflict_keys[conflict_keys > 1].index

        if len(has_conflict) > 0:
            # 标记冲突行
            df["_key_tuple"] = df[feature_only_cols].apply(lambda r: tuple(r.values.tolist()), axis=1)
            conflict_key_tuples = set(tuple(x) if not isinstance(x, tuple) else x for x in has_conflict)
            conflict_mask = df["_key_tuple"].isin(conflict_key_tuples)

            # 聚合冲突组：目标取中位数；每个ID列取第一条；并生成 ids_merged_* 追溯
            agg_dict = {target: "median"}
            for c in id_cols:
                agg_dict[c] = "first"

            agg_conflict = (
                df[conflict_mask]
                .groupby(feature_only_cols, dropna=False)
                .agg(agg_dict)
                .reset_index()
            )

            # 生成 ids_merged_* 列
            ids_merged_frames = []
            for c in id_cols:
                merged = (
                    df[conflict_mask]
                    .groupby(feature_only_cols, dropna=False)[c]
                    .apply(lambda s: "|".join(map(str, pd.unique(s.astype(str)))))
                    .reset_index(name=f"ids_merged_{c}")
                )
                ids_merged_frames.append(merged)

            # 合并所有 ids_merged_* 到聚合结果
            for merged in ids_merged_frames:
                agg_conflict = agg_conflict.merge(merged, on=feature_only_cols, how="left")

            # 非冲突部分：完全去重后保留，并设置 ids_merged_* = 自身ID值
            non_conflict = df[~conflict_mask].drop_duplicates(subset=key_cols, keep="first").copy()
            for c in id_cols:
                non_conflict[f"ids_merged_{c}"] = non_conflict[c].astype(str)

            # 对齐列并合并
            common_cols = list(agg_conflict.columns)
            non_conflict = non_conflict.reindex(columns=common_cols, fill_value=None)

            out = pd.concat([non_conflict, agg_conflict], ignore_index=True)
            out.drop(columns=["_key_tuple"], errors="ignore", inplace=True)
        else:
            # 无冲突：简单去重 + ids_merged_* = 自身ID值
            for c in id_cols:
                dedup_simple[f"ids_merged_{c}"] = dedup_simple[c].astype(str)
            out = dedup_simple

    # 把 ID 与 ids_merged_* 放前面，方便查看
    front = id_cols + [f"ids_merged_{c}" for c in id_cols]
    cols = front + [c for c in out.columns if c not in front]
    out = out[cols]

    out.to_csv(outfile, index=False)

    print(f"原始: {len(df)} | 完全去重后: {len(dedup_simple)} | 最终输出: {len(out)}")
    print(f"已保留 {id_cols}，并新增 {['ids_merged_'+c for c in id_cols]} 用于追溯。输出: {outfile}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=str, default="origin_data.csv")
    ap.add_argument("--outfile", type=str, default="data_clean_dedup.csv")
    args = ap.parse_args()
    main(args.infile, args.outfile)
