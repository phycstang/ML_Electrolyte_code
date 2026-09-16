#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMAP/HDBSCAN 可视化脚本（增强：指定样本红圈高亮）
- 输入：clusters.csv, embedding_umap.csv, prototypes.json(可选), params_used.json
- 新增：
  --highlight "AlCl3_mp-25469.cif,TaCl5_mp-29831,ZrCl4_mp-569175,HfCl4_mp-29422,ZnCl2_mp-22889"
  --highlight-filelist /path/to/list.txt   # 文本文件每行一个条目
- 匹配规则：对 df['file'] 同时支持
    1) 完整文件名（含或不含 .cif）
    2) stem（去掉扩展名）
    3) 子串（如 mp-29831）
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import matplotlib as mpl
from matplotlib.colors import Normalize

def load_data(outdir: Path):
    df = pd.read_csv(outdir / "clusters.csv")
    emb = pd.read_csv(outdir / "embedding_umap.csv")
    with open(outdir / "params_used.json", "r") as f:
        params = json.load(f)
    protos = {}
    pfile = outdir / "prototypes.json"
    if pfile.exists():
        with open(pfile, "r") as f:
            protos = json.load(f)

    if len(df) != len(emb):
        raise RuntimeError(f"Row mismatch: clusters.csv({len(df)}) vs embedding_umap.csv({len(emb)})")
    df = df.reset_index(drop=True).copy()
    emb = emb.reset_index(drop=True).copy()
    df["umap_0"] = emb.iloc[:, 0].values
    df["umap_1"] = emb.iloc[:, 1].values
    df["row_id"] = np.arange(len(df))

    # 预生成便于匹配的列
    df["file_norm"] = df["file"].astype(str)
    df["stem_norm"] = df["file_norm"].str.replace(r"\.cif$", "", regex=True)
    df["lower_file"] = df["file_norm"].str.lower()
    df["lower_stem"] = df["stem_norm"].str.lower()
    return df, protos, params

def make_cluster_palette(labels: np.ndarray, cmap_name="tab20"):
    """
    为整数簇构造稳定调色板；-1 噪声固定为浅灰。
    兼容 Matplotlib 3.7+（使用 mpl.colormaps.get_cmap），
    同时对旧版提供降级方案。
    """
    uniq = np.unique(labels)
    uniq_pos = [u for u in uniq if u >= 0]
    n = max(len(uniq_pos), 1)

    # Matplotlib 3.6+ 推荐接口
    try:
        cmap = mpl.colormaps.get_cmap(cmap_name).resampled(n)
    except Exception:
        # 旧版降级：使用 pyplot.get_cmap 并离散采样
        from matplotlib.cm import get_cmap
        base = get_cmap(cmap_name)
        xs = np.linspace(0, 1, n)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(base(xs))

    color_map = {lab: cmap(i) for i, lab in enumerate(sorted(uniq_pos))}
    color_map[-1] = (0.75, 0.75, 0.75, 0.8)  # 噪声
    return color_map

def scatter_by_cluster(ax, df, color_map, s=10, alpha=0.95):
    for lab, sub in df.groupby("cluster_id"):
        c = color_map.get(lab, (0.5,0.5,0.5,0.8))
        ax.scatter(sub["umap_0"], sub["umap_1"], s=s, c=[c], alpha=alpha, edgecolors="none", linewidths=0.2)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP clusters (HDBSCAN)")
    ax.grid(True, alpha=0.2)

def scatter_by_probability(ax, df, base_alpha=0.15, s=10):
    # 背景淡灰
    ax.scatter(df["umap_0"], df["umap_1"], s=s, c=[(0,0,0,base_alpha)], edgecolors="none")
    m = df["cluster_id"] >= 0
    sub = df.loc[m]
    norm = Normalize(vmin=0.0, vmax=1.0)
    sc = ax.scatter(sub["umap_0"], sub["umap_1"], s=s, c=sub["soft_prob"], cmap="viridis", norm=norm, edgecolors="none")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("cluster membership probability")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("UMAP colored by soft probability")
    ax.grid(True, alpha=0.2)

def annotate_centroids(ax, df, fontsize=9, box=True):
    for lab, sub in df.groupby("cluster_id"):
        if lab < 0:
            continue
        x, y = sub["umap_0"].mean(), sub["umap_1"].mean()
        txt = f"{int(lab)} (n={len(sub)})"
        kw = dict(ha="center", va="center", fontsize=fontsize, color="k")
        if box:
            kw["bbox"] = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
        ax.text(x, y, txt, **kw)

def mark_prototypes(ax, df, prototypes: dict, size=80, edgecolor="k"):
    for lab_str, idx_list in prototypes.items():
        if not idx_list: 
            continue
        sub = df[df["row_id"].isin(idx_list)]
        if sub.empty: 
            continue
        ax.scatter(sub["umap_0"], sub["umap_1"], s=size, facecolors="none",
                   edgecolors=edgecolor, linewidths=1.0, zorder=6)

# ===== 新增：高亮匹配与绘制 =====

def parse_highlight_args(args) -> list[str]:
    items = []
    if args.highlight:
        for part in args.highlight.split(","):
            t = part.strip()
            if t:
                items.append(t)
    if args.highlight_filelist:
        p = Path(args.highlight_filelist)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                t = line.strip()
                if t and not t.startswith("#"):
                    items.append(t)
    # 去重，统一小写匹配键
    # 保留原始（用于报告），同时构造 lower 版本
    return items

def match_highlights(df: pd.DataFrame, queries: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """返回匹配到的子表，以及未匹配列表"""
    if not queries:
        return df.iloc[0:0], []
    q_norm = [q.strip() for q in queries if q.strip()]
    q_lower = [q.lower().removesuffix(".cif") for q in q_norm]
    hits_mask = np.zeros(len(df), dtype=bool)
    for q in q_lower:
        # 1) 完整文件名匹配（含/不含 .cif）
        hits_mask |= (df["lower_file"] == q) | (df["lower_file"] == q + ".cif")
        # 2) stem 精确匹配
        hits_mask |= (df["lower_stem"] == q)
        # 3) 子串（例如 mp-29831）
        hits_mask |= df["lower_file"].str.contains(q, regex=False)
    found = df.loc[hits_mask].copy()
    # 反查未匹配
    found_lowers = set(found["lower_file"].tolist()) | set(found["lower_stem"].tolist())
    not_found = []
    for q in q_lower:
        matched_any = False
        # 只要出现在任一行的 file/stem/子串，就视为匹配
        if (q in found["lower_file"].values) or (q in found["lower_stem"].values) or any(q in s for s in found["lower_file"].values):
            matched_any = True
        if not matched_any:
            not_found.append(q)
    return found, not_found

def draw_highlights(ax, sub: pd.DataFrame, edge="red", lw=1.5, size=120, annotate=True):
    if sub.empty:
        return
    ax.scatter(sub["umap_0"], sub["umap_1"], s=size, facecolors="none",
               edgecolors=edge, linewidths=lw, zorder=7)
    if annotate:
        for _, r in sub.iterrows():
            label = r["stem_norm"]
            ax.text(r["umap_0"], r["umap_1"], label, fontsize=8, color="red",
                    ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6),
                    zorder=8)

# ===== 摘要导出 =====

def save_cluster_summary(df: pd.DataFrame, outdir: Path):
    rows = []
    for lab, sub in df.groupby("cluster_id"):
        rows.append({
            "cluster_id": int(lab),
            "size": len(sub),
            "noise": int(lab < 0),
            "prob_mean": sub["soft_prob"].mean(),
            "prob_median": sub["soft_prob"].median(),
            "prob_q10": sub["soft_prob"].quantile(0.10),
            "prob_q90": sub["soft_prob"].quantile(0.90),
        })
    pd.DataFrame(rows).sort_values(["noise","size","cluster_id"]).to_csv(outdir / "cluster_summary.csv", index=False)

def maybe_plotly(df: pd.DataFrame, outpath: Path, highlights: pd.DataFrame | None = None):
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception:
        print("[INFO] plotly 未安装，跳过交互图。pip install plotly 可启用。")
        return
    fig = px.scatter(
        df, x="umap_0", y="umap_1",
        color=df["cluster_id"].astype(str),
        hover_data=["file","composition","cluster_id","soft_prob","is_noise"],
        title="UMAP clusters (interactive)",
        opacity=0.9
    )
    # 高亮图层（以空心 marker 表示）
    if highlights is not None and len(highlights) > 0:
        fig.add_trace(go.Scatter(
            x=highlights["umap_0"], y=highlights["umap_1"],
            mode="markers+text",
            marker=dict(size=14, symbol="circle-open", line=dict(width=2)),
            text=highlights["stem_norm"],
            textposition="top left",
            name="highlights"
        ))
    fig.write_html(str(outpath))
    print(f"[OK] 交互图写入：{outpath}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, required=True, help="无监督管线的输出目录")
    p.add_argument("--figsize", type=float, nargs=2, default=[9, 7], help="图尺寸 (W H)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--format", type=str, default="png", choices=["png","pdf","svg"])
    p.add_argument("--annotate-centroid", action="store_true", help="标注每簇质心与规模")
    p.add_argument("--mark-prototypes", action="store_true", help="圈出 prototypes.json 中的原型样本")
    p.add_argument("--plotly", action="store_true", help="额外导出交互版 HTML（需要 plotly）")

    # 新增：高亮选项
    p.add_argument("--highlight", type=str, default="", 
                   help="逗号分隔的文件名/stem/mp-id 片段。如：AlCl3_mp-25469.cif,TaCl5_mp-29831,...")
    p.add_argument("--highlight-filelist", type=str, default="", 
                   help="TXT 文件路径，每行一个条目（可注释行以 # 开头）")
    p.add_argument("--no-label-highlights", action="store_true", help="只画红圈，不标注文字")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df, prototypes, params = load_data(outdir)

    # 解析与匹配高亮
    queries = parse_highlight_args(args)
    hl_df, not_found = match_highlights(df, queries)
    if queries:
        print(f"[INFO] 请求高亮 {len(queries)} 项；匹配到 {len(hl_df)} 项。")
        if not_found:
            print("[WARN] 未匹配到的条目（请检查拼写/是否存在于 clusters.csv 的 file 列）：")
            for q in not_found:
                print("  -", q)

    # ===== 图 1：簇着色 =====
    fig, ax = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)
    color_map = make_cluster_palette(df["cluster_id"].values, cmap_name="tab20")
    scatter_by_cluster(ax, df, color_map, s=10, alpha=0.95)
    if args.annotate_centroid:
        annotate_centroids(ax, df)
    if args.mark_prototypes and len(prototypes) > 0:
        mark_prototypes(ax, df, prototypes, size=80, edgecolor="k")
    # 红圈高亮
    if len(hl_df) > 0:
        draw_highlights(ax, hl_df, edge="red", lw=1.6, size=140, annotate=not args.no_label_highlights)
    fig.tight_layout()
    f1 = outdir / f"umap_clusters.{args.format}"
    fig.savefig(f1); plt.close(fig)
    print(f"[OK] 写入：{f1}")

    # ===== 图 2：soft 概率着色 =====
    fig2, ax2 = plt.subplots(figsize=tuple(args.figsize), dpi=args.dpi)
    scatter_by_probability(ax2, df, base_alpha=0.15, s=10)
    if args.annotate_centroid:
        annotate_centroids(ax2, df)
    if len(hl_df) > 0:
        draw_highlights(ax2, hl_df, edge="red", lw=1.6, size=140, annotate=not args.no_label_highlights)
    fig2.tight_layout()
    f2 = outdir / f"umap_clusters_by_prob.{args.format}"
    fig2.savefig(f2); plt.close(fig2)
    print(f"[OK] 写入：{f2}")

    # ===== 簇摘要 =====
    save_cluster_summary(df, outdir)
    print(f"[OK] 写入：{outdir/'cluster_summary.csv'}")

    # ===== 交互版（可选）=====
    if args.plotly:
        maybe_plotly(df, outdir / "umap_clusters_interactive.html", highlights=hl_df)

if __name__ == "__main__":
    main()
