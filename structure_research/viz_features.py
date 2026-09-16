#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_features.py

This script reads feature-analysis outputs produced by the study pipeline—
such as feature rankings, group contributions, cross‑validation metrics,
stability‑selection frequencies, variance‑inflation factors (VIF) and
missing‑rate reports—and generates a set of diagnostic plots along
with a simple HTML report summarising the results. It can also
generate a correlation heatmap for the most important features if
provided with the full feature matrix.

Inputs (all expected in the same directory):

* ``feature_rank.csv`` – Contains at least two columns, ``Unnamed: 0``
  holding feature names and ``fused_rank`` holding their combined
  importance across models. Produced by study_features.py.
* ``group_rank.csv`` – Columns: ``group`` and ``score``. Aggregated
  importance of feature groups.
* ``cv_metrics.csv`` – Columns: ``model``, ``R2`` and ``MAE``. Cross
  validation metrics for each estimator.
* ``stability_select_freq.csv`` – Columns: ``feature`` and
  ``stability_freq``. The fraction of bootstraps in which each
  candidate feature appears among the top K predictors.
* ``vif_scores.csv`` – Columns: ``feature`` and ``vif``. Estimated
  variance‑inflation factors for the remaining features after
  correlation and VIF pruning.
* ``missing_rate.csv`` – Columns: ``feature`` and ``missing_rate``. The
  proportion of missing values per feature (post processing).
* ``selected_features.txt`` – A list of final candidate features,
  each on its own line.

Optional:

* ``extra_features_csv`` – If provided (via ``--extra_features_csv``),
  the script will compute a correlation heatmap for the top few
  features in ``feature_rank.csv`` to visualise redundancies and
  relationships.

Outputs:

* A set of PNG plots stored in the specified output directory (``--out_dir``)
  showing top features, group contributions, cross‑validation metrics,
  stability frequency, VIF distribution and top VIF values, and
  missing rate distribution.
* An HTML report combining the figures and listing the selected
  features for quick inspection.

Usage example:

    python viz_features.py --in_dir path/to/results --out_dir viz_out \
        --extra_features_csv path/to/extra_features_max.csv

If ``extra_features_csv`` is omitted, the heatmap is skipped.

"""

import os
import argparse
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_bar(df: pd.DataFrame, xcol: str, ycol: str, title: str, outpng: str, topn: int | None = None) -> None:
    """Helper to save a horizontal bar chart of the top entries in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to plot.
    xcol : str
        Column name for the labels (drawn on the y‑axis).
    ycol : str
        Column name for the values (drawn on the x‑axis).
    title : str
        Title for the chart.
    outpng : str
        Output file path for the PNG.
    topn : int | None, optional
        If given, only the top N rows of ``df`` (sorted as provided) are used.
    """
    d = df.copy()
    if topn is not None:
        d = d.head(topn)
    plt.figure(figsize=(10, 5))
    # Reverse order so that the highest value appears at the top
    plt.barh(d[xcol].astype(str)[::-1], d[ycol].values[::-1])
    plt.xlabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=160)
    plt.close()


def save_heatmap_from_table(
    feat_csv: str, feature_rank_csv: str, outpng: str, topn: int = 15
) -> bool:
    """Generate and save a correlation heatmap for the top features.

    Parameters
    ----------
    feat_csv : str
        Path to the full feature matrix in CSV format. Must contain
        numeric columns corresponding to those in ``feature_rank_csv``.
    feature_rank_csv : str
        Path to the feature rank file. Must contain a column
        ``fused_rank`` and a column with the feature names (typically
        ``Unnamed: 0``).
    outpng : str
        Output file path for the heatmap image.
    topn : int, optional
        Number of top features to include in the heatmap. Default is 15.

    Returns
    -------
    bool
        True if the heatmap was successfully generated; False if
        insufficient columns are available or an error occurs.
    """
    try:
        X = pd.read_csv(feat_csv)
        fr = pd.read_csv(feature_rank_csv)
        top = fr.sort_values("fused_rank", ascending=False).head(topn)["Unnamed: 0"].tolist()
        use = [c for c in top if c in X.columns]
        if len(use) < 2:
            return False
        corr = X[use].corr(method="spearman")
        plt.figure(figsize=(8, 6))
        im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        plt.xticks(range(len(use)), use, rotation=90)
        plt.yticks(range(len(use)), use)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"Spearman correlation (Top {len(use)})")
        plt.tight_layout()
        plt.savefig(outpng, dpi=160)
        plt.close()
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualizations and report from feature‑study outputs.")
    parser.add_argument("--in_dir", default=".", help="Directory containing study outputs (CSV files).")
    parser.add_argument("--out_dir", default="viz_out", help="Directory to store plots and report.")
    parser.add_argument(
        "--extra_features_csv",
        default="",
        help="Optional: path to the full feature matrix (e.g. extra_features_max.csv) to draw correlation heatmap.",
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=30,
        help="Number of top items to display in bar charts and the heatmap.",
    )
    args = parser.parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir
    topn = args.topn
    os.makedirs(out_dir, exist_ok=True)

    # Load inputs
    # Feature rank
    fr_path = os.path.join(in_dir, "feature_rank.csv")
    fr = pd.read_csv(fr_path)
    fr = fr.rename(columns={"Unnamed: 0": "feature"})
    fr_sorted = fr.sort_values("fused_rank", ascending=False)
    # Group rank
    gr = pd.read_csv(os.path.join(in_dir, "group_rank.csv"))
    gr = gr.rename(columns={"group": "feature_group"})
    gr_sorted = gr.sort_values("score", ascending=False)
    # CV metrics
    cv = pd.read_csv(os.path.join(in_dir, "cv_metrics.csv"))
    # Stability selection
    sf = pd.read_csv(os.path.join(in_dir, "stability_select_freq.csv"))
    sf.columns = ["feature", "stability_freq"]
    sf_sorted = sf.sort_values("stability_freq", ascending=False)
    # VIF scores
    vf = pd.read_csv(os.path.join(in_dir, "vif_scores.csv"))
    vf.columns = ["feature", "vif"]
    # Missing rate
    mr = pd.read_csv(os.path.join(in_dir, "missing_rate.csv"))
    mr.columns = ["feature", "missing_rate"]
    # Selected features
    with open(os.path.join(in_dir, "selected_features.txt"), "r") as f:
        sel = [line.strip() for line in f if line.strip()]

    # Plot Top‑N features
    save_bar(
        fr_sorted,
        "feature",
        "fused_rank",
        f"Top-{topn} Features (fused rank)",
        os.path.join(out_dir, "top_features.png"),
        topn=topn,
    )
    # Plot group contributions
    save_bar(
        gr_sorted,
        "feature_group",
        "score",
        "Feature Group Contribution",
        os.path.join(out_dir, "group_contrib.png"),
    )
    # Plot CV metrics: overlay R2 and MAE
    plt.figure(figsize=(6, 4))
    x = np.arange(len(cv["model"]))
    plt.bar(x - 0.2, cv["R2"], width=0.4, label="R2")
    plt.bar(x + 0.2, cv["MAE"], width=0.4, label="MAE")
    plt.xticks(x, cv["model"])
    plt.title("Cross‑validation metrics by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cv_metrics.png"), dpi=160)
    plt.close()
    # Plot stability frequency
    save_bar(
        sf_sorted,
        "feature",
        "stability_freq",
        f"Stability Selection Frequency (Top-{topn})",
        os.path.join(out_dir, "stability_freq.png"),
        topn=topn,
    )
    # Plot VIF distribution and top VIF values
    plt.figure(figsize=(6, 4))
    plt.hist(vf["vif"].clip(0, 1e3), bins=40)
    plt.title("VIF distribution (clipped at 1e3)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "vif_hist.png"), dpi=160)
    plt.close()
    top_vif = vf.sort_values("vif", ascending=False).head(topn)
    save_bar(
        top_vif,
        "feature",
        "vif",
        f"Top-{topn} VIF",
        os.path.join(out_dir, "vif_top.png"),
    )
    # Plot missing rate
    save_bar(
        mr.sort_values("missing_rate", ascending=False),
        "feature",
        "missing_rate",
        f"Missing rate (Top-{topn})",
        os.path.join(out_dir, "missing_top.png"),
        topn=topn,
    )
    # Optional heatmap
    heatmap_ok = False
    if args.extra_features_csv:
        heatmap_ok = save_heatmap_from_table(
            args.extra_features_csv,
            fr_path,
            os.path.join(out_dir, "top_corr_heatmap.png"),
            topn=15,
        )
    # Prepare portions of the HTML outside of the f-string to avoid backslash issues
    # Format the list of selected features with indentation
    sel_formatted = textwrap.indent("\n".join(sel), "  ")
    # Prepare optional heatmap HTML snippet
    heatmap_html = (
        "<h3>Top Feature Correlation Heatmap</h3><img src=\"top_corr_heatmap.png\" width=\"900\" />"
        if heatmap_ok
        else ""
    )
    # Build HTML report using the precomputed variables
    report_html = f"""
    <html><head><meta charset=\"utf-8\"><title>Feature Analysis Report</title></head><body>
    <h1>Feature Study Visualization Report</h1>
    <p>This report summarises feature importance and model diagnostics.
    See the accompanying plots for details.</p>
    <h2>Summary of Selected Features</h2>
    <p>{len(sel)} features were selected in the final subset.</p>
    <pre>{sel_formatted}</pre>
    <h2>Plots</h2>
    <h3>Top Features by Fused Rank</h3>
    <img src=\"top_features.png\" width=\"900\" />
    <h3>Feature Group Contribution</h3>
    <img src=\"group_contrib.png\" width=\"900\" />
    <h3>Cross‑validation Metrics by Model</h3>
    <img src=\"cv_metrics.png\" width=\"600\" />
    <h3>Stability Selection Frequency</h3>
    <img src=\"stability_freq.png\" width=\"900\" />
    <h3>VIF Distribution and Top VIF</h3>
    <img src=\"vif_hist.png\" width=\"600\" />
    <img src=\"vif_top.png\" width=\"600\" />
    <h3>Missing Rate (Top {topn})</h3>
    <img src=\"missing_top.png\" width=\"900\" />
    {heatmap_html}
    </body></html>
    """
    # Write the HTML file
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"[OK] Plots and report written to {out_dir}")


if __name__ == "__main__":
    main()