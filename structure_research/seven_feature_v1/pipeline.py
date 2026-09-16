#!/usr/bin/env python3
"""Develop, freeze, and evaluate the independent seven-feature clustering model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    MODEL_FEATURES,
    canonical_formula,
    load_formula_boundary,
    load_json,
    sha256_file,
    stable_json_hash,
    validate_config,
)
from .dataset import (
    add_eligibility,
    add_structure_families,
    select_prototype_anchors,
)
from .model import (
    SevenFeatureTransform,
    fit_hdbscan_ensemble,
    predict_hdbscan_ensemble,
    stable_families,
)
from .retrieval import (
    PrototypeReference,
    aggregate_formula_ranking,
    build_prototype_references,
    build_unique_prototype_portfolio,
    parameter_top_fraction_frequency,
    score_prototype_neighbourhoods,
)


FROZEN_MODEL_NAME = "frozen_model.joblib"
FROZEN_MANIFEST_NAME = "frozen_manifest.json"
PREDEVELOPMENT_COMMITMENT_NAME = "predevelopment_commitment.json"
SOURCE_FILES = (
    "common.py",
    "radii.py",
    "extract.py",
    "dataset.py",
    "model.py",
    "retrieval.py",
    "pipeline.py",
    "validate.py",
)


def _write_json(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    mode = "x" if exclusive else "w"
    with Path(path).open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    hashes = {
        f"src/seven_feature_v1/{name}": sha256_file(directory / name)
        for name in SOURCE_FILES
    }
    hashes["src/discovery/bonding_vesta.py"] = sha256_file(
        directory.parent / "discovery" / "bonding_vesta.py"
    )
    return hashes


def _load_feature_bundle(
    directory: Path, config_path: Path, expected_mode: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features_path = Path(directory) / "seven_features.csv"
    provenance_path = Path(directory) / "provenance.json"
    if not features_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"incomplete seven-feature bundle: {directory}")
    provenance = load_json(provenance_path)
    if provenance.get("isolation_mode") != expected_mode:
        raise ValueError(
            f"feature isolation mode is {provenance.get('isolation_mode')!r}, "
            f"expected {expected_mode!r}"
        )
    if provenance.get("seven_features_sha256") != sha256_file(features_path):
        raise RuntimeError("seven_features.csv differs from extraction provenance")
    if provenance.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("feature bundle and clustering config differ")
    source_hashes = _source_hashes()
    extraction_commitments = {
        "extractor_sha256": source_hashes["src/seven_feature_v1/extract.py"],
        "common_module_sha256": source_hashes["src/seven_feature_v1/common.py"],
        "radii_module_sha256": source_hashes["src/seven_feature_v1/radii.py"],
        "bonding_module_sha256": source_hashes[
            "src/discovery/bonding_vesta.py"
        ],
    }
    drift = {
        key: {"feature_bundle": provenance.get(key), "runtime": value}
        for key, value in extraction_commitments.items()
        if provenance.get(key) != value
    }
    if drift:
        raise RuntimeError(f"feature-extraction source changed: {drift}")
    frame = pd.read_csv(features_path, low_memory=False)
    required = {
        "cif_file",
        "material_id",
        "formula",
        "center_element",
        "halogen_element",
        "feature_status",
        "is_ordered",
        *MODEL_FEATURES,
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"seven-feature bundle lacks columns: {missing}")
    if frame["cif_file"].duplicated().any():
        raise ValueError("seven-feature bundle has duplicate CIF rows")
    frame["formula"] = frame["formula"].map(canonical_formula)
    return frame, provenance


def _prototype_formulas(config: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): canonical_formula(value)
        for key, value in config["positive_prototypes"].items()
    }


def _prepare_analysis(
    source: pd.DataFrame,
    config: dict[str, Any],
    cif_dir: Path,
    radius_variant: str,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audited = add_eligibility(source, config, radius_variant=radius_variant)
    audited = add_structure_families(audited, cif_dir, config, n_jobs=n_jobs)
    analysis = audited.loc[audited["eligible_analysis"].astype(bool)].copy()
    analysis = analysis.sort_values("cif_file", kind="mergesort").reset_index(drop=True)
    if analysis.empty:
        raise RuntimeError("no complete, hard-screen-eligible seven-feature rows")
    return audited, analysis


def _merge_fit_and_predictions(
    fitted_labels: np.ndarray,
    fit_mask: np.ndarray,
    predicted_labels: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(fitted_labels, dtype=int).copy()
    predicted = np.asarray(predicted_labels, dtype=int)
    if labels.shape != predicted.shape:
        raise ValueError("fitted and predicted label matrices differ in shape")
    nonfit = ~np.asarray(fit_mask, dtype=bool)
    labels[nonfit] = predicted[nonfit]
    return labels


def _grade_frequency(values: np.ndarray) -> np.ndarray:
    frequency = np.asarray(values, dtype=float)
    return np.select(
        [frequency >= 0.80, frequency >= 0.50], ["A", "B"], default="C"
    )


def _score_analysis(
    analysis: pd.DataFrame,
    matrix: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    anchors: dict[str, int],
    references: dict[str, PrototypeReference],
    config: dict[str, Any],
    consensus: np.ndarray,
) -> pd.DataFrame:
    scored = score_prototype_neighbourhoods(
        analysis,
        matrix,
        labels,
        references,
        anchors,
        config,
    )
    prototype_formula_set = set(_prototype_formulas(config).values())
    candidate_mask = (
        scored["is_structure_representative"].astype(bool)
        & ~scored["formula"].isin(prototype_formula_set)
    ).to_numpy()
    frequency = parameter_top_fraction_frequency(
        scored,
        labels,
        references,
        anchors,
        candidate_mask,
        config,
    )
    scored["top_fraction_frequency"] = frequency
    scored["robustness_grade"] = _grade_frequency(frequency)

    fit_families = stable_families(
        consensus,
        threshold=float(config["hdbscan"]["consensus_threshold"]),
        min_family_size=2,
    )
    family = np.full(len(scored), -1, dtype=int)
    family[np.asarray(fit_mask, dtype=bool)] = fit_families
    scored["stable_consensus_family"] = family
    eligible_rep = scored["is_structure_representative"].astype(bool)
    scored["rank_eligible_structures"] = np.nan
    scored.loc[eligible_rep, "rank_eligible_structures"] = scored.loc[
        eligible_rep, "seven_feature_score"
    ].rank(method="min", ascending=False)
    for position, name in enumerate(MODEL_FEATURES):
        scored[f"scaled7__{name.removeprefix('feature__')}"] = matrix[:, position]
    return scored


def _labels_frame(analysis: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    result = analysis[["cif_file", "material_id", "formula"]].copy()
    for run in range(labels.shape[1]):
        result[f"run_{run:02d}"] = labels[:, run]
    return result


def _fit_umap(
    matrix: np.ndarray, fit_mask: np.ndarray, config: dict[str, Any]
) -> tuple[Any, np.ndarray]:
    import umap

    visual = config["visualization"]
    if visual.get("umap_only_not_used_for_clustering") is not True:
        raise ValueError("UMAP must be visualization-only")
    fit_values = matrix[np.asarray(fit_mask, dtype=bool)]
    n_neighbors = min(int(visual["umap_n_neighbors"]), max(len(fit_values) - 1, 2))
    model = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(visual["umap_min_dist"]),
        metric=str(visual["umap_metric"]),
        random_state=int(config["random_seed"]),
        transform_seed=int(config["random_seed"]),
        n_jobs=1,
    )
    model.fit(fit_values)
    return model, np.asarray(model.transform(matrix), dtype=float)


def _plot_umap(
    frame: pd.DataFrame,
    embedding: np.ndarray,
    anchors: dict[str, int],
    output: Path,
    *,
    evaluation_mask: np.ndarray | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(10.8, 7.4), dpi=170)
    dimensions = frame["feature__D"].astype(int).to_numpy()
    scatter = axis.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=dimensions,
        cmap="viridis",
        s=16,
        alpha=0.62,
        linewidths=0,
    )
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.01)
    colorbar.set_label("VESTA M–X network dimension D")
    for key, index in anchors.items():
        axis.scatter(
            embedding[int(index), 0],
            embedding[int(index), 1],
            marker="*",
            s=230,
            facecolor="#ffcc33",
            edgecolor="#541f1f",
            linewidth=1.2,
            zorder=6,
        )
        axis.annotate(
            key.split("_", 1)[-1],
            (embedding[int(index), 0], embedding[int(index), 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
        )
    if evaluation_mask is not None and np.any(evaluation_mask):
        selected = np.asarray(evaluation_mask, dtype=bool)
        axis.scatter(
            embedding[selected, 0],
            embedding[selected, 1],
            marker="o",
            s=95,
            facecolor="none",
            edgecolor="#d62728",
            linewidth=1.8,
            label="post-freeze evaluation",
            zorder=5,
        )
        axis.legend(frameon=False, loc="best")
    axis.set_xlabel("UMAP-1 (display only)")
    axis.set_ylabel("UMAP-2 (display only)")
    axis.set_title("Seven-feature space: UMAP is not used for HDBSCAN")
    axis.grid(alpha=0.12)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def _formula_ranking_and_portfolio(
    scored: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prototypes = _prototype_formulas(config)
    formula_ranking = aggregate_formula_ranking(scored, prototypes)
    portfolio = build_unique_prototype_portfolio(
        formula_ranking,
        set(prototypes.values()),
        prototypes,
        int(config["retrieval"]["portfolio_per_prototype"]),
    )
    return formula_ranking, portfolio


def _run_lopo(
    analysis: pd.DataFrame,
    base_anchors: dict[str, int],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Refit the full transform and ensemble after hiding each positive formula."""

    prototypes = _prototype_formulas(config)
    records: list[dict[str, Any]] = []
    representative = analysis["is_structure_representative"].astype(bool).to_numpy()
    for held_key, held_formula in prototypes.items():
        held_rows = analysis["formula"].eq(held_formula).to_numpy()
        fit_mask = representative & ~held_rows
        transform = SevenFeatureTransform.fit(analysis, fit_mask, config)
        matrix = transform.transform(analysis)
        ensemble = fit_hdbscan_ensemble(
            matrix,
            fit_mask,
            config,
            validity_fn=lambda *_args, **_kwargs: 0.0,
        )
        prediction = predict_hdbscan_ensemble(ensemble, matrix, ~fit_mask)
        labels = _merge_fit_and_predictions(
            ensemble.labels, fit_mask, prediction.labels
        )
        anchors = {
            key: index for key, index in base_anchors.items() if key != held_key
        }
        remaining = {key: value for key, value in prototypes.items() if key != held_key}
        references = build_prototype_references(
            analysis,
            matrix,
            fit_mask,
            anchors,
            remaining,
        )
        scored = score_prototype_neighbourhoods(
            analysis, matrix, labels, references, anchors, config
        )
        scored["top_fraction_frequency"] = parameter_top_fraction_frequency(
            scored,
            labels,
            references,
            anchors,
            representative,
            config,
        )
        scored["robustness_grade"] = _grade_frequency(
            scored["top_fraction_frequency"].to_numpy()
        )
        ranking = aggregate_formula_ranking(scored, remaining)
        hidden = ranking.loc[ranking["formula"].eq(held_formula)]
        if hidden.empty:
            raise RuntimeError(f"LOPO hidden formula became unrankable: {held_key}")
        best = hidden.iloc[0]
        records.append(
            {
                "held_out_prototype": held_key,
                "held_out_formula": held_formula,
                "formula_rank": int(best["formula_rank"]),
                "n_ranked_formulas": int(len(ranking)),
                "formula_percentile": float(best["formula_percentile"]),
                "top_10_recalled": bool(best["formula_rank"] <= 10),
                "top_20_recalled": bool(best["formula_rank"] <= 20),
                "top_5_percent_recalled": bool(
                    best["formula_rank"] <= max(1, math.ceil(0.05 * len(ranking)))
                ),
                "seven_feature_score": float(best["seven_feature_score"]),
                "nearest_remaining_prototype": str(best["nearest_prototype"]),
            }
        )
    return pd.DataFrame(records)


def _coverage_summary(audited: pd.DataFrame) -> dict[str, Any]:
    hard = audited["eligible_hard_screen"].astype(bool)
    complete = audited["eligible_complete_case"].astype(bool)
    analysis = audited["eligible_analysis"].astype(bool)
    return {
        "n_extracted_structures": int(len(audited)),
        "n_extracted_formulas": int(audited["formula"].nunique()),
        "n_hard_screen_eligible_structures": int(hard.sum()),
        "n_complete_case_structures_before_hard_screen": int(complete.sum()),
        "n_analysis_structures": int(analysis.sum()),
        "n_analysis_formulas": int(audited.loc[analysis, "formula"].nunique()),
        "n_missing_complete_case_after_hard_screen": int((hard & ~complete).sum()),
        "missing_formal_oxidation_state_after_hard_screen": int(
            (
                hard
                & pd.to_numeric(
                    audited["audit__formal_oxidation_state_M"], errors="coerce"
                ).isna()
            ).sum()
        ),
    }


def _development_commitment(
    args: argparse.Namespace, config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method_name": "seven_feature_v1",
        "created_before_feature_values_loaded": True,
        "development_features_sha256": sha256_file(
            args.features_dir / "seven_features.csv"
        ),
        "development_provenance_sha256": sha256_file(
            args.features_dir / "provenance.json"
        ),
        "config_sha256": sha256_file(args.config),
        "formula_isolation_file_sha256": sha256_file(args.isolation_file),
        "source_hashes": _source_hashes(),
        "radius_variant": args.radius_variant,
        "model_features": list(MODEL_FEATURES),
        "hdbscan_run_count": 30,
        "target_formulas_used_for": ["identity_isolation_only"],
        "target_formulas_not_used_for": [
            "feature_definition",
            "preprocessing_fit",
            "clustering",
            "weights",
            "thresholds",
            "portfolio_rule",
        ],
        "config_contract_hash": stable_json_hash(config),
    }


def develop(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    validate_config(config)
    args.outdir.mkdir(parents=True, exist_ok=False)
    commitment = _development_commitment(args, config)
    _write_json(
        args.outdir / PREDEVELOPMENT_COMMITMENT_NAME,
        commitment,
        exclusive=True,
    )

    source, feature_provenance = _load_feature_bundle(
        args.features_dir, args.config, "development_exclude"
    )
    isolated = load_formula_boundary(args.isolation_file)
    overlap = set(source["formula"]) & isolated
    if overlap:
        raise RuntimeError("development feature table crosses formula isolation boundary")
    if feature_provenance.get("boundary_file_sha256") != sha256_file(
        args.isolation_file
    ):
        raise RuntimeError("development extraction used a different isolation boundary")

    audited, analysis = _prepare_analysis(
        source,
        config,
        args.cif_dir,
        args.radius_variant,
        args.n_jobs,
    )
    anchors = select_prototype_anchors(analysis, config)
    fit_mask = analysis["is_structure_representative"].astype(bool).to_numpy()
    transform = SevenFeatureTransform.fit(analysis, fit_mask, config)
    matrix = transform.transform(analysis)
    ensemble = fit_hdbscan_ensemble(matrix, fit_mask, config)
    predicted = predict_hdbscan_ensemble(ensemble, matrix, ~fit_mask)
    labels = _merge_fit_and_predictions(ensemble.labels, fit_mask, predicted.labels)
    prototypes = _prototype_formulas(config)
    references = build_prototype_references(
        analysis, matrix, fit_mask, anchors, prototypes
    )
    scored = _score_analysis(
        analysis,
        matrix,
        labels,
        fit_mask,
        anchors,
        references,
        config,
        ensemble.consensus,
    )
    formula_ranking, portfolio = _formula_ranking_and_portfolio(scored, config)
    lopo = _run_lopo(analysis, anchors, config)
    lopo_summary = {
        "mean_reciprocal_rank": float(np.mean(1.0 / lopo["formula_rank"])),
        "top_10_recall": float(lopo["top_10_recalled"].mean()),
        "top_20_recall": float(lopo["top_20_recalled"].mean()),
        "top_5_percent_recall": float(lopo["top_5_percent_recalled"].mean()),
    }

    umap_model, embedding = _fit_umap(matrix, fit_mask, config)
    umap_frame = analysis[["cif_file", "material_id", "formula"]].copy()
    umap_frame["umap_1"] = embedding[:, 0]
    umap_frame["umap_2"] = embedding[:, 1]
    umap_frame["is_fit_representative"] = fit_mask
    _plot_umap(
        analysis,
        embedding,
        anchors,
        args.outdir / "umap_seven_feature_development.png",
    )

    audited.to_csv(args.outdir / "feature_eligibility_audit.csv", index=False)
    scored.to_csv(args.outdir / "ranked_structures.csv", index=False)
    formula_ranking.to_csv(args.outdir / "ranked_formulas.csv", index=False)
    portfolio.to_csv(args.outdir / "candidate_portfolio.csv", index=False)
    lopo.to_csv(args.outdir / "leave_one_positive_out.csv", index=False)
    _write_json(args.outdir / "leave_one_positive_out.summary.json", lopo_summary)
    ensemble.diagnostics_frame().to_csv(
        args.outdir / "hdbscan_run_diagnostics.csv", index=False
    )
    _labels_frame(analysis, labels).to_csv(
        args.outdir / "hdbscan_labels.csv", index=False
    )
    umap_frame.to_csv(args.outdir / "umap_visualization.csv", index=False)
    np.savez_compressed(
        args.outdir / "consensus_matrix.npz",
        consensus=ensemble.consensus,
        fit_cif_files=analysis.loc[fit_mask, "cif_file"].astype(str).to_numpy(),
    )

    state = {
        "schema_version": 1,
        "radius_variant": args.radius_variant,
        "transform": transform,
        "ensemble": ensemble,
        "development_analysis_frame": analysis,
        "development_matrix": matrix,
        "development_labels": labels,
        "development_prediction_strengths": predicted.strengths,
        "fit_mask": fit_mask,
        "anchors": anchors,
        "prototype_formulas": prototypes,
        "prototype_references": references,
        "umap_model": umap_model,
        "umap_embedding": embedding,
        "config_sha256": sha256_file(args.config),
        "source_hashes": _source_hashes(),
        "development_feature_sha256": sha256_file(
            args.features_dir / "seven_features.csv"
        ),
    }
    model_path = args.outdir / FROZEN_MODEL_NAME
    joblib.dump(state, model_path, compress=3)
    output_names = [
        PREDEVELOPMENT_COMMITMENT_NAME,
        "feature_eligibility_audit.csv",
        "ranked_structures.csv",
        "ranked_formulas.csv",
        "candidate_portfolio.csv",
        "leave_one_positive_out.csv",
        "leave_one_positive_out.summary.json",
        "hdbscan_run_diagnostics.csv",
        "hdbscan_labels.csv",
        "umap_visualization.csv",
        "umap_seven_feature_development.png",
        "consensus_matrix.npz",
        FROZEN_MODEL_NAME,
    ]
    manifest = {
        "schema_version": 1,
        "method_name": "seven_feature_v1",
        "status": "frozen_development_model",
        "radius_variant": args.radius_variant,
        "model_features": list(MODEL_FEATURES),
        "development_features_dir": str(args.features_dir.resolve()),
        "development_features_sha256": sha256_file(
            args.features_dir / "seven_features.csv"
        ),
        "development_provenance_sha256": sha256_file(
            args.features_dir / "provenance.json"
        ),
        "config_sha256": sha256_file(args.config),
        "formula_isolation_file_sha256": sha256_file(args.isolation_file),
        "source_hashes": _source_hashes(),
        "output_hashes": {
            name: sha256_file(args.outdir / name) for name in output_names
        },
        "coverage": _coverage_summary(audited),
        "n_fit_structure_representatives": int(fit_mask.sum()),
        "n_hdbscan_runs": int(ensemble.n_runs),
        "prototype_anchors": {
            key: {
                "formula": prototypes[key],
                "cif_file": str(analysis.iloc[index]["cif_file"]),
                "material_id": str(analysis.iloc[index]["material_id"]),
            }
            for key, index in anchors.items()
        },
        "leave_one_positive_out": lopo_summary,
        "blind_acceptance_used_for_features_parameters_weights_thresholds": False,
        "software": {
            name: importlib.metadata.version(name)
            for name in [
                "pymatgen",
                "numpy",
                "pandas",
                "scikit-learn",
                "hdbscan",
                "umap-learn",
                "joblib",
            ]
        },
    }
    _write_json(args.outdir / FROZEN_MANIFEST_NAME, manifest, exclusive=True)
    print(
        f"[OK] frozen seven-feature development model: "
        f"{fit_mask.sum()} fit structures, {len(formula_ranking)} formulas"
    )


def _verify_frozen_manifest(
    manifest_path: Path,
    config_path: Path,
    isolation_file: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("status") != "frozen_development_model":
        raise RuntimeError("manifest is not a frozen development model")
    checks = {
        "config_sha256": sha256_file(config_path),
        "formula_isolation_file_sha256": sha256_file(isolation_file),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"frozen manifest mismatch for {key}")
    if manifest.get("source_hashes") != _source_hashes():
        raise RuntimeError("seven-feature source changed after freeze")
    development_features = Path(str(manifest["development_features_dir"]))
    if sha256_file(development_features / "seven_features.csv") != manifest.get(
        "development_features_sha256"
    ):
        raise RuntimeError("development feature values changed after freeze")
    if sha256_file(development_features / "provenance.json") != manifest.get(
        "development_provenance_sha256"
    ):
        raise RuntimeError("development feature provenance changed after freeze")
    result_dir = manifest_path.parent
    for name, expected in manifest["output_hashes"].items():
        path = result_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen development output changed: {name}")
    return manifest


def evaluate(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    validate_config(config)
    manifest = _verify_frozen_manifest(
        args.frozen_manifest, args.config, args.isolation_file
    )
    if args.radius_variant != manifest.get("radius_variant"):
        raise ValueError(
            "evaluation --radius-variant must equal the frozen development variant"
        )
    args.outdir.mkdir(parents=True, exist_ok=False)
    state = joblib.load(args.frozen_manifest.parent / FROZEN_MODEL_NAME)
    if state["source_hashes"] != _source_hashes():
        raise RuntimeError("serialized model source commitment differs")
    if state["config_sha256"] != sha256_file(args.config):
        raise RuntimeError("serialized model config differs")
    radius_variant = str(manifest["radius_variant"])
    source, provenance = _load_feature_bundle(
        args.features_dir, args.config, "evaluation_include_only"
    )
    if provenance.get("boundary_file_sha256") != sha256_file(args.isolation_file):
        raise RuntimeError("evaluation extraction used a different isolation boundary")
    if provenance.get("frozen_manifest_sha256") != sha256_file(
        args.frozen_manifest
    ):
        raise RuntimeError("evaluation features were not authorized by this freeze")
    targets = load_formula_boundary(args.isolation_file)
    exclusion_path = args.features_dir / "structure_exclusions.csv"
    exclusions = pd.read_csv(exclusion_path) if exclusion_path.is_file() else pd.DataFrame()
    observed_formulas = set(source["formula"])
    if "formula" in exclusions:
        observed_formulas |= {
            canonical_formula(value)
            for value in exclusions["formula"].dropna().astype(str)
            if value.strip()
        }
    if observed_formulas != targets:
        raise RuntimeError(
            "evaluation inventory does not contain exactly the isolated formulas"
        )

    audited, evaluation_analysis = _prepare_analysis(
        source,
        config,
        args.cif_dir,
        radius_variant,
        args.n_jobs,
    )
    transform: SevenFeatureTransform = state["transform"]
    evaluation_matrix = transform.transform(evaluation_analysis)
    prediction = predict_hdbscan_ensemble(
        state["ensemble"], evaluation_matrix
    )
    development = state["development_analysis_frame"].copy()
    all_frame = pd.concat([development, evaluation_analysis], ignore_index=True)
    all_matrix = np.vstack([state["development_matrix"], evaluation_matrix])
    all_labels = np.vstack([state["development_labels"], prediction.labels])
    development_count = len(development)
    anchors = {key: int(value) for key, value in state["anchors"].items()}
    fit_mask = np.r_[state["fit_mask"], np.zeros(len(evaluation_analysis), dtype=bool)]
    scored = _score_analysis(
        all_frame,
        all_matrix,
        all_labels,
        fit_mask,
        anchors,
        state["prototype_references"],
        config,
        state["ensemble"].consensus,
    )
    formula_ranking, portfolio = _formula_ranking_and_portfolio(scored, config)
    portfolio_formulas = set(portfolio.get("formula", pd.Series(dtype=str)))
    rank_lookup = formula_ranking.set_index("formula")
    report_rows: list[dict[str, Any]] = []
    for formula in sorted(targets):
        extracted = audited.loc[audited["formula"].eq(formula)]
        eligible = extracted.loc[extracted["eligible_analysis"].astype(bool)]
        if formula in rank_lookup.index:
            ranked = rank_lookup.loc[formula]
            if isinstance(ranked, pd.DataFrame):
                ranked = ranked.iloc[0]
            status = "ranked"
            rank = int(ranked["formula_rank"])
            percentile = float(ranked["formula_percentile"])
            score = float(ranked["seven_feature_score"])
            best_cif = str(ranked["cif_file"])
            nearest = str(ranked["nearest_prototype"])
        else:
            rank = None
            percentile = None
            score = None
            best_cif = None
            nearest = None
            if extracted.empty:
                status = "not_extracted"
            elif not extracted["eligible_hard_screen"].any():
                status = "failed_hard_screen"
            elif not extracted["eligible_complete_case"].any():
                status = "missing_strict_seven_feature_value"
            else:
                status = "not_ranked"
        report_rows.append(
            {
                "formula": formula,
                "evaluation_status": status,
                "n_extracted_structures": int(len(extracted)),
                "n_eligible_structures": int(len(eligible)),
                "best_cif_file": best_cif,
                "formula_rank": rank,
                "n_ranked_formulas": int(len(formula_ranking)),
                "formula_percentile": percentile,
                "seven_feature_score": score,
                "nearest_prototype": nearest,
                "in_predeclared_portfolio": formula in portfolio_formulas,
            }
        )
    evaluation_report = pd.DataFrame(report_rows)

    combined_embedding = np.vstack(
        [
            state["umap_embedding"],
            state["umap_model"].transform(evaluation_matrix),
        ]
    )
    evaluation_mask = np.arange(len(all_frame)) >= development_count
    _plot_umap(
        all_frame,
        combined_embedding,
        anchors,
        args.outdir / "umap_seven_feature_postfreeze_evaluation.png",
        evaluation_mask=evaluation_mask,
    )
    scored.to_csv(args.outdir / "ranked_all_structures.csv", index=False)
    formula_ranking.to_csv(args.outdir / "ranked_all_formulas.csv", index=False)
    portfolio.to_csv(args.outdir / "candidate_portfolio.csv", index=False)
    audited.to_csv(args.outdir / "evaluation_feature_audit.csv", index=False)
    evaluation_report.to_csv(args.outdir / "postfreeze_evaluation_report.csv", index=False)
    embedding_frame = all_frame[["cif_file", "material_id", "formula"]].copy()
    embedding_frame["umap_1"] = combined_embedding[:, 0]
    embedding_frame["umap_2"] = combined_embedding[:, 1]
    embedding_frame["is_postfreeze_evaluation"] = evaluation_mask
    embedding_frame.to_csv(args.outdir / "umap_visualization.csv", index=False)
    summary = {
        "schema_version": 1,
        "method_name": "seven_feature_v1",
        "evidence_status": "posthoc_mechanism_calibration_not_new_blind_discovery",
        "frozen_manifest_sha256": sha256_file(args.frozen_manifest),
        "evaluation_feature_bundle_sha256": sha256_file(
            args.features_dir / "seven_features.csv"
        ),
        "n_evaluation_formulas": int(len(targets)),
        "n_ranked_evaluation_formulas": int(
            evaluation_report["evaluation_status"].eq("ranked").sum()
        ),
        "n_evaluation_formulas_in_predeclared_portfolio": int(
            evaluation_report["in_predeclared_portfolio"].sum()
        ),
        "all_evaluation_formulas_in_predeclared_portfolio": bool(
            evaluation_report["in_predeclared_portfolio"].all()
        ),
        "no_refit_during_evaluation": True,
        "evaluation_targets_used_for": ["postfreeze_reporting_only"],
    }
    _write_json(args.outdir / "postfreeze_evaluation.summary.json", summary)
    print(
        "[OK] post-freeze evaluation: "
        f"ranked={summary['n_ranked_evaluation_formulas']}/{len(targets)}, "
        "portfolio="
        f"{summary['n_evaluation_formulas_in_predeclared_portfolio']}/{len(targets)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("develop", "evaluate"):
        sub = subparsers.add_parser(stage)
        sub.add_argument("--features-dir", type=Path, required=True)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--isolation-file", type=Path, required=True)
        sub.add_argument("--cif-dir", type=Path, required=True)
        sub.add_argument("--outdir", type=Path, required=True)
        sub.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
        sub.add_argument(
            "--radius-variant", choices=["strict", "nearest_cn"], default="strict"
        )
        if stage == "evaluate":
            sub.add_argument("--frozen-manifest", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "develop":
        develop(args)
    elif args.stage == "evaluate":
        evaluate(args)
    else:  # pragma: no cover - argparse guarantees this
        raise ValueError(args.stage)


if __name__ == "__main__":
    main()
