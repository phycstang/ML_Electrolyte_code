"""Prototype-neighbour scoring using only the standardized seven-feature space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PrototypeReference:
    key: str
    formula: str
    anchor_cif: str
    anchor_vector: np.ndarray
    sorted_reference_distances: np.ndarray


def empirical_survival_similarity(
    distances: np.ndarray, sorted_reference: np.ndarray
) -> np.ndarray:
    """Map distance to a development-frozen empirical survival fraction.

    A value near one means at least as close as almost every non-prototype
    development structure.  The +1 finite-sample correction prevents exact zero.
    """

    query = np.asarray(distances, dtype=float)
    reference = np.sort(np.asarray(sorted_reference, dtype=float))
    reference = reference[np.isfinite(reference)]
    if not len(reference):
        raise ValueError("prototype distance reference distribution is empty")
    left = np.searchsorted(reference, query, side="left")
    return (len(reference) - left + 1.0) / (len(reference) + 1.0)


def build_prototype_references(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    fit_mask: np.ndarray,
    anchors: dict[str, int],
    prototype_formulas: dict[str, str],
) -> dict[str, PrototypeReference]:
    """Freeze anchors and distance calibration using non-prototype fit rows only."""

    values = np.asarray(matrix, dtype=float)
    mask = np.asarray(fit_mask, dtype=bool)
    if len(frame) != len(values) or len(mask) != len(frame):
        raise ValueError("frame/matrix/fit_mask row mismatch")
    all_prototype_formulas = set(prototype_formulas.values())
    reference_mask = mask & ~frame["formula"].isin(all_prototype_formulas).to_numpy()
    if not reference_mask.any():
        raise ValueError("no non-prototype development rows for distance calibration")
    references: dict[str, PrototypeReference] = {}
    for key, index in anchors.items():
        anchor = values[int(index)]
        distances = np.linalg.norm(values[reference_mask] - anchor, axis=1)
        references[key] = PrototypeReference(
            key=key,
            formula=str(prototype_formulas[key]),
            anchor_cif=str(frame.iloc[int(index)]["cif_file"]),
            anchor_vector=np.asarray(anchor, dtype=float),
            sorted_reference_distances=np.sort(distances),
        )
    return references


def _prototype_support_from_labels(
    labels: np.ndarray, anchor_position: int
) -> np.ndarray:
    values = np.asarray(labels, dtype=int)
    if values.ndim != 2:
        raise ValueError("labels must be rows x runs")
    anchor_labels = values[int(anchor_position)]
    same = (values == anchor_labels[None, :]) & (values >= 0) & (
        anchor_labels[None, :] >= 0
    )
    return same.mean(axis=1)


def score_prototype_neighbourhoods(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    labels: np.ndarray,
    references: dict[str, PrototypeReference],
    anchor_positions: dict[str, int],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Fuse distance and co-clustering inside each prototype, then take max."""

    values = np.asarray(matrix, dtype=float)
    if len(frame) != len(values) or np.asarray(labels).shape[0] != len(frame):
        raise ValueError("score inputs have inconsistent rows")
    retrieval = config["retrieval"]
    distance_weight = float(retrieval["distance_component_weight"])
    cluster_weight = float(retrieval["prototype_cocluster_component_weight"])
    if not np.isclose(distance_weight + cluster_weight, 1.0):
        raise ValueError("retrieval component weights must sum to one")
    if retrieval.get("prototype_fusion") != "within_prototype_then_max_across_prototypes":
        raise ValueError("prototype fusion contract changed")

    result = frame.copy()
    score_columns: list[str] = []
    for key, reference in references.items():
        distances = np.linalg.norm(values - reference.anchor_vector, axis=1)
        similarity = empirical_survival_similarity(
            distances, reference.sorted_reference_distances
        )
        support = _prototype_support_from_labels(labels, anchor_positions[key])
        score = distance_weight * similarity + cluster_weight * support
        result[f"distance__{key}"] = distances
        result[f"distance_similarity__{key}"] = similarity
        result[f"cocluster_support__{key}"] = support
        result[f"prototype_score__{key}"] = score
        score_columns.append(f"prototype_score__{key}")

    score_matrix = result[score_columns].to_numpy(dtype=float)
    best = np.argmax(score_matrix, axis=1)
    keys = list(references)
    result["nearest_prototype"] = [keys[index] for index in best]
    result["seven_feature_score"] = score_matrix[np.arange(len(result)), best]
    result["nearest_prototype_distance"] = np.asarray(
        [result.iloc[row][f"distance__{keys[column]}"] for row, column in enumerate(best)]
    )
    result["max_prototype_cocluster_support"] = np.asarray(
        [
            result.iloc[row][f"cocluster_support__{keys[column]}"]
            for row, column in enumerate(best)
        ]
    )
    result["hdbscan_noise_frequency"] = (np.asarray(labels) < 0).mean(axis=1)
    return result


def parameter_top_fraction_frequency(
    scored: pd.DataFrame,
    labels: np.ndarray,
    references: dict[str, PrototypeReference],
    anchor_positions: dict[str, int],
    candidate_mask: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Frequency of entering Top-f under individual preregistered HDBSCAN runs."""

    labels = np.asarray(labels, dtype=int)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if len(candidate_mask) != len(scored):
        raise ValueError("candidate_mask row mismatch")
    top_fraction = float(config["retrieval"]["top_fraction_for_stability"])
    distance_weight = float(config["retrieval"]["distance_component_weight"])
    cluster_weight = float(
        config["retrieval"]["prototype_cocluster_component_weight"]
    )
    frequencies = np.zeros(len(scored), dtype=float)
    candidates = np.flatnonzero(candidate_mask)
    if not len(candidates):
        return frequencies
    top_n = max(1, int(np.ceil(top_fraction * len(candidates))))
    distance_similarity = {
        key: scored[f"distance_similarity__{key}"].to_numpy(dtype=float)
        for key in references
    }
    for run in range(labels.shape[1]):
        per_prototype = []
        for key in references:
            anchor_label = labels[int(anchor_positions[key]), run]
            same = (
                (labels[:, run] == anchor_label) & (labels[:, run] >= 0)
                if anchor_label >= 0
                else np.zeros(len(scored), dtype=bool)
            )
            per_prototype.append(
                distance_weight * distance_similarity[key]
                + cluster_weight * same.astype(float)
            )
        run_score = np.max(np.stack(per_prototype, axis=1), axis=1)
        order = candidates[
            np.lexsort(
                (
                    scored.iloc[candidates]["cif_file"].astype(str).to_numpy(),
                    -run_score[candidates],
                )
            )
        ]
        frequencies[order[:top_n]] += 1.0
    return frequencies / max(labels.shape[1], 1)


def aggregate_formula_ranking(
    scored_structures: pd.DataFrame,
    prototype_keys: Iterable[str],
) -> pd.DataFrame:
    """Take the best representative structure per formula with deterministic ties."""

    eligible = scored_structures.loc[
        scored_structures["eligible_analysis"].astype(bool)
        & scored_structures["is_structure_representative"].astype(bool)
    ].copy()
    if eligible.empty:
        raise ValueError("no eligible representative structures to rank")
    eligible = eligible.sort_values(
        ["seven_feature_score", "energy_above_hull_used_eV_atom", "cif_file"],
        ascending=[False, True, True],
        na_position="last",
        kind="mergesort",
    )
    best = eligible.drop_duplicates("formula", keep="first").copy()
    best = best.sort_values(
        ["seven_feature_score", "formula", "cif_file"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best["formula_rank"] = np.arange(1, len(best) + 1)
    best["formula_percentile"] = 1.0 - (best["formula_rank"] - 1) / max(len(best), 1)
    columns = [
        "formula",
        "material_id",
        "cif_file",
        "center_element",
        "halogen_element",
        "formula_rank",
        "formula_percentile",
        "seven_feature_score",
        "nearest_prototype",
        "max_prototype_cocluster_support",
        "hdbscan_noise_frequency",
        "top_fraction_frequency",
        "robustness_grade",
        "energy_above_hull_used_eV_atom",
        "stability_tier",
    ]
    for key in prototype_keys:
        columns.extend(
            [
                f"distance__{key}",
                f"distance_similarity__{key}",
                f"cocluster_support__{key}",
                f"prototype_score__{key}",
            ]
        )
    return best[[name for name in columns if name in best]]


def build_unique_prototype_portfolio(
    formula_ranking: pd.DataFrame,
    prototype_formulas: set[str],
    prototype_keys: Iterable[str],
    per_prototype: int,
) -> pd.DataFrame:
    """Assign each formula to its best prototype and select unique section leaders."""

    candidates = formula_ranking.loc[
        ~formula_ranking["formula"].isin(prototype_formulas)
    ].copy()
    rows: list[pd.DataFrame] = []
    for key in prototype_keys:
        section = candidates.loc[candidates["nearest_prototype"].eq(key)].copy()
        section = section.sort_values(
            [
                f"prototype_score__{key}",
                f"cocluster_support__{key}",
                f"distance__{key}",
                "formula",
            ],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).head(int(per_prototype))
        section.insert(0, "portfolio_section", key)
        section.insert(1, "section_rank", np.arange(1, len(section) + 1))
        rows.append(section)
    portfolio = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(portfolio) and portfolio["formula"].duplicated().any():
        raise RuntimeError("prototype portfolio must be formula-unique")
    return portfolio
