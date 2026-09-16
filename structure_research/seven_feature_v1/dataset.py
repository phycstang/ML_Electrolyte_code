"""Eligibility and structure-family handling for the seven-feature workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Structure

from .common import MODEL_FEATURES, canonical_formula


SENSITIVITY_RADIUS_FEATURES = {
    "feature__radius_ratio_M_X": "sensitivity__radius_ratio_M_X_nearest_CN",
    "feature__field_strength_M": "sensitivity__field_strength_M_nearest_CN",
}


def _as_bool(series: pd.Series, *, missing: bool = False) -> pd.Series:
    """Parse CSV booleans without treating the string ``False`` as true."""

    normalized = series.astype("string").str.strip().str.lower()
    result = normalized.map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    ).astype("boolean")
    return result.fillna(bool(missing)).astype(bool)


def add_eligibility(
    frame: pd.DataFrame, config: dict[str, Any], *, radius_variant: str = "strict"
) -> pd.DataFrame:
    """Attach hard-screen and complete-case flags without imputing any feature."""

    if radius_variant not in {"strict", "nearest_cn"}:
        raise ValueError("radius_variant must be strict or nearest_cn")
    out = frame.copy()
    out["formula"] = out["formula"].map(canonical_formula)
    eligibility = config["eligibility"]
    toxic = set(eligibility["strict_toxic_elements"])
    precious = set(eligibility["precious_elements"])
    out["risk_radioactive"] = out["center_element"].map(
        lambda value: bool(Element(str(value)).is_radioactive)
    )
    out["risk_strict_toxic"] = out["center_element"].isin(toxic)
    out["flag_precious"] = out["center_element"].isin(precious)

    hull = pd.to_numeric(out.get("energy_above_hull"), errors="coerce")
    out["energy_above_hull_used_eV_atom"] = hull
    main = float(eligibility["main_hull_max_eV_atom"])
    exploratory = float(eligibility["exploratory_hull_max_eV_atom"])
    out["stability_tier"] = np.select(
        [hull.isna(), hull <= main, hull <= exploratory],
        ["unknown", "main", "exploratory"],
        default="above_exploratory",
    )
    stability_allowed = hull.le(exploratory) | (
        hull.isna() & bool(eligibility["allow_unknown_stability"])
    )
    ordered = _as_bool(out["is_ordered"], missing=False)
    if not bool(eligibility["require_ordered_structure"]):
        ordered[:] = True
    if "deprecated" in out and bool(eligibility.get("exclude_deprecated", True)):
        not_deprecated = ~_as_bool(out["deprecated"], missing=False)
    else:
        not_deprecated = pd.Series(True, index=out.index)

    out["eligible_hard_screen"] = (
        out["feature_status"].eq("ok")
        & ~out["risk_radioactive"]
        & ~out["risk_strict_toxic"]
        & stability_allowed
        & ordered
        & not_deprecated
    )

    source_columns = list(MODEL_FEATURES)
    if radius_variant == "nearest_cn":
        for strict_name, sensitivity_name in SENSITIVITY_RADIUS_FEATURES.items():
            if sensitivity_name not in out:
                raise ValueError(f"missing nearest-CN sensitivity column: {sensitivity_name}")
            out[strict_name] = pd.to_numeric(out[sensitivity_name], errors="coerce")
    for name in MODEL_FEATURES:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    finite = np.isfinite(out[list(MODEL_FEATURES)].to_numpy(dtype=float)).all(axis=1)
    out["eligible_complete_case"] = finite
    out["eligible_analysis"] = out["eligible_hard_screen"] & finite
    out["analysis_radius_variant"] = radius_variant
    out["analysis_feature_source_columns"] = ",".join(source_columns)
    return out


def _selection_order(group: pd.DataFrame) -> list[int]:
    hull = pd.to_numeric(group.get("energy_above_hull"), errors="coerce").fillna(np.inf)
    formation = pd.to_numeric(
        group.get("formation_energy_per_atom"), errors="coerce"
    ).fillna(np.inf)
    ordering = pd.DataFrame(
        {
            "hull": hull,
            "formation": formation,
            "cif_file": group["cif_file"].astype(str),
        },
        index=group.index,
    ).sort_values(["hull", "formation", "cif_file"], kind="mergesort")
    return [int(value) for value in ordering.index]


@dataclass(frozen=True)
class FormulaDeduplication:
    assignments: tuple[tuple[int, str, bool, str, str], ...]


def _deduplicate_formula(
    formula: str,
    group: pd.DataFrame,
    cif_dir: Path,
    config: dict[str, Any],
) -> FormulaDeduplication:
    cfg = config["deduplication"]
    matcher = StructureMatcher(
        ltol=float(cfg["ltol"]),
        stol=float(cfg["stol"]),
        angle_tol=float(cfg["angle_tol_deg"]),
        primitive_cell=bool(cfg["primitive_cell"]),
        scale=bool(cfg["scale"]),
        attempt_supercell=bool(cfg["attempt_supercell"]),
        allow_subset=bool(cfg["allow_subset"]),
    )
    signature_columns = list(cfg["structure_signature_features"])
    decimals = int(cfg["signature_round_decimals"])
    structures: dict[int, Structure | Exception] = {}
    for index, row in group.iterrows():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structures[int(index)] = Structure.from_file(
                    cif_dir / str(row["cif_file"])
                )
        except Exception as exc:  # keep the row unique and expose the failure
            structures[int(index)] = exc

    assignments: list[tuple[int, str, bool, str, str]] = []
    family_counter = 0
    for _, signature_group in group.groupby(
        [group[name].round(decimals) for name in signature_columns],
        sort=True,
        dropna=False,
    ):
        representatives: list[tuple[int, str]] = []
        for index in _selection_order(signature_group):
            structure = structures[index]
            matched_family = ""
            matched_cif = ""
            match_error = ""
            if isinstance(structure, Exception):
                match_error = f"{type(structure).__name__}: {structure}"
            else:
                for representative_index, family_id in representatives:
                    representative = structures[representative_index]
                    if isinstance(representative, Exception):
                        continue
                    try:
                        if matcher.fit(structure, representative):
                            matched_family = family_id
                            matched_cif = str(group.loc[representative_index, "cif_file"])
                            break
                    except Exception as exc:
                        match_error = f"{type(exc).__name__}: {exc}"
            if matched_family:
                assignments.append(
                    (index, matched_family, False, matched_cif, match_error)
                )
            else:
                family_counter += 1
                family_id = f"{formula}::sf{family_counter:04d}"
                representatives.append((index, family_id))
                assignments.append((index, family_id, True, "", match_error))
    return FormulaDeduplication(tuple(assignments))


def add_structure_families(
    frame: pd.DataFrame,
    cif_dir: Path,
    config: dict[str, Any],
    *,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Deduplicate only within formula and the four-feature structure signature."""

    out = frame.copy()
    out["structure_family_id"] = ""
    out["is_structure_representative"] = False
    out["duplicate_of_cif"] = ""
    out["structure_match_error"] = ""
    eligible = out.loc[out["eligible_analysis"].astype(bool)]
    groups = [
        (str(formula), group.copy())
        for formula, group in eligible.groupby("formula", sort=True)
    ]
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_deduplicate_formula)(formula, group, Path(cif_dir), config)
        for formula, group in groups
    )
    for result in results:
        for index, family_id, representative, duplicate_of, error in result.assignments:
            out.at[index, "structure_family_id"] = family_id
            out.at[index, "is_structure_representative"] = bool(representative)
            out.at[index, "duplicate_of_cif"] = duplicate_of
            out.at[index, "structure_match_error"] = error
    return out


def select_prototype_anchors(
    frame: pd.DataFrame, config: dict[str, Any]
) -> dict[str, int]:
    """Select one deterministic, eligible representative for each known prototype."""

    expected = "lowest_energy_above_hull_then_formation_energy_then_cif"
    if config.get("prototype_anchor_policy") != expected:
        raise ValueError(f"prototype_anchor_policy must be {expected}")
    anchors: dict[str, int] = {}
    for key, value in config["positive_prototypes"].items():
        formula = canonical_formula(value)
        candidates = frame.loc[
            frame["formula"].eq(formula)
            & frame["eligible_analysis"].astype(bool)
            & frame["is_structure_representative"].astype(bool)
        ]
        if candidates.empty:
            raise ValueError(f"prototype has no eligible representative: {key}={formula}")
        anchors[str(key)] = _selection_order(candidates)[0]
    return anchors
