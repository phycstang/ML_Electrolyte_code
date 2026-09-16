"""Shared, audit-focused helpers for the independent seven-feature workflow."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Structure


SCHEMA_VERSION = 1
MODEL_FEATURES = (
    "feature__D",
    "feature__CN_M",
    "feature__Z_poly",
    "feature__n_shared_mean",
    "feature__delta_chi_MX",
    "feature__radius_ratio_M_X",
    "feature__field_strength_M",
)
STRUCTURE_FEATURES = MODEL_FEATURES[:4]
CHEMISTRY_FEATURES = MODEL_FEATURES[4:]
HALOGENS = frozenset({"F", "Cl", "Br", "I"})
ID_RE = re.compile(r"(mp-\d+)")


def canonical_formula(value: str) -> str:
    """Return pymatgen's deterministic reduced formula."""

    return Composition(str(value)).reduced_formula


def formula_from_cif_name(filename: str) -> str:
    """Parse only the human-readable formula portion of an MP CIF filename."""

    name = Path(str(filename)).name
    raw = name.split("_mp-", 1)[0] if "_mp-" in name else name.rsplit(".", 1)[0]
    return canonical_formula(raw)


def material_id_from_cif_name(filename: str) -> str:
    match = ID_RE.search(Path(str(filename)).name)
    return match.group(1) if match else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    serialized = frame.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def load_formula_boundary(path: Path) -> set[str]:
    payload = load_json(path)
    values = payload.get("formulas")
    if not isinstance(values, list) or not values:
        raise ValueError(f"non-empty formulas list required: {path}")
    return {canonical_formula(value) for value in values}


def identify_binary_halide(structure: Structure) -> tuple[str, str, float, float]:
    """Return M, X and their reduced amounts for a single-X binary halide."""

    composition = structure.composition.reduced_composition
    amounts = {element.symbol: float(amount) for element, amount in composition.items()}
    halogens = sorted(set(amounts) & HALOGENS)
    centres = sorted(set(amounts) - HALOGENS)
    if len(amounts) != 2 or len(halogens) != 1 or len(centres) != 1:
        raise ValueError(
            "expected exactly one non-halogen and one F/Cl/Br/I species; "
            f"found {composition.formula}"
        )
    m_symbol, x_symbol = centres[0], halogens[0]
    return m_symbol, x_symbol, amounts[m_symbol], amounts[x_symbol]


def standardize_structure(structure: Structure, config: dict[str, Any]) -> Structure:
    structure_config = config["structure"]
    if structure_config.get("descriptor_cell_standardization") != "primitive_then_niggli":
        raise ValueError("only primitive_then_niggli standardization is supported")
    tolerance = float(structure_config["primitive_tolerance_A"])
    primitive = structure.get_primitive_structure(
        tolerance=tolerance, use_site_props=False
    )
    return primitive.get_reduced_structure(reduction_algo="niggli")


def validate_config(config: dict[str, Any]) -> None:
    """Fail closed if a purported v1 config changes the seven-feature contract."""

    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"seven-feature schema_version must be {SCHEMA_VERSION}")
    contract = config.get("feature_contract")
    if not isinstance(contract, dict):
        raise ValueError("feature_contract object is required")
    if tuple(contract.get("model_features", ())) != MODEL_FEATURES:
        raise ValueError(f"model_features must be exactly {list(MODEL_FEATURES)}")
    if int(contract.get("minimum_cn_for_polyhedron", -1)) != 3:
        raise ValueError("the original Xcn/Pcn/Xsh population requires minimum CN=3")
    if contract.get("delta_chi_definition") != "absolute_pauling_difference":
        raise ValueError("v1 delta chi must be the absolute Pauling difference")
    if contract.get("missing_radius_policy") != (
        "exclude_from_corresponding_complete_case_analysis"
    ):
        raise ValueError("v1 forbids silent radius imputation")
    preprocessing = config.get("preprocessing", {})
    if preprocessing.get("feature_weighting") != "equal_per_feature":
        raise ValueError("v1 clustering uses equal per-feature weights")
    if preprocessing.get("PCA_for_clustering") is not False:
        raise ValueError("PCA must not be used for seven-dimensional clustering")


def finite_or_nan(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if np.isfinite(numeric) else float("nan")
