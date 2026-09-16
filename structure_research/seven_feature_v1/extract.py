#!/usr/bin/env python3
"""Extract the independent seven-feature binary-halide descriptor table.

Chemical identity is resolved from each CIF before the isolation boundary is
applied.  In development mode, holdout formulas are physically removed before
any VESTA graph, coordination number, electronegativity, or radius is evaluated.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.core import Structure

from src.discovery.bonding_vesta import build_vesta_mx_graph, vesta_rule_provenance

from .common import (
    MODEL_FEATURES,
    canonical_formula,
    dataframe_sha256,
    formula_from_cif_name,
    identify_binary_halide,
    load_formula_boundary,
    load_json,
    material_id_from_cif_name,
    sha256_file,
    stable_json_hash,
    standardize_structure,
    validate_config,
)
from .radii import compute_chemical_features, shannon_source_provenance


SUPPORTED_SUFFIXES = frozenset({".cif", ".vasp", ".poscar"})


def _empty_identity(cif_file: str) -> dict[str, Any]:
    return {
        "cif_file": str(cif_file),
        "material_id": material_id_from_cif_name(cif_file),
        "cif_sha256": "",
        "parsed_formula": "",
        "filename_formula": "",
        "identity_status": "ok",
        "identity_error": "",
    }


def inspect_identity(cif_file: str, cif_dir: Path) -> dict[str, Any]:
    """Read composition and content identity, but evaluate no descriptor."""

    record = _empty_identity(cif_file)
    path = Path(cif_dir) / str(cif_file)
    try:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported structure suffix: {path.suffix}")
        record["cif_sha256"] = sha256_file(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_file(path)
        parsed = canonical_formula(structure.composition.reduced_formula)
        filename_formula = formula_from_cif_name(cif_file)
        record["parsed_formula"] = parsed
        record["filename_formula"] = filename_formula
        if parsed != filename_formula:
            raise ValueError(
                "CIF composition/filename formula mismatch: "
                f"parsed={parsed}, filename={filename_formula}"
            )
    except Exception as exc:
        record["identity_status"] = "error"
        record["identity_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _flatten_chemical_audit(result: dict[str, Any]) -> dict[str, Any]:
    audit = dict(result.get("audit", {}))
    flattened: dict[str, Any] = {}
    for key, value in audit.items():
        name = f"audit__{key}"
        if isinstance(value, (dict, list, tuple)):
            flattened[name] = json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        else:
            flattened[name] = value
    return flattened


def extract_one(
    identity: dict[str, Any], cif_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Compute exactly four VESTA and three chemistry model features."""

    path = Path(cif_dir) / str(identity["cif_file"])
    base: dict[str, Any] = {
        "cif_file": str(identity["cif_file"]),
        "material_id": str(identity["material_id"]),
        "formula": str(identity["parsed_formula"]),
        "cif_sha256": str(identity["cif_sha256"]),
        "feature_status": "ok",
        "feature_error": "",
    }
    try:
        if sha256_file(path) != base["cif_sha256"]:
            raise RuntimeError("CIF changed after identity/isolation pass")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            source = Structure.from_file(path)
        parsed = canonical_formula(source.composition.reduced_formula)
        if parsed != base["formula"]:
            raise RuntimeError(
                f"CIF formula changed after isolation: {base['formula']} -> {parsed}"
            )
        base["source_n_sites"] = int(len(source))
        base["source_cell_volume_A3"] = float(source.volume)
        base["is_ordered"] = bool(source.is_ordered)
        if not source.is_ordered:
            raise ValueError("partially occupied/disordered structure is not supported")
        structure = standardize_structure(source, config)
        base["descriptor_n_sites"] = int(len(structure))
        base["descriptor_cell_standardization"] = "primitive_then_niggli"
        m_symbol, x_symbol, n_m, n_x = identify_binary_halide(structure)
        base["center_element"] = m_symbol
        base["halogen_element"] = x_symbol
        base["audit__reduced_amount_M"] = float(n_m)
        base["audit__reduced_amount_X"] = float(n_x)

        graph = build_vesta_mx_graph(structure)
        if not graph.bonds:
            raise ValueError(
                "VESTA-2019 pair is defined but identifies no M-X bond; "
                "zero bonds are not interpreted as a physical 0D network"
            )
        minimum_cn = int(config["feature_contract"]["minimum_cn_for_polyhedron"])
        poly = graph.polyhedron_metrics(minimum_cn=minimum_cn)
        grouped = graph.bonds_by_center()
        eligible_indices = [int(value) for value in poly["eligible_center_indices"]]
        site_cns = {index: len(grouped[index]) for index in eligible_indices}

        base.update(
            {
                "feature__D": float(poly["dim"]),
                "feature__CN_M": float(poly["st1"]),
                "feature__Z_poly": float(poly["st3"]),
                "feature__n_shared_mean": float(poly["st2"]),
                "audit__vesta_cutoff_A": float(graph.cutoff_A),
                "audit__vesta_bond_orbit_count": int(len(graph.bonds)),
                "audit__polyhedron_count_CN_ge_3": int(len(eligible_indices)),
                "audit__eligible_center_indices": json.dumps(eligible_indices),
                "audit__eligible_center_CNs": json.dumps(
                    site_cns, sort_keys=True, separators=(",", ":")
                ),
                "audit__all_center_CNs": json.dumps(
                    {int(index): len(bonds) for index, bonds in grouped.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
        if site_cns:
            observed_mean = float(np.mean(list(site_cns.values())))
            if not np.isclose(observed_mean, base["feature__CN_M"], atol=1e-12):
                raise RuntimeError("Xcn population and per-site CN audit disagree")
        elif base["feature__CN_M"] != 0.0:
            raise RuntimeError("empty Xcn population must have CN_M=0")

        chemistry = compute_chemical_features(
            m_symbol,
            x_symbol,
            n_m,
            n_x,
            site_cns,
            delta_mode="absolute",
        )
        base["feature__delta_chi_MX"] = chemistry["delta_chi"]
        base["feature__radius_ratio_M_X"] = chemistry["strict_radius_ratio"]
        base["feature__field_strength_M"] = chemistry["strict_field_strength"]
        base["sensitivity__radius_ratio_M_X_nearest_CN"] = chemistry[
            "sensitivity_radius_ratio"
        ]
        base["sensitivity__field_strength_M_nearest_CN"] = chemistry[
            "sensitivity_field_strength"
        ]
        base["audit__formal_oxidation_state_M"] = chemistry[
            "formal_oxidation_state"
        ]
        base["audit__strict_radius_available"] = bool(
            chemistry["strict_radius_available"]
        )
        base["audit__nearest_CN_radius_available"] = bool(
            chemistry["sensitivity_radius_available"]
        )
        base.update(_flatten_chemical_audit(chemistry))
        if sha256_file(path) != base["cif_sha256"]:
            raise RuntimeError("CIF changed during feature extraction")
    except Exception as exc:
        base["feature_status"] = "error"
        base["feature_error"] = f"{type(exc).__name__}: {exc}"
        for name in MODEL_FEATURES:
            base.setdefault(name, np.nan)
        base.setdefault("sensitivity__radius_ratio_M_X_nearest_CN", np.nan)
        base.setdefault("sensitivity__field_strength_M_nearest_CN", np.nan)
    return base


def merge_metadata(
    features: pd.DataFrame, metadata_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach only metadata rows whose identities crossed the isolation boundary."""

    header = pd.read_csv(metadata_path, nrows=0)
    required = {"material_id", "cif_file"}
    if not required.issubset(header.columns):
        raise ValueError(f"metadata requires columns {sorted(required)}")
    identities = pd.read_csv(metadata_path, usecols=["material_id"])
    if identities["material_id"].duplicated().any():
        raise ValueError("metadata material_id is not unique")
    allowed = set(features["material_id"].astype(str))
    keep_lines = {
        position + 1
        for position, value in enumerate(identities["material_id"].astype(str))
        if value in allowed
    }
    metadata = pd.read_csv(
        metadata_path,
        skiprows=lambda number: number > 0 and number not in keep_lines,
    )
    if metadata["material_id"].duplicated().any():
        raise ValueError("selected metadata material_id is not unique")
    identity = features[["material_id", "cif_file"]].merge(
        metadata[["material_id", "cif_file"]].rename(
            columns={"cif_file": "metadata_cif_file"}
        ),
        on="material_id",
        how="left",
        validate="one_to_one",
    )
    mismatch = identity[
        identity["metadata_cif_file"].notna()
        & identity["cif_file"].ne(identity["metadata_cif_file"])
    ]
    if len(mismatch):
        raise ValueError(f"metadata CIF mismatch: {mismatch.head().to_dict('records')}")
    metadata = metadata.drop(columns=["cif_file"])
    collisions = sorted((set(features) & set(metadata)) - {"material_id"})
    if collisions:
        raise ValueError(f"metadata columns collide with features: {collisions}")
    before = features[["cif_file", "material_id", "formula"]].copy()
    merged = features.merge(
        metadata, on="material_id", how="left", validate="one_to_one", sort=False
    )
    if not before.equals(merged[["cif_file", "material_id", "formula"]]):
        raise RuntimeError("metadata merge changed row identity/order")
    return merged, {
        "source_sha256": sha256_file(metadata_path),
        "source_rows": int(len(identities)),
        "selected_rows": int(len(metadata)),
        "selected_values_sha256": dataframe_sha256(metadata),
    }


def validate_isolation_arguments(
    exclude_formulas_file: Path | None, include_only_formulas_file: Path | None
) -> str:
    if (exclude_formulas_file is None) == (include_only_formulas_file is None):
        raise ValueError(
            "exactly one of --exclude-formulas-file and --include-only-formulas-file "
            "is required"
        )
    return "development_exclude" if exclude_formulas_file else "evaluation_include_only"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    isolation_mode = validate_isolation_arguments(
        args.exclude_formulas_file, args.include_only_formulas_file
    )
    boundary_file = args.exclude_formulas_file or args.include_only_formulas_file
    assert boundary_file is not None
    if isolation_mode == "evaluation_include_only":
        if args.frozen_manifest is None:
            raise ValueError(
                "evaluation feature extraction requires --frozen-manifest and "
                "is forbidden before the development model is frozen"
            )
        # Import lazily so development extraction remains independent of the
        # clustering runtime.  This gate executes before the target formula file
        # is parsed or any evaluation CIF is inspected.
        from .pipeline import _verify_frozen_manifest

        _verify_frozen_manifest(args.frozen_manifest, args.config, boundary_file)
    elif args.frozen_manifest is not None:
        raise ValueError("development extraction must not receive --frozen-manifest")
    config = load_json(args.config)
    validate_config(config)
    runtime_bonding = vesta_rule_provenance(verify=True)
    contract = config["feature_contract"]
    bonding_checks = {
        "backend": contract["bonding_backend"],
        "cutoff_table_sha256": contract["vesta_cutoff_table_sha256"],
        "cutoff_pair_count": int(contract["vesta_cutoff_pair_count"]),
        "distance_comparison": contract["distance_comparison"],
        "periodic_images_retained": bool(contract["retain_periodic_images"]),
    }
    mismatch = {
        key: {"configured": value, "runtime": runtime_bonding.get(key)}
        for key, value in bonding_checks.items()
        if runtime_bonding.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"VESTA bonding contract mismatch: {mismatch}")

    inventory_source = pd.read_csv(args.inventory, usecols=["cif_file"])
    if inventory_source["cif_file"].duplicated().any():
        raise ValueError("inventory contains duplicate cif_file values")
    identity_rows = Parallel(n_jobs=args.n_jobs, prefer="processes", verbose=5)(
        delayed(inspect_identity)(str(value), args.cif_dir)
        for value in inventory_source["cif_file"].astype(str)
    )
    full_identity = pd.DataFrame(identity_rows).sort_values("cif_file").reset_index(drop=True)
    boundary = load_formula_boundary(boundary_file)
    inventory = full_identity.copy()
    inventory["formula"] = inventory["parsed_formula"].fillna("")
    if isolation_mode == "development_exclude":
        inventory = inventory.loc[~inventory["formula"].isin(boundary)].copy()
    else:
        inventory = inventory.loc[inventory["formula"].isin(boundary)].copy()
    inventory = inventory.sort_values("cif_file").reset_index(drop=True)
    if inventory.empty:
        raise RuntimeError("no CIF remains after formula isolation")

    args.outdir.mkdir(parents=True, exist_ok=False)
    identity_failures = inventory.loc[inventory["identity_status"].ne("ok")].copy()
    ready = inventory.loc[inventory["identity_status"].eq("ok")].copy()
    extracted = Parallel(n_jobs=args.n_jobs, prefer="processes", verbose=5)(
        delayed(extract_one)(row, args.cif_dir, config)
        for row in ready.to_dict("records")
    )
    frame = pd.DataFrame(extracted).sort_values("cif_file").reset_index(drop=True)
    metadata_info: dict[str, Any] | None = None
    if args.metadata is not None:
        frame, metadata_info = merge_metadata(frame, args.metadata)

    feature_errors = frame.loc[frame["feature_status"].ne("ok")].copy()
    feature_success = frame.loc[frame["feature_status"].eq("ok")].copy()
    identity_ledger = pd.DataFrame(
        {
            "cif_file": identity_failures.get("cif_file", pd.Series(dtype=str)),
            "material_id": identity_failures.get("material_id", pd.Series(dtype=str)),
            "formula": identity_failures.get("formula", pd.Series(dtype=str)),
            "exclusion_stage": "identity",
            "exclusion_reason": identity_failures.get(
                "identity_error", pd.Series(dtype=str)
            ),
        }
    )
    error_ledger = pd.DataFrame(
        {
            "cif_file": feature_errors.get("cif_file", pd.Series(dtype=str)),
            "material_id": feature_errors.get("material_id", pd.Series(dtype=str)),
            "formula": feature_errors.get("formula", pd.Series(dtype=str)),
            "exclusion_stage": "descriptor",
            "exclusion_reason": feature_errors.get(
                "feature_error", pd.Series(dtype=str)
            ),
        }
    )
    exclusions = pd.concat([identity_ledger, error_ledger], ignore_index=True)
    output_path = args.outdir / "seven_features.csv"
    exclusions_path = args.outdir / "structure_exclusions.csv"
    feature_success.to_csv(output_path, index=False)
    exclusions.to_csv(exclusions_path, index=False)

    strict_complete = np.isfinite(
        feature_success[list(MODEL_FEATURES)].apply(pd.to_numeric, errors="coerce")
    ).all(axis=1)
    sensitivity_columns = [
        *MODEL_FEATURES[:5],
        "sensitivity__radius_ratio_M_X_nearest_CN",
        "sensitivity__field_strength_M_nearest_CN",
    ]
    sensitivity_complete = np.isfinite(
        feature_success[sensitivity_columns].apply(pd.to_numeric, errors="coerce")
    ).all(axis=1)
    provenance = {
        "schema_version": 1,
        "method_name": "seven_feature_v1",
        "isolation_mode": isolation_mode,
        "boundary_file_sha256": sha256_file(boundary_file),
        "frozen_manifest_sha256": (
            sha256_file(args.frozen_manifest) if args.frozen_manifest else None
        ),
        "boundary_formula_names_written_to_development_outputs": False,
        "config_sha256": sha256_file(args.config),
        "extractor_sha256": sha256_file(Path(__file__)),
        "common_module_sha256": sha256_file(Path(__file__).with_name("common.py")),
        "radii_module_sha256": sha256_file(Path(__file__).with_name("radii.py")),
        "bonding_module_sha256": sha256_file(
            Path(__file__).resolve().parents[1] / "discovery" / "bonding_vesta.py"
        ),
        "inventory_source_sha256": sha256_file(args.inventory),
        "metadata": metadata_info,
        "full_inventory_count": int(len(full_identity)),
        "isolated_inventory_count": int(len(inventory)),
        "identity_success_count": int(len(ready)),
        "feature_success_count": int(len(feature_success)),
        "feature_error_count": int(len(exclusions)),
        "strict_complete_count_before_hard_screen": int(strict_complete.sum()),
        "nearest_CN_complete_count_before_hard_screen": int(sensitivity_complete.sum()),
        "full_identity_inventory_sha256": stable_json_hash(
            full_identity[
                [
                    "cif_file",
                    "cif_sha256",
                    "parsed_formula",
                    "filename_formula",
                    "identity_status",
                ]
            ].to_dict("records")
        ),
        "isolated_inventory_sha256": stable_json_hash(
            inventory[["cif_file", "cif_sha256", "formula"]].to_dict("records")
        ),
        "seven_features_sha256": sha256_file(output_path),
        "structure_exclusions_sha256": sha256_file(exclusions_path),
        "model_features": list(MODEL_FEATURES),
        "bonding": runtime_bonding,
        "shannon_source": shannon_source_provenance(),
        "software": {
            name: importlib.metadata.version(name)
            for name in ["pymatgen", "numpy", "pandas", "joblib"]
        },
    }
    _write_json(args.outdir / "provenance.json", provenance)
    print(
        f"[OK] {isolation_mode}: {len(feature_success)} structures; "
        f"strict complete={strict_complete.sum()}, nearest-CN complete={sensitivity_complete.sum()}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cif-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--exclude-formulas-file", type=Path)
    parser.add_argument("--include-only-formulas-file", type=Path)
    parser.add_argument(
        "--frozen-manifest",
        type=Path,
        help="required gate for post-freeze evaluation extraction",
    )
    parser.add_argument("--n-jobs", type=int, default=min(16, os.cpu_count() or 1))
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
