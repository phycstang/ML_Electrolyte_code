#!/usr/bin/env python3
"""Run frozen multi-view, multi-prototype retrieval for binary halides.

Development excludes the blind formulas before fitting every transformer, clusterer,
distance scale, and empirical flexibility mapping.  Reveal requires the exact frozen
development manifest and only then transforms/ranks the isolated structures.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
from functools import wraps
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import re
import stat
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from joblib import Parallel, delayed, dump, load
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Structure
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler


CHEMISTRY_FEATURES = [
    "chem__x_over_m",
    "chem__ionic_radius_m_A",
    "chem__radius_x_A",
    "chem__q_over_r",
    "chem__q_over_r2",
    "chem__radius_ratio_m_x",
    "chem__chi_m_pauling",
    "chem__chi_x_pauling",
    "chem__delta_chi_x_m",
    "chem__halogen_polarizability_a0_3",
    "chem__atomic_number_m",
    "chem__group_m",
    "chem__period_m",
    "chem__n_common_positive_oxidation_states",
    "chem__n_positive_oxidation_states",
    "chem__n_shannon_coordination_environments",
    "chem__softness_proxy_px_over_field",
]

TOPOLOGY_FEATURES = [
    "vesta__dim",
    "vesta__Xcn",
    "vesta__Xsh",
    "vesta__Pcn",
    "local__cn_mean",
    "local__cn_std",
    "local__cn_entropy",
    "local__cn3_fraction",
    "local__cn4_fraction",
    "local__cn5_fraction",
    "local__cn6_fraction",
    "local__cn7_fraction",
    "local__cn8_fraction",
    "local__bond_length_mean_A",
    "local__bond_distortion_mean",
    "local__angle_variance_deg2",
    "local__polyhedron_volume_cv",
    "local__tetra_like_fraction",
    "local__trigonal_bipyramid_like_fraction",
    "local__square_pyramid_like_fraction",
    "local__octa_like_fraction",
    "local__irregular_fraction",
    "local__geometry_classifiable_fraction",
    "topology__corner_share_fraction",
    "topology__edge_share_fraction",
    "topology__face_share_fraction",
    "topology__connection_type_entropy",
    "topology__polyhedron_network_dimension",
    "topology__edge_network_dimension",
    "topology__center_degree_mean_all_centers",
    "topology__center_degree_std_all_centers",
    "topology__polyhedron_eligible_fraction",
    "topology__halogen_degree_mean",
    "topology__bridge_halogen_fraction",
    "topology__terminal_halogen_fraction",
    "topology__unbonded_halogen_fraction",
    "topology__bridge_terminal_coexistence",
    "topology__structure_component_fraction_max",
    "topology__dimension_coexistence_count",
    "topology__dim0_fraction",
    "topology__dim1_fraction",
    "topology__dim2_fraction",
    "topology__dim3_fraction",
    "tri__is_applicable",
    "tri__layer_coverage_fraction",
    "tri__center_pass_fraction",
    "tri__psi6_median",
    "tri__psi6_p10",
    "tri__psi6_iqr",
    "tri__psi4_median",
    "tri__angular_gap_rmse_deg",
    "tri__unique_same_layer_core6_fraction",
    "tri__xx_inplane_mean_A",
    "tri__xx_inplane_cv",
    "tri__buckling_mad_A",
    "tri__buckling_over_xx",
    "tri__normal_coherence",
    "stack__is_applicable",
    "stack__coverage_fraction",
    "stack__spacing_mean_A",
    "stack__spacing_cv",
    "stack__spacing_over_xx",
    "stack__inter_intra_xx_ratio",
]

FLEX_FEATURES = [
    "flex__volume_per_atom_A3",
    "flex__density_g_cm3",
    "flex__packing_fraction_proxy",
    "flex__voronoi_volume_mean_A3",
    "flex__voronoi_volume_cv",
    "flex__max_cavity_clearance_proxy_A",
    "flex__cavity_clearance_std_A",
    "flex__lattice_anisotropy",
    "flex__xx_min_A",
    "flex__xx_distance_cv",
]

FLEXIBILITY_SCORE_INPUTS = [
    "topology__bridge_terminal_coexistence",
    "local__bond_distortion_mean",
    "local__cn_entropy",
    "flex__voronoi_volume_cv",
    "topology__connection_type_entropy",
    "vesta__dim",
]
FLEXIBILITY_PERCENTILE_OPTIMA = {
    "local__bond_distortion_mean": 0.60,
    "local__cn_entropy": 0.60,
    "flex__voronoi_volume_cv": 0.60,
    "topology__connection_type_entropy": 0.60,
}
FLEXIBILITY_PERCENTILE_SIGMA = 0.25

ID_RE = re.compile(r"(mp-\d+)")
FROZEN_MODEL_NAME = "frozen_model.joblib"
PREDEVELOPMENT_COMMITMENT_NAME = "predevelopment_commitment.json"
DEVELOPMENT_RESULT_LOCK_NAME = ".discovery_pipeline.lock"
PREDEVELOPMENT_GUARDED_OUTPUTS = (
    "ranked_structures.csv",
    "candidate_portfolio.csv",
    "leave_one_positive_out.csv",
    "leave_one_positive_out.summary.json",
    "consensus_fit_matrices.npz",
    "umap_visualization.csv",
    "umap_prototype_neighborhoods.png",
    "hdbscan_labels_topology.csv",
    "hdbscan_labels_chemistry.csv",
    "hdbscan_labels_soap.csv",
    FROZEN_MODEL_NAME,
    "frozen_manifest.json",
)
EXPECTED_SOAP_R_CUT_VARIANTS_A = (5.0, 6.0, 7.0)
DISCOVERY_SCHEMA_VERSION = 4
ROBUSTNESS_WEIGHT_KEYS = (
    "topology",
    "soap",
    "chemistry",
    "flexibility",
    "stability",
)
FEATURE_EXTRACTOR_SOURCE = Path(__file__).with_name("extract_features.py")
VESTA_BONDING_SOURCE = Path(__file__).with_name("bonding_vesta.py")
TRIANGULAR_REFERENCE_RELATIVE_PATH = (
    "experiments/triangular_lattice/t23_reference.py"
)
TRIANGULAR_REFERENCE_SOURCE = (
    Path(__file__).resolve().parents[2] / TRIANGULAR_REFERENCE_RELATIVE_PATH
)
FEATURE_BUNDLE_FILES = (
    "interpretable_features.csv",
    "soap_rows.csv",
    "soap_pseudo_mx.npy",
    "structure_exclusions.csv",
    "provenance.json",
)
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)


@contextmanager
def development_result_lock(
    directory: Path, *, create_directory: bool = False
) -> Iterable[Path]:
    """Hold the single-writer lock shared by develop, freeze-gate, and reveal."""

    directory = Path(directory)
    if create_directory:
        directory.mkdir(parents=True, exist_ok=True)
    elif not directory.is_dir():
        raise FileNotFoundError(f"development result directory does not exist: {directory}")
    lock_path = directory / DEVELOPMENT_RESULT_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o644)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError(f"development lock is not a regular file: {lock_path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another develop/reveal process holds the development lock: {lock_path}"
            ) from exc
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def locked_discovery_run(function):
    """Decorate the complete discovery run with its development-directory lock."""

    @wraps(function)
    def wrapped(args: argparse.Namespace):
        if args.stage == "develop":
            directory = Path(args.outdir)
            create_directory = True
        elif args.stage == "reveal":
            if args.frozen_manifest is None:
                raise ValueError("reveal requires --frozen-manifest")
            directory = Path(args.frozen_manifest).parent
            create_directory = False
        else:
            raise ValueError(f"unsupported discovery stage: {args.stage!r}")
        with development_result_lock(
            directory, create_directory=create_directory
        ):
            return function(args)

    return wrapped


def canonical_formula(value: str) -> str:
    return Composition(str(value)).reduced_formula


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def soap_variant_key(r_cut_A: float) -> str:
    value = float(r_cut_A)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"SOAP r_cut must be positive and finite, got {r_cut_A!r}")
    return f"r_cut_{value:.1f}A".replace(".", "p")


def validate_soap_cutoff_config(config: dict[str, Any]) -> tuple[float, ...]:
    """Require identical pre-registered extraction and robustness cutoff lists."""

    soap = config.get("soap")
    robustness = config.get("robustness")
    if not isinstance(soap, dict) or not isinstance(robustness, dict):
        raise ValueError("config must contain soap and robustness objects")
    try:
        extraction = tuple(float(value) for value in soap["r_cut_variants_A"])
        sensitivity = tuple(float(value) for value in robustness["soap_r_cut_A"])
        base = float(soap["r_cut_A"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "SOAP cutoff contract requires numeric soap.r_cut_A, "
            "soap.r_cut_variants_A and robustness.soap_r_cut_A"
        ) from exc
    if extraction != sensitivity:
        raise ValueError(
            "SOAP extraction cutoffs must exactly equal robustness SOAP cutoffs"
        )
    if extraction != EXPECTED_SOAP_R_CUT_VARIANTS_A:
        raise ValueError(
            "pre-registered SOAP r_cut variants must equal "
            f"{list(EXPECTED_SOAP_R_CUT_VARIANTS_A)}"
        )
    if not math.isclose(base, 6.0, abs_tol=1e-12) or base not in extraction:
        raise ValueError("base SOAP r_cut must be the registered 6.0 A variant")
    return extraction


def validate_formal_feature_schema() -> None:
    """Reject repeated formal columns and the known exact triangular alias."""

    for name, columns in {
        "chemistry": CHEMISTRY_FEATURES,
        "topology": TOPOLOGY_FEATURES,
        "flexibility_audit": FLEX_FEATURES,
        "flexibility_score": FLEXIBILITY_SCORE_INPUTS,
    }.items():
        duplicates = sorted(
            column for column, count in Counter(columns).items() if count > 1
        )
        if duplicates:
            raise RuntimeError(f"duplicate {name} feature columns: {duplicates}")
    exact_aliases = {
        "tri__applicable_center_fraction",
        "tri__unique_same_layer_core6_fraction",
    }
    if exact_aliases <= set(TOPOLOGY_FEATURES):
        raise RuntimeError(
            "formal topology cannot include both identical core-6 coverage aliases"
        )


def _normalized_robustness_weights(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict) or set(payload) != set(ROBUSTNESS_WEIGHT_KEYS):
        raise RuntimeError(
            "robustness weights must contain exactly "
            f"{list(ROBUSTNESS_WEIGHT_KEYS)}"
        )
    weights = {key: float(payload[key]) for key in ROBUSTNESS_WEIGHT_KEYS}
    if not all(np.isfinite(value) and value >= 0.0 for value in weights.values()):
        raise RuntimeError("robustness weights must be finite and non-negative")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
        raise RuntimeError("robustness weights must sum to one")
    return weights


def registered_robustness_scenario_specs(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the exact ordered 2x3x3x3 scenario identity contract."""

    cfg = config.get("robustness")
    if not isinstance(cfg, dict):
        raise RuntimeError("config has no robustness object")
    soap_r_cuts = validate_soap_cutoff_config(config)
    pca_variances = tuple(float(value) for value in cfg.get("pca_variance", []))
    distance_quantiles = tuple(
        float(value) for value in cfg.get("distance_scale_quantile", [])
    )
    if (
        len(pca_variances) != 2
        or len(set(pca_variances)) != 2
        or not all(np.isfinite(value) and 0.0 < value <= 1.0 for value in pca_variances)
    ):
        raise RuntimeError("robustness PCA contract requires two unique values in (0, 1]")
    if (
        len(distance_quantiles) != 3
        or len(set(distance_quantiles)) != 3
        or not all(
            np.isfinite(value) and 0.0 < value < 1.0
            for value in distance_quantiles
        )
    ):
        raise RuntimeError(
            "robustness distance-scale contract requires three unique quantiles in (0, 1)"
        )
    weight_sets = cfg.get("weight_sets")
    if not isinstance(weight_sets, list) or len(weight_sets) != 3:
        raise RuntimeError("robustness.weight_sets must contain exactly three entries")
    normalized_weight_sets: list[tuple[str, dict[str, float]]] = []
    for item in weight_sets:
        if not isinstance(item, dict) or set(item) != {"name", "weights"}:
            raise RuntimeError(
                "each robustness weight set must contain exactly name and weights"
            )
        normalized_weight_sets.append(
            (str(item["name"]), _normalized_robustness_weights(item["weights"]))
        )
    names = [name for name, _weights in normalized_weight_sets]
    weight_hashes = [
        stable_json_hash(weights) for _name, weights in normalized_weight_sets
    ]
    if (
        any(not name for name in names)
        or len(names) != len(set(names))
        or len(weight_hashes) != len(set(weight_hashes))
    ):
        raise RuntimeError(
            "robustness weight sets must have unique non-empty names and unique vectors"
        )
    specs = [
        {
            "pca_variance": float(pca_variance),
            "distance_scale_quantile": float(distance_quantile),
            "weight_name": weight_name,
            "weights": dict(weights),
            "soap_variant": soap_variant_key(r_cut_A),
            "soap_r_cut_A": float(r_cut_A),
        }
        for pca_variance, distance_quantile, (weight_name, weights), r_cut_A in itertools.product(
            pca_variances,
            distance_quantiles,
            normalized_weight_sets,
            soap_r_cuts,
        )
    ]
    signatures = [stable_json_hash(spec) for spec in specs]
    if len(specs) != 54 or len(signatures) != len(set(signatures)):
        raise RuntimeError(
            "robustness scenario identity contract must be exactly 54 unique "
            "ordered Cartesian-product entries"
        )
    return specs


def validate_frozen_robustness_scenarios(
    payload: Any, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Fail closed unless frozen fitted scenarios exactly extend the registered grid."""

    if not isinstance(payload, list):
        raise RuntimeError("frozen robustness scenarios must be a list")
    expected = registered_robustness_scenario_specs(config)
    identity_keys = set(expected[0])
    fitted_keys = {
        "topology_components",
        "chemistry_components",
        "tau_topology",
        "tau_chemistry",
        "top_score_threshold",
    }
    normalized: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict) or set(item) != identity_keys | fitted_keys:
            raise RuntimeError(
                "each frozen robustness scenario must contain exactly the registered "
                "identity and fitted-state fields"
            )
        identity = {
            "pca_variance": float(item["pca_variance"]),
            "distance_scale_quantile": float(item["distance_scale_quantile"]),
            "weight_name": str(item["weight_name"]),
            "weights": _normalized_robustness_weights(item["weights"]),
            "soap_variant": str(item["soap_variant"]),
            "soap_r_cut_A": float(item["soap_r_cut_A"]),
        }
        identities.append(identity)
        raw_topology_components = item["topology_components"]
        raw_chemistry_components = item["chemistry_components"]
        if (
            isinstance(raw_topology_components, bool)
            or isinstance(raw_chemistry_components, bool)
            or not float(raw_topology_components).is_integer()
            or not float(raw_chemistry_components).is_integer()
        ):
            raise RuntimeError("frozen robustness component counts must be integers")
        topology_components = int(raw_topology_components)
        chemistry_components = int(raw_chemistry_components)
        tau_topology = float(item["tau_topology"])
        tau_chemistry = float(item["tau_chemistry"])
        threshold = float(item["top_score_threshold"])
        if topology_components <= 0 or chemistry_components <= 0:
            raise RuntimeError("frozen robustness component counts must be positive")
        if not (
            np.isfinite(tau_topology)
            and tau_topology > 0.0
            and np.isfinite(tau_chemistry)
            and tau_chemistry > 0.0
        ):
            raise RuntimeError("frozen robustness distance scales must be finite and positive")
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise RuntimeError("frozen robustness threshold must be finite and in [0, 1]")
        normalized.append(
            {
                **identity,
                "topology_components": topology_components,
                "chemistry_components": chemistry_components,
                "tau_topology": tau_topology,
                "tau_chemistry": tau_chemistry,
                "top_score_threshold": threshold,
            }
        )
    if identities != expected:
        raise RuntimeError(
            "frozen robustness scenarios do not exactly equal the unique registered "
            "ordered Cartesian-product contract"
        )
    return normalized


def verify_manifest_robustness_contract(
    manifest: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    scenarios = validate_frozen_robustness_scenarios(
        state.get("robustness_scenarios"), config
    )
    summary = manifest.get("robustness")
    if not isinstance(summary, dict):
        raise RuntimeError("frozen manifest has no robustness summary")
    actual_hash = stable_json_hash(scenarios)
    if summary.get("n_runs") != len(scenarios):
        raise RuntimeError("manifest robustness run count differs from frozen state")
    if summary.get("scenario_contract_sha256") != actual_hash:
        raise RuntimeError("manifest robustness hash differs from frozen state scenarios")
    return scenarios


def provenance_soap_variant_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize the SOAP matrix inventory recorded by extraction."""

    raw = payload.get("soap_variants")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("feature provenance has no SOAP cutoff-variant inventory")
    records: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise RuntimeError("SOAP cutoff-variant provenance records must be objects")
        required = {
            "variant",
            "r_cut_A",
            "filename",
            "shape",
            "sha256",
            "l2_normalized_rows",
        }
        missing = sorted(required - set(item))
        if missing:
            raise RuntimeError(f"SOAP cutoff-variant provenance is missing {missing}")
        r_cut_A = float(item["r_cut_A"])
        variant = soap_variant_key(r_cut_A)
        filename = str(item["filename"])
        if Path(filename).name != filename or not filename.endswith(".npy"):
            raise RuntimeError(f"unsafe SOAP cutoff-variant filename: {filename!r}")
        shape = item["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(value, bool) or int(value) <= 0 for value in shape)
        ):
            raise RuntimeError(f"invalid SOAP cutoff-variant shape: {shape!r}")
        if item["variant"] != variant:
            raise RuntimeError(
                f"SOAP variant key/r_cut mismatch: {item['variant']!r} != {variant!r}"
            )
        if item["l2_normalized_rows"] is not True:
            raise RuntimeError("SOAP cutoff-variant rows must be declared L2-normalized")
        digest = str(item["sha256"])
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError("invalid SOAP cutoff-variant SHA256")
        records.append(
            {
                "variant": variant,
                "r_cut_A": r_cut_A,
                "filename": filename,
                "shape": [int(shape[0]), int(shape[1])],
                "sha256": digest,
                "l2_normalized_rows": True,
            }
        )
    for field in ["variant", "r_cut_A", "filename"]:
        values = [record[field] for record in records]
        if len(values) != len(set(values)):
            raise RuntimeError(f"duplicate SOAP cutoff-variant {field}")
    base = float(payload.get("soap_base_r_cut_A", np.nan))
    base_records = [
        record for record in records if math.isclose(record["r_cut_A"], base, abs_tol=1e-12)
    ]
    if len(base_records) != 1 or base_records[0]["filename"] != "soap_pseudo_mx.npy":
        raise RuntimeError("SOAP base cutoff must map uniquely to soap_pseudo_mx.npy")
    if payload.get("soap_shape") != base_records[0]["shape"]:
        raise RuntimeError("legacy SOAP shape differs from the base cutoff-variant shape")
    if payload.get("soap_array_sha256") != base_records[0]["sha256"]:
        raise RuntimeError("legacy SOAP hash differs from the base cutoff-variant hash")
    contract = [
        {
            "variant": record["variant"],
            "r_cut_A": record["r_cut_A"],
            "filename": record["filename"],
            "width": record["shape"][1],
            "l2_normalized_rows": True,
        }
        for record in records
    ]
    if payload.get("soap_variants_contract_sha256") != stable_json_hash(contract):
        raise RuntimeError("SOAP cutoff-variant contract hash mismatch")
    return records


def verify_soap_provenance_config(
    payload: dict[str, Any], config: dict[str, Any]
) -> None:
    expected = validate_soap_cutoff_config(config)
    records = provenance_soap_variant_records(payload)
    observed = tuple(record["r_cut_A"] for record in records)
    if observed != expected:
        raise RuntimeError(
            f"feature SOAP variants differ from config: {observed} != {expected}"
        )
    if not math.isclose(
        float(payload.get("soap_base_r_cut_A", np.nan)),
        float(config["soap"]["r_cut_A"]),
        abs_tol=1e-12,
    ):
        raise RuntimeError("feature SOAP base cutoff differs from config")


def feature_bundle_hashes(path: Path) -> dict[str, str]:
    provenance_path = path / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"feature bundle has no provenance: {path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    variant_files = [
        record["filename"] for record in provenance_soap_variant_records(provenance)
    ]
    bundle_files = list(dict.fromkeys([*FEATURE_BUNDLE_FILES, *variant_files]))
    missing = [name for name in bundle_files if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete feature bundle {path}: {missing}")
    return {name: sha256_file(path / name) for name in bundle_files}


def load_provenance(path: Path) -> dict[str, Any]:
    return json.loads((path / "provenance.json").read_text(encoding="utf-8"))


def triangular_reference_record(payload: dict[str, Any]) -> dict[str, str]:
    """Validate the immutable reference-snapshot relationship in provenance."""

    record = payload.get("triangular_layer_reference")
    if not isinstance(record, dict):
        raise RuntimeError("feature provenance has no triangular-layer reference")
    expected = {
        "path": TRIANGULAR_REFERENCE_RELATIVE_PATH,
        "relationship": "adapted_reference_not_executed",
    }
    mismatches = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    digest = record.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        mismatches["sha256"] = (digest, "64 lowercase hexadecimal characters")
    if mismatches:
        raise RuntimeError(f"invalid triangular-layer reference record: {mismatches}")
    return {
        "path": str(record["path"]),
        "sha256": str(record["sha256"]),
        "relationship": str(record["relationship"]),
    }


def provenance_contract(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") or {}
    soap_shape = payload.get("soap_shape") or [None, None]
    soap_variants = provenance_soap_variant_records(payload)
    soap_variant_contract = [
        {
            "variant": record["variant"],
            "r_cut_A": record["r_cut_A"],
            "filename": record["filename"],
            "width": record["shape"][1],
            "l2_normalized_rows": True,
        }
        for record in soap_variants
    ]
    return {
        "schema_version": payload.get("schema_version"),
        "extractor_sha256": payload.get("extractor_sha256"),
        "bonding_module_sha256": payload.get("bonding_module_sha256"),
        "config_sha256": payload.get("config_sha256"),
        "metrics_source_commitment_sha256": payload.get(
            "metrics_source_commitment_sha256"
        ),
        "metrics_cif_inventory_sha256": payload.get("metrics_cif_inventory_sha256"),
        "metadata_source_commitment_sha256": metadata.get(
            "metadata_source_commitment_sha256"
        ),
        "metadata_source_columns_sha256": metadata.get(
            "metadata_source_columns_sha256"
        ),
        "bonding": payload.get("bonding"),
        "triangular_layer_reference": triangular_reference_record(payload),
        "identity_source": payload.get("identity_source"),
        "filename_formula_mismatch_policy": payload.get(
            "filename_formula_mismatch_policy"
        ),
        "full_cif_content_inventory_sha256": payload.get(
            "full_cif_content_inventory_sha256"
        ),
        "full_cif_identity_inventory_sha256": payload.get(
            "full_cif_identity_inventory_sha256"
        ),
        "n_inventory_structures_before_isolation": payload.get(
            "n_inventory_structures_before_isolation"
        ),
        "feature_columns_sha256": payload.get("feature_columns_sha256"),
        "soap_width": soap_shape[1] if len(soap_shape) > 1 else None,
        "soap_base_r_cut_A": payload.get("soap_base_r_cut_A"),
        "soap_variant_contract": soap_variant_contract,
        "soap_variants_contract_sha256": payload.get(
            "soap_variants_contract_sha256"
        ),
    }


def feature_source_commitment(payload: dict[str, Any]) -> dict[str, str | None]:
    """Return the immutable inputs that define a feature-extraction run."""
    metadata = payload.get("metadata") or {}
    triangular_reference = triangular_reference_record(payload)
    return {
        "config_sha256": payload.get("config_sha256"),
        "extractor_sha256": payload.get("extractor_sha256"),
        "bonding_module_sha256": payload.get("bonding_module_sha256"),
        "triangular_reference_sha256": triangular_reference["sha256"],
        "metrics_source_commitment_sha256": payload.get(
            "metrics_source_commitment_sha256"
        ),
        "metadata_source_commitment_sha256": metadata.get(
            "metadata_source_commitment_sha256"
        ),
    }


def current_feature_source_commitment(
    args: argparse.Namespace,
) -> dict[str, str | None]:
    """Hash the live sources rather than trusting paths recorded by a caller."""
    metrics = getattr(args, "metrics", None)
    if metrics is None:
        raise RuntimeError("feature source verification requires --metrics")
    metrics = Path(metrics)
    if not metrics.is_file():
        raise FileNotFoundError(f"metrics source does not exist: {metrics}")
    metadata = getattr(args, "metadata", None)
    if metadata is not None:
        metadata = Path(metadata)
        if not metadata.is_file():
            raise FileNotFoundError(f"metadata source does not exist: {metadata}")
    source_paths = {
        "config": Path(args.config),
        "extractor": FEATURE_EXTRACTOR_SOURCE,
        "bonding module": VESTA_BONDING_SOURCE,
        "triangular reference": TRIANGULAR_REFERENCE_SOURCE,
    }
    missing = [label for label, path in source_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"feature source file(s) missing: {missing}")
    return {
        "config_sha256": sha256_file(source_paths["config"]),
        "extractor_sha256": sha256_file(source_paths["extractor"]),
        "bonding_module_sha256": sha256_file(source_paths["bonding module"]),
        "triangular_reference_sha256": sha256_file(
            source_paths["triangular reference"]
        ),
        "metrics_source_commitment_sha256": sha256_file(metrics),
        "metadata_source_commitment_sha256": (
            sha256_file(metadata) if metadata is not None else None
        ),
    }


def verify_feature_source_commitment(
    args: argparse.Namespace,
    provenance: dict[str, Any],
    *,
    context: str,
) -> dict[str, str | None]:
    """Fail closed if any live extraction input differs from provenance."""
    expected = feature_source_commitment(provenance)
    live = current_feature_source_commitment(args)
    missing = sorted(
        key
        for key, value in expected.items()
        if value is None and key != "metadata_source_commitment_sha256"
    )
    if (
        provenance.get("metadata") is not None
        and expected["metadata_source_commitment_sha256"] is None
    ):
        missing.append("metadata_source_commitment_sha256")
    mismatches = {
        key: (expected.get(key), value)
        for key, value in live.items()
        if expected.get(key) != value
    }
    if missing:
        mismatches["missing_provenance_commitments"] = (missing, [])
    if mismatches:
        raise RuntimeError(f"{context} feature source commitment mismatch: {mismatches}")
    return live


def predevelopment_live_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Build the live, outcome-free input commitment used before any LOPO run."""
    provenance = load_provenance(args.features_dir)
    source_commitment = verify_feature_source_commitment(
        args, provenance, context="predevelopment"
    )
    bundle = feature_bundle_hashes(args.features_dir)
    live_cif_inventory = full_cif_content_inventory_hash(args.cif_dir)
    committed_cif_inventory = provenance.get("full_cif_content_inventory_sha256")
    if committed_cif_inventory != live_cif_inventory:
        raise RuntimeError(
            "predevelopment CIF source commitment mismatch: "
            f"{(committed_cif_inventory, live_cif_inventory)}"
        )
    extratrees = getattr(args, "extratrees_scores", None)
    if extratrees is not None:
        extratrees = Path(extratrees)
        if not extratrees.is_file():
            raise FileNotFoundError(f"ExtraTrees score source does not exist: {extratrees}")
    return {
        "config_sha256": source_commitment["config_sha256"],
        "extractor_sha256": source_commitment["extractor_sha256"],
        "bonding_module_sha256": source_commitment["bonding_module_sha256"],
        "triangular_reference_sha256": source_commitment[
            "triangular_reference_sha256"
        ],
        "run_discovery_sha256": sha256_file(Path(__file__)),
        "metrics_source_commitment_sha256": source_commitment[
            "metrics_source_commitment_sha256"
        ],
        "metadata_source_commitment_sha256": source_commitment[
            "metadata_source_commitment_sha256"
        ],
        "blind_isolation_sha256": sha256_file(args.blind_file),
        "development_feature_provenance_sha256": sha256_file(
            args.features_dir / "provenance.json"
        ),
        "development_feature_bundle_sha256": bundle,
        "development_feature_bundle_commitment_sha256": stable_json_hash(bundle),
        "development_feature_contract_sha256": stable_json_hash(
            provenance_contract(provenance)
        ),
        "development_feature_source_commitment_sha256": stable_json_hash(
            source_commitment
        ),
        "full_cif_content_inventory_sha256": live_cif_inventory,
        "extratrees_scores_sha256": (
            sha256_file(extratrees) if extratrees is not None else None
        ),
        "environment_versions": environment_versions(),
    }


def _predevelopment_payload_hash(payload: dict[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("commitment_payload_sha256", None)
    return stable_json_hash(unhashed)


def verify_predevelopment_commitment(
    args: argparse.Namespace, path: Path
) -> dict[str, Any]:
    """Verify the sealed pre-LOPO commitment against every live input."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"predevelopment commitment is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"predevelopment commitment is unreadable: {path}") from exc
    mismatches: dict[str, Any] = {}
    if payload.get("schema_version") != DISCOVERY_SCHEMA_VERSION:
        mismatches["schema_version"] = (
            payload.get("schema_version"),
            DISCOVERY_SCHEMA_VERSION,
        )
    if payload.get("stage") != "sealed_before_development_fit_and_lopo":
        mismatches["stage"] = (
            payload.get("stage"),
            "sealed_before_development_fit_and_lopo",
        )
    created = payload.get("created_at_utc")
    try:
        parsed = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        offset = parsed.utcoffset()
        if parsed.tzinfo is None or offset is None or offset.total_seconds() != 0:
            raise ValueError("timestamp is not UTC")
    except (TypeError, ValueError) as exc:
        mismatches["created_at_utc"] = (created, f"valid timezone-aware timestamp: {exc}")
    stored_hash = payload.get("commitment_payload_sha256")
    actual_hash = _predevelopment_payload_hash(payload)
    if stored_hash != actual_hash:
        mismatches["commitment_payload_sha256"] = (stored_hash, actual_hash)
    live = predevelopment_live_inputs(args)
    for key, value in live.items():
        if payload.get(key) != value:
            mismatches[key] = (payload.get(key), value)
    expected_keys = set(live) | {
        "schema_version",
        "stage",
        "created_at_utc",
        "commitment_payload_sha256",
    }
    if set(payload) != expected_keys:
        mismatches["payload_keys"] = (
            sorted(payload),
            sorted(expected_keys),
        )
    if mismatches:
        raise RuntimeError(f"predevelopment commitment verification failed: {mismatches}")
    return payload


def ensure_predevelopment_commitment(args: argparse.Namespace) -> Path:
    """Create once, never overwrite, and verify the commitment before development."""
    path = args.outdir / PREDEVELOPMENT_COMMITMENT_NAME
    if path.exists():
        verify_predevelopment_commitment(args, path)
        return path
    preexisting_outputs = [
        name
        for name in PREDEVELOPMENT_GUARDED_OUTPUTS
        if (args.outdir / name).exists()
    ]
    if preexisting_outputs:
        raise RuntimeError(
            "development output exists without a prior predevelopment commitment: "
            f"{preexisting_outputs}"
        )
    payload = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "stage": "sealed_before_development_fit_and_lopo",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **predevelopment_live_inputs(args),
    }
    payload["commitment_payload_sha256"] = _predevelopment_payload_hash(payload)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        verify_predevelopment_commitment(args, path)
        return path
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        # A partial exclusive file is intentionally retained: subsequent runs fail
        # closed instead of silently replacing a possibly pre-LOPO commitment.
        raise
    verify_predevelopment_commitment(args, path)
    return path


def feature_inventory_keys(path: Path) -> set[str]:
    feature_rows = pd.read_csv(path / "soap_rows.csv", usecols=["cif_file"])
    exclusions = pd.read_csv(path / "structure_exclusions.csv", usecols=["cif_file"])
    keys = feature_rows["cif_file"].astype(str).tolist() + exclusions["cif_file"].astype(str).tolist()
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate CIF identity across success/exclusion ledgers in {path}")
    return set(keys)


def cif_inventory_hash(frame: pd.DataFrame, cif_dir: Path) -> str:
    require_columns(frame, ["cif_file", "cif_sha256"])
    records = []
    for cif_file, expected in frame[["cif_file", "cif_sha256"]].itertuples(index=False):
        actual = sha256_file(cif_dir / str(cif_file))
        if str(expected) != actual:
            raise RuntimeError(f"CIF hash mismatch for frozen development key {cif_file}")
        records.append({"cif_file": str(cif_file), "sha256": actual})
    return stable_json_hash(sorted(records, key=lambda item: item["cif_file"]))


def full_cif_content_inventory_hash(cif_dir: Path) -> str:
    records = [
        {"cif_file": path.name, "cif_sha256": sha256_file(path)}
        for path in sorted(cif_dir.glob("*.cif"), key=lambda item: item.name)
    ]
    return stable_json_hash(records)


def environment_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "hdbscan",
        "umap-learn",
        "joblib",
        "pymatgen",
        "dscribe",
        "ase",
        "scipy",
        "networkx",
    ]
    versions = {name: importlib.metadata.version(name) for name in packages}
    versions["python"] = sys.version.split()[0]
    return versions


def require_columns(frame: pd.DataFrame, names: Iterable[str]) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"missing required feature columns: {missing}")


def bool_values(series: pd.Series, *, missing: bool) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(missing).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map({"true": True, "1": True, "false": False, "0": False})
    return mapped.fillna(missing).astype(bool)


@dataclass
class ViewTransform:
    features: list[str]
    imputer: SimpleImputer
    variance: VarianceThreshold
    power: PowerTransformer | None
    scaler: RobustScaler | None
    pca: PCA
    clip_value: float | None

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        features: list[str],
        fit_mask: np.ndarray,
        variance_fraction: float,
        use_power: bool,
        use_scaler: bool,
        clip_value: float | None,
    ) -> tuple["ViewTransform", np.ndarray]:
        require_columns(frame, features)
        raw = frame[features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        imputer = SimpleImputer(strategy="median")
        fit_imputed = imputer.fit_transform(raw[fit_mask])
        all_imputed = imputer.transform(raw)
        variance = VarianceThreshold(threshold=1e-12)
        fit_v = variance.fit_transform(fit_imputed)
        all_v = variance.transform(all_imputed)
        if fit_v.shape[1] == 0:
            raise ValueError("all features are constant after imputation")
        power: PowerTransformer | None = None
        if use_power:
            power = PowerTransformer(method="yeo-johnson", standardize=False)
            fit_v = power.fit_transform(fit_v)
            all_v = power.transform(all_v)
        scaler: RobustScaler | None = None
        if use_scaler:
            scaler = RobustScaler(quantile_range=(25.0, 75.0))
            fit_s = scaler.fit_transform(fit_v)
            all_s = scaler.transform(all_v)
        else:
            fit_s, all_s = fit_v, all_v
        if clip_value is not None:
            fit_s = np.clip(fit_s, -float(clip_value), float(clip_value))
            all_s = np.clip(all_s, -float(clip_value), float(clip_value))
        pca = PCA(n_components=variance_fraction, svd_solver="full", random_state=42)
        pca.fit(fit_s)
        return cls(features, imputer, variance, power, scaler, pca, clip_value), pca.transform(all_s)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame[self.features].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        values = self.variance.transform(self.imputer.transform(raw))
        if self.power is not None:
            values = self.power.transform(values)
        if self.scaler is not None:
            values = self.scaler.transform(values)
        if self.clip_value is not None:
            values = np.clip(values, -float(self.clip_value), float(self.clip_value))
        return self.pca.transform(values)


@dataclass
class PreparedData:
    frame: pd.DataFrame
    soap: np.ndarray
    development_count: int
    fit_mask: np.ndarray
    prototype_indices: dict[str, np.ndarray]
    transforms: dict[str, ViewTransform]
    flexibility_calibration: FlexibilityCalibration
    embeddings: dict[str, np.ndarray]
    soap_variants: dict[str, np.ndarray] | None = None


def select_prototype_indices(
    frame: pd.DataFrame, is_dev: np.ndarray, config: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Choose exactly one predeclared, deterministic structure anchor per formula."""

    policy = config.get("prototype_anchor_policy")
    expected = "lowest_energy_above_hull_then_formation_energy_then_cif"
    if policy != expected:
        raise ValueError(f"prototype_anchor_policy must be {expected!r}, got {policy!r}")
    anchors: dict[str, np.ndarray] = {}
    for key, value in config["positive_prototypes"].items():
        formula = canonical_formula(value)
        mask = is_dev & frame["formula"].eq(formula).to_numpy()
        mask &= frame["eligible_analysis"].to_numpy(bool)
        candidates = frame.loc[mask].copy()
        representatives = candidates.loc[candidates["is_structure_representative"]]
        if len(representatives):
            candidates = representatives
        if candidates.empty:
            raise ValueError(f"prototype structure missing or ineligible: {key}={formula}")
        hull = pd.to_numeric(
            candidates["energy_above_hull_used_eV_atom"], errors="coerce"
        ).fillna(np.inf)
        if "formation_energy_per_atom" in candidates:
            formation = pd.to_numeric(
                candidates["formation_energy_per_atom"], errors="coerce"
            ).fillna(np.inf)
        else:
            formation = pd.Series(np.inf, index=candidates.index)
        order = pd.DataFrame(
            {
                "hull": hull,
                "formation": formation,
                "cif_file": candidates["cif_file"].astype(str),
            },
            index=candidates.index,
        ).sort_values(["hull", "formation", "cif_file"], kind="mergesort")
        anchors[key] = np.asarray([int(order.index[0])], dtype=int)
    return anchors


def load_feature_dir(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(path / "interpretable_features.csv", low_memory=False)
    soap_rows = pd.read_csv(path / "soap_rows.csv")
    soap = np.load(path / "soap_pseudo_mx.npy")
    if len(frame) != len(soap_rows) or len(frame) != len(soap):
        raise ValueError(f"row mismatch in {path}")
    if not frame["cif_file"].equals(soap_rows["cif_file"]):
        raise ValueError(f"SOAP row order mismatch in {path}")
    frame["formula"] = frame["formula"].map(canonical_formula)
    return frame, np.asarray(soap, dtype=np.float64)


def load_soap_variant_matrices(
    path: Path,
    config: dict[str, Any],
    *,
    base_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load every pre-registered raw L2 SOAP matrix in the shared row order."""

    provenance = load_provenance(path)
    verify_soap_provenance_config(provenance, config)
    records = provenance_soap_variant_records(provenance)
    expected_rows = len(pd.read_csv(path / "soap_rows.csv", usecols=["cif_file"]))
    matrices: dict[str, np.ndarray] = {}
    for record in records:
        matrix = np.asarray(np.load(path / record["filename"]), dtype=np.float64)
        if list(matrix.shape) != record["shape"] or len(matrix) != expected_rows:
            raise RuntimeError(
                f"SOAP cutoff-variant shape mismatch for {record['filename']}"
            )
        if not np.isfinite(matrix).all():
            raise RuntimeError(f"non-finite SOAP values in {record['filename']}")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, rtol=2e-5, atol=2e-6):
            raise RuntimeError(
                f"SOAP cutoff-variant rows are not L2 normalized: {record['filename']}"
            )
        matrices[record["variant"]] = matrix
    base_key = soap_variant_key(float(config["soap"]["r_cut_A"]))
    if base_key not in matrices or not np.array_equal(matrices[base_key], base_matrix):
        raise RuntimeError("loaded base SOAP matrix differs from soap_pseudo_mx.npy")
    return matrices


def load_blind_formulas(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {canonical_formula(v) for v in payload["formulas"]}


def energy_column(frame: pd.DataFrame) -> str | None:
    candidates = [
        "energy_above_hull",
        "energy_above_hull_eV_atom",
        "energy_above_hull_eV_per_atom",
    ]
    return next((name for name in candidates if name in frame.columns), None)


def add_eligibility(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    eligibility = config["eligibility"]
    toxic = set(eligibility["strict_toxic_elements"])
    precious = set(eligibility["precious_elements"])
    out["risk_radioactive"] = out["center_element"].map(lambda s: bool(Element(str(s)).is_radioactive))
    out["risk_strict_toxic"] = out["center_element"].isin(toxic)
    out["flag_precious"] = out["center_element"].isin(precious)
    ecol = energy_column(out)
    if ecol is None:
        out["energy_above_hull_used_eV_atom"] = np.nan
    else:
        out["energy_above_hull_used_eV_atom"] = pd.to_numeric(out[ecol], errors="coerce")
    hull = out["energy_above_hull_used_eV_atom"]
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
    ordered_allowed = out["is_ordered"].astype(bool) | (
        not bool(eligibility["require_ordered_structure"])
    )
    out["eligible_chemistry"] = ~(out["risk_radioactive"] | out["risk_strict_toxic"])
    out["eligible_analysis"] = (
        out["feature_status"].eq("ok")
        & out["eligible_chemistry"]
        & stability_allowed
        & ordered_allowed
    )
    out["stability_score"] = np.where(
        hull.notna(), np.exp(-np.maximum(hull.fillna(0.0), 0.0) / 0.10), 0.50
    )
    return out


STRUCTURE_MATCHER_CONFIG_KEYS = frozenset(
    {
        "backend",
        "ltol",
        "stol",
        "angle_tol_deg",
        "primitive_cell",
        "scale",
        "attempt_supercell",
        "allow_subset",
        "require_descriptor_signature_match",
        "descriptor_signature_features",
        "descriptor_signature_round_decimals",
    }
)

DEDUPLICATION_SIGNATURE_FEATURES = (
    "vesta__dim",
    "vesta__Xcn",
    "vesta__Xsh",
    "vesta__Pcn",
    "local__cn_mean",
    "local__cn_std",
    "local__cn_min",
    "local__cn_max",
    "local__cn3_fraction",
    "local__cn4_fraction",
    "local__cn5_fraction",
    "local__cn6_fraction",
    "local__cn7_fraction",
    "local__cn8_fraction",
    "topology__polyhedron_network_dimension",
    "topology__edge_network_dimension",
    "topology__polyhedron_eligible_fraction",
    "topology__corner_share_fraction",
    "topology__edge_share_fraction",
    "topology__face_share_fraction",
    "topology__halogen_degree_mean",
    "topology__bridge_halogen_fraction",
    "topology__terminal_halogen_fraction",
    "topology__unbonded_halogen_fraction",
    "tri__is_applicable",
    "stack__is_applicable",
)


def validate_structure_matcher_config(matcher_config: dict[str, Any]) -> None:
    """Fail closed rather than silently ignore a deduplication configuration field."""

    if not isinstance(matcher_config, dict):
        raise ValueError("deduplication configuration must be an object")
    keys = set(matcher_config)
    missing = sorted(STRUCTURE_MATCHER_CONFIG_KEYS - keys)
    unknown = sorted(keys - STRUCTURE_MATCHER_CONFIG_KEYS)
    if missing or unknown:
        raise ValueError(
            "deduplication configuration keys differ from the frozen contract; "
            f"missing={missing}, unknown={unknown}"
        )
    if matcher_config["backend"] != "pymatgen.StructureMatcher":
        raise ValueError("deduplication backend must be 'pymatgen.StructureMatcher'")
    for name in ["ltol", "stol", "angle_tol_deg"]:
        value = matcher_config[name]
        if isinstance(value, bool):
            raise ValueError(f"deduplication {name} must be a positive finite number")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"deduplication {name} must be a positive finite number"
            ) from exc
        if not np.isfinite(numeric) or numeric <= 0:
            raise ValueError(f"deduplication {name} must be a positive finite number")
    for name in ["primitive_cell", "scale", "attempt_supercell", "allow_subset"]:
        if not isinstance(matcher_config[name], bool):
            raise ValueError(f"deduplication {name} must be a JSON boolean")
    if matcher_config["allow_subset"]:
        raise ValueError("StructureMatcher.group_structures requires allow_subset=false")
    if matcher_config["require_descriptor_signature_match"] is not True:
        raise ValueError("deduplication must require descriptor-signature agreement")
    if matcher_config["descriptor_signature_features"] != list(
        DEDUPLICATION_SIGNATURE_FEATURES
    ):
        raise ValueError(
            "deduplication descriptor_signature_features differ from the frozen contract"
        )
    decimals = matcher_config["descriptor_signature_round_decimals"]
    if isinstance(decimals, bool) or decimals != 8:
        raise ValueError("deduplication descriptor signature rounding must equal 8")


def _deduplication_descriptor_signature(
    row: pd.Series, matcher_config: dict[str, Any]
) -> tuple[float, ...]:
    """Return the frozen discrete VESTA/topology identity used after matching."""

    decimals = int(matcher_config["descriptor_signature_round_decimals"])
    values: list[float] = []
    for feature in DEDUPLICATION_SIGNATURE_FEATURES:
        try:
            value = float(row[feature])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"deduplication signature feature is unavailable: {feature}"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"deduplication signature feature is non-finite: {feature}"
            )
        rounded = round(value, decimals)
        values.append(0.0 if rounded == 0.0 else rounded)
    return tuple(values)


def _dedup_formula(
    rows: list[tuple[int, str]],
    cif_dir: Path,
    energy: dict[int, float],
    matcher_config: dict[str, Any],
) -> list[tuple[int, str, int, bool]]:
    validate_structure_matcher_config(matcher_config)
    structures: list[Structure] = []
    identity_by_object: dict[int, tuple[int, str]] = {}
    groups: list[list[tuple[int, str]]] = []
    for index, filename in sorted(rows, key=lambda item: item[1]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structure = Structure.from_file(cif_dir / filename)
        except Exception:
            groups.append([(index, filename)])
            continue
        structures.append(structure)
        identity_by_object[id(structure)] = (index, filename)
    if structures:
        matcher = StructureMatcher(
            ltol=float(matcher_config["ltol"]),
            stol=float(matcher_config["stol"]),
            angle_tol=float(matcher_config["angle_tol_deg"]),
            primitive_cell=matcher_config["primitive_cell"],
            scale=matcher_config["scale"],
            attempt_supercell=matcher_config["attempt_supercell"],
            allow_subset=matcher_config["allow_subset"],
        )
        matched = matcher.group_structures(structures, anonymous=False)
        expected_object_ids = [id(structure) for structure in structures]
        returned_object_ids = [id(structure) for group in matched for structure in group]
        if (
            len(expected_object_ids) != len(set(expected_object_ids))
            or len(returned_object_ids) != len(expected_object_ids)
            or set(returned_object_ids) != set(expected_object_ids)
        ):
            raise RuntimeError(
                "StructureMatcher.group_structures did not return each original "
                "Structure object exactly once; object-identity mapping is unsafe"
            )
        groups.extend(
            [[identity_by_object[id(structure)] for structure in group] for group in matched]
        )
    result = []
    groups = sorted(groups, key=lambda group: min(name for _, name in group))
    for group_number, group in enumerate(groups, start=1):
        def key(item: tuple[int, str]) -> tuple[float, str]:
            e = energy.get(item[0], np.nan)
            return (e if np.isfinite(e) else np.inf, item[1])

        representative_index = min(group, key=key)[0]
        group_id = f"sm-{group_number:03d}"
        for index, _filename in group:
            result.append((index, group_id, len(group), index == representative_index))
    return result


def add_structure_duplicates(
    frame: pd.DataFrame, cif_dir: Path, n_jobs: int, config: dict[str, Any]
) -> pd.DataFrame:
    out = frame.copy()
    if "eligible_analysis" not in out:
        raise ValueError("structure deduplication requires eligible_analysis")
    energy = out["energy_above_hull_used_eV_atom"].to_dict()
    # Only structures that can enter a fit, rank, or portfolio need expensive
    # same-formula StructureMatcher comparisons.  Keeping excluded structures as
    # explicit singleton representatives is lossless for the reported inventory
    # and avoids pathological all-pairs matching for large toxic/radioactive
    # formula families that are never eligible downstream.
    out["duplicate_fingerprint_group"] = [
        f"{formula}::excluded-singleton::{cif_file}"
        for formula, cif_file in zip(out["formula"], out["cif_file"])
    ]
    out["duplicate_fingerprint_group_size"] = 1
    out["is_structure_representative"] = True
    eligible = out.loc[out["eligible_analysis"].astype(bool)]
    tasks = [
        (formula, list(zip(group.index.tolist(), group["cif_file"].tolist())))
        for formula, group in eligible.groupby("formula", sort=True)
    ]
    grouped = (
        Parallel(n_jobs=n_jobs, prefer="processes", verbose=3)(
            delayed(_dedup_formula)(rows, cif_dir, energy, config["deduplication"])
            for _formula, rows in tasks
        )
        if tasks
        else []
    )
    records = [item for group in grouped for item in group]
    for index, group_id, size, is_rep in records:
        out.loc[index, "duplicate_fingerprint_group"] = f"{out.loc[index, 'formula']}::{group_id}"
        out.loc[index, "duplicate_fingerprint_group_size"] = int(size)
        out.loc[index, "is_structure_representative"] = bool(is_rep)

    # StructureMatcher deliberately tolerates uniform scale changes.  A fixed
    # VESTA cutoff can nevertheless make two such structures different for this
    # discovery problem (including a change in 0D/1D/2D/3D connectivity).  Split
    # every geometric group by a predeclared, rounded discrete descriptor
    # signature so real topology polymorphs are never discarded as duplicates.
    eligible_indices = out.index[out["eligible_analysis"].astype(bool)]
    initial_groups = out.loc[eligible_indices].groupby(
        "duplicate_fingerprint_group", sort=True
    )
    for initial_group, members in initial_groups:
        buckets: dict[tuple[float, ...], list[int]] = {}
        for index, row in members.sort_values("cif_file").iterrows():
            signature = _deduplication_descriptor_signature(
                row, config["deduplication"]
            )
            buckets.setdefault(signature, []).append(int(index))
        for number, signature in enumerate(sorted(buckets), start=1):
            indices = buckets[signature]

            def representative_key(index: int) -> tuple[float, str]:
                value = energy.get(index, np.nan)
                return (
                    float(value) if np.isfinite(value) else np.inf,
                    str(out.loc[index, "cif_file"]),
                )

            representative = min(indices, key=representative_key)
            refined_group = f"{initial_group}::sig-{number:03d}"
            out.loc[indices, "duplicate_fingerprint_group"] = refined_group
            out.loc[indices, "duplicate_fingerprint_group_size"] = len(indices)
            out.loc[indices, "is_structure_representative"] = False
            out.loc[representative, "is_structure_representative"] = True
    out["duplicate_fingerprint_group_size"] = out["duplicate_fingerprint_group_size"].astype(int)
    out["is_structure_representative"] = out["is_structure_representative"].astype(bool)
    return out


def empirical_percentile(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    train = np.sort(train[np.isfinite(train)])
    if not len(train):
        return np.full(len(values), 0.5)
    # Mid-rank percentiles avoid treating every member of a large discrete tie as
    # if it occupied the upper edge of that tie (for example, CN entropy == 0).
    lower = np.searchsorted(train, values, side="left")
    upper = np.searchsorted(train, values, side="right")
    return (lower + upper) / (2.0 * len(train))


@dataclass
class FlexibilityCalibration:
    """Frozen empirical maps for the static flexibility/reconstructability proxy."""

    distortion_reference: np.ndarray
    percentile_references: dict[str, np.ndarray]
    fill_values: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, fit_mask: np.ndarray) -> "FlexibilityCalibration":
        require_columns(frame, FLEXIBILITY_SCORE_INPUTS)
        distortion = pd.to_numeric(
            frame["local__bond_distortion_mean"], errors="coerce"
        ).to_numpy(float)
        distortion = np.where(np.isfinite(distortion), distortion, 0.0)
        references: dict[str, np.ndarray] = {}
        fill_values: dict[str, float] = {}
        for column in [
            "local__cn_entropy",
            "flex__voronoi_volume_cv",
            "topology__connection_type_entropy",
        ]:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            finite_fit = values[fit_mask & np.isfinite(values)]
            median = float(np.median(finite_fit)) if len(finite_fit) else 0.0
            filled = np.where(np.isfinite(values), values, median)
            fill_values[column] = median
            references[column] = np.sort(filled[fit_mask])
        return cls(
            distortion_reference=np.sort(distortion[fit_mask]),
            percentile_references=references,
            fill_values=fill_values,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        components = []
        coexist = pd.to_numeric(
            frame["topology__bridge_terminal_coexistence"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        components.append(np.clip(coexist, 0.0, 1.0))
        distortion = pd.to_numeric(
            frame["local__bond_distortion_mean"], errors="coerce"
        ).fillna(0.0).to_numpy(float)
        p_dist = empirical_percentile(self.distortion_reference, distortion)
        components.append(
            np.exp(
                -(
                    (p_dist - FLEXIBILITY_PERCENTILE_OPTIMA["local__bond_distortion_mean"])
                    ** 2
                )
                / (2 * FLEXIBILITY_PERCENTILE_SIGMA**2)
            )
        )
        for column, reference in self.percentile_references.items():
            values = pd.to_numeric(frame[column], errors="coerce").fillna(
                self.fill_values[column]
            ).to_numpy(float)
            percentile = empirical_percentile(reference, values)
            components.append(
                np.exp(
                    -((percentile - FLEXIBILITY_PERCENTILE_OPTIMA[column]) ** 2)
                    / (2 * FLEXIBILITY_PERCENTILE_SIGMA**2)
                )
            )
        dimension = pd.to_numeric(frame["vesta__dim"], errors="coerce").to_numpy(float)
        if not np.isfinite(dimension).all():
            raise ValueError("VESTA dimensionality contains missing/non-finite values")
        components.append(
            np.select(
                [dimension == 0, dimension == 1, dimension == 2],
                [0.80, 1.00, 0.85],
                default=0.30,
            )
        )
        return np.mean(np.vstack(components), axis=0)


def serialize_view_transform(transform: ViewTransform) -> dict[str, Any]:
    return {
        "features": transform.features,
        "imputer": transform.imputer,
        "variance": transform.variance,
        "power": transform.power,
        "scaler": transform.scaler,
        "pca": transform.pca,
        "clip_value": transform.clip_value,
    }


def deserialize_view_transform(payload: dict[str, Any]) -> ViewTransform:
    return ViewTransform(
        features=list(payload["features"]),
        imputer=payload["imputer"],
        variance=payload["variance"],
        power=payload["power"],
        scaler=payload["scaler"],
        pca=payload["pca"],
        clip_value=payload["clip_value"],
    )


def serialize_flexibility(calibration: FlexibilityCalibration) -> dict[str, Any]:
    return {
        "distortion_reference": calibration.distortion_reference,
        "percentile_references": calibration.percentile_references,
        "fill_values": calibration.fill_values,
    }


def deserialize_flexibility(payload: dict[str, Any]) -> FlexibilityCalibration:
    return FlexibilityCalibration(
        distortion_reference=np.asarray(payload["distortion_reference"], dtype=float),
        percentile_references={
            key: np.asarray(value, dtype=float)
            for key, value in payload["percentile_references"].items()
        },
        fill_values={key: float(value) for key, value in payload["fill_values"].items()},
    )


def prepare(
    development_dir: Path,
    reveal_dir: Path | None,
    config: dict[str, Any],
    cif_dir: Path,
    n_jobs: int,
    frozen_state: dict[str, Any] | None = None,
) -> PreparedData:
    validate_soap_cutoff_config(config)
    dev, dev_soap = load_feature_dir(development_dir)
    dev_soap_variants = load_soap_variant_matrices(
        development_dir, config, base_matrix=dev_soap
    )
    development_count = len(dev)
    if reveal_dir is not None:
        blind, blind_soap = load_feature_dir(reveal_dir)
        blind_soap_variants = load_soap_variant_matrices(
            reveal_dir, config, base_matrix=blind_soap
        )
        if set(dev_soap_variants) != set(blind_soap_variants):
            raise RuntimeError("development/reveal SOAP cutoff variants differ")
        overlap = set(dev["cif_file"]) & set(blind["cif_file"])
        if overlap:
            raise ValueError(f"development/reveal overlap: {sorted(overlap)[:5]}")
        frame = pd.concat([dev, blind], ignore_index=True)
        soap = np.vstack([dev_soap, blind_soap])
        soap_variants = {
            key: np.vstack([matrix, blind_soap_variants[key]])
            for key, matrix in dev_soap_variants.items()
        }
    else:
        frame, soap = dev, dev_soap
        soap_variants = dev_soap_variants
    frame = add_eligibility(frame, config)
    frame = add_structure_duplicates(frame, cif_dir, n_jobs, config)
    is_dev = np.arange(len(frame)) < development_count
    fit_mask = (
        is_dev
        & frame["eligible_analysis"].to_numpy(bool)
        & frame["is_structure_representative"].to_numpy(bool)
    )
    prototype_indices = select_prototype_indices(frame, is_dev, config)
    prep = config["preprocessing"]
    if prep.get("power_transform") != "yeo-johnson" or prep.get("scaler") != "robust":
        raise ValueError("only the frozen Yeo-Johnson + RobustScaler contract is supported")
    if prep.get("soap_scaler") != "none_after_l2_normalization":
        raise ValueError("SOAP must remain L2-normalized without a fitted scaler")
    robust_clip = float(prep["robust_scaled_clip"])
    soap_frame = pd.DataFrame(soap, columns=[f"soap_{i}" for i in range(soap.shape[1])])
    transforms: dict[str, ViewTransform]
    embeddings: dict[str, np.ndarray] = {}
    if frozen_state is None:
        transforms = {}
        transforms["topology"], embeddings["topology"] = ViewTransform.fit(
            frame,
            TOPOLOGY_FEATURES,
            fit_mask,
            float(prep["interpretable_pca_variance"]),
            use_power=True,
            use_scaler=True,
            clip_value=robust_clip,
        )
        transforms["chemistry"], embeddings["chemistry"] = ViewTransform.fit(
            frame,
            CHEMISTRY_FEATURES,
            fit_mask,
            float(prep["interpretable_pca_variance"]),
            use_power=True,
            use_scaler=True,
            clip_value=robust_clip,
        )
        transforms["soap"], embeddings["soap"] = ViewTransform.fit(
            soap_frame,
            list(soap_frame.columns),
            fit_mask,
            float(prep["soap_pca_variance"]),
            use_power=False,
            use_scaler=False,
            clip_value=None,
        )
        flexibility_calibration = FlexibilityCalibration.fit(frame, fit_mask)
        frame["flexibility_score"] = flexibility_calibration.transform(frame)
    else:
        if int(frozen_state["development_count"]) != development_count:
            raise RuntimeError("frozen development row count changed")
        if frame.loc[: development_count - 1, "cif_file"].tolist() != list(
            frozen_state["development_cif_files"]
        ):
            raise RuntimeError("frozen development row identity/order changed")
        if not np.array_equal(fit_mask[:development_count], frozen_state["fit_mask"]):
            raise RuntimeError("frozen development fit/representative mask changed")
        for key in prototype_indices:
            if not np.array_equal(prototype_indices[key], frozen_state["prototype_indices"][key]):
                raise RuntimeError(f"frozen prototype anchor changed: {key}")
        transforms = {
            key: deserialize_view_transform(value)
            for key, value in frozen_state["transforms"].items()
        }
        reveal_slice = slice(development_count, len(frame))
        for view, source in {
            "topology": frame,
            "chemistry": frame,
            "soap": soap_frame,
        }.items():
            dev_embedding = np.asarray(
                frozen_state["development_embeddings"][view], dtype=float
            )
            new_embedding = transforms[view].transform(source.iloc[reveal_slice])
            embeddings[view] = np.vstack([dev_embedding, new_embedding])
        flexibility_calibration = deserialize_flexibility(
            frozen_state["flexibility_calibration"]
        )
        dev_flex = np.asarray(frozen_state["development_flexibility_score"], dtype=float)
        reveal_flex = flexibility_calibration.transform(frame.iloc[reveal_slice])
        frame["flexibility_score"] = np.concatenate([dev_flex, reveal_flex])
    return PreparedData(
        frame=frame,
        soap=soap,
        development_count=development_count,
        fit_mask=fit_mask,
        prototype_indices=prototype_indices,
        transforms=transforms,
        flexibility_calibration=flexibility_calibration,
        embeddings=embeddings,
        soap_variants=soap_variants,
    )


def subspace(matrix: np.ndarray, name: str) -> np.ndarray:
    if name == "all" or matrix.shape[1] <= 2:
        return matrix
    if name == "even":
        return matrix[:, np.arange(matrix.shape[1]) % 2 == 0]
    if name == "odd":
        return matrix[:, np.arange(matrix.shape[1]) % 2 == 1]
    raise ValueError(f"unknown component subspace: {name}")


def hdbscan_runs(
    matrix: np.ndarray, fit_mask: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, list[hdbscan.HDBSCAN]]:
    fit_indices = np.flatnonzero(fit_mask)
    predict_indices = np.flatnonzero(~fit_mask)
    cfg = config["hdbscan"]
    grid = list(
        itertools.product(
            cfg["min_cluster_size"],
            cfg["min_samples"],
            cfg["cluster_selection_method"],
            cfg["component_subspaces"],
        )
    )
    labels_all, params, clusterers = [], [], []
    consensus_counts = np.zeros((len(fit_indices), len(fit_indices)), dtype=np.uint16)
    for min_size, min_samples, method, space in grid:
        view = subspace(matrix, space)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(min_size),
            min_samples=int(min_samples),
            cluster_selection_method=str(method),
            prediction_data=True,
            core_dist_n_jobs=1,
        )
        fit_labels = clusterer.fit_predict(view[fit_indices])
        labels = np.full(len(matrix), -1, dtype=int)
        labels[fit_indices] = fit_labels
        if len(predict_indices):
            predicted, _strength = hdbscan.approximate_predict(
                clusterer, view[predict_indices]
            )
            labels[predict_indices] = predicted
        labels_all.append(labels)
        clusterers.append(clusterer)
        params.append(
            {
                "min_cluster_size": int(min_size),
                "min_samples": int(min_samples),
                "cluster_selection_method": method,
                "component_subspace": space,
            }
        )
        for cluster_id in np.unique(fit_labels[fit_labels >= 0]):
            members = np.flatnonzero(fit_labels == cluster_id)
            consensus_counts[np.ix_(members, members)] += 1
    consensus = consensus_counts.astype(np.float32) / max(len(grid), 1)
    np.fill_diagonal(consensus, 1.0)
    return np.asarray(labels_all, dtype=int).T, params, consensus, clusterers


def prototype_consensus(
    labels: np.ndarray, prototype_indices: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    scores = {}
    for key, indices in prototype_indices.items():
        per_run = []
        for run in range(labels.shape[1]):
            proto_labels = set(labels[indices, run]) - {-1}
            per_run.append(np.isin(labels[:, run], list(proto_labels)) & (labels[:, run] >= 0))
        scores[key] = np.mean(np.asarray(per_run, dtype=float), axis=0)
    return scores


def stable_families(
    consensus_by_view: dict[str, np.ndarray], fit_mask: np.ndarray, threshold: float
) -> np.ndarray:
    fit_indices = np.flatnonzero(fit_mask)
    combined = np.mean(np.stack(list(consensus_by_view.values())), axis=0)
    labels = np.full(len(fit_mask), -1, dtype=int)
    if len(fit_indices) < 2:
        return labels
    distance = np.clip(1.0 - combined, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=True), method="complete")
    raw = fcluster(tree, t=1.0 - float(threshold), criterion="distance")
    family = 0
    for cluster_id in sorted(np.unique(raw)):
        members = np.flatnonzero(raw == cluster_id)
        if len(members) < 2:
            continue
        labels[fit_indices[members]] = family
        family += 1
    return labels


def distances_by_prototype(
    matrix: np.ndarray, prototype_indices: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    out = {}
    for key, indices in prototype_indices.items():
        differences = matrix[:, None, :] - matrix[indices][None, :, :]
        out[key] = np.sqrt(np.sum(differences * differences, axis=2)).min(axis=1)
    return out


def soap_kernel_by_prototype(
    soap: np.ndarray, prototype_indices: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        key: np.clip(np.max(soap @ soap[indices].T, axis=1), 0.0, 1.0)
        for key, indices in prototype_indices.items()
    }


def distance_tau(
    distances: dict[str, np.ndarray],
    frame: pd.DataFrame,
    fit_mask: np.ndarray,
    excluded_formulas: set[str],
    quantile: float = 0.10,
) -> float:
    nearest = np.min(np.vstack(list(distances.values())), axis=0)
    eligible = fit_mask & ~frame["formula"].isin(excluded_formulas).to_numpy()
    values = nearest[eligible & np.isfinite(nearest) & (nearest > 1e-10)]
    tau = float(np.quantile(values, float(quantile))) if len(values) else 1.0
    return max(tau, 1e-8)


def score_candidates(
    data: PreparedData,
    config: dict[str, Any],
    prototype_indices: dict[str, np.ndarray] | None = None,
    distance_scales: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prototypes = data.prototype_indices if prototype_indices is None else prototype_indices
    top_dist = distances_by_prototype(data.embeddings["topology"], prototypes)
    chem_dist = distances_by_prototype(data.embeddings["chemistry"], prototypes)
    soap_kernel = soap_kernel_by_prototype(data.soap, prototypes)
    all_positive_formulas = {
        canonical_formula(value) for value in config["positive_prototypes"].values()
    }
    tau_quantile = float(config.get("distance_scale_quantile", 0.10))
    if distance_scales is None:
        tau_top = distance_tau(
            top_dist, data.frame, data.fit_mask, all_positive_formulas, tau_quantile
        )
        tau_chem = distance_tau(
            chem_dist, data.frame, data.fit_mask, all_positive_formulas, tau_quantile
        )
    else:
        tau_top = float(distance_scales["tau_topology"])
        tau_chem = float(distance_scales["tau_chemistry"])
    top_similarity = {key: np.exp(-((value / tau_top) ** 2)) for key, value in top_dist.items()}
    chem_similarity = {key: np.exp(-((value / tau_chem) ** 2)) for key, value in chem_dist.items()}
    weights = config["score_weights"]
    if not math.isclose(sum(float(value) for value in weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("score_weights must sum to exactly 1")
    result = data.frame.copy()
    for key in prototypes:
        result[f"distance_topology__{key}"] = top_dist[key]
        result[f"distance_chemistry__{key}"] = chem_dist[key]
        result[f"similarity_topology__{key}"] = top_similarity[key]
        result[f"similarity_chemistry__{key}"] = chem_similarity[key]
        result[f"similarity_soap__{key}"] = soap_kernel[key]
        result[f"mechanism_similarity__{key}"] = (
            float(weights["topology"]) * top_similarity[key]
            + float(weights["soap"]) * soap_kernel[key]
            + float(weights["chemistry"]) * chem_similarity[key]
        ) / (float(weights["topology"]) + float(weights["soap"]) + float(weights["chemistry"]))
        result[f"prototype_score__{key}"] = (
            float(weights["topology"]) * top_similarity[key]
            + float(weights["soap"]) * soap_kernel[key]
            + float(weights["chemistry"]) * chem_similarity[key]
            + float(weights["flexibility"]) * result["flexibility_score"]
            + float(weights["stability"]) * result["stability_score"]
        )
    result["topology_similarity"] = np.max(np.vstack(list(top_similarity.values())), axis=0)
    result["chemistry_similarity"] = np.max(np.vstack(list(chem_similarity.values())), axis=0)
    result["soap_similarity"] = np.max(np.vstack(list(soap_kernel.values())), axis=0)
    mechanism_columns = [f"mechanism_similarity__{key}" for key in prototypes]
    sorted_mechanism = np.sort(result[mechanism_columns].to_numpy(), axis=1)
    result["mechanism_best_similarity"] = sorted_mechanism[:, -1]
    result["mechanism_second_best_similarity"] = sorted_mechanism[:, -2]
    result["prototype_bridge_margin"] = sorted_mechanism[:, -1] - sorted_mechanism[:, -2]
    prototype_score_columns = [f"prototype_score__{key}" for key in prototypes]
    prototype_score_matrix = result[prototype_score_columns].to_numpy()
    best_index = np.argmax(prototype_score_matrix, axis=1)
    keys = list(prototypes)
    result["nearest_prototype"] = [keys[i] for i in best_index]
    sorted_scores = np.sort(prototype_score_matrix, axis=1)
    result["prototype_best_score"] = sorted_scores[:, -1]
    result["prototype_second_best_score"] = sorted_scores[:, -2]
    result["prototype_total_score_margin"] = sorted_scores[:, -1] - sorted_scores[:, -2]
    # Fuse all views inside each physical prototype first.  Taking this maximum
    # afterwards prevents a candidate from borrowing topology, chemistry and SOAP
    # similarity from three unrelated mechanisms.
    result["discovery_score"] = result["prototype_best_score"]
    eligible = result["eligible_analysis"].to_numpy(bool)
    result["rank_all_structures"] = result["discovery_score"].rank(ascending=False, method="min").astype(int)
    result["rank_eligible_structures"] = np.nan
    result.loc[eligible, "rank_eligible_structures"] = result.loc[eligible, "discovery_score"].rank(
        ascending=False, method="min"
    )
    summary = {"tau_topology": tau_top, "tau_chemistry": tau_chem}
    return result, summary


def _pca_component_count(transform: ViewTransform, variance_fraction: float) -> int:
    cumulative = np.cumsum(transform.pca.explained_variance_ratio_)
    return min(
        len(cumulative),
        max(1, int(np.searchsorted(cumulative, float(variance_fraction), side="left") + 1)),
    )


def robustness_ensemble(
    data: PreparedData,
    scored: pd.DataFrame,
    config: dict[str, Any],
    frozen_scenarios: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Compute predeclared score stability without changing the VESTA bond rule.

    VESTA remains a fixed scientific definition.  The sensitivity ensemble perturbs
    PCA retention, distance calibration, physically plausible group weights and the
    pre-registered 5/6/7 A SOAP cutoff variants.  HDBSCAN and the primary score keep
    the 6 A base representation; HDBSCAN parameter stability is reported separately.
    """

    cfg = config["robustness"]
    soap_r_cuts = validate_soap_cutoff_config(config)
    registered_specs = registered_robustness_scenario_specs(config)
    all_positive = {
        canonical_formula(value) for value in config["positive_prototypes"].values()
    }
    scenario_states: list[dict[str, Any]] = []
    scenario_scores = []
    selected = []
    if frozen_scenarios is None:
        scenario_specs = registered_specs
    else:
        scenario_specs = validate_frozen_robustness_scenarios(
            frozen_scenarios, config
        )
    expected_n_scenarios = len(registered_specs)
    if expected_n_scenarios != 54 or len(scenario_specs) != expected_n_scenarios:
        raise RuntimeError(
            "robustness ensemble must replay exactly 54 pre-registered scenarios; "
            f"expected={expected_n_scenarios}, observed={len(scenario_specs)}"
        )

    if not isinstance(data.soap_variants, dict):
        raise RuntimeError("PreparedData has no SOAP cutoff-variant matrices")
    expected_variant_keys = {soap_variant_key(value) for value in soap_r_cuts}
    if set(data.soap_variants) != expected_variant_keys:
        raise RuntimeError(
            "PreparedData SOAP cutoff variants differ from the pre-registered set"
        )
    soap_similarities = {
        variant: soap_kernel_by_prototype(matrix, data.prototype_indices)
        for variant, matrix in data.soap_variants.items()
    }

    for spec in scenario_specs:
        if "soap_variant" not in spec or "soap_r_cut_A" not in spec:
            raise RuntimeError("robustness scenario is missing its frozen SOAP variant")
        scenario_r_cut = float(spec["soap_r_cut_A"])
        scenario_variant = str(spec["soap_variant"])
        if (
            scenario_variant != soap_variant_key(scenario_r_cut)
            or scenario_r_cut not in soap_r_cuts
            or scenario_variant not in soap_similarities
        ):
            raise RuntimeError(f"invalid frozen SOAP variant in robustness scenario: {spec}")
        soap_similarity = soap_similarities[scenario_variant]
        weights = spec["weights"]
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12):
            raise ValueError(f"robustness weights do not sum to one: {spec}")
        if frozen_scenarios is None:
            k_top = _pca_component_count(
                data.transforms["topology"], spec["pca_variance"]
            )
            k_chem = _pca_component_count(
                data.transforms["chemistry"], spec["pca_variance"]
            )
        else:
            k_top = int(spec["topology_components"])
            k_chem = int(spec["chemistry_components"])
            if k_top > data.embeddings["topology"].shape[1] or k_chem > data.embeddings[
                "chemistry"
            ].shape[1]:
                raise RuntimeError(
                    "frozen robustness component count exceeds the frozen embedding width"
                )
        top_dist = distances_by_prototype(
            data.embeddings["topology"][:, :k_top], data.prototype_indices
        )
        chem_dist = distances_by_prototype(
            data.embeddings["chemistry"][:, :k_chem], data.prototype_indices
        )
        if frozen_scenarios is None:
            tau_top = distance_tau(
                top_dist,
                data.frame,
                data.fit_mask,
                all_positive,
                spec["distance_scale_quantile"],
            )
            tau_chem = distance_tau(
                chem_dist,
                data.frame,
                data.fit_mask,
                all_positive,
                spec["distance_scale_quantile"],
            )
        else:
            tau_top = float(spec["tau_topology"])
            tau_chem = float(spec["tau_chemistry"])
        per_prototype = []
        for key in data.prototype_indices:
            sim_top = np.exp(-((top_dist[key] / tau_top) ** 2))
            sim_chem = np.exp(-((chem_dist[key] / tau_chem) ** 2))
            per_prototype.append(
                weights["topology"] * sim_top
                + weights["soap"] * soap_similarity[key]
                + weights["chemistry"] * sim_chem
                + weights["flexibility"] * scored["flexibility_score"].to_numpy(float)
                + weights["stability"] * scored["stability_score"].to_numpy(float)
            )
        score = np.max(np.vstack(per_prototype), axis=0)
        if frozen_scenarios is None:
            calibration_mask = (
                scored["eligible_analysis"].to_numpy(bool)
                & scored["is_structure_representative"].to_numpy(bool)
                & ~scored["formula"].isin(all_positive).to_numpy()
            )
            threshold = float(
                np.quantile(score[calibration_mask], 1.0 - float(cfg["top_fraction"]))
            )
            state = {
                **spec,
                "topology_components": k_top,
                "chemistry_components": k_chem,
                "tau_topology": tau_top,
                "tau_chemistry": tau_chem,
                "top_score_threshold": threshold,
            }
            scenario_states.append(state)
        else:
            threshold = float(spec["top_score_threshold"])
            scenario_states.append(spec)
        scenario_scores.append(score)
        selected.append(score >= threshold)

    score_matrix = np.vstack(scenario_scores)
    selected_matrix = np.vstack(selected)
    out = scored.copy()
    out["selection_frequency_top5"] = selected_matrix.mean(axis=0)
    out["robustness_score_mean"] = score_matrix.mean(axis=0)
    out["robustness_score_std"] = score_matrix.std(axis=0)
    out["robustness_n_runs"] = score_matrix.shape[0]
    a_min = float(cfg["grade_A_min_frequency"])
    b_min = float(cfg["grade_B_min_frequency"])
    out["robustness_grade"] = np.select(
        [out["selection_frequency_top5"] >= a_min, out["selection_frequency_top5"] >= b_min],
        ["A", "B"],
        default="C",
    )
    summary = {
        "n_runs": len(scenario_states),
        "scenario_contract_sha256": stable_json_hash(scenario_states),
        "grade_A_min_frequency": a_min,
        "grade_B_min_frequency": b_min,
        "top_fraction": float(cfg["top_fraction"]),
        "fixed_vesta_cutoff_table": True,
        "base_soap_radial_cutoff_A": float(config["soap"]["r_cut_A"]),
        "soap_radial_cutoff_sensitivity_A": list(soap_r_cuts),
        "soap_radial_cutoff_varied_in_robustness": True,
        "primary_score_and_hdbscan_use_base_soap_only": True,
        "hdbscan_stability_reported_separately": True,
    }
    return out, scenario_states, summary


def merge_extratrees(frame: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    if path is None:
        return frame
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ExtraTrees score source does not exist: {path}")
    if "cif_file" not in frame or frame["cif_file"].isna().any():
        raise ValueError("discovery frame requires non-missing cif_file identities")
    frame_keys = frame["cif_file"].astype(str)
    if frame_keys.duplicated().any():
        duplicates = frame_keys[frame_keys.duplicated(False)].head().tolist()
        raise ValueError(f"discovery frame has duplicate CIF identities: {duplicates}")

    header = pd.read_csv(path, nrows=0)
    required = {"cif_file", "score"}
    if not required.issubset(header.columns):
        raise ValueError(
            f"ExtraTrees score source is missing columns: {sorted(required - set(header.columns))}"
        )
    source_identity = pd.read_csv(path, usecols=["cif_file"])
    if source_identity["cif_file"].isna().any():
        raise ValueError("ExtraTrees score source has missing CIF identities")
    source_keys = source_identity["cif_file"].astype(str)
    if source_keys.duplicated().any():
        duplicates = source_keys[source_keys.duplicated(False)].head().tolist()
        raise ValueError(f"ExtraTrees score source has duplicate CIF identities: {duplicates}")
    allowed_keys = set(frame_keys)
    available_keys = set(source_keys)
    missing_allowed = sorted(allowed_keys - available_keys)
    if missing_allowed:
        raise ValueError(
            "ExtraTrees score source is missing allowed development CIF identities: "
            f"{missing_allowed[:5]}"
        )
    keep_lines = {
        int(position) + 1
        for position, value in enumerate(source_keys)
        if value in allowed_keys
    }
    keep = ["cif_file", "score"] + [
        name
        for name in ["formula", "seen_train_cif", "seen_train_formula", "seen_train_group_key"]
        if name in header.columns
    ]
    # Only the immutable CIF identity is parsed for the complete source.  Every
    # non-development row is physically skipped before score/formula values are
    # materialized by pandas.
    extra = pd.read_csv(
        path,
        usecols=keep,
        skiprows=lambda line_number: line_number > 0 and line_number not in keep_lines,
    )
    if extra["cif_file"].isna().any():
        raise ValueError("selected ExtraTrees rows have missing CIF identities")
    parsed_keys = extra["cif_file"].astype(str)
    parsed_nonallowed = sorted(set(parsed_keys) - allowed_keys)
    if parsed_nonallowed:
        raise RuntimeError(
            "ExtraTrees selective parser materialized non-allowed CIF identities: "
            f"{parsed_nonallowed[:5]}"
        )
    parsed_missing = sorted(allowed_keys - set(parsed_keys))
    if parsed_missing:
        raise RuntimeError(
            "ExtraTrees selective parser omitted allowed CIF identities: "
            f"{parsed_missing[:5]}"
        )
    if parsed_keys.duplicated().any() or len(extra) != len(allowed_keys):
        raise RuntimeError("ExtraTrees selective parser did not return one row per allowed CIF")
    extra = extra.rename(
        columns={
            "score": "extratrees_expert_priority_score",
            "formula": "extratrees_formula_audit",
            "seen_train_cif": "extratrees_seen_train_cif",
            "seen_train_formula": "extratrees_seen_train_formula",
            "seen_train_group_key": "extratrees_seen_train_group_key",
        }
    )
    out = frame.merge(extra, on="cif_file", how="left", validate="one_to_one")
    if not frame_keys.reset_index(drop=True).equals(
        out["cif_file"].astype(str).reset_index(drop=True)
    ):
        raise RuntimeError("ExtraTrees merge changed discovery row identity or order")
    if "extratrees_formula_audit" in out:
        comparable = out["extratrees_formula_audit"].notna()
        parsed = pd.Series(index=out.index, dtype="object")
        for index, value in out.loc[comparable, "extratrees_formula_audit"].items():
            try:
                parsed.loc[index] = canonical_formula(value)
            except Exception:
                parsed.loc[index] = None
        out["extratrees_formula_audit_parseable"] = parsed.notna()
        mismatch = comparable & parsed.notna() & parsed.ne(out["formula"])
        if mismatch.any():
            raise ValueError("ExtraTrees score file formula/CIF identity mismatch")
    for column in [
        "extratrees_seen_train_cif",
        "extratrees_seen_train_formula",
        "extratrees_seen_train_group_key",
    ]:
        if column not in out:
            out[column] = np.nan
    unseen = ~bool_values(out["extratrees_seen_train_formula"], missing=True)
    independent = unseen & out["eligible_analysis"] & out["is_structure_representative"]
    out["extratrees_evidence_independent"] = independent
    out["extratrees_evidence_status"] = np.select(
        [independent, bool_values(out["extratrees_seen_train_cif"], missing=False)],
        ["independent_unseen_formula", "training_exact_cif_overlap"],
        default="training_formula_or_unknown_overlap",
    )
    out["extratrees_rank_independent"] = np.nan
    out.loc[independent, "extratrees_rank_independent"] = out.loc[
        independent, "extratrees_expert_priority_score"
    ].rank(ascending=False, method="min")
    return out


def refit_prepared_for_fold(
    data: PreparedData,
    fit_mask: np.ndarray,
    prototype_indices: dict[str, np.ndarray],
    config: dict[str, Any],
) -> PreparedData:
    """Fit a fresh development-only representation for one strict LOPO fold."""

    prep = config["preprocessing"]
    robust_clip = float(prep["robust_scaled_clip"])
    transforms: dict[str, ViewTransform] = {}
    embeddings: dict[str, np.ndarray] = {}
    transforms["topology"], embeddings["topology"] = ViewTransform.fit(
        data.frame,
        TOPOLOGY_FEATURES,
        fit_mask,
        float(prep["interpretable_pca_variance"]),
        use_power=True,
        use_scaler=True,
        clip_value=robust_clip,
    )
    transforms["chemistry"], embeddings["chemistry"] = ViewTransform.fit(
        data.frame,
        CHEMISTRY_FEATURES,
        fit_mask,
        float(prep["interpretable_pca_variance"]),
        use_power=True,
        use_scaler=True,
        clip_value=robust_clip,
    )
    soap_frame = pd.DataFrame(
        data.soap, columns=[f"soap_{i}" for i in range(data.soap.shape[1])]
    )
    transforms["soap"], embeddings["soap"] = ViewTransform.fit(
        soap_frame,
        list(soap_frame.columns),
        fit_mask,
        float(prep["soap_pca_variance"]),
        use_power=False,
        use_scaler=False,
        clip_value=None,
    )
    calibration = FlexibilityCalibration.fit(data.frame, fit_mask)
    frame = data.frame.copy()
    frame["flexibility_score"] = calibration.transform(frame)
    return PreparedData(
        frame=frame,
        soap=data.soap,
        development_count=data.development_count,
        fit_mask=fit_mask,
        prototype_indices=prototype_indices,
        transforms=transforms,
        flexibility_calibration=calibration,
        embeddings=embeddings,
        soap_variants=data.soap_variants,
    )


def leave_one_positive_out(data: PreparedData, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for hidden_key, _hidden_anchor in data.prototype_indices.items():
        hidden_formula = canonical_formula(config["positive_prototypes"][hidden_key])
        hidden_indices = np.flatnonzero(
            data.frame["formula"].eq(hidden_formula).to_numpy()
            & data.frame["eligible_analysis"].to_numpy(bool)
        )
        remaining = {key: value for key, value in data.prototype_indices.items() if key != hidden_key}
        fold_fit_mask = data.fit_mask & ~data.frame["formula"].eq(hidden_formula).to_numpy()
        fold_data = refit_prepared_for_fold(data, fold_fit_mask, remaining, config)
        scored, _ = score_candidates(fold_data, config, remaining)
        candidate = (
            scored["eligible_analysis"].to_numpy(bool)
            & scored["is_structure_representative"].to_numpy(bool)
        )
        scores = scored.loc[candidate, "discovery_score"]
        rank_columns = [
            "topology_similarity",
            "chemistry_similarity",
            "soap_similarity",
            "discovery_score",
        ]
        ranks_by_column = {
            column: scored.loc[candidate, column].rank(ascending=False, method="min")
            for column in rank_columns
        }
        formula_scores = scored.loc[candidate].groupby("formula")["discovery_score"].max()
        formula_ranks = formula_scores.rank(ascending=False, method="min")
        formula_rank = int(formula_ranks.loc[hidden_formula])
        evaluated_hidden = hidden_indices[candidate[hidden_indices]]
        for index in evaluated_hidden:
            rows.append(
                {
                    "hidden_prototype": hidden_key,
                    "hidden_formula": hidden_formula,
                    "cif_file": scored.loc[index, "cif_file"],
                    "discovery_score": scored.loc[index, "discovery_score"],
                    "structure_rank": int(ranks_by_column["discovery_score"].loc[index]),
                    "structure_percentile": 1.0
                    - (int(ranks_by_column["discovery_score"].loc[index]) - 1) / max(len(scores), 1),
                    "topology_view_rank": int(ranks_by_column["topology_similarity"].loc[index]),
                    "chemistry_view_rank": int(ranks_by_column["chemistry_similarity"].loc[index]),
                    "soap_view_rank": int(ranks_by_column["soap_similarity"].loc[index]),
                    "formula_rank_best_structure": formula_rank,
                    "formula_percentile": 1.0 - (formula_rank - 1) / max(len(formula_scores), 1),
                }
            )
    report = pd.DataFrame(rows)
    by_formula = report.groupby("hidden_prototype").agg(
        best_structure_rank=("structure_rank", "min"),
        best_topology_view_rank=("topology_view_rank", "min"),
        best_chemistry_view_rank=("chemistry_view_rank", "min"),
        best_soap_view_rank=("soap_view_rank", "min"),
        formula_rank=("formula_rank_best_structure", "first"),
        formula_percentile=("formula_percentile", "first"),
    )
    n_ranked_formulas = int(scored.loc[candidate, "formula"].nunique())
    top_5_percent_limit = max(1, int(math.floor(0.05 * n_ranked_formulas)))
    top_10_percent_limit = max(1, int(math.floor(0.10 * n_ranked_formulas)))
    summary = {
        "mrr_formula": float(np.mean(1.0 / by_formula["formula_rank"])),
        "all_recalled_top_10": bool((by_formula["formula_rank"] <= 10).all()),
        "all_recalled_top_20": bool((by_formula["formula_rank"] <= 20).all()),
        "all_recalled_top_5_percent": bool(
            (by_formula["formula_rank"] <= top_5_percent_limit).all()
        ),
        "all_recalled_top_10_percent": bool(
            (by_formula["formula_rank"] <= top_10_percent_limit).all()
        ),
        "n_ranked_formulas": n_ranked_formulas,
        "top_5_percent_rank_limit_floor": top_5_percent_limit,
        "top_10_percent_rank_limit_floor": top_10_percent_limit,
        "per_prototype": by_formula.reset_index().to_dict("records"),
        "validation_used_for_parameter_selection": False,
        "strict_fold_refits_exclude_hidden_formula": True,
    }
    return report, summary


def committed_leave_one_positive_out(
    args: argparse.Namespace,
    commitment_path: Path,
    data: PreparedData,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recheck the immutable commitment immediately before LOPO calculation."""
    verify_predevelopment_commitment(args, commitment_path)
    return leave_one_positive_out(data, config)


def add_clustering(
    data: PreparedData,
    scored: pd.DataFrame,
    config: dict[str, Any],
    outdir: Path,
    frozen_clustering: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    labels_by_view, consensus_by_view, run_params = {}, {}, {}
    clustering_state: dict[str, Any] = {"views": {}}
    prototype_support_by_view = {}
    for view in ["topology", "chemistry", "soap"]:
        if frozen_clustering is None:
            labels, params, consensus, clusterers = hdbscan_runs(
                data.embeddings[view], data.fit_mask, config
            )
            clustering_state["views"][view] = {
                "development_labels": labels,
                "params": params,
                "consensus": consensus,
                "clusterers": clusterers,
            }
        else:
            frozen_view = frozen_clustering["views"][view]
            params = frozen_view["params"]
            clusterers = frozen_view["clusterers"]
            dev_labels = np.asarray(frozen_view["development_labels"], dtype=int)
            if len(dev_labels) != data.development_count:
                raise RuntimeError(f"frozen HDBSCAN development rows changed for {view}")
            blind_matrix = data.embeddings[view][data.development_count :]
            predicted_runs = []
            for params_one, clusterer in zip(params, clusterers):
                view_matrix = subspace(blind_matrix, params_one["component_subspace"])
                if len(view_matrix):
                    predicted, _strength = hdbscan.approximate_predict(
                        clusterer, view_matrix
                    )
                else:
                    predicted = np.empty(0, dtype=int)
                predicted_runs.append(np.asarray(predicted, dtype=int))
            predicted_matrix = np.asarray(predicted_runs, dtype=int).T
            labels = np.vstack([dev_labels, predicted_matrix])
            consensus = np.asarray(frozen_view["consensus"], dtype=np.float32)
            clustering_state["views"][view] = frozen_view
        labels_by_view[view] = labels
        consensus_by_view[view] = consensus
        run_params[view] = params
        support = prototype_consensus(labels, data.prototype_indices)
        prototype_support_by_view[view] = support
        pd.DataFrame(
            labels,
            columns=[f"run_{i:03d}" for i in range(labels.shape[1])],
        ).assign(cif_file=data.frame["cif_file"].values).to_csv(
            outdir / f"hdbscan_labels_{view}.csv", index=False
        )
    result = scored.copy()
    for view, supports in prototype_support_by_view.items():
        for key, values in supports.items():
            result[f"consensus_{view}__{key}"] = values
        result[f"consensus_{view}_prototype_max"] = np.max(np.vstack(list(supports.values())), axis=0)
    per_prototype_support = []
    for key in data.prototype_indices:
        column = f"consensus_multiview__{key}"
        result[column] = result[
            [f"consensus_{view}__{key}" for view in labels_by_view]
        ].mean(axis=1)
        per_prototype_support.append(column)
    result["consensus_prototype_support"] = result[per_prototype_support].max(axis=1)
    result["consensus_nearest_prototype"] = result[per_prototype_support].idxmax(axis=1).str.replace(
        "consensus_multiview__", "", regex=False
    )
    if frozen_clustering is None:
        stable_labels = stable_families(
            consensus_by_view,
            data.fit_mask,
            float(config["hdbscan"]["consensus_threshold"]),
        )
        clustering_state["development_stable_family"] = stable_labels
    else:
        stable_labels = np.concatenate(
            [
                np.asarray(frozen_clustering["development_stable_family"], dtype=int),
                np.full(len(result) - data.development_count, -1, dtype=int),
            ]
        )
    result["stable_family"] = stable_labels
    np.savez_compressed(
        outdir / "consensus_fit_matrices.npz",
        fit_indices=np.flatnonzero(data.fit_mask),
        **{view: matrix for view, matrix in consensus_by_view.items()},
    )
    summary = {
        "runs_per_view": {key: len(value) for key, value in run_params.items()},
        "parameters": run_params,
        "consensus_threshold": float(config["hdbscan"]["consensus_threshold"]),
        "stable_family_linkage": "complete",
    }
    clustering_state["summary"] = summary
    return result, summary, clustering_state


def make_umap(
    data: PreparedData,
    result: pd.DataFrame,
    outdir: Path,
    config: dict[str, Any],
    frozen_umap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blocks = []
    norms = [] if frozen_umap is None else list(frozen_umap["block_norms"])
    for position, view in enumerate(["topology", "chemistry", "soap"]):
        block = data.embeddings[view]
        norm = (
            float(np.sqrt(np.mean(block[data.fit_mask] ** 2)))
            if frozen_umap is None
            else float(norms[position])
        )
        if frozen_umap is None:
            norms.append(norm)
        blocks.append(block / max(norm, 1e-8))
    combined = np.hstack(blocks)
    xy = np.zeros((len(result), 2), dtype=float)
    if frozen_umap is None:
        seed = int(config["random_seed"])
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.08,
            n_components=2,
            metric="euclidean",
            random_state=seed,
            transform_seed=seed,
        )
        fit_indices = np.flatnonzero(data.fit_mask)
        xy[fit_indices] = reducer.fit_transform(combined[fit_indices])
        other = np.flatnonzero(~data.fit_mask)
        if len(other):
            xy[other] = reducer.transform(combined[other])
        state = {
            "reducer": reducer,
            "block_norms": norms,
            "development_xy": xy.copy(),
        }
    else:
        reducer = frozen_umap["reducer"]
        dev_xy = np.asarray(frozen_umap["development_xy"], dtype=float)
        if len(dev_xy) != data.development_count:
            raise RuntimeError("frozen UMAP development row count changed")
        xy[: data.development_count] = dev_xy
        if len(result) > data.development_count:
            xy[data.development_count :] = reducer.transform(
                combined[data.development_count :]
            )
        state = frozen_umap
    result[["cif_file", "formula"]].assign(umap_x=xy[:, 0], umap_y=xy[:, 1]).to_csv(
        outdir / "umap_visualization.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xy[:, 0], xy[:, 1], c=result["discovery_score"], s=9, cmap="viridis", alpha=0.65)
    for key, indices in data.prototype_indices.items():
        ax.scatter(xy[indices, 0], xy[indices, 1], s=70, marker="*", label=key)
    ax.set_xlabel("UMAP 1 (visualization only)")
    ax.set_ylabel("UMAP 2 (visualization only)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "umap_prototype_neighborhoods.png", dpi=220)
    plt.close(fig)
    return state


def _diverse_formula_head(
    frame: pd.DataFrame, n: int, used_formulas: set[str], max_per_center: int
) -> pd.DataFrame:
    chosen = []
    center_counts: dict[str, int] = {}
    for index, row in frame.drop_duplicates("formula").iterrows():
        formula = str(row["formula"])
        center = str(row["center_element"])
        if formula in used_formulas or center_counts.get(center, 0) >= max_per_center:
            continue
        chosen.append(index)
        used_formulas.add(formula)
        center_counts[center] = center_counts.get(center, 0) + 1
        if len(chosen) >= n:
            break
    return frame.loc[chosen].copy()


def build_portfolio(
    result: pd.DataFrame,
    config: dict[str, Any],
    frozen_thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    known = {canonical_formula(v) for v in config["positive_prototypes"].values()}
    base = result.loc[
        result["eligible_analysis"]
        & result["is_structure_representative"]
        & ~result["formula"].isin(known)
    ].copy()
    if frozen_thresholds is None:
        thresholds = {
            "high_potential_score": float(base["discovery_score"].quantile(0.95))
            if len(base)
            else 1.0,
            "bridge_second_score": float(
                base["mechanism_second_best_similarity"].quantile(
                    float(config.get("bridge_min_score_quantile", 0.80))
                )
            )
            if len(base)
            else 1.0,
        }
    else:
        thresholds = {key: float(value) for key, value in frozen_thresholds.items()}
    selections = []
    used_formulas: set[str] = set()
    max_per_center = int(config.get("portfolio_max_per_center_per_section", 2))
    per = int(config["portfolio_per_prototype"])
    for key in config["positive_prototypes"]:
        support_column = f"consensus_multiview__{key}"
        part = base.loc[
            base["nearest_prototype"].eq(key)
            & base[support_column].ge(float(config.get("prototype_min_consensus_support", 0.0)))
        ].sort_values(
            ["selection_frequency_top5", "discovery_score"], ascending=[False, False]
        )
        part = _diverse_formula_head(part, per, used_formulas, max_per_center)
        part["portfolio_section"] = f"prototype::{key}"
        selections.append(part)
    bridge_pool = base.loc[
        base["mechanism_second_best_similarity"].ge(thresholds["bridge_second_score"])
    ].sort_values(
        [
            "selection_frequency_top5",
            "mechanism_second_best_similarity",
            "prototype_bridge_margin",
            "discovery_score",
        ],
        ascending=[False, False, True, False],
    )
    bridge = _diverse_formula_head(
        bridge_pool,
        int(config["portfolio_bridge"]),
        used_formulas,
        max_per_center,
    )
    bridge["portfolio_section"] = "bridge_between_prototypes"
    selections.append(bridge)
    rare_pool = base.loc[
        base["discovery_score"].ge(thresholds["high_potential_score"])
        & base["consensus_prototype_support"].le(
            float(config.get("rare_max_consensus_support", 0.20))
        )
    ].sort_values(
        ["selection_frequency_top5", "consensus_prototype_support", "discovery_score"],
        ascending=[False, True, False],
    )
    rare = _diverse_formula_head(
        rare_pool,
        int(config["portfolio_rare_high_potential"]),
        used_formulas,
        max_per_center,
    )
    rare["portfolio_section"] = "rare_high_potential"
    selections.append(rare)
    if not selections:
        return base.iloc[0:0], thresholds
    return pd.concat(selections, ignore_index=True), thresholds


def build_frozen_state(
    data: PreparedData,
    score_summary: dict[str, Any],
    clustering_state: dict[str, Any],
    umap_state: dict[str, Any],
    portfolio_thresholds: dict[str, float],
    robustness_scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "development_count": data.development_count,
        "development_cif_files": data.frame["cif_file"].tolist(),
        "fit_mask": data.fit_mask.copy(),
        "prototype_indices": {
            key: np.asarray(value, dtype=int) for key, value in data.prototype_indices.items()
        },
        "transforms": {
            key: serialize_view_transform(value) for key, value in data.transforms.items()
        },
        "development_embeddings": {
            key: np.asarray(value, dtype=float) for key, value in data.embeddings.items()
        },
        "flexibility_calibration": serialize_flexibility(data.flexibility_calibration),
        "development_flexibility_score": data.frame["flexibility_score"].to_numpy(float),
        "distance_scales": {key: float(value) for key, value in score_summary.items()},
        "clustering": clustering_state,
        "umap": umap_state,
        "portfolio_thresholds": portfolio_thresholds,
        "robustness_scenarios": robustness_scenarios,
    }


def save_frozen_state(state: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    dump(state, temporary, compress=3)
    os.replace(temporary, path)


def freeze_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    data: PreparedData,
    score_summary: dict[str, Any],
    clustering_summary: dict[str, Any],
    frozen_model_path: Path,
    portfolio_thresholds: dict[str, float],
    robustness_summary: dict[str, Any],
) -> dict[str, Any]:
    provenance = load_provenance(args.features_dir)
    source_commitment = verify_feature_source_commitment(
        args, provenance, context="development freeze"
    )
    predevelopment_path = args.outdir / PREDEVELOPMENT_COMMITMENT_NAME
    predevelopment = verify_predevelopment_commitment(args, predevelopment_path)
    anchor_rows = []
    for key, indices in data.prototype_indices.items():
        index = int(indices[0])
        anchor_rows.append(
            {
                "prototype": key,
                "formula": data.frame.loc[index, "formula"],
                "cif_file": data.frame.loc[index, "cif_file"],
                "material_id": data.frame.loc[index, "material_id"],
            }
        )
    inventory_keys = sorted(feature_inventory_keys(args.features_dir))
    decision_files = [
        "ranked_structures.csv",
        "candidate_portfolio.csv",
        "leave_one_positive_out.csv",
        "leave_one_positive_out.summary.json",
        "consensus_fit_matrices.npz",
        "umap_visualization.csv",
        "umap_prototype_neighborhoods.png",
        "hdbscan_labels_topology.csv",
        "hdbscan_labels_chemistry.csv",
        "hdbscan_labels_soap.csv",
    ]
    manifest = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "stage": "development_frozen",
        "predevelopment_commitment": {
            "filename": PREDEVELOPMENT_COMMITMENT_NAME,
            "file_sha256": sha256_file(predevelopment_path),
            "commitment_payload_sha256": predevelopment[
                "commitment_payload_sha256"
            ],
            "created_at_utc": predevelopment["created_at_utc"],
        },
        "config_sha256": sha256_file(args.config),
        "blind_isolation_sha256": sha256_file(args.blind_file),
        "development_feature_provenance_sha256": sha256_file(args.features_dir / "provenance.json"),
        "development_feature_bundle_sha256": feature_bundle_hashes(args.features_dir),
        "development_feature_contract": provenance_contract(provenance),
        "development_feature_contract_sha256": stable_json_hash(provenance_contract(provenance)),
        "development_feature_source_commitment": source_commitment,
        "development_feature_source_commitment_sha256": stable_json_hash(
            source_commitment
        ),
        "development_selected_metadata_values_sha256": (
            (provenance.get("metadata") or {}).get("metadata_selected_values_sha256")
        ),
        "development_isolated_inventory_sha256": provenance.get(
            "isolated_inventory_sha256"
        ),
        "development_inventory_keys_sha256": stable_json_hash(inventory_keys),
        "development_success_cif_inventory_sha256": cif_inventory_hash(data.frame, args.cif_dir),
        "frozen_full_cif_content_inventory_sha256": full_cif_content_inventory_hash(
            args.cif_dir
        ),
        "runner_sha256": sha256_file(Path(__file__)),
        "frozen_model": {
            "filename": frozen_model_path.name,
            "sha256": sha256_file(frozen_model_path),
        },
        "environment_versions": environment_versions(),
        "extratrees_scores_sha256": (
            sha256_file(args.extratrees_scores)
            if args.extratrees_scores is not None and args.extratrees_scores.is_file()
            else None
        ),
        "development_structure_count": data.development_count,
        "fit_representative_count": int(data.fit_mask.sum()),
        "prototype_anchors": anchor_rows,
        "prototype_anchor_policy": config["prototype_anchor_policy"],
        "feature_schema": {
            "chemistry": CHEMISTRY_FEATURES,
            "topology": TOPOLOGY_FEATURES,
            # ``flexibility`` is the exact six-field scalar-score input contract.
            # The ten additional free-volume descriptors are extracted and kept for
            # audit/future analysis, but they do not silently enter this frozen rank.
            "flexibility": FLEXIBILITY_SCORE_INPUTS,
            "extracted_flexibility_audit_descriptors": FLEX_FEATURES,
            "flexibility_percentile_optima": FLEXIBILITY_PERCENTILE_OPTIMA,
            "flexibility_percentile_sigma": FLEXIBILITY_PERCENTILE_SIGMA,
            "schema_hash": stable_json_hash(
                {
                    "chemistry": CHEMISTRY_FEATURES,
                    "topology": TOPOLOGY_FEATURES,
                    "flexibility": FLEXIBILITY_SCORE_INPUTS,
                    "extracted_flexibility_audit_descriptors": FLEX_FEATURES,
                    "flexibility_percentile_optima": FLEXIBILITY_PERCENTILE_OPTIMA,
                    "flexibility_percentile_sigma": FLEXIBILITY_PERCENTILE_SIGMA,
                }
            ),
        },
        "pca_components": {key: int(value.shape[1]) for key, value in data.embeddings.items()},
        "distance_scales": score_summary,
        "portfolio_thresholds": portfolio_thresholds,
        "robustness": robustness_summary,
        "clustering": clustering_summary,
        "development_decision_output_sha256": {
            name: sha256_file(args.outdir / name) for name in decision_files
        },
        "positive_validation_is_diagnostic_only": True,
        "blind_acceptance_used_for_features_parameters_weights_thresholds": False,
        "operator_note": "Blind identities are supplied only to remove them before development; names are not copied into development outputs.",
    }
    manifest["freeze_hash"] = stable_json_hash(manifest)
    return manifest


def verify_manifest_self_hash(manifest: dict[str, Any]) -> None:
    """Authenticate manifest integrity before any pickle-compatible artifact load."""

    stored_hash = manifest.get("freeze_hash")
    payload = dict(manifest)
    payload.pop("freeze_hash", None)
    actual_hash = stable_json_hash(payload)
    if stored_hash != actual_hash:
        raise RuntimeError(
            "frozen-manifest verification failed before model load: "
            f"freeze_hash={(stored_hash, actual_hash)}"
        )


def frozen_model_path(frozen_manifest: Path, manifest: dict[str, Any]) -> Path:
    """Resolve only the fixed local model filename; never trust a pickle path."""

    model_info = manifest.get("frozen_model")
    if not isinstance(model_info, dict):
        raise RuntimeError("frozen manifest has no frozen_model object")
    filename = model_info.get("filename")
    if (
        filename != FROZEN_MODEL_NAME
        or Path(str(filename)).name != str(filename)
        or Path(str(filename)).is_absolute()
    ):
        raise RuntimeError(
            f"frozen model filename must be exactly {FROZEN_MODEL_NAME!r}"
        )
    return Path(frozen_manifest).parent / FROZEN_MODEL_NAME


def verify_frozen(args: argparse.Namespace, manifest: dict[str, Any]) -> Path:
    verify_manifest_self_hash(manifest)
    checks = {
        "config_sha256": sha256_file(args.config),
        "blind_isolation_sha256": sha256_file(args.blind_file),
        "development_feature_provenance_sha256": sha256_file(args.features_dir / "provenance.json"),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    mismatches = {key: (manifest.get(key), value) for key, value in checks.items() if manifest.get(key) != value}
    if manifest.get("schema_version") != DISCOVERY_SCHEMA_VERSION:
        mismatches["schema_version"] = (
            manifest.get("schema_version"),
            DISCOVERY_SCHEMA_VERSION,
        )
    if manifest.get("stage") != "development_frozen":
        mismatches["stage"] = (manifest.get("stage"), "development_frozen")
    predevelopment_info = manifest.get("predevelopment_commitment") or {}
    if predevelopment_info.get("filename") != PREDEVELOPMENT_COMMITMENT_NAME:
        mismatches["predevelopment_commitment_filename"] = (
            predevelopment_info.get("filename"),
            PREDEVELOPMENT_COMMITMENT_NAME,
        )
    predevelopment_path = (
        args.frozen_manifest.parent / PREDEVELOPMENT_COMMITMENT_NAME
    )
    if not predevelopment_path.is_file():
        mismatches["predevelopment_commitment_missing"] = (
            predevelopment_info.get("filename"),
            None,
        )
    else:
        predevelopment = verify_predevelopment_commitment(
            args, predevelopment_path
        )
        actual_predevelopment_file_hash = sha256_file(predevelopment_path)
        if predevelopment_info.get("file_sha256") != actual_predevelopment_file_hash:
            mismatches["predevelopment_commitment_file_sha256"] = (
                predevelopment_info.get("file_sha256"),
                actual_predevelopment_file_hash,
            )
        for key in ["commitment_payload_sha256", "created_at_utc"]:
            if predevelopment_info.get(key) != predevelopment.get(key):
                mismatches[f"predevelopment_commitment_{key}"] = (
                    predevelopment_info.get(key),
                    predevelopment.get(key),
                )
    bundle = feature_bundle_hashes(args.features_dir)
    if manifest.get("development_feature_bundle_sha256") != bundle:
        mismatches["development_feature_bundle_sha256"] = (
            manifest.get("development_feature_bundle_sha256"),
            bundle,
        )
    provenance = load_provenance(args.features_dir)
    live_source_commitment = verify_feature_source_commitment(
        args, provenance, context="frozen development"
    )
    recorded_source_commitment = feature_source_commitment(provenance)
    if manifest.get("development_feature_source_commitment") != recorded_source_commitment:
        mismatches["development_feature_source_commitment"] = (
            manifest.get("development_feature_source_commitment"),
            recorded_source_commitment,
        )
    source_commitment_hash = stable_json_hash(recorded_source_commitment)
    if manifest.get("development_feature_source_commitment_sha256") != source_commitment_hash:
        mismatches["development_feature_source_commitment_sha256"] = (
            manifest.get("development_feature_source_commitment_sha256"),
            source_commitment_hash,
        )
    if recorded_source_commitment != live_source_commitment:
        mismatches["live_feature_source_commitment"] = (
            recorded_source_commitment,
            live_source_commitment,
        )
    contract = provenance_contract(provenance)
    if manifest.get("development_feature_contract_sha256") != stable_json_hash(contract):
        mismatches["development_feature_contract_sha256"] = (
            manifest.get("development_feature_contract_sha256"),
            stable_json_hash(contract),
        )
    dev_frame, _ = load_feature_dir(args.features_dir)
    live_cif_hash = cif_inventory_hash(dev_frame, args.cif_dir)
    if manifest.get("development_success_cif_inventory_sha256") != live_cif_hash:
        mismatches["development_success_cif_inventory_sha256"] = (
            manifest.get("development_success_cif_inventory_sha256"),
            live_cif_hash,
        )
    full_live_cif_hash = full_cif_content_inventory_hash(args.cif_dir)
    if manifest.get("frozen_full_cif_content_inventory_sha256") != full_live_cif_hash:
        mismatches["frozen_full_cif_content_inventory_sha256"] = (
            manifest.get("frozen_full_cif_content_inventory_sha256"),
            full_live_cif_hash,
        )
    if contract.get("full_cif_content_inventory_sha256") != full_live_cif_hash:
        mismatches["feature_contract_full_cif_content_inventory_sha256"] = (
            contract.get("full_cif_content_inventory_sha256"),
            full_live_cif_hash,
        )
    inventory_hash = stable_json_hash(sorted(feature_inventory_keys(args.features_dir)))
    if manifest.get("development_inventory_keys_sha256") != inventory_hash:
        mismatches["development_inventory_keys_sha256"] = (
            manifest.get("development_inventory_keys_sha256"),
            inventory_hash,
        )
    if manifest.get("environment_versions") != environment_versions():
        mismatches["environment_versions"] = (
            manifest.get("environment_versions"),
            environment_versions(),
        )
    model_info = manifest.get("frozen_model") or {}
    model_path = frozen_model_path(args.frozen_manifest, manifest)
    if not model_path.is_file():
        mismatches["frozen_model_missing"] = (model_info.get("filename"), None)
    elif model_info.get("sha256") != sha256_file(model_path):
        mismatches["frozen_model_sha256"] = (
            model_info.get("sha256"),
            sha256_file(model_path),
        )
    else:
        try:
            state = load(model_path)
            required_state = {
                "schema_version",
                "development_count",
                "development_cif_files",
                "fit_mask",
                "prototype_indices",
                "transforms",
                "development_embeddings",
                "flexibility_calibration",
                "development_flexibility_score",
                "distance_scales",
                "clustering",
                "umap",
                "portfolio_thresholds",
                "robustness_scenarios",
            }
            missing_state = sorted(required_state - set(state))
            if missing_state:
                mismatches["frozen_model_required_keys"] = (missing_state, [])
            if state.get("schema_version") != DISCOVERY_SCHEMA_VERSION:
                mismatches["frozen_model_schema_version"] = (
                    state.get("schema_version"),
                    DISCOVERY_SCHEMA_VERSION,
                )
            count = int(state.get("development_count", -1))
            if count != int(manifest.get("development_structure_count", -2)):
                mismatches["frozen_model_development_count"] = (
                    count,
                    manifest.get("development_structure_count"),
                )
            if len(state.get("development_cif_files", [])) != count:
                mismatches["frozen_model_cif_rows"] = (
                    len(state.get("development_cif_files", [])),
                    count,
                )
            if any(not isinstance(value, dict) for value in state.get("transforms", {}).values()):
                mismatches["frozen_model_transform_serialization"] = (
                    "contains non-dictionary custom state",
                    "portable dictionary state",
                )
            live_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
            try:
                verify_manifest_robustness_contract(manifest, state, live_config)
            except Exception as exc:
                mismatches["frozen_model_robustness_contract"] = (
                    f"{type(exc).__name__}: {exc}",
                    "exact registered Cartesian product bound to manifest",
                )
        except Exception as exc:
            mismatches["frozen_model_deserialization"] = (
                f"{type(exc).__name__}: {exc}",
                "loadable",
            )
    extra_hash = (
        sha256_file(args.extratrees_scores)
        if args.extratrees_scores is not None and args.extratrees_scores.is_file()
        else None
    )
    if manifest.get("extratrees_scores_sha256") != extra_hash:
        mismatches["extratrees_scores_sha256"] = (
            manifest.get("extratrees_scores_sha256"),
            extra_hash,
        )
    for name, expected in manifest.get("development_decision_output_sha256", {}).items():
        path = args.frozen_manifest.parent / name
        actual = sha256_file(path) if path.is_file() else None
        if expected != actual:
            mismatches[f"development_decision_output::{name}"] = (expected, actual)
    if mismatches:
        raise RuntimeError(f"frozen-manifest verification failed: {mismatches}")
    return model_path


def verify_feature_provenance_outputs(path: Path) -> None:
    provenance = load_provenance(path)
    records = provenance_soap_variant_records(provenance)
    expected = {
        "interpretable_features_sha256": sha256_file(path / "interpretable_features.csv"),
        "soap_rows_sha256": sha256_file(path / "soap_rows.csv"),
        "soap_array_sha256": sha256_file(path / "soap_pseudo_mx.npy"),
        "structure_exclusions_sha256": sha256_file(path / "structure_exclusions.csv"),
    }
    mismatch = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    soap_row_count = len(pd.read_csv(path / "soap_rows.csv", usecols=["cif_file"]))
    for record in records:
        matrix_path = path / record["filename"]
        actual_hash = sha256_file(matrix_path) if matrix_path.is_file() else None
        if actual_hash != record["sha256"]:
            mismatch[f"soap_variant_sha256::{record['variant']}"] = (
                record["sha256"],
                actual_hash,
            )
            continue
        try:
            matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
            actual_shape = list(matrix.shape)
        except Exception as exc:
            mismatch[f"soap_variant_read::{record['variant']}"] = (
                "readable NPY",
                f"{type(exc).__name__}: {exc}",
            )
            continue
        if actual_shape != record["shape"] or actual_shape[0] != soap_row_count:
            mismatch[f"soap_variant_shape::{record['variant']}"] = (
                record["shape"],
                actual_shape,
            )
    if mismatch:
        raise RuntimeError(f"feature provenance/output mismatch in {path}: {mismatch}")


def verify_reveal_partition(
    args: argparse.Namespace, manifest: dict[str, Any], blind: set[str]
) -> None:
    assert args.reveal_features_dir is not None
    reveal_dir = args.reveal_features_dir
    verify_feature_provenance_outputs(reveal_dir)
    dev_provenance = load_provenance(args.features_dir)
    reveal_provenance = load_provenance(reveal_dir)
    verify_feature_source_commitment(
        args, dev_provenance, context="frozen development"
    )
    verify_feature_source_commitment(args, reveal_provenance, context="reveal")
    live_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    verify_soap_provenance_config(dev_provenance, live_config)
    verify_soap_provenance_config(reveal_provenance, live_config)
    if provenance_contract(reveal_provenance) != manifest["development_feature_contract"]:
        raise RuntimeError("reveal feature/extractor contract differs from frozen development")
    blind_hash = sha256_file(args.blind_file)
    if dev_provenance.get("formula_isolation_file_sha256") != blind_hash:
        raise RuntimeError("development feature bundle was not isolated with the frozen blind file")
    if dev_provenance.get("include_only_file_sha256") is not None:
        raise RuntimeError("development feature bundle unexpectedly used include-only isolation")
    if reveal_provenance.get("include_only_file_sha256") != blind_hash:
        raise RuntimeError("reveal feature bundle was not selected by the frozen blind file")
    if reveal_provenance.get("formula_isolation_file_sha256") is not None:
        raise RuntimeError("reveal bundle unexpectedly used an exclusion file")
    if reveal_provenance.get("formula_isolation_mode") != "reveal_include_only":
        raise RuntimeError("reveal feature provenance has the wrong isolation mode")
    if reveal_provenance.get("isolated_formula_names_written_to_feature_rows") is not True:
        raise RuntimeError("reveal feature provenance does not declare isolated formula rows")

    dev_keys = feature_inventory_keys(args.features_dir)
    reveal_keys = feature_inventory_keys(reveal_dir)
    if dev_keys & reveal_keys:
        raise RuntimeError("development/reveal inventories overlap")
    live_keys = {path.name for path in args.cif_dir.glob("*.cif")}
    if dev_keys | reveal_keys != live_keys:
        missing = sorted(live_keys - (dev_keys | reveal_keys))[:5]
        extra = sorted((dev_keys | reveal_keys) - live_keys)[:5]
        raise RuntimeError(
            f"development + reveal do not partition the complete CIF inventory; "
            f"missing={missing}, extra={extra}"
        )
    if len(dev_keys | reveal_keys) != int(
        manifest["development_feature_contract"]["n_inventory_structures_before_isolation"]
    ):
        raise RuntimeError("partition size differs from the frozen source inventory")

    reveal_success = pd.read_csv(reveal_dir / "interpretable_features.csv", usecols=["formula"])
    reveal_errors = pd.read_csv(reveal_dir / "structure_exclusions.csv", usecols=["formula"])
    inventory_formulas = {
        canonical_formula(value)
        for value in pd.concat([reveal_success["formula"], reveal_errors["formula"]]).dropna()
    }
    if inventory_formulas != blind:
        raise RuntimeError("reveal inventory formulas do not exactly equal the isolated set")
    success_formulas = {canonical_formula(value) for value in reveal_success["formula"]}
    if success_formulas != blind:
        raise RuntimeError("one or more isolated formulas has no valid VESTA/SOAP descriptor")


def verify_development_partition(
    args: argparse.Namespace, blind: set[str]
) -> None:
    provenance = load_provenance(args.features_dir)
    verify_feature_source_commitment(args, provenance, context="development")
    live_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    verify_soap_provenance_config(provenance, live_config)
    blind_hash = sha256_file(args.blind_file)
    if provenance.get("formula_isolation_file_sha256") != blind_hash:
        raise RuntimeError("development descriptors were not built with the frozen blind file")
    if provenance.get("include_only_file_sha256") is not None:
        raise RuntimeError("development descriptors unexpectedly used include-only isolation")
    if provenance.get("formula_isolation_mode") != "development_exclude":
        raise RuntimeError("development feature provenance has the wrong isolation mode")
    if provenance.get("isolated_formula_names_written_to_feature_rows") is not False:
        raise RuntimeError("development feature rows declare isolated formula disclosure")
    success = pd.read_csv(
        args.features_dir / "interpretable_features.csv", usecols=["formula"]
    )
    exclusions = pd.read_csv(
        args.features_dir / "structure_exclusions.csv", usecols=["formula"]
    )
    present = {
        canonical_formula(value)
        for value in pd.concat([success["formula"], exclusions["formula"]]).dropna()
    }
    leaked = sorted(present & blind)
    if leaked:
        raise RuntimeError("blind isolation failure in development success/exclusion inventory")
    selected_keys = feature_inventory_keys(args.features_dir)
    if len(selected_keys) != int(provenance["n_input_structures"]):
        raise RuntimeError("development success + exclusion ledger is incomplete")
    live_hash = full_cif_content_inventory_hash(args.cif_dir)
    if provenance.get("full_cif_content_inventory_sha256") != live_hash:
        raise RuntimeError("full CIF inventory changed after development feature isolation")
    if len(list(args.cif_dir.glob("*.cif"))) != int(
        provenance["n_inventory_structures_before_isolation"]
    ):
        raise RuntimeError("full CIF inventory count differs from feature provenance")


def validate_runtime_config(config: dict[str, Any]) -> None:
    validate_formal_feature_schema()
    if int(config.get("schema_version", -1)) != DISCOVERY_SCHEMA_VERSION:
        raise ValueError(
            "multiview discovery config schema_version must be "
            f"{DISCOVERY_SCHEMA_VERSION}"
        )
    if config.get("distance_scale_rule") != "development_nonprototype_positive_nearest_distance_q10":
        raise ValueError("unexpected distance-scale rule")
    if not math.isclose(float(config.get("distance_scale_quantile", -1)), 0.10):
        raise ValueError("base distance-scale quantile must be frozen at 0.10")
    retained = float(config["preprocessing"]["interpretable_pca_variance"])
    soap_retained = float(config["preprocessing"]["soap_pca_variance"])
    if max(config["robustness"]["pca_variance"]) > retained:
        raise ValueError("robustness PCA retention cannot exceed the fitted base transform")
    if soap_retained != 0.95:
        raise ValueError("SOAP clustering PCA retention must be frozen at 0.95")
    robustness_runs = len(registered_robustness_scenario_specs(config))
    if robustness_runs != 54:
        raise ValueError("pre-registered robustness ensemble must contain 54 scenarios")
    expected_subspaces = ["all", "even", "odd"]
    if config["hdbscan"]["component_subspaces"] != expected_subspaces:
        raise ValueError(f"HDBSCAN component_subspaces must equal {expected_subspaces}")
    validate_structure_matcher_config(config.get("deduplication"))


@locked_discovery_run
def run(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_runtime_config(config)
    blind = load_blind_formulas(args.blind_file)
    args.outdir.mkdir(parents=True, exist_ok=True)
    frozen_state: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    model_path: Path | None = None
    predevelopment_path: Path | None = None
    if args.stage == "develop":
        verify_feature_provenance_outputs(args.features_dir)
        verify_development_partition(args, blind)
        predevelopment_path = ensure_predevelopment_commitment(args)
        dev_frame, _ = load_feature_dir(args.features_dir)
        leaked = sorted(set(dev_frame["formula"].map(canonical_formula)) & blind)
        if leaked:
            raise RuntimeError("blind isolation failure: development descriptors contain isolated formulas")
        data = prepare(args.features_dir, None, config, args.cif_dir, args.n_jobs)
    else:
        if args.reveal_features_dir is None or args.frozen_manifest is None:
            raise ValueError("reveal requires --reveal-features-dir and --frozen-manifest")
        manifest = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
        model_path = verify_frozen(args, manifest)
        verify_reveal_partition(args, manifest, blind)
        frozen_state = load(model_path)
        if frozen_state.get("schema_version") != DISCOVERY_SCHEMA_VERSION:
            raise RuntimeError("unsupported frozen discovery model schema")
        frozen_robustness_scenarios = frozen_state.get("robustness_scenarios")
        if (
            not isinstance(frozen_robustness_scenarios, list)
            or not frozen_robustness_scenarios
        ):
            raise RuntimeError(
                "reveal requires a non-empty list of frozen robustness scenarios"
            )
        data = prepare(
            args.features_dir,
            args.reveal_features_dir,
            config,
            args.cif_dir,
            args.n_jobs,
            frozen_state=frozen_state,
        )

    scored, score_summary = score_candidates(
        data,
        config,
        distance_scales=(frozen_state["distance_scales"] if frozen_state else None),
    )
    scored, robustness_scenarios, robustness_summary = robustness_ensemble(
        data,
        scored,
        config,
        frozen_scenarios=(frozen_state["robustness_scenarios"] if frozen_state else None),
    )
    scored, clustering_summary, clustering_state = add_clustering(
        data,
        scored,
        config,
        args.outdir,
        frozen_clustering=(frozen_state["clustering"] if frozen_state else None),
    )
    scored = merge_extratrees(scored, args.extratrees_scores)
    umap_state = make_umap(
        data,
        scored,
        args.outdir,
        config,
        frozen_umap=(frozen_state["umap"] if frozen_state else None),
    )
    scored.sort_values("discovery_score", ascending=False).to_csv(args.outdir / "ranked_structures.csv", index=False)
    portfolio, portfolio_thresholds = build_portfolio(
        scored,
        config,
        frozen_thresholds=(frozen_state["portfolio_thresholds"] if frozen_state else None),
    )
    portfolio.to_csv(args.outdir / "candidate_portfolio.csv", index=False)

    if args.stage == "develop":
        if predevelopment_path is None:
            raise RuntimeError("predevelopment commitment was not established")
        loo, loo_summary = committed_leave_one_positive_out(
            args, predevelopment_path, data, config
        )
        loo.to_csv(args.outdir / "leave_one_positive_out.csv", index=False)
        (args.outdir / "leave_one_positive_out.summary.json").write_text(
            json.dumps(loo_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        frozen_state = build_frozen_state(
            data,
            score_summary,
            clustering_state,
            umap_state,
            portfolio_thresholds,
            robustness_scenarios,
        )
        model_path = args.outdir / FROZEN_MODEL_NAME
        save_frozen_state(frozen_state, model_path)
        manifest = freeze_manifest(
            args,
            config,
            data,
            score_summary,
            clustering_summary,
            model_path,
            portfolio_thresholds,
            robustness_summary,
        )
        (args.outdir / "frozen_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        reveal_mask = np.arange(len(scored)) >= data.development_count
        blind_report = scored.loc[reveal_mask].sort_values("discovery_score", ascending=False).copy()
        formula_rank_mask = scored["eligible_analysis"] & scored["is_structure_representative"]
        representative_blind_best = (
            scored.loc[reveal_mask & formula_rank_mask]
            .groupby("formula")["discovery_score"]
            .max()
        )
        all_formula_scores = scored.loc[formula_rank_mask].groupby("formula")["discovery_score"].max()
        formula_ranks = all_formula_scores.rank(ascending=False, method="min")
        blind_report["blind_formula_best_score"] = blind_report["formula"].map(
            representative_blind_best
        )
        blind_report["blind_formula_rank"] = blind_report["formula"].map(formula_ranks)
        blind_report["blind_formula_percentile"] = blind_report["blind_formula_rank"].map(
            lambda rank: 1.0 - (rank - 1) / max(len(all_formula_scores), 1)
        )
        blind_report.to_csv(args.outdir / "blind_acceptance_report.csv", index=False)
        audit = {
            "schema_version": DISCOVERY_SCHEMA_VERSION,
            "frozen_manifest_sha256": sha256_file(args.frozen_manifest),
            "frozen_model_sha256": sha256_file(model_path),
            "reveal_feature_provenance_sha256": sha256_file(args.reveal_features_dir / "provenance.json"),
            "reveal_feature_bundle_sha256": feature_bundle_hashes(args.reveal_features_dir),
            "n_revealed_structures": int(reveal_mask.sum()),
            "blind_outcomes_used_to_refit": False,
            "execution_path": "frozen_transform_approximate_predict_only",
            "development_reveal_complete_partition_verified": True,
        }
        (args.outdir / "reveal_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"{args.stage} complete: {args.outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["develop", "reveal"], required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--reveal-features-dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blind-file", type=Path, required=True)
    parser.add_argument("--cif-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--extratrees-scores", type=Path)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
