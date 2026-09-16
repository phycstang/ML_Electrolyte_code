#!/usr/bin/env python3
"""Extract interpretable and pseudo-element SOAP features for binary halides.

The extractor is deliberately label-free.  In development mode the CIF composition is
read in a lightweight identity pass so blind formulas can be removed *before* any
descriptor is evaluated.  Filename formulas are checked, but never trusted as the
chemical identity of a structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import warnings
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import ConvexHull

from ase import Atoms
from dscribe.descriptors import SOAP
from pymatgen.analysis.dimensionality import get_structure_components
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Composition, Element, Structure
from pymatgen.io.ase import AseAtomsAdaptor

try:
    from src.discovery.bonding_vesta import (
        PeriodicMXGraph,
        build_vesta_mx_graph,
        vesta_rule_provenance,
    )
except (
    ModuleNotFoundError
):  # direct execution: python src/discovery/extract_features.py
    from bonding_vesta import (  # type: ignore[no-redef]
        PeriodicMXGraph,
        build_vesta_mx_graph,
        vesta_rule_provenance,
    )


HALOGENS = {"F", "Cl", "Br", "I"}
# Static free-atom polarizabilities in a0^3.  They are intentionally kept in the
# configuration-independent feature definition so a database-version change cannot
# silently change the four values.
HALOGEN_POLARIZABILITY_AU = {"F": 3.74, "Cl": 14.6, "Br": 21.0, "I": 32.9}
ID_RE = re.compile(r"(mp-\d+)")
EXPECTED_SOAP_R_CUT_VARIANTS_A = (5.0, 6.0, 7.0)
DISCOVERY_SCHEMA_VERSION = 4
TRIANGULAR_REFERENCE_RELATIVE_PATH = Path(
    "experiments/triangular_lattice/t23_reference.py"
)
EXPECTED_TRIANGULAR_REFERENCE_SHA256 = (
    "d9bf4e9f4ffc306e5440a3f42f4f3a2f4c02f92e2c0cfb7f04015940205188cc"
)


def canonical_formula(value: str) -> str:
    return Composition(str(value)).reduced_formula


def formula_from_cif_name(filename: str) -> str:
    name = Path(str(filename)).name
    raw = name.split("_mp-", 1)[0] if "_mp-" in name else name.rsplit(".", 1)[0]
    return canonical_formula(raw)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Any) -> str:
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def triangular_reference_provenance() -> dict[str, str]:
    """Bind the adapted descriptor to the immutable, non-executed snapshot."""

    repository_root = Path(__file__).resolve().parents[2]
    reference = repository_root / TRIANGULAR_REFERENCE_RELATIVE_PATH
    if not reference.is_file():
        raise FileNotFoundError(
            f"triangular reference snapshot is missing: {reference}"
        )
    digest = sha256_file(reference)
    if digest != EXPECTED_TRIANGULAR_REFERENCE_SHA256:
        raise RuntimeError(
            "triangular reference snapshot hash mismatch: "
            f"expected {EXPECTED_TRIANGULAR_REFERENCE_SHA256}, got {digest}"
        )
    return {
        "path": TRIANGULAR_REFERENCE_RELATIVE_PATH.as_posix(),
        "sha256": digest,
        "relationship": "adapted_reference_not_executed",
    }


def soap_variant_key(r_cut_A: float) -> str:
    """Return the stable JSON key used for a pre-registered SOAP cutoff."""

    value = float(r_cut_A)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"SOAP r_cut must be positive and finite, got {r_cut_A!r}")
    return f"r_cut_{value:.1f}A".replace(".", "p")


def soap_variant_filename(r_cut_A: float, base_r_cut_A: float) -> str:
    """Keep the historical 6 A filename while naming sensitivity arrays explicitly."""

    if math.isclose(float(r_cut_A), float(base_r_cut_A), abs_tol=1e-12):
        return "soap_pseudo_mx.npy"
    token = f"{float(r_cut_A):.1f}".replace(".", "p")
    return f"soap_pseudo_mx_rcut_{token}.npy"


def validate_soap_cutoff_contract(config: dict[str, Any]) -> tuple[float, ...]:
    """Validate the extraction and robustness SOAP-cutoff declarations together."""

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
    if len({soap_variant_key(value) for value in extraction}) != len(extraction):
        raise ValueError("SOAP r_cut variants do not have unique serialized keys")
    return extraction


def dataframe_sha256(frame: pd.DataFrame) -> str:
    text = frame.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def finite(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def validate_bonding_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Fail closed if config and the installed frozen VESTA rule disagree."""

    declared = config.get("bonding")
    if not isinstance(declared, dict):
        raise ValueError("config must contain a bonding contract")
    runtime = vesta_rule_provenance(verify=True)
    checks = {
        "backend": runtime["backend"],
        "preset": runtime["preset"],
        "cutoff_table_sha256": runtime["cutoff_table_sha256"],
        "cutoff_pair_count": runtime["cutoff_pair_count"],
        "distance_comparison": runtime["distance_comparison"],
        "fallback": None,
        "retain_periodic_images": True,
    }
    mismatch = {
        key: {"declared": declared.get(key), "runtime": expected}
        for key, expected in checks.items()
        if declared.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"bonding contract mismatch: {mismatch}")
    if int(declared.get("minimum_cn_for_polyhedron", -1)) != 3:
        raise ValueError("discovery-v2 polyhedron minimum CN is frozen at 3")
    eligibility = config.get("eligibility", {})
    for policy in (
        "unsupported_vesta_pair_policy",
        "no_vesta_mx_bond_policy",
        "structure_feature_error_policy",
    ):
        if eligibility.get(policy) != "exclude_with_ledger":
            raise ValueError(f"{policy} must be frozen as exclude_with_ledger")
    return runtime


def element_radius_A(element: Element) -> float:
    """Reproducible size proxy, preferring atomic then calculated atomic radius."""
    for key in ("atomic_radius", "atomic_radius_calculated", "van_der_waals_radius"):
        val = finite(getattr(element, key, None))
        if np.isfinite(val) and val > 0:
            return val
    return 1.5


def ionic_radius_A(element: Element, nominal_charge: float) -> float:
    radii = []
    try:
        for charge, radius in element.ionic_radii.items():
            if float(charge) > 0 and finite(radius) > 0:
                radii.append((abs(float(charge) - nominal_charge), finite(radius)))
    except Exception:
        pass
    if radii:
        return min(radii)[1]
    val = finite(getattr(element, "average_ionic_radius", None))
    return val if np.isfinite(val) and val > 0 else element_radius_A(element)


def shannon_coordination_environment_count(
    element: Element, nominal_charge: float
) -> float:
    """Count tabulated Shannon coordination environments near the nominal charge.

    Pymatgen's immutable element table stores coordination labels (for example IV,
    IVSQ, VI) by oxidation state.  Selecting the closest positive tabulated state
    gives a reproducible static proxy for the centre element's coordination
    adaptability without learning anything from the candidate or acceptance set.
    """

    table = element.data.get("Shannon radii") or {}
    candidates: list[tuple[float, str]] = []
    if isinstance(table, dict):
        for charge in table:
            try:
                numeric = float(charge)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                candidates.append((abs(numeric - nominal_charge), str(charge)))
    if not candidates:
        return 0.0
    selected = min(candidates)[1]
    environments = table.get(selected) or {}
    return float(len(environments)) if isinstance(environments, dict) else 0.0


def identify_binary_halide(structure: Structure) -> tuple[str, str, float, float]:
    comp = structure.composition.reduced_composition
    amounts = {el.symbol: float(amount) for el, amount in comp.items()}
    hal = [symbol for symbol in amounts if symbol in HALOGENS]
    center = [symbol for symbol in amounts if symbol not in HALOGENS]
    if len(amounts) != 2 or len(hal) != 1 or len(center) != 1:
        raise ValueError(f"not a binary single-halogen structure: {comp.formula}")
    m, x = center[0], hal[0]
    return m, x, amounts[m], amounts[x]


def standardize_descriptor_structure(
    structure: Structure, config: dict[str, Any]
) -> Structure:
    contract = config["structure"].get("descriptor_cell_standardization")
    if contract != "primitive_then_niggli":
        raise ValueError(
            "descriptor_cell_standardization must be primitive_then_niggli"
        )
    tolerance = float(config["structure"]["primitive_tolerance_A"])
    primitive = structure.get_primitive_structure(
        tolerance=tolerance, use_site_props=False
    )
    return primitive.get_reduced_structure(reduction_algo="niggli")


def chemistry_features(structure: Structure) -> dict[str, float | str]:
    m, x, n_m, n_x = identify_binary_halide(structure)
    em, ex = Element(m), Element(x)
    ratio = n_x / n_m
    q = ratio  # nominal charge balance with X^-; explicitly a proxy
    r_m = ionic_radius_A(em, q)
    r_x = element_radius_A(ex)
    chi_m, chi_x = finite(em.X), finite(ex.X)
    common_ox = [v for v in em.common_oxidation_states if v > 0]
    all_ox = [v for v in em.oxidation_states if v > 0]
    return {
        "center_element": m,
        "halogen_element": x,
        "chem__x_over_m": ratio,
        "chem__nominal_q_m": q,
        "chem__ionic_radius_m_A": r_m,
        "chem__radius_x_A": r_x,
        "chem__q_over_r": q / max(r_m, 1e-8),
        "chem__q_over_r2": q / max(r_m * r_m, 1e-8),
        "chem__radius_ratio_m_x": r_m / max(r_x, 1e-8),
        "chem__chi_m_pauling": chi_m,
        "chem__chi_x_pauling": chi_x,
        "chem__delta_chi_x_m": chi_x - chi_m,
        "chem__halogen_polarizability_a0_3": HALOGEN_POLARIZABILITY_AU[x],
        "chem__atomic_number_m": float(em.Z),
        "chem__group_m": finite(em.group),
        "chem__period_m": finite(em.row),
        "chem__n_common_positive_oxidation_states": float(len(common_ox)),
        "chem__n_positive_oxidation_states": float(len(all_ox)),
        "chem__n_shannon_coordination_environments": (
            shannon_coordination_environment_count(em, q)
        ),
        "chem__softness_proxy_px_over_field": HALOGEN_POLARIZABILITY_AU[x]
        / max(q / max(r_m * r_m, 1e-8), 1e-8),
    }


def _neighbor_key(info: dict[str, Any]) -> tuple[int, int, int, int]:
    image = tuple(int(round(float(v))) for v in info.get("image", (0, 0, 0)))
    return (int(info["site_index"]), *image)


def _geometry_similarity(
    vectors: np.ndarray, ideal_sorted_cosines: np.ndarray
) -> float:
    if len(vectors) < 2:
        return 0.0
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    observed = np.sort(
        np.asarray(
            [np.dot(unit[i], unit[j]) for i, j in combinations(range(len(unit)), 2)]
        )
    )
    if observed.shape != ideal_sorted_cosines.shape:
        return 0.0
    mse = float(np.mean((observed - ideal_sorted_cosines) ** 2))
    return float(np.exp(-mse / 0.08))


IDEAL_COSINES = {
    "tetra": np.sort(np.full(6, -1.0 / 3.0)),
    "trigonal_bipyramid": np.sort(np.asarray([-1.0] + [0.0] * 6 + [-0.5] * 3)),
    "square_pyramid": np.sort(np.asarray([-1.0] * 2 + [0.0] * 8)),
    "octa": np.sort(np.asarray([-1.0] * 3 + [0.0] * 12)),
}


def _percentile_summary(values: Iterable[float], prefix: str) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(arr):
        return {f"{prefix}mean": np.nan, f"{prefix}std": np.nan}
    return {f"{prefix}mean": float(arr.mean()), f"{prefix}std": float(arr.std())}


def _periodic_patch_metrics(
    center_indices: list[int],
    pair_shared: Counter[tuple[int, int, int, int, int]],
) -> tuple[float, float, float]:
    """Return image-aware clustering, ring participation, and shortest ring.

    A finite open patch is constructed from the periodic edge orbits.  It has no
    wraparound edges, so a 1D translational chain cannot become an artificial
    ring.  Metrics are evaluated only on nodes in the central cell.
    """

    if not pair_shared or not center_indices:
        return 0.0, 0.0, 0.0
    max_image = max(max(abs(dx), abs(dy), abs(dz)) for _, _, dx, dy, dz in pair_shared)
    radius = max(1, 2 * max_image)
    cells = list(product(range(-radius, radius + 1), repeat=3))
    cell_set = set(cells)
    graph = nx.Graph()
    graph.add_nodes_from((idx, *cell) for cell in cells for idx in center_indices)
    for i, j, dx, dy, dz in pair_shared:
        delta = np.asarray((dx, dy, dz), dtype=int)
        for cell in cells:
            target = tuple((np.asarray(cell, dtype=int) + delta).tolist())
            if target in cell_set:
                graph.add_edge((i, *cell), (j, *target))

    central = [(idx, 0, 0, 0) for idx in center_indices]
    clustering = float(np.mean([nx.clustering(graph, node) for node in central]))
    ring_sizes: list[int] = []
    for node in central:
        neighbors = list(graph.neighbors(node))
        if len(neighbors) < 2:
            continue
        without = nx.restricted_view(graph, [node], [])
        best: int | None = None
        for pos, source in enumerate(neighbors[:-1]):
            lengths = nx.single_source_shortest_path_length(without, source)
            for target in neighbors[pos + 1 :]:
                if target in lengths:
                    size = int(lengths[target] + 2)
                    best = size if best is None else min(best, size)
        if best is not None:
            ring_sizes.append(best)
    participation = len(ring_sizes) / len(central)
    shortest = float(min(ring_sizes)) if ring_sizes else 0.0
    return clustering, float(participation), shortest


def local_topology_features(
    structure: Structure,
    x_symbol: str,
    config: dict[str, Any],
    mx_graph: PeriodicMXGraph | None = None,
) -> tuple[dict[str, float], list[list[dict[str, Any]]]]:
    minimum_cn = int(config["bonding"]["minimum_cn_for_polyhedron"])
    mx_graph = mx_graph or build_vesta_mx_graph(structure)
    if mx_graph.halogen_symbol != x_symbol:
        raise ValueError(
            f"halogen identity mismatch: {x_symbol} != {mx_graph.halogen_symbol}"
        )
    center_indices = list(mx_graph.center_indices)
    x_indices = list(mx_graph.halogen_indices)
    all_infos: list[list[dict[str, Any]]] = []
    cn_values: list[float] = []
    center_bond_means: list[float] = []
    center_distortions: list[float] = []
    center_angle_vars: list[float] = []
    center_volumes: list[float] = []
    geom_scores = {name: [] for name in IDEAL_COSINES}
    geom_labels: list[str] = []

    # X(j@0) -> M(i@translation) is an exact periodic inversion of all M@0-X bonds.
    x_to_metals = mx_graph.x_to_center_images()
    sg = mx_graph.structure_graph()
    by_center = mx_graph.bonds_by_center()

    for center_index in center_indices:
        infos = mx_graph.neighbor_info(center_index)
        all_infos.append(infos)
        cn = len(infos)
        cn_values.append(float(cn))
        vectors, distances = [], []
        for info in infos:
            key = _neighbor_key(info)
            j, tx, ty, tz = key
            vec = np.asarray(info["site"].coords) - np.asarray(
                structure[center_index].coords
            )
            vectors.append(vec)
            distances.append(float(np.linalg.norm(vec)))

        d = np.asarray(distances, dtype=float)
        if len(d):
            center_bond_means.append(float(d.mean()))
            center_distortions.append(
                float(np.mean(np.abs(d - d.mean())) / max(d.mean(), 1e-8))
            )
        if len(vectors) >= 2:
            unit = np.asarray(vectors) / np.maximum(
                np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12
            )
            angles = [
                np.degrees(np.arccos(np.clip(np.dot(unit[i], unit[j]), -1.0, 1.0)))
                for i, j in combinations(range(len(unit)), 2)
            ]
            center_angle_vars.append(float(np.var(angles)))
        if len(vectors) >= 4:
            try:
                center_volumes.append(float(ConvexHull(np.asarray(vectors)).volume))
            except Exception:
                pass

        possible: dict[str, float] = {}
        if cn == 4:
            possible["tetra"] = _geometry_similarity(
                np.asarray(vectors), IDEAL_COSINES["tetra"]
            )
        elif cn == 5:
            possible["trigonal_bipyramid"] = _geometry_similarity(
                np.asarray(vectors), IDEAL_COSINES["trigonal_bipyramid"]
            )
            possible["square_pyramid"] = _geometry_similarity(
                np.asarray(vectors), IDEAL_COSINES["square_pyramid"]
            )
        elif cn == 6:
            possible["octa"] = _geometry_similarity(
                np.asarray(vectors), IDEAL_COSINES["octa"]
            )
        for name in geom_scores:
            geom_scores[name].append(possible.get(name, 0.0))
        if possible and max(possible.values()) >= 0.50:
            geom_labels.append(max(possible, key=possible.get))
        elif possible:
            geom_labels.append("irregular")
        else:
            # CN values outside the pre-registered 4/5/6 templates are not
            # evidence for an irregular tetrahedron/pyramid/octahedron.  Keep
            # them outside the geometry denominator and expose the coverage
            # explicitly below.
            geom_labels.append("unclassifiable")

    eligible_centers = {
        idx for idx in center_indices if len(by_center.get(idx, ())) >= minimum_cn
    }
    pair_shared = mx_graph.shared_polyhedron_pairs(eligible_centers=eligible_centers)
    poly_metrics = mx_graph.polyhedron_metrics(minimum_cn=minimum_cn)
    poly_network_dim = mx_graph.pair_network_dimension(pair_shared)
    edge_pairs = Counter(
        {key: shared for key, shared in pair_shared.items() if shared == 2}
    )
    edge_network_dim = mx_graph.pair_network_dimension(edge_pairs)

    n_pairs = len(pair_shared)
    corner = sum(v == 1 for v in pair_shared.values())
    edge = sum(v == 2 for v in pair_shared.values())
    face = sum(v >= 3 for v in pair_shared.values())
    degrees = mx_graph.periodic_degrees(center_indices, pair_shared)
    quotient = nx.Graph()
    quotient.add_nodes_from(center_indices)
    for i, j, dx, dy, dz in pair_shared:
        quotient.add_edge(i, j)

    x_degrees = [len(x_to_metals.get(j, set())) for j in x_indices]
    terminal_fraction = float(np.mean(np.asarray(x_degrees) == 1)) if x_degrees else 0.0
    bridge_fraction = float(np.mean(np.asarray(x_degrees) >= 2)) if x_degrees else 0.0
    active_nodes = [idx for idx in center_indices if degrees[idx]]
    active_quotient = quotient.subgraph(active_nodes)
    components = list(nx.connected_components(active_quotient)) if active_nodes else []
    clustering, ring_participation, shortest_ring = _periodic_patch_metrics(
        center_indices, pair_shared
    )
    connection_probs = np.asarray([corner, edge, face], dtype=float)
    if connection_probs.sum() > 0:
        connection_probs /= connection_probs.sum()
        connection_entropy = float(
            -np.sum(
                connection_probs[connection_probs > 0]
                * np.log(connection_probs[connection_probs > 0])
            )
            / math.log(3.0)
        )
    else:
        connection_entropy = 0.0

    dim_fractions = {d: 0.0 for d in range(4)}
    component_count = 0
    max_component_fraction = 0.0
    dominant_dim = float(mx_graph.dimensionality())
    structural_components = list(
        get_structure_components(sg, inc_orientation=True, inc_site_ids=True)
    )
    component_count = len(structural_components)
    for component in structural_components:
        dim = int(component["dimensionality"])
        weight = len(component.get("site_ids", [])) / max(len(structure), 1)
        dim_fractions[dim] += weight
        max_component_fraction = max(max_component_fraction, weight)

    cn_arr = np.asarray(cn_values, dtype=float)
    cn_counts = Counter(int(round(v)) for v in cn_values)
    total_centers = max(len(center_indices), 1)
    cn_probs = np.asarray(list(cn_counts.values()), dtype=float) / total_centers
    cn_entropy = (
        float(-np.sum(cn_probs * np.log(cn_probs)) / math.log(max(len(cn_probs), 2)))
        if len(cn_probs) > 1
        else 0.0
    )
    labels = Counter(geom_labels)
    result: dict[str, float] = {
        "local__cn_mean": float(cn_arr.mean()) if len(cn_arr) else 0.0,
        "local__cn_std": float(cn_arr.std()) if len(cn_arr) else 0.0,
        "local__cn_min": float(cn_arr.min()) if len(cn_arr) else 0.0,
        "local__cn_max": float(cn_arr.max()) if len(cn_arr) else 0.0,
        "local__cn_entropy": cn_entropy,
        "local__bond_length_mean_A": float(np.mean(center_bond_means))
        if center_bond_means
        else np.nan,
        "local__bond_length_between_center_std_A": float(np.std(center_bond_means))
        if center_bond_means
        else np.nan,
        "local__bond_distortion_mean": float(np.mean(center_distortions))
        if center_distortions
        else np.nan,
        "local__bond_distortion_std": float(np.std(center_distortions))
        if center_distortions
        else np.nan,
        "local__angle_variance_deg2": float(np.mean(center_angle_vars))
        if center_angle_vars
        else np.nan,
        "local__polyhedron_volume_mean_A3": float(np.mean(center_volumes))
        if center_volumes
        else np.nan,
        "local__polyhedron_volume_cv": (
            float(np.std(center_volumes) / max(np.mean(center_volumes), 1e-8))
            if center_volumes
            else np.nan
        ),
        "local__tetra_like_fraction": labels["tetra"] / total_centers,
        "local__trigonal_bipyramid_like_fraction": labels["trigonal_bipyramid"]
        / total_centers,
        "local__square_pyramid_like_fraction": labels["square_pyramid"] / total_centers,
        "local__octa_like_fraction": labels["octa"] / total_centers,
        "local__irregular_fraction": labels["irregular"] / total_centers,
        "local__geometry_classifiable_fraction": (
            (total_centers - labels["unclassifiable"]) / total_centers
        ),
        "local__tetra_similarity_mean": float(np.mean(geom_scores["tetra"]))
        if geom_scores["tetra"]
        else 0.0,
        "local__octa_similarity_mean": float(np.mean(geom_scores["octa"]))
        if geom_scores["octa"]
        else 0.0,
        "vesta__dim": float(poly_metrics["dim"]),
        "vesta__Xcn": float(poly_metrics["st1"]),
        "vesta__Xsh": float(poly_metrics["st2"]),
        "vesta__Pcn": float(poly_metrics["st3"]),
        "vesta__mx_cutoff_A": float(mx_graph.cutoff_A),
        "vesta__mx_bond_orbit_count": float(len(mx_graph.bonds)),
        "vesta__polyhedron_count": float(poly_metrics["n_polyhedra"]),
        "topology__corner_share_fraction": corner / max(n_pairs, 1),
        "topology__edge_share_fraction": edge / max(n_pairs, 1),
        "topology__face_share_fraction": face / max(n_pairs, 1),
        "topology__connection_type_entropy": connection_entropy,
        "topology__polyhedron_network_dimension": float(poly_network_dim),
        "topology__edge_network_dimension": float(edge_network_dim),
        "topology__center_degree_mean_all_centers": (
            float(np.mean([len(v) for v in degrees.values()])) if degrees else 0.0
        ),
        "topology__center_degree_std_all_centers": (
            float(np.std([len(v) for v in degrees.values()])) if degrees else 0.0
        ),
        "topology__polyhedron_eligible_fraction": len(eligible_centers) / total_centers,
        "topology__halogen_degree_mean": float(np.mean(x_degrees))
        if x_degrees
        else 0.0,
        "topology__bridge_halogen_fraction": bridge_fraction,
        "topology__terminal_halogen_fraction": terminal_fraction,
        "topology__unbonded_halogen_fraction": (
            float(np.mean(np.asarray(x_degrees) == 0)) if x_degrees else 0.0
        ),
        "topology__bridge_terminal_coexistence": 4.0
        * bridge_fraction
        * terminal_fraction,
        "audit__poly_graph_component_fraction_max": (
            max((len(c) for c in components), default=0) / max(len(center_indices), 1)
        ),
        "topology__component_count_per_atom": component_count / max(len(structure), 1),
        "topology__structure_component_fraction_max": max_component_fraction,
        "audit__periodic_patch_graph_clustering": clustering,
        "audit__periodic_patch_ring_participation": ring_participation,
        "audit__periodic_patch_shortest_ring_size": shortest_ring,
        "audit__edge_component_fraction": 0.0,
        "topology__dominant_dimension": dominant_dim,
        "topology__dimension_coexistence_count": float(
            sum(v > 1e-8 for v in dim_fractions.values())
        ),
    }
    for cn in range(3, 9):
        result[f"local__cn{cn}_fraction"] = cn_counts[cn] / total_centers
    for dim in range(4):
        result[f"topology__dim{dim}_fraction"] = dim_fractions[dim]

    edge_graph = nx.Graph()
    for (i, j, _dx, _dy, _dz), shared in pair_shared.items():
        if shared == 2:
            edge_graph.add_edge(i, j)
    if edge_graph.number_of_edges() > 0:
        result["audit__edge_component_fraction"] = max(
            (len(c) for c in nx.connected_components(edge_graph)), default=0
        ) / max(len(center_indices), 1)
    return result, all_infos


def _canonical_layer_normal(vector: np.ndarray) -> np.ndarray:
    """Return a deterministic representative of an unoriented plane normal."""

    normal = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError("zero vector cannot define a layer normal")
    normal = normal / norm
    pivot = int(np.argmax(np.abs(normal)))
    return -normal if normal[pivot] < 0 else normal


def _plane_basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _canonical_layer_normal(normal)
    axes = np.eye(3)
    reference = axes[int(np.argmin(np.abs(axes @ normal)))]
    ex = np.cross(normal, reference)
    ex /= max(float(np.linalg.norm(ex)), 1e-12)
    ey = np.cross(normal, ex)
    ey /= max(float(np.linalg.norm(ey)), 1e-12)
    return ex, ey


def _projective_weighted_center(
    weighted_vectors: list[tuple[np.ndarray, int]],
) -> np.ndarray:
    """Average unoriented plane normals without sign cancellation.

    A layer normal represents a projective direction: ``n`` and ``-n`` are the
    same plane.  An ordinary vector sum can therefore become exactly zero when
    individually canonicalized representatives switch hemispheres near a
    component tie.  The second-moment tensor is invariant to that arbitrary
    sign and its leading eigenvector supplies a deterministic unit centre.
    """

    support = sum(int(weight) for _vector, weight in weighted_vectors)
    if support <= 0:
        raise ValueError("projective normal cluster must have positive support")
    tensor = (
        sum(
            int(weight) * np.outer(vector, vector)
            for vector, weight in weighted_vectors
        )
        / support
    )
    _values, eigenvectors = np.linalg.eigh(tensor)
    return _canonical_layer_normal(eigenvectors[:, -1])


def _periodic_halogen_cloud(
    structure: Structure, x_indices: list[int], repetitions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    """Build an odd, origin-centred periodic X patch and central-row map."""

    if repetitions < 3 or repetitions % 2 != 1:
        raise ValueError("triangular_supercell_repetitions must be an odd integer >= 3")
    half = repetitions // 2
    positions: list[np.ndarray] = []
    site_ids: list[int] = []
    images: list[tuple[int, int, int]] = []
    central_rows: dict[int, int] = {}
    for image in product(range(-half, half + 1), repeat=3):
        shift = np.asarray(image, dtype=float)
        for site_id in x_indices:
            row = len(positions)
            frac = np.asarray(structure[site_id].frac_coords, dtype=float) + shift
            positions.append(np.asarray(structure.lattice.get_cartesian_coords(frac)))
            site_ids.append(site_id)
            images.append(image)
            if image == (0, 0, 0):
                central_rows[site_id] = row
    return (
        np.asarray(positions, dtype=float),
        np.asarray(site_ids, dtype=int),
        np.asarray(images, dtype=int),
        central_rows,
    )


def _vesta_patch_components(
    mx_graph: PeriodicMXGraph,
    cloud_site_ids: np.ndarray,
    cloud_images: np.ndarray,
    repetitions: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Label finite periodic X nodes by their exact VESTA M-X component.

    Base-cell site ids alone cannot distinguish translated copies of a 2D slab:
    adjacent, disconnected slabs share the same crystallographic ids.  The open
    patch below retains image-labelled nodes, so a core-6 shell cannot borrow X
    atoms from another translated slab.  The patch has a two-cell safety margin
    around every descriptor point to keep central-cell reachability away from an
    artificial boundary.
    """

    component_dimensions: dict[int, int] = {}
    for component in get_structure_components(
        mx_graph.structure_graph(), inc_orientation=True, inc_site_ids=True
    ):
        dimension = int(component["dimensionality"])
        for site_id in component.get("site_ids", ()):
            previous = component_dimensions.setdefault(int(site_id), dimension)
            if previous != dimension:
                raise RuntimeError("one VESTA site was assigned conflicting dimensions")

    half = repetitions // 2
    max_bond_image = max(
        (max(abs(v) for v in bond.image) for bond in mx_graph.bonds), default=0
    )
    radius = half + max_bond_image + 2
    cells = list(product(range(-radius, radius + 1), repeat=3))
    cell_set = set(cells)
    graph = nx.Graph()
    graph.add_nodes_from(
        (site_id, *cell) for cell in cells for site_id in range(len(mx_graph.structure))
    )
    for bond in mx_graph.bonds:
        delta = np.asarray(bond.image, dtype=int)
        for cell in cells:
            target_cell = tuple((np.asarray(cell, dtype=int) + delta).tolist())
            if target_cell in cell_set:
                graph.add_edge(
                    (bond.center_index, *cell),
                    (bond.halogen_index, *target_cell),
                )
    node_component: dict[tuple[int, int, int, int], int] = {}
    for component_id, nodes in enumerate(nx.connected_components(graph)):
        for node in nodes:
            node_component[node] = component_id
    cloud_components = np.asarray(
        [
            node_component[(int(site_id), *tuple(int(v) for v in image))]
            for site_id, image in zip(cloud_site_ids, cloud_images)
        ],
        dtype=int,
    )
    return cloud_components, component_dimensions


def _candidate_layer_normals(
    positions: np.ndarray,
    site_ids: np.ndarray,
    images: np.ndarray,
    central_rows: dict[int, int],
    patch_components: np.ndarray,
    component_dimensions: dict[int, int],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Generate and projectively cluster pair-plane normal candidates."""

    cfg = config["structure"]
    cutoff = float(cfg["xx_neighbor_cutoff_A"])
    pair_kmax = int(cfg["triangular_pair_kmax"])
    pair_min_angle = float(cfg["triangular_pair_min_angle_deg"])
    max_seeds = int(cfg["triangular_seed_max_variants_per_center"])
    seed_sep = float(cfg["triangular_seed_min_normal_separation_deg"])
    cluster_tol = float(cfg["triangular_normal_cluster_tolerance_deg"])
    min_support = int(cfg["triangular_min_normal_support"])
    cos_pair = math.cos(math.radians(pair_min_angle))
    cos_seed_sep = math.cos(math.radians(seed_sep))
    raw: list[tuple[np.ndarray, int]] = []

    for site_id, center_row in sorted(central_rows.items()):
        if component_dimensions.get(site_id, 0) < 2:
            continue
        center_component = int(patch_components[center_row])
        vectors = positions - positions[center_row]
        distances = np.linalg.norm(vectors, axis=1)
        rows = [
            row
            for row in range(len(positions))
            if row != center_row
            and int(patch_components[row]) == center_component
            and 1e-8 < float(distances[row]) <= cutoff
        ]
        rows.sort(
            key=lambda row: (
                float(distances[row]),
                int(site_ids[row]),
                tuple(int(v) for v in images[row]),
            )
        )
        rows = rows[:pair_kmax]
        local: list[dict[str, Any]] = []
        for left, right in combinations(rows, 2):
            va = vectors[left] / max(float(distances[left]), 1e-12)
            vb = vectors[right] / max(float(distances[right]), 1e-12)
            if abs(float(np.dot(va, vb))) >= cos_pair:
                continue
            cross = np.cross(va, vb)
            if float(np.linalg.norm(cross)) < 1e-10:
                continue
            normal = _canonical_layer_normal(cross)
            match = next(
                (
                    idx
                    for idx, item in enumerate(local)
                    if abs(float(np.dot(normal, item["normal"]))) >= cos_seed_sep
                ),
                None,
            )
            if match is not None:
                local[match]["votes"] += 1
                continue
            if len(local) < max_seeds:
                local.append({"normal": normal, "votes": 1})
        raw.extend((item["normal"], int(item["votes"])) for item in local)

    raw.sort(key=lambda value: tuple(float(v) for v in np.round(value[0], 12)))
    clusters: list[list[tuple[np.ndarray, int]]] = []
    centers: list[np.ndarray] = []
    cos_cluster = math.cos(math.radians(cluster_tol))
    for normal, votes in raw:
        match = next(
            (
                idx
                for idx, center in enumerate(centers)
                if abs(float(np.dot(normal, center))) >= cos_cluster
            ),
            None,
        )
        if match is None:
            clusters.append([(normal, votes)])
            centers.append(normal)
            continue
        clusters[match].append((normal, votes))
        centers[match] = _projective_weighted_center(clusters[match])

    candidates: list[dict[str, Any]] = []
    for weighted_vectors in clusters:
        support = sum(weight for _vector, weight in weighted_vectors)
        if support < min_support:
            continue
        tensor = (
            sum(
                weight * np.outer(vector, vector) for vector, weight in weighted_vectors
            )
            / support
        )
        values, eigenvectors = np.linalg.eigh(tensor)
        normal = _canonical_layer_normal(eigenvectors[:, -1])
        candidates.append(
            {
                "normal": normal,
                "support": support,
                "coherence": float(values[-1]),
            }
        )
    candidates.sort(
        key=lambda item: tuple(float(v) for v in np.round(item["normal"], 12))
    )
    return candidates, sum(votes for _normal, votes in raw)


def _robust_projected_layers(
    positions: np.ndarray, normal: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, list[dict[str, float]], float]:
    """Robust 1D layer partition adapted from the read-only t23 reference.

    The implementation deliberately fails closed instead of treating an
    unresolved projection as one giant layer.  Returned memberships are rebuilt
    from the final labels after outlier rejection and soft reassignment.
    """

    cfg = config["structure"]
    eps_min = float(cfg["triangular_layer_eps_min_A"])
    eps_max = float(cfg["triangular_layer_eps_max_A"])
    min_size = int(cfg["triangular_min_layer_size"])
    merge_gain = float(cfg["triangular_layer_merge_gain"])
    outlier_sigma = float(cfg["triangular_layer_outlier_sigma"])
    reassign_factor = float(cfg["triangular_layer_reassign_margin_factor"])
    flatness_max = float(cfg["triangular_layer_flatness_max_A"])
    z = np.asarray(positions @ _canonical_layer_normal(normal), dtype=float)
    if not len(z):
        return np.asarray([], dtype=int), [], eps_min

    ordered = np.sort(z)
    gaps = np.abs(np.diff(ordered))
    if len(gaps):
        lower_count = max(1, int(0.70 * len(gaps)))
        lower = np.sort(gaps)[:lower_count]
        eps = float(np.clip(1.1 * np.quantile(lower, 0.75), eps_min, eps_max))
    else:
        eps = eps_max

    order = np.argsort(z, kind="mergesort")
    groups: list[np.ndarray] = []
    start = 0
    for position, gap in enumerate(np.diff(z[order])):
        if float(gap) > eps:
            groups.append(order[start : position + 1])
            start = position + 1
    groups.append(order[start:])

    cleaned: list[np.ndarray] = []
    for group in groups:
        if len(group) < min_size:
            continue
        values = z[group]
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        sigma = 1.4826 * mad
        keep = np.abs(values - median) <= max(eps_min, outlier_sigma * sigma)
        retained = np.asarray(group[keep], dtype=int)
        if len(retained) >= min_size:
            cleaned.append(retained)
    cleaned.sort(key=lambda group: float(np.median(z[group])))

    merged: list[np.ndarray] = []
    for group in cleaned:
        if not merged:
            merged.append(group)
            continue
        previous = merged[-1]
        med_a = float(np.median(z[previous]))
        med_b = float(np.median(z[group]))
        sig_a = 1.4826 * float(np.median(np.abs(z[previous] - med_a)))
        sig_b = 1.4826 * float(np.median(np.abs(z[group] - med_b)))
        if med_b - med_a <= max(merge_gain * eps, 2.0 * sig_a, 2.0 * sig_b):
            merged[-1] = np.asarray(sorted(set(previous) | set(group)), dtype=int)
        else:
            merged.append(group)
    if not merged:
        return np.full(len(z), -1, dtype=int), [], eps

    medians = np.asarray([np.median(z[group]) for group in merged], dtype=float)
    sigmas = np.asarray(
        [
            1.4826 * np.median(np.abs(z[group] - med))
            for group, med in zip(merged, medians)
        ],
        dtype=float,
    )
    labels = np.full(len(z), -1, dtype=int)
    for row, value in enumerate(z):
        layer = int(np.argmin(np.abs(medians - value)))
        threshold = max(eps, outlier_sigma * float(sigmas[layer])) * reassign_factor
        if abs(float(value - medians[layer])) <= threshold:
            labels[row] = layer

    # Rebuild from final labels.  This fixes the reference implementation's
    # subtle mismatch between soft-reassigned labels and stale member arrays.
    final_layers: list[dict[str, float]] = []
    rebuilt_labels = np.full(len(z), -1, dtype=int)
    for old_layer in range(len(merged)):
        rows = np.where(labels == old_layer)[0]
        if len(rows) < min_size:
            continue
        median = float(np.median(z[rows]))
        mad = float(np.median(np.abs(z[rows] - median)))
        sigma = 1.4826 * mad
        new_layer = len(final_layers)
        rebuilt_labels[rows] = new_layer
        final_layers.append(
            {
                "median_A": median,
                "mad_A": mad,
                "sigma_A": sigma,
                "reliable": float(sigma <= flatness_max),
            }
        )
    return rebuilt_labels, final_layers, eps


def _triangular_empty_features() -> dict[str, float]:
    """Return explicit applicability/coverage and NaN physical quantities."""

    return {
        "tri__is_applicable": 0.0,
        "tri__layer_coverage_fraction": 0.0,
        "tri__applicable_center_fraction": 0.0,
        "tri__center_valid_fraction": 0.0,
        "tri__center_pass_fraction": 0.0,
        "tri__unique_same_layer_core6_fraction": 0.0,
        "tri__psi6_median": np.nan,
        "tri__psi6_p10": np.nan,
        "tri__psi6_iqr": np.nan,
        "tri__psi4_median": np.nan,
        "tri__angular_gap_rmse_deg": np.nan,
        "tri__xx_inplane_mean_A": np.nan,
        "tri__xx_inplane_cv": np.nan,
        "tri__buckling_mad_A": np.nan,
        "tri__buckling_over_xx": np.nan,
        "tri__normal_coherence": np.nan,
        "stack__is_applicable": 0.0,
        "audit__stack_reliable_patch_layer_count": 0.0,
        "stack__coverage_fraction": 0.0,
        "stack__spacing_mean_A": np.nan,
        "stack__spacing_cv": np.nan,
        "stack__spacing_over_xx": np.nan,
        "stack__inter_intra_xx_ratio": np.nan,
        "audit__tri_candidate_normal_count": 0.0,
        "audit__tri_raw_normal_count": 0.0,
        "audit__tri_selected_normal_support": 0.0,
        "audit__tri_vesta_dim_ge2_center_fraction": 0.0,
    }


def _unique_same_layer_core6_indices(
    projected_radii: np.ndarray,
    full_distances: np.ndarray,
    tie_keys: list[tuple[int, tuple[int, int, int]]],
    cutoff_A: float,
    tie_tolerance_A: float,
) -> list[int] | None:
    """Select a physical, uniquely defined same-layer nearest-six shell."""

    usable = [row for row, radius in enumerate(projected_radii) if float(radius) > 1e-8]
    usable.sort(
        key=lambda row: (
            float(projected_radii[row]),
            float(full_distances[row]),
            tie_keys[row],
        )
    )
    if len(usable) < 6:
        return None
    first_six = usable[:6]
    if float(np.max(full_distances[first_six])) > cutoff_A:
        return None
    if (
        len(usable) > 6
        and float(projected_radii[usable[6]]) - float(projected_radii[first_six[5]])
        <= tie_tolerance_A
    ):
        return None
    return first_six


def _select_layer_direction(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen target-independent normal-selection ordering."""

    if not evaluated:
        raise ValueError("no evaluated layer direction")
    return min(
        evaluated,
        key=lambda item: (
            -float(item["coverage"]),
            -int(item["support"]),
            float(item["layer_mad_A"]),
            tuple(float(v) for v in np.round(item["normal"], 12)),
        ),
    )


def triangular_features(
    structure: Structure,
    x_symbol: str,
    config: dict[str, Any],
    *,
    mx_graph: PeriodicMXGraph | None = None,
) -> dict[str, float]:
    """Continuous triangular-X descriptors using layer-first semantics.

    A candidate plane is selected solely by reliable-layer coverage, clustered
    normal support, and layer MAD.  No psi/pass statistic participates in that
    selection.  Only afterwards are the nearest six X sites chosen, and every
    member must share both the final layer label and the same image-aware VESTA
    M-X component (whose periodic dimension must be at least two).
    """

    result = _triangular_empty_features()
    cfg = config["structure"]
    repetitions = int(cfg["triangular_supercell_repetitions"])
    cutoff = float(cfg["xx_neighbor_cutoff_A"])
    tie_tolerance = float(cfg["triangular_shell_tie_tolerance_A"])
    min_layer_size = int(cfg["triangular_min_layer_size"])
    x_indices = [
        i for i, site in enumerate(structure) if site.specie.symbol == x_symbol
    ]
    if not x_indices:
        return result
    mx_graph = mx_graph or build_vesta_mx_graph(structure)
    if mx_graph.halogen_symbol != x_symbol:
        raise ValueError("triangular descriptor halogen differs from VESTA graph")

    positions, site_ids, images, central_rows = _periodic_halogen_cloud(
        structure, x_indices, repetitions
    )
    patch_components, component_dimensions = _vesta_patch_components(
        mx_graph, site_ids, images, repetitions
    )
    eligible_count = sum(
        component_dimensions.get(site_id, 0) >= 2 for site_id in x_indices
    )
    result["audit__tri_vesta_dim_ge2_center_fraction"] = eligible_count / len(x_indices)
    if eligible_count == 0:
        return result

    candidates, raw_count = _candidate_layer_normals(
        positions,
        site_ids,
        images,
        central_rows,
        patch_components,
        component_dimensions,
        config,
    )
    result["audit__tri_candidate_normal_count"] = float(len(candidates))
    result["audit__tri_raw_normal_count"] = float(raw_count)
    if not candidates:
        return result

    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        labels, layers, _eps = _robust_projected_layers(
            positions, candidate["normal"], config
        )
        covered: list[int] = []
        layer_mads: list[float] = []
        for site_id, center_row in sorted(central_rows.items()):
            if component_dimensions.get(site_id, 0) < 2:
                continue
            layer_id = int(labels[center_row])
            if layer_id < 0 or not bool(layers[layer_id]["reliable"]):
                continue
            same_component_layer = np.where(
                (labels == layer_id)
                & (patch_components == int(patch_components[center_row]))
            )[0]
            if len(same_component_layer) < min_layer_size:
                continue
            covered.append(center_row)
            layer_mads.append(float(layers[layer_id]["mad_A"]))
        coverage = len(covered) / len(x_indices)
        thickness = float(np.median(layer_mads)) if layer_mads else np.inf
        evaluated.append(
            {
                **candidate,
                "labels": labels,
                "layers": layers,
                "covered_rows": covered,
                "coverage": coverage,
                "layer_mad_A": thickness,
            }
        )

    # Pre-registered, target-independent direction choice.  Most importantly,
    # neither psi6 nor the eventual pass fraction occurs in this key.
    best = _select_layer_direction(evaluated)
    if float(best["coverage"]) <= 0:
        return result

    result["tri__is_applicable"] = 1.0
    result["tri__layer_coverage_fraction"] = float(best["coverage"])
    result["tri__normal_coherence"] = float(best["coherence"])
    result["audit__tri_selected_normal_support"] = float(best["support"])
    normal = np.asarray(best["normal"], dtype=float)
    labels = np.asarray(best["labels"], dtype=int)
    layers = best["layers"]
    ex, ey = _plane_basis_from_normal(normal)

    psi6s: list[float] = []
    psi4s: list[float] = []
    radial_cvs: list[float] = []
    planarity: list[float] = []
    gap_rmse: list[float] = []
    inplane_d: list[float] = []
    interlayer_distances: list[float] = []
    passed = 0
    valid = 0
    for site_id, center_row in sorted(central_rows.items()):
        if component_dimensions.get(site_id, 0) < 2:
            continue
        layer_id = int(labels[center_row])
        if layer_id < 0 or not bool(layers[layer_id]["reliable"]):
            continue
        center_component = int(patch_components[center_row])
        rows = np.where((labels == layer_id) & (patch_components == center_component))[
            0
        ]
        vectors = positions[rows] - positions[center_row]
        z_components = vectors @ normal
        projected = vectors - np.outer(z_components, normal)
        radii = np.linalg.norm(projected, axis=1)
        full_distances = np.linalg.norm(vectors, axis=1)
        first_six = _unique_same_layer_core6_indices(
            radii,
            full_distances,
            [
                (
                    int(site_ids[cloud_row]),
                    tuple(int(v) for v in images[cloud_row]),
                )
                for cloud_row in rows
            ],
            cutoff,
            tie_tolerance,
        )
        if first_six is None:
            continue

        core = projected[first_six]
        core_z = z_components[first_six]
        core_radii = radii[first_six]
        angles = np.mod(np.arctan2(core @ ey, core @ ex), 2.0 * np.pi)
        sorted_angles = np.sort(angles)
        angle_gaps = np.diff(np.r_[sorted_angles, sorted_angles[0] + 2.0 * np.pi])
        p6 = float(abs(np.mean(np.exp(6j * angles))))
        p4 = float(abs(np.mean(np.exp(4j * angles))))
        radial_cv = float(np.std(core_radii) / max(np.mean(core_radii), 1e-8))
        planar = float(
            np.sqrt(np.mean(core_z * core_z)) / max(np.mean(core_radii), 1e-8)
        )
        angular_rmse = float(np.sqrt(np.mean((np.degrees(angle_gaps) - 60.0) ** 2)))
        valid += 1
        passed += int(
            p6 >= float(cfg["triangular_psi6_gate"])
            and p4 <= float(cfg["triangular_psi4_max"])
            and planar <= float(cfg["triangular_planarity_gate"])
            and radial_cv <= float(cfg["triangular_radial_cv_gate"])
            and angular_rmse <= float(cfg["triangular_angular_gap_rmse_gate_deg"])
        )
        psi6s.append(p6)
        psi4s.append(p4)
        radial_cvs.append(radial_cv)
        planarity.append(planar)
        gap_rmse.append(angular_rmse)
        inplane_d.append(float(np.mean(core_radii)))

        other_layer_rows = np.where(
            (labels >= 0)
            & (labels != layer_id)
            & np.asarray(
                [
                    bool(layers[int(label)]["reliable"]) if label >= 0 else False
                    for label in labels
                ]
            )
        )[0]
        if len(other_layer_rows):
            distances = np.linalg.norm(
                positions[other_layer_rows] - positions[center_row], axis=1
            )
            interlayer_distances.append(float(np.min(distances)))

    denominator = len(x_indices)
    valid_fraction = valid / denominator
    result["tri__applicable_center_fraction"] = valid_fraction
    result["tri__center_valid_fraction"] = valid_fraction
    result["tri__unique_same_layer_core6_fraction"] = valid_fraction
    result["tri__center_pass_fraction"] = passed / denominator

    def q(values: list[float], quantile: float) -> float:
        return float(np.quantile(values, quantile)) if values else np.nan

    result["tri__psi6_median"] = q(psi6s, 0.50)
    result["tri__psi6_p10"] = q(psi6s, 0.10)
    result["tri__psi6_iqr"] = q(psi6s, 0.75) - q(psi6s, 0.25) if psi6s else np.nan
    result["tri__psi4_median"] = q(psi4s, 0.50)
    result["tri__angular_gap_rmse_deg"] = q(gap_rmse, 0.50)
    result["tri__xx_inplane_mean_A"] = q(inplane_d, 0.50)
    result["tri__xx_inplane_cv"] = q(radial_cvs, 0.50)
    result["tri__buckling_mad_A"] = float(best["layer_mad_A"])
    result["tri__buckling_over_xx"] = q(planarity, 0.50)

    reliable_layers = [layer for layer in layers if bool(layer["reliable"])]
    medians = np.sort(
        np.asarray([float(layer["median_A"]) for layer in reliable_layers], dtype=float)
    )
    spacings = np.diff(medians)
    spacings = spacings[spacings > max(1e-8, float(cfg["triangular_layer_eps_min_A"]))]
    if len(reliable_layers) >= 2 and len(spacings):
        spacing_mean = float(np.mean(spacings))
        result["stack__is_applicable"] = 1.0
        result["audit__stack_reliable_patch_layer_count"] = float(len(reliable_layers))
        result["stack__coverage_fraction"] = float(best["coverage"])
        result["stack__spacing_mean_A"] = spacing_mean
        result["stack__spacing_cv"] = float(np.std(spacings) / max(spacing_mean, 1e-8))
        if inplane_d:
            result["stack__spacing_over_xx"] = spacing_mean / max(
                q(inplane_d, 0.50), 1e-8
            )
        if interlayer_distances and inplane_d:
            result["stack__inter_intra_xx_ratio"] = float(
                np.median(interlayer_distances) / max(q(inplane_d, 0.50), 1e-8)
            )
    return result


def free_volume_features(
    structure: Structure, config: dict[str, Any]
) -> dict[str, float]:
    radii = np.asarray(
        [element_radius_A(site.specie) for site in structure], dtype=float
    )
    sphere_volume = float(np.sum(4.0 * np.pi * radii**3 / 3.0))
    lattice_lengths = np.asarray(structure.lattice.abc, dtype=float)
    voronoi_volumes: list[float] = []
    try:
        all_poly = VoronoiNN(cutoff=10.0).get_all_voronoi_polyhedra(structure)
        voronoi_volumes = [
            float(sum(face.get("volume", 0.0) for face in poly.values()))
            for poly in all_poly
        ]
    except Exception:
        pass
    vv = np.asarray(voronoi_volumes, dtype=float)
    ngrid = int(config["structure"]["cavity_grid_points_per_axis"])
    axis = (np.arange(ngrid, dtype=float) + 0.5) / ngrid
    base_grid = np.asarray(list(product(axis, repeat=3)), dtype=float)
    # Anchor the fractional sampling grid to every M site.  Translating the entire
    # periodic structure then translates atoms and grid together, while aggregation
    # over all M anchors avoids dependence on site ordering.  A lattice-origin-fixed
    # grid changes its extrema under an arbitrary CIF origin shift.
    m_symbol, x_symbol, _, _ = identify_binary_halide(structure)
    anchors = np.asarray(
        [site.frac_coords for site in structure if site.specie.symbol == m_symbol],
        dtype=float,
    )
    if not len(anchors):
        raise ValueError("binary halide contains no M site for cavity-grid anchoring")
    grid = np.mod(anchors[:, None, :] + base_grid[None, :, :], 1.0).reshape(-1, 3)
    distances = structure.lattice.get_all_distances(grid, structure.frac_coords)
    clearance = np.min(distances - radii[None, :], axis=1)
    x_indices = [
        i for i, site in enumerate(structure) if site.specie.symbol == x_symbol
    ]
    xx_distances = []
    for i in x_indices:
        xx_distances.extend(
            n.nn_distance
            for n in structure.get_neighbors(structure[i], 7.0)
            if n.specie.symbol == x_symbol and n.nn_distance > 1e-6
        )
    xx = np.asarray(xx_distances, dtype=float)
    return {
        "flex__volume_per_atom_A3": structure.volume / max(len(structure), 1),
        "flex__density_g_cm3": finite(structure.density),
        "flex__packing_fraction_proxy": sphere_volume / max(structure.volume, 1e-8),
        "flex__voronoi_volume_mean_A3": float(vv.mean()) if len(vv) else np.nan,
        "flex__voronoi_volume_cv": float(vv.std() / max(vv.mean(), 1e-8))
        if len(vv)
        else np.nan,
        "flex__max_cavity_clearance_proxy_A": float(np.max(clearance)),
        "flex__cavity_clearance_std_A": float(np.std(clearance)),
        "flex__lattice_anisotropy": float(
            lattice_lengths.max() / max(lattice_lengths.min(), 1e-8)
        ),
        "flex__xx_min_A": float(xx.min()) if len(xx) else np.nan,
        "flex__xx_distance_cv": float(xx.std() / max(xx.mean(), 1e-8))
        if len(xx)
        else np.nan,
    }


def inspect_cif_identity(row: dict[str, Any], cif_dir: Path) -> dict[str, Any]:
    """Resolve formula identity from the CIF without computing descriptors."""

    cif_file = str(row["cif_file"])
    cif_path = cif_dir / cif_file
    record = dict(row)
    record.update(
        {
            "filename_formula": "",
            "parsed_formula": "",
            "cif_sha256": "",
            "identity_status": "ok",
            "identity_error": "",
        }
    )
    try:
        record["cif_sha256"] = sha256_file(cif_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_file(cif_path)
        parsed = canonical_formula(structure.composition.reduced_formula)
        record["parsed_formula"] = parsed
        # The filename is only a consistency check, never the source of chemical
        # identity.  Keep its parsing inside this fail-closed identity pass so a
        # malformed name is written to the exclusion ledger instead of aborting
        # the complete batch before CIF isolation.
        record["filename_formula"] = formula_from_cif_name(cif_file)
        if parsed != record["filename_formula"]:
            raise ValueError(
                "CIF composition/filename formula mismatch: "
                f"parsed={parsed}, filename={record['filename_formula']}"
            )
    except Exception as exc:
        record["identity_status"] = "error"
        record["identity_error"] = f"{type(exc).__name__}: {exc}"
    return record


def extract_one(
    row: dict[str, Any], cif_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    cif_path = cif_dir / row["cif_file"]
    base: dict[str, Any] = {
        "cif_file": row["cif_file"],
        "material_id": ID_RE.search(row["cif_file"]).group(1)
        if ID_RE.search(row["cif_file"])
        else "",
        "formula": canonical_formula(row["formula"]),
        "cif_sha256": row["cif_sha256"],
        "feature_status": "ok",
        "feature_error": "",
    }
    try:
        if sha256_file(cif_path) != row["cif_sha256"]:
            raise RuntimeError("CIF changed after the identity/isolation pass")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = Structure.from_file(cif_path)
            parsed_formula = canonical_formula(structure.composition.reduced_formula)
            if parsed_formula != base["formula"]:
                raise RuntimeError(
                    f"parsed CIF formula changed: expected {base['formula']}, got {parsed_formula}"
                )
            base["source_n_sites"] = len(structure)
            base["source_cell_volume_A3"] = float(structure.volume)
            structure = standardize_descriptor_structure(structure, config)
            _, x_symbol, _, _ = identify_binary_halide(structure)
            base["n_sites"] = len(structure)
            base["descriptor_cell_standardization"] = "primitive_then_niggli"
            base["is_ordered"] = bool(structure.is_ordered)
            base.update(chemistry_features(structure))
            mx_graph = build_vesta_mx_graph(structure)
            if not mx_graph.bonds:
                raise ValueError(
                    "VESTA-2019 pair is defined but identifies no M-X bond; "
                    "zero bonds cannot be interpreted as a physical 0D network"
                )
            base["bonding_backend"] = "vesta_2019"
            topology, _ = local_topology_features(
                structure, x_symbol, config, mx_graph=mx_graph
            )
            base.update(topology)
            base.update(
                triangular_features(structure, x_symbol, config, mx_graph=mx_graph)
            )
            base.update(free_volume_features(structure, config))
    except Exception as exc:
        base["feature_status"] = "error"
        base["feature_error"] = f"{type(exc).__name__}: {exc}"
    return base


def pseudo_atoms(cif_path: Path, expected_sha256: str, config: dict[str, Any]) -> Atoms:
    if sha256_file(cif_path) != expected_sha256:
        raise RuntimeError(f"CIF changed before SOAP extraction: {cif_path.name}")
    structure = standardize_descriptor_structure(Structure.from_file(cif_path), config)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    numbers = np.asarray(
        [2 if atom.symbol in HALOGENS else 1 for atom in atoms], dtype=int
    )
    result = Atoms(
        numbers=numbers, positions=atoms.positions, cell=atoms.cell, pbc=True
    )
    if sha256_file(cif_path) != expected_sha256:
        raise RuntimeError(f"CIF changed during SOAP extraction: {cif_path.name}")
    return result


def compute_soap(
    frame: pd.DataFrame,
    cif_dir: Path,
    config: dict[str, Any],
    n_jobs: int,
    *,
    r_cut_A: float | None = None,
) -> np.ndarray:
    cfg = config["soap"]
    cutoff = float(cfg["r_cut_A"] if r_cut_A is None else r_cut_A)
    soap = SOAP(
        species=cfg["pseudo_species"],
        periodic=bool(cfg["periodic"]),
        r_cut=cutoff,
        n_max=int(cfg["n_max"]),
        l_max=int(cfg["l_max"]),
        sigma=float(cfg["sigma_A"]),
        average=cfg["average"],
        sparse=False,
    )
    atoms = [
        pseudo_atoms(cif_dir / row.cif_file, str(row.cif_sha256), config)
        for row in frame[["cif_file", "cif_sha256"]].itertuples(index=False)
    ]
    matrix = np.asarray(soap.create(atoms, n_jobs=n_jobs), dtype=np.float32)
    if matrix.ndim == 1 and len(frame) == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != len(frame):
        raise RuntimeError(
            f"unexpected SOAP matrix shape {matrix.shape} for {len(frame)} structures"
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def load_excluded_formulas(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {canonical_formula(v) for v in payload.get("formulas", [])}


def validate_formula_isolation(
    exclude_formulas_file: Path | None, include_only_formulas_file: Path | None
) -> str:
    """Require exactly one mutually exclusive formula-isolation boundary."""

    if (exclude_formulas_file is None) == (include_only_formulas_file is None):
        raise ValueError(
            "exactly one isolation mode is required: --exclude-formulas-file for "
            "development or --include-only-formulas-file for reveal"
        )
    return (
        "development_exclude"
        if exclude_formulas_file is not None
        else "reveal_include_only"
    )


def merge_metadata(
    features: pd.DataFrame, path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach structure-level metadata without silently colliding with feature keys."""
    header = pd.read_csv(path, nrows=0)
    if "material_id" not in header:
        raise ValueError("metadata must contain material_id")
    source_commitment = sha256_file(path)
    source_columns = header.columns.tolist()
    # First pass parses only the identity key.  The second pass physically skips every
    # non-selected CSV row before its descriptor values are materialized by pandas.
    metadata_ids = pd.read_csv(path, usecols=["material_id"])
    source_rows = int(len(metadata_ids))
    allowed_ids = set(features["material_id"].astype(str))
    keep_lines = {
        int(position) + 1
        for position, value in enumerate(metadata_ids["material_id"].astype(str))
        if value in allowed_ids
    }
    metadata = pd.read_csv(
        path,
        skiprows=lambda line_number: line_number > 0 and line_number not in keep_lines,
    )
    if metadata["material_id"].duplicated().any():
        duplicates = metadata.loc[
            metadata["material_id"].duplicated(False), "material_id"
        ].head()
        raise ValueError(f"metadata material_id is not unique: {duplicates.tolist()}")
    if "cif_file" in metadata:
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
            raise ValueError(
                f"metadata CIF identity mismatch: {mismatch.head().to_dict('records')}"
            )
        metadata = metadata.drop(columns="cif_file")
    if "formula" in metadata:
        metadata_formula = metadata["formula"].map(canonical_formula)
        formula_map = features.set_index("material_id")["formula"]
        aligned = metadata["material_id"].map(formula_map)
        mismatch = aligned.notna() & metadata_formula.ne(aligned)
        if mismatch.any():
            raise ValueError(
                f"metadata formula mismatch: {metadata.loc[mismatch, ['material_id', 'formula']].head().to_dict('records')}"
            )
        metadata = metadata.rename(columns={"formula": "metadata_formula"})
    if "formula_pretty" in metadata:
        metadata_formula = metadata["formula_pretty"].dropna().map(canonical_formula)
        aligned = metadata.loc[metadata_formula.index, "material_id"].map(
            features.set_index("material_id")["formula"]
        )
        mismatch = aligned.notna() & metadata_formula.ne(aligned)
        if mismatch.any():
            raise ValueError(
                "metadata formula_pretty mismatch: "
                f"{metadata.loc[metadata_formula.index[mismatch], ['material_id', 'formula_pretty']].head().to_dict('records')}"
            )
    if "is_ordered" in metadata and "is_ordered" in features:
        ordered = features[["material_id", "is_ordered"]].merge(
            metadata[["material_id", "is_ordered"]].rename(
                columns={"is_ordered": "metadata_is_ordered"}
            ),
            on="material_id",
            how="left",
            validate="one_to_one",
        )
        comparable = ordered["metadata_is_ordered"].notna()
        parsed = (
            ordered.loc[comparable, "metadata_is_ordered"]
            .astype(str)
            .str.lower()
            .eq("true")
        )
        mismatch = parsed.ne(ordered.loc[comparable, "is_ordered"].astype(bool))
        if mismatch.any():
            raise ValueError("metadata is_ordered conflicts with parsed CIF")
        metadata = metadata.drop(columns="is_ordered")
    collisions = sorted(
        (set(features.columns) & set(metadata.columns)) - {"material_id"}
    )
    if collisions:
        raise ValueError(
            f"metadata columns collide with extracted feature columns: {collisions}"
        )
    before = features[["cif_file", "material_id", "formula"]].copy()
    merged = features.merge(
        metadata, on="material_id", how="left", validate="one_to_one", sort=False
    )
    if not before.equals(merged[["cif_file", "material_id", "formula"]]):
        raise RuntimeError("metadata merge changed feature row identity or order")
    info = {
        "metadata_source_commitment_sha256": source_commitment,
        "metadata_source_rows": source_rows,
        "metadata_source_columns_sha256": stable_json_hash(source_columns),
        "metadata_selected_values_sha256": dataframe_sha256(metadata),
        "metadata_selected_rows": int(len(metadata)),
        "metadata_matched_feature_rows": int(
            merged["material_id"].isin(metadata["material_id"]).sum()
        ),
        "metadata_columns_attached": sorted(set(metadata.columns) - {"material_id"}),
        "metadata_status_counts": (
            {
                str(key): int(value)
                for key, value in merged["mp_metadata_status"]
                .fillna("missing_row")
                .value_counts(dropna=False)
                .sort_index()
                .items()
            }
            if "mp_metadata_status" in merged
            else None
        ),
    }
    return merged, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--cif-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--exclude-formulas-file", type=Path)
    parser.add_argument("--include-only-formulas-file", type=Path)
    parser.add_argument(
        "--metadata", type=Path, help="optional MP metadata CSV keyed by material_id"
    )
    parser.add_argument("--n-jobs", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--skip-soap", action="store_true")
    args = parser.parse_args()

    isolation_mode = validate_formula_isolation(
        args.exclude_formulas_file, args.include_only_formulas_file
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != DISCOVERY_SCHEMA_VERSION:
        raise ValueError(
            "multiview discovery config schema_version must be "
            f"{DISCOVERY_SCHEMA_VERSION}"
        )
    bonding_provenance = validate_bonding_contract(config)
    triangular_reference = triangular_reference_provenance()
    soap_r_cut_variants = validate_soap_cutoff_contract(config)
    metrics_source_commitment = sha256_file(args.metrics)
    # Only the immutable CIF key is admitted across the isolation boundary.  Legacy
    # CrystalNN dim/st columns remain in the historical file but are neither parsed
    # nor copied into the V2 feature table.
    frame = pd.read_csv(args.metrics, usecols=["cif_file"])
    metrics_cif_inventory_sha256 = stable_json_hash(
        sorted(frame["cif_file"].astype(str).tolist())
    )
    if frame["cif_file"].duplicated().any():
        duplicate = (
            frame.loc[frame["cif_file"].duplicated(False), "cif_file"].head().tolist()
        )
        raise ValueError(f"metrics inventory contains duplicate CIF keys: {duplicate}")

    # Resolve identity from the actual CIF before blind isolation.  This pass reads
    # only composition and file identity; it does not evaluate any descriptor.
    identity_rows = Parallel(n_jobs=args.n_jobs, prefer="processes", verbose=5)(
        delayed(inspect_cif_identity)(row, args.cif_dir)
        for row in frame.to_dict("records")
    )
    full_inventory = pd.DataFrame(identity_rows)
    full_cif_content_inventory_sha256 = stable_json_hash(
        full_inventory[["cif_file", "cif_sha256"]]
        .sort_values("cif_file")
        .to_dict("records")
    )
    full_cif_identity_inventory_sha256 = stable_json_hash(
        full_inventory[
            [
                "cif_file",
                "parsed_formula",
                "filename_formula",
                "cif_sha256",
                "identity_status",
            ]
        ]
        .sort_values("cif_file")
        .to_dict("records")
    )
    inventory = full_inventory.copy()
    inventory["formula"] = np.where(
        inventory["parsed_formula"].astype(str).ne(""),
        inventory["parsed_formula"],
        inventory["filename_formula"],
    )
    # A CIF that could not be parsed has no trustworthy formula.  Preserve the
    # empty sentinel so it can be ledgered (and can never pass include-only
    # isolation) rather than raising a second, batch-wide Composition error.
    inventory["formula"] = inventory["formula"].map(
        lambda value: canonical_formula(value) if str(value).strip() else ""
    )
    excluded = load_excluded_formulas(args.exclude_formulas_file)
    included = load_excluded_formulas(args.include_only_formulas_file)
    if args.exclude_formulas_file is not None:
        inventory = inventory.loc[~inventory["formula"].isin(excluded)].copy()
    if args.include_only_formulas_file is not None:
        inventory = inventory.loc[inventory["formula"].isin(included)].copy()
    inventory = inventory.sort_values("cif_file").reset_index(drop=True)
    if inventory.empty:
        raise SystemExit("no structures remain after formula isolation")

    args.outdir.mkdir(parents=True, exist_ok=True)
    identity_failed = inventory.loc[inventory["identity_status"].ne("ok")].copy()
    identity_cif = identity_failed.get("cif_file", pd.Series(dtype=str))
    identity_ledger = pd.DataFrame(
        {
            "cif_file": identity_cif,
            "material_id": identity_cif.map(
                lambda value: ID_RE.search(str(value)).group(1)
                if ID_RE.search(str(value))
                else ""
            ),
            "formula": identity_failed.get("formula", pd.Series(dtype=str)),
            "cif_sha256": identity_failed.get("cif_sha256", pd.Series(dtype=str)),
            "feature_status": "error",
            "feature_error": identity_failed.get(
                "identity_error", pd.Series(dtype=str)
            ),
        }
    )
    frame = inventory.loc[inventory["identity_status"].eq("ok")].copy()
    rows = Parallel(n_jobs=args.n_jobs, prefer="processes", verbose=5)(
        delayed(extract_one)(row, args.cif_dir, config)
        for row in frame.to_dict("records")
    )
    extracted = pd.DataFrame(rows)
    failed = pd.concat(
        [identity_ledger, extracted.loc[extracted["feature_status"].ne("ok")].copy()],
        ignore_index=True,
        sort=False,
    )
    failed.to_csv(args.outdir / "structure_exclusions.csv", index=False)
    features = (
        extracted.loc[extracted["feature_status"].eq("ok")]
        .copy()
        .reset_index(drop=True)
    )
    if features.empty:
        raise RuntimeError("all structures were excluded during feature extraction")
    metadata_provenance = None
    if args.metadata:
        if not args.metadata.exists():
            raise FileNotFoundError(f"metadata file does not exist: {args.metadata}")
        features, metadata_provenance = merge_metadata(features, args.metadata)
    features.to_csv(args.outdir / "interpretable_features.csv", index=False)

    soap_shape = None
    soap_variant_records: list[dict[str, Any]] = []
    if not args.skip_soap:
        features[["cif_file", "material_id", "formula"]].to_csv(
            args.outdir / "soap_rows.csv", index=False
        )
        base_r_cut = float(config["soap"]["r_cut_A"])
        for r_cut_A in soap_r_cut_variants:
            matrix = compute_soap(
                features,
                args.cif_dir,
                config,
                args.n_jobs,
                r_cut_A=r_cut_A,
            )
            filename = soap_variant_filename(r_cut_A, base_r_cut)
            matrix_path = args.outdir / filename
            np.save(matrix_path, matrix)
            record = {
                "variant": soap_variant_key(r_cut_A),
                "r_cut_A": float(r_cut_A),
                "filename": filename,
                "shape": list(matrix.shape),
                "sha256": sha256_file(matrix_path),
                "l2_normalized_rows": True,
            }
            soap_variant_records.append(record)
            if math.isclose(r_cut_A, base_r_cut, abs_tol=1e-12):
                soap_shape = list(matrix.shape)
        if soap_shape is None:
            raise RuntimeError("base SOAP cutoff was not extracted")

    provenance = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "extractor_sha256": sha256_file(Path(__file__)),
        "bonding_module_sha256": sha256_file(
            Path(__file__).with_name("bonding_vesta.py")
        ),
        "config_sha256": sha256_file(args.config),
        "metrics_source_commitment_sha256": metrics_source_commitment,
        "metrics_cif_inventory_sha256": metrics_cif_inventory_sha256,
        "formula_isolation_file_sha256": (
            sha256_file(args.exclude_formulas_file)
            if args.exclude_formulas_file
            else None
        ),
        "include_only_file_sha256": (
            sha256_file(args.include_only_formulas_file)
            if args.include_only_formulas_file
            else None
        ),
        "metadata": metadata_provenance,
        "bonding": bonding_provenance,
        "triangular_layer_reference": triangular_reference,
        "identity_source": "parsed_cif_composition",
        "filename_formula_mismatch_policy": "exclude_with_ledger",
        "full_cif_content_inventory_sha256": full_cif_content_inventory_sha256,
        "full_cif_identity_inventory_sha256": full_cif_identity_inventory_sha256,
        "isolated_inventory_sha256": stable_json_hash(
            inventory[["cif_file", "formula", "cif_sha256", "identity_status"]].to_dict(
                "records"
            )
        ),
        "n_inventory_structures_before_isolation": len(identity_rows),
        "n_input_structures": len(inventory),
        "n_structures": len(features),
        "n_structure_exclusions": len(failed),
        "structure_exclusions_sha256": sha256_file(
            args.outdir / "structure_exclusions.csv"
        ),
        "interpretable_features_sha256": sha256_file(
            args.outdir / "interpretable_features.csv"
        ),
        "soap_rows_sha256": (
            sha256_file(args.outdir / "soap_rows.csv") if not args.skip_soap else None
        ),
        "soap_array_sha256": (
            sha256_file(args.outdir / "soap_pseudo_mx.npy")
            if not args.skip_soap
            else None
        ),
        "soap_base_r_cut_A": float(config["soap"]["r_cut_A"]),
        "soap_variants": soap_variant_records,
        "soap_variants_contract_sha256": (
            stable_json_hash(
                [
                    {
                        "variant": record["variant"],
                        "r_cut_A": record["r_cut_A"],
                        "filename": record["filename"],
                        "width": record["shape"][1],
                        "l2_normalized_rows": record["l2_normalized_rows"],
                    }
                    for record in soap_variant_records
                ]
            )
            if not args.skip_soap
            else None
        ),
        "feature_columns_sha256": stable_json_hash(features.columns.tolist()),
        "soap_shape": soap_shape,
        "formula_isolation_mode": isolation_mode,
        "isolated_formula_names_written_to_feature_rows": bool(
            args.include_only_formulas_file
        ),
    }
    (args.outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"wrote {args.outdir} ({len(features)} structures, SOAP={soap_shape}, "
        f"r_cut_A={list(soap_r_cut_variants)})"
    )


if __name__ == "__main__":
    main()
