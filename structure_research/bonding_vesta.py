#!/usr/bin/env python3
"""Periodic M-X bonding graph using pymatgen's frozen VESTA-2019 cutoffs.

This module is the single structural-bonding contract for the discovery v2
pipeline.  It deliberately does not fall back to CrystalNN, VoronoiNN, or a
nearest-distance shell: a structure either has bonds under the frozen VESTA
table or it does not.

The graph stores every periodic image explicitly.  This is essential for a
primitive cell containing one crystallographic M site: an M-X-M chain can be
represented by two bonds to the same X index in different lattice images.
Dropping those images would incorrectly turn a periodic chain or layer into a
0D object.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CutOffDictNN
from pymatgen.core import Structure
import pymatgen.analysis.local_env as pymatgen_local_env


HALOGENS = frozenset({"F", "Cl", "Br", "I"})
VESTA_PRESET = "vesta_2019"
VESTA_RULE_ID = "pymatgen.CutOffDictNN.from_preset:vesta_2019"

# Freeze the exact table used to define discovery v2.  This prevents a future
# pymatgen upgrade from silently changing every coordination and topology
# feature.  A different table requires an explicit schema/version change.
EXPECTED_VESTA_CUTOFF_SHA256 = (
    "f2493332b712fabc7ecb1965753ac58fb3dd735603b41d32d25f1cd85361b144"
)
EXPECTED_VESTA_PAIR_COUNT = 914

Image = tuple[int, int, int]
CenterImage = tuple[int, int, int, int]
PolyPair = tuple[int, int, int, int, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vesta_cutoff_path() -> Path:
    """Return the installed pymatgen VESTA cutoff-table path."""

    return Path(pymatgen_local_env.__file__).resolve().with_name("vesta_cutoffs.yaml")


@lru_cache(maxsize=2)
def vesta_rule_provenance(*, verify: bool = True) -> dict[str, Any]:
    """Describe and optionally verify the immutable bonding-rule source."""

    path = vesta_cutoff_path()
    if not path.is_file():
        raise FileNotFoundError(f"pymatgen VESTA cutoff table is missing: {path}")
    digest = _sha256(path)
    nn = CutOffDictNN.from_preset(VESTA_PRESET)
    pair_count = len(nn.cut_off_dict)
    if verify and digest != EXPECTED_VESTA_CUTOFF_SHA256:
        raise RuntimeError(
            "VESTA cutoff-table hash differs from the discovery-v2 contract: "
            f"expected {EXPECTED_VESTA_CUTOFF_SHA256}, got {digest} ({path})"
        )
    if verify and pair_count != EXPECTED_VESTA_PAIR_COUNT:
        raise RuntimeError(
            "VESTA cutoff-table pair count differs from the discovery-v2 contract: "
            f"expected {EXPECTED_VESTA_PAIR_COUNT}, got {pair_count}"
        )
    return {
        "backend": VESTA_RULE_ID,
        "preset": VESTA_PRESET,
        "pymatgen_version": importlib.metadata.version("pymatgen"),
        "cutoff_table_path": str(path),
        "cutoff_table_sha256": digest,
        "cutoff_pair_count": pair_count,
        "distance_comparison": "strictly_less_than_cutoff",
        "fallback": None,
        "periodic_images_retained": True,
    }


@lru_cache(maxsize=1)
def _vesta_nn() -> CutOffDictNN:
    return CutOffDictNN.from_preset(VESTA_PRESET)


def identify_binary_halide_symbols(structure: Structure) -> tuple[str, str]:
    """Return the unique non-halogen centre and halogen symbols."""

    symbols = {site.specie.symbol for site in structure}
    halogens = sorted(symbols & HALOGENS)
    centers = sorted(symbols - HALOGENS)
    if len(symbols) != 2 or len(halogens) != 1 or len(centers) != 1:
        raise ValueError(
            "VESTA M-X backend requires one non-halogen species and one of "
            f"F/Cl/Br/I; found {sorted(symbols)}"
        )
    return centers[0], halogens[0]


def _image(value: Iterable[float | int]) -> Image:
    vals = tuple(int(round(float(v))) for v in value)
    if len(vals) != 3:
        raise ValueError(f"periodic image must have length 3, got {vals}")
    return vals  # type: ignore[return-value]


def canonical_poly_pair(a: CenterImage, b: CenterImage) -> PolyPair:
    """Canonicalize an undirected periodic centre-centre pair orbit."""

    ia, ib = a[0], b[0]
    delta = tuple(int(b[k] - a[k]) for k in range(1, 4))
    forward: PolyPair = (ia, ib, *delta)
    reverse: PolyPair = (ib, ia, *(-v for v in delta))
    return min(forward, reverse)


@dataclass(frozen=True, order=True)
class PeriodicMXBond:
    """One M@0 to X@image bond orbit."""

    center_index: int
    halogen_index: int
    image: Image
    distance_A: float

    @property
    def key(self) -> tuple[int, int, int, int, int]:
        return (self.center_index, self.halogen_index, *self.image)


@dataclass
class PeriodicMXGraph:
    """Binary-halide M-X graph with image-aware local and topology helpers."""

    structure: Structure
    center_symbol: str
    halogen_symbol: str
    cutoff_A: float
    center_indices: tuple[int, ...]
    halogen_indices: tuple[int, ...]
    bonds: tuple[PeriodicMXBond, ...]

    def bonds_by_center(self) -> dict[int, tuple[PeriodicMXBond, ...]]:
        grouped: dict[int, list[PeriodicMXBond]] = {
            idx: [] for idx in self.center_indices
        }
        for bond in self.bonds:
            grouped[bond.center_index].append(bond)
        return {idx: tuple(sorted(values)) for idx, values in grouped.items()}

    def neighbor_info(self, center_index: int) -> list[dict[str, Any]]:
        """Return pymatgen-like neighbor dictionaries for geometry code."""

        out: list[dict[str, Any]] = []
        for bond in self.bonds_by_center().get(center_index, ()):
            site = self.structure[bond.halogen_index]
            frac = np.asarray(site.frac_coords, dtype=float) + np.asarray(
                bond.image, dtype=float
            )
            periodic_site = site.__class__(
                site.species,
                frac,
                self.structure.lattice,
                coords_are_cartesian=False,
                properties=site.properties,
            )
            out.append(
                {
                    "site": periodic_site,
                    "site_index": bond.halogen_index,
                    "image": bond.image,
                    "weight": bond.distance_A,
                    "distance_A": bond.distance_A,
                }
            )
        return out

    def all_neighbor_info(self) -> list[list[dict[str, Any]]]:
        return [self.neighbor_info(idx) for idx in self.center_indices]

    def structure_graph(self) -> StructureGraph:
        graph = StructureGraph.from_empty_graph(
            self.structure, name=f"M-X bonds ({VESTA_PRESET})"
        )
        for bond in self.bonds:
            graph.add_edge(
                bond.center_index,
                bond.halogen_index,
                from_jimage=(0, 0, 0),
                to_jimage=bond.image,
                warn_duplicates=False,
            )
        return graph

    def dimensionality(self) -> int:
        if not self.bonds:
            return 0
        return int(get_dimensionality_larsen(self.structure_graph()))

    def x_to_center_images(
        self, *, eligible_centers: set[int] | None = None
    ) -> dict[int, set[CenterImage]]:
        """Invert M@0-X@t bonds into X@0-M@-t attachments."""

        result: dict[int, set[CenterImage]] = {
            idx: set() for idx in self.halogen_indices
        }
        for bond in self.bonds:
            if eligible_centers is not None and bond.center_index not in eligible_centers:
                continue
            tx, ty, tz = bond.image
            result[bond.halogen_index].add(
                (bond.center_index, -tx, -ty, -tz)
            )
        return result

    def shared_polyhedron_pairs(
        self, *, eligible_centers: set[int] | None = None
    ) -> Counter[PolyPair]:
        shared: Counter[PolyPair] = Counter()
        for center_images in self.x_to_center_images(
            eligible_centers=eligible_centers
        ).values():
            for left, right in combinations(sorted(center_images), 2):
                shared[canonical_poly_pair(left, right)] += 1
        return shared

    @staticmethod
    def periodic_degrees(
        center_indices: Iterable[int], pair_shared: Counter[PolyPair]
    ) -> dict[int, set[CenterImage]]:
        degrees: dict[int, set[CenterImage]] = {
            int(idx): set() for idx in center_indices
        }
        for i, j, dx, dy, dz in pair_shared:
            degrees[i].add((j, dx, dy, dz))
            degrees[j].add((i, -dx, -dy, -dz))
        return degrees

    def polyhedron_metrics(self, *, minimum_cn: int = 3) -> dict[str, Any]:
        """Compute image-aware counterparts of historical st1/st2/st3."""

        grouped = self.bonds_by_center()
        eligible = {
            idx for idx in self.center_indices if len(grouped.get(idx, ())) >= minimum_cn
        }
        cn = [len(grouped[idx]) for idx in sorted(eligible)]
        pair_shared = self.shared_polyhedron_pairs(eligible_centers=eligible)
        degrees = self.periodic_degrees(sorted(eligible), pair_shared)
        return {
            "dim": self.dimensionality(),
            "st1": float(np.mean(cn)) if cn else 0.0,
            "st2": float(np.mean(list(pair_shared.values()))) if pair_shared else 0.0,
            "st3": (
                float(np.mean([len(degrees[idx]) for idx in sorted(eligible)]))
                if eligible
                else 0.0
            ),
            "n_polyhedra": len(eligible),
            "eligible_center_indices": sorted(eligible),
            "degrees": {str(idx): len(degrees[idx]) for idx in sorted(eligible)},
            "shared_pair_counts": {
                ",".join(map(str, key)): int(value)
                for key, value in sorted(pair_shared.items())
            },
        }

    def pair_network_dimension(self, pair_shared: Counter[PolyPair]) -> int:
        """Larsen dimension of a periodic centre-centre pair-orbit graph."""

        if not pair_shared:
            return 0
        graph = StructureGraph.from_empty_graph(
            self.structure, name=f"polyhedron pairs ({VESTA_PRESET})"
        )
        for i, j, dx, dy, dz in pair_shared:
            graph.add_edge(
                i,
                j,
                from_jimage=(0, 0, 0),
                to_jimage=(dx, dy, dz),
                warn_duplicates=False,
            )
        return int(get_dimensionality_larsen(graph))


def build_vesta_mx_graph(
    structure: Structure, *, verify_contract: bool = True
) -> PeriodicMXGraph:
    """Build the frozen VESTA-2019 M-X graph for an ordered binary halide."""

    if not structure.is_ordered:
        raise ValueError("VESTA M-X backend requires an ordered structure")
    if verify_contract:
        vesta_rule_provenance(verify=True)
    center_symbol, halogen_symbol = identify_binary_halide_symbols(structure)
    nn = _vesta_nn()
    cutoff = nn._lookup_dict.get(center_symbol, {}).get(halogen_symbol, 0.0)
    if cutoff <= 0:
        raise KeyError(
            f"VESTA-2019 has no cutoff for {center_symbol}-{halogen_symbol}; "
            "no alternative neighbor rule is permitted"
        )
    center_indices = tuple(
        i for i, site in enumerate(structure) if site.specie.symbol == center_symbol
    )
    halogen_indices = tuple(
        i for i, site in enumerate(structure) if site.specie.symbol == halogen_symbol
    )

    unique: dict[tuple[int, int, int, int, int], PeriodicMXBond] = {}
    if cutoff > 0:
        for center_index in center_indices:
            for neighbor in structure.get_neighbors(structure[center_index], cutoff):
                if neighbor.specie.symbol != halogen_symbol:
                    continue
                distance = float(neighbor.nn_distance)
                # Match CutOffDictNN.get_nn_info exactly: equality is not a bond.
                if not distance < cutoff:
                    continue
                image = _image(neighbor.image)
                bond = PeriodicMXBond(
                    center_index=center_index,
                    halogen_index=int(neighbor.index),
                    image=image,
                    distance_A=distance,
                )
                previous = unique.get(bond.key)
                if previous is None or distance < previous.distance_A:
                    unique[bond.key] = bond

    return PeriodicMXGraph(
        structure=structure,
        center_symbol=center_symbol,
        halogen_symbol=halogen_symbol,
        cutoff_A=float(cutoff),
        center_indices=center_indices,
        halogen_indices=halogen_indices,
        bonds=tuple(sorted(unique.values())),
    )


__all__ = [
    "EXPECTED_VESTA_CUTOFF_SHA256",
    "EXPECTED_VESTA_PAIR_COUNT",
    "HALOGENS",
    "PeriodicMXBond",
    "PeriodicMXGraph",
    "VESTA_PRESET",
    "VESTA_RULE_ID",
    "build_vesta_mx_graph",
    "canonical_poly_pair",
    "identify_binary_halide_symbols",
    "vesta_cutoff_path",
    "vesta_rule_provenance",
]
