"""Periodic union volume of convex ligand polyhedra by randomized Sobol sampling.

Input coordinates are fractional coordinates in one declared periodic cell.
Integrating a union indicator over [0, 1)^3 directly gives the physical volume
fraction, including for skew cells.  Periodic ligand vertices may lie outside
that interval.  The caller is responsible for supplying all centre-associated
full-dimensional hulls in the cell, and for reporting excluded low-rank sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import ceil, floor, prod
from typing import Any, Iterable

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import qmc


@dataclass
class _PeriodicCopy:
    lower: np.ndarray
    upper: np.ndarray
    normals: np.ndarray
    offsets: np.ndarray
    bounding_box_fraction: float


def _prepare_copies(
    hulls: list[Any], max_periodic_copies: int
) -> tuple[list[_PeriodicCopy], float]:
    copies: list[_PeriodicCopy] = []
    summed_volume = 0.0
    for hull_index, raw_points in enumerate(hulls):
        points = np.asarray(raw_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
            raise ValueError(f"hull {hull_index}: expected at least four 3D vertices")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"hull {hull_index}: non-finite fractional coordinates")
        # A common integer translation preserves the periodic body.  Remove a
        # large arbitrary origin before Qhull to improve numerical conditioning.
        points = points - np.floor(points[0])
        if not np.all(np.isfinite(points)):
            raise ValueError(f"hull {hull_index}: fractional span overflows")
        hull = ConvexHull(points)  # Never use QJ to manufacture a 3D body.
        hull_volume = float(hull.volume)
        if not np.isfinite(hull_volume) or hull_volume <= 0:
            raise ValueError(f"hull {hull_index}: nonpositive or invalid 3D volume")
        summed_volume += hull_volume
        if not np.isfinite(summed_volume):
            raise ValueError("summed fractional hull volume overflows")

        # Qhull triangulates coplanar facets.  Exact equation duplicates can be
        # removed without changing any halfspace or relaxing its tolerance.
        equations = np.unique(hull.equations, axis=0)
        normals, original_offsets = equations[:, :3], equations[:, 3]
        lower, upper = points.min(axis=0), points.max(axis=0)

        # For positive-volume overlap of bounding intervals:
        # upper + t > 0 and lower + t < 1, with t an integer translation.
        translation_ranges = [
            range(floor(-float(upper[d])) + 1, ceil(1.0 - float(lower[d])))
            for d in range(3)
        ]
        proposed = prod(len(values) for values in translation_ranges)
        if len(copies) + proposed > max_periodic_copies:
            raise ValueError(
                "periodic bounding-box copy count exceeds max_periodic_copies "
                f"({max_periodic_copies}); hull {hull_index} requires {proposed} copies"
            )
        for integer_shift in product(*translation_ranges):
            shift = np.asarray(integer_shift, dtype=float)
            clipped_lower = np.maximum(lower + shift, 0.0)
            clipped_upper = np.minimum(upper + shift, 1.0)
            if np.any(clipped_upper <= clipped_lower):
                continue
            copies.append(
                _PeriodicCopy(
                    lower=clipped_lower,
                    upper=clipped_upper,
                    normals=normals,
                    offsets=original_offsets - normals @ shift,
                    bounding_box_fraction=float(np.prod(clipped_upper - clipped_lower)),
                )
            )
    # Visiting large boxes first often labels most occupied points early.
    copies.sort(key=lambda copy: copy.bounding_box_fraction, reverse=True)
    return copies, float(summed_volume)


def _count_union_points(
    samples: np.ndarray,
    copies: list[_PeriodicCopy],
    chunk_size: int,
    containment_tolerance: float,
) -> int:
    count = 0
    for start in range(0, len(samples), chunk_size):
        points = samples[start : start + chunk_size]
        covered = np.zeros(len(points), dtype=bool)
        for copy in copies:
            candidates = ~covered
            if not np.any(candidates):
                break
            for axis in range(3):
                candidates &= points[:, axis] >= copy.lower[axis] - containment_tolerance
                candidates &= points[:, axis] <= copy.upper[axis] + containment_tolerance
            indices = np.flatnonzero(candidates)
            if len(indices):
                signed_distances = points[indices] @ copy.normals.T + copy.offsets
                inside = np.all(signed_distances <= containment_tolerance, axis=1)
                covered[indices[inside]] = True
        count += int(np.count_nonzero(covered))
    return count


def periodic_polyhedron_union_fraction(
    hulls_fractional: Iterable[Any],
    *,
    sobol_power: int = 14,
    n_replicates: int = 4,
    seed: int = 0,
    chunk_size: int = 8192,
    containment_tolerance: float = 1e-12,
    max_periodic_copies: int = 20000,
) -> dict[str, Any]:
    """Estimate union volume/cell volume without double-counting any overlap.

    Each input array contains the fractional vertices of one full-dimensional
    ligand hull centred on one of the cell's centre sites.  Periodic translates
    are included whenever their bounding boxes intersect [0, 1)^3.  Input hulls
    themselves may overlap or be duplicates; the union indicator counts each
    sampled position once.  ``sum_volume_fraction`` remains the uncorrected sum
    of input convex-hull volumes and can exceed one.

    Replicates use independent scrambled Sobol sequences with reproducible child
    seeds.  ``fraction_standard_error`` is sample SD of randomized estimates
    divided by sqrt(n_replicates); ``fraction_replicate_range`` is max minus min.
    These are numerical integration diagnostics, not physical uncertainty or a
    guaranteed confidence bound.  A zero diagnostic is not proof of convergence.
    Increase sobol_power and compare estimates for a stronger convergence check.

    n_periodic_copies counts bounding-box candidate copies, some of whose bodies
    may miss the cell.  Tests against their exact convex halfspaces decide point
    membership.  No Qhull jitter, projection, or artificial volume is introduced.
    """

    result: dict[str, Any] = {
        "status": "error",
        "error_reason": None,
        "method": "independent_scrambled_sobol_periodic_union",
        "fraction_mean": None,
        "fraction_replicates": [],
        "fraction_standard_error": None,
        "fraction_replicate_range": None,
        "sum_volume_fraction": None,
        "n_input_hulls": 0,
        "n_periodic_copies": 0,
        "sobol_power": None,
        "n_replicates": None,
        "n_points_per_replicate": None,
        "n_points_evaluated": 0,
        "seed": None,
        "containment_tolerance_fractional": None,
        "diagnostic_interpretation": (
            "Randomized numerical integration diagnostics only; not physical "
            "uncertainty or a guaranteed confidence interval. Zero replicate "
            "spread does not prove convergence."
        ),
    }
    try:
        if not isinstance(sobol_power, (int, np.integer)) or not 1 <= sobol_power <= 22:
            raise ValueError("sobol_power must be an integer between 1 and 22")
        if not isinstance(n_replicates, (int, np.integer)) or not 2 <= n_replicates <= 64:
            raise ValueError("n_replicates must be an integer between 2 and 64")
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if not isinstance(max_periodic_copies, (int, np.integer)) or max_periodic_copies < 1:
            raise ValueError("max_periodic_copies must be a positive integer")
        if not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(containment_tolerance) or containment_tolerance < 0:
            raise ValueError("containment_tolerance must be finite and nonnegative")
        n_points = 2 ** int(sobol_power)
        result.update(
            sobol_power=int(sobol_power),
            n_replicates=int(n_replicates),
            n_points_per_replicate=n_points,
            seed=int(seed),
            containment_tolerance_fractional=float(containment_tolerance),
        )
        hulls = list(hulls_fractional)
        result["n_input_hulls"] = len(hulls)
        if not hulls:
            result.update(
                status="empty",
                method="exact_empty_set",
                fraction_mean=0.0,
                fraction_replicates=[0.0] * int(n_replicates),
                fraction_standard_error=0.0,
                fraction_replicate_range=0.0,
                sum_volume_fraction=0.0,
            )
            return result
        copies, summed_volume = _prepare_copies(hulls, int(max_periodic_copies))
        result["n_periodic_copies"] = len(copies)
        result["sum_volume_fraction"] = summed_volume
        seed_sequences = np.random.SeedSequence(int(seed)).spawn(int(n_replicates))
        estimates: list[float] = []
        for sequence in seed_sequences:
            sampler = qmc.Sobol(d=3, scramble=True, seed=np.random.default_rng(sequence))
            samples = sampler.random_base2(int(sobol_power))
            count = _count_union_points(samples, copies, int(chunk_size), float(containment_tolerance))
            estimates.append(float(count / n_points))
        result.update(
            status="ok",
            fraction_mean=float(np.mean(estimates)),
            fraction_replicates=estimates,
            fraction_standard_error=float(np.std(estimates, ddof=1) / np.sqrt(n_replicates)),
            fraction_replicate_range=float(np.ptp(estimates)),
            n_points_evaluated=int(n_points * n_replicates),
        )
    except (TypeError, ValueError, OverflowError, QhullError, np.linalg.LinAlgError) as exc:
        reason = str(exc).splitlines()[0] if str(exc) else "unspecified numerical failure"
        result["error_reason"] = f"{type(exc).__name__}: {reason}"
    return result


__all__ = ["periodic_polyhedron_union_fraction"]
