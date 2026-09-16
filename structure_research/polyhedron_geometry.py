"""Local ligand geometry, independent of labels and the periodic-cell convention.

The caller must supply the explicit periodic ligand images belonging to one
centre.  Only ligands form the convex hull: the centre is never a hull vertex.
Planar coordination polygons are reported separately from closed 3D surfaces.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def cell_surface_area_A2(lattice_matrix: Any) -> float:
    """Six-face area of a nondegenerate cell whose rows are lattice vectors.

    This quantity is invariant to rigid motions but changes under general
    equivalent basis changes and supercells; it is not an intensive quantity.
    """

    matrix = np.asarray(lattice_matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("lattice_matrix must be a finite (3, 3) matrix")
    determinant = float(np.linalg.det(matrix))
    if not np.isfinite(determinant) or determinant == 0.0:
        raise ValueError("lattice_matrix must describe a nonzero-volume cell")
    a, b, c = matrix
    result = 2.0 * sum(
        float(np.linalg.norm(np.cross(left, right)))
        for left, right in ((a, b), (b, c), (c, a))
    )
    if not np.isfinite(result) or result <= 0:
        raise ValueError("cell surface area must be finite and positive")
    return float(result)


def _unique_points(points: np.ndarray, tolerance_A: float) -> np.ndarray:
    """Keep lexicographically first vertices, removing Cartesian duplicates.

    Exact duplicates are removed first.  Remaining vertices within the absolute
    distance tolerance of an already retained vertex are merged.  The relative
    rank tolerance is not used to merge geometrically separate vertices.
    """

    exact_unique = np.unique(points, axis=0)
    retained: list[np.ndarray] = []
    for point in exact_unique:
        if not retained or all(
            float(np.linalg.norm(point - existing)) > tolerance_A
            for existing in retained
        ):
            retained.append(point)
    return np.asarray(retained, dtype=float).reshape((-1, 3))


def describe_ligand_hull(
    points_cart: Any,
    center_cart: Any = None,
    abs_tol_A: float = 1e-8,
    rel_tol: float = 1e-7,
) -> dict[str, Any]:
    """Describe a finite ligand cloud using its affine rank and convex hull.

    Coordinates are in angstroms.  The rank is calculated after subtracting the
    ligand centroid, with threshold ``max(abs_tol_A, rel_tol * sigma_max)``.
    ``surface_area_A2`` and ``volume_A3`` are defined only for a full 3D hull.
    ``planar_area_A2`` is the single polygon area in the best-fit ligand plane.
    No centre atom, synthetic thickness, or Qhull jitter is introduced.

    Results contain Python scalars/lists/None and can be serialized as strict
    JSON.  Empty input is a distinct status.  Invalid numerical inputs and hull
    failures are retained as ``hull_error`` with an ``error_reason``.
    """

    result: dict[str, Any] = {
        "n_input_vertices": 0,
        "n_unique_vertices": 0,
        "affine_rank": None,
        "svd_singular_values_A": [],
        "rank_threshold_A": None,
        "geometry_status": "empty",
        "surface_area_A2": None,
        "volume_A3": None,
        "planar_area_A2": None,
        "hull_vertex_count": 0,
        "center_inside_hull": None,
        "near_planar": False,
        "error_reason": None,
    }
    try:
        if not np.isfinite(abs_tol_A) or abs_tol_A <= 0:
            raise ValueError("abs_tol_A must be finite and positive")
        if not np.isfinite(rel_tol) or rel_tol < 0:
            raise ValueError("rel_tol must be finite and nonnegative")
        points = np.asarray(points_cart, dtype=float)
        if points.ndim > 0:
            result["n_input_vertices"] = int(len(points))
        if points.size == 0 and points.shape in ((0,), (0, 3)):
            return result
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_cart must have shape (n, 3)")
        if not np.all(np.isfinite(points)):
            raise ValueError("points_cart contains non-finite coordinates")
        center = None
        if center_cart is not None:
            center = np.asarray(center_cart, dtype=float)
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                raise ValueError("center_cart must be a finite length-3 vector")

        # Subtract an anchor before the centroid to reduce loss of precision
        # when CIF coordinates or chosen periodic images have a large origin.
        anchor = points[0].copy()
        relative = points - anchor
        if not np.all(np.isfinite(relative)):
            raise ValueError("ligand coordinate differences exceed finite precision")
        unique = _unique_points(relative, float(abs_tol_A))
        result["n_unique_vertices"] = int(len(unique))
        centroid = np.mean(unique, axis=0)
        shifted = unique - centroid
        if not np.all(np.isfinite(shifted)):
            raise ValueError("centred ligand coordinates exceed finite precision")
        _, singular, axes = np.linalg.svd(shifted, full_matrices=False)
        if not np.all(np.isfinite(singular)):
            raise ValueError("ligand singular values exceed finite precision")
        singular_values = np.zeros(3, dtype=float)
        singular_values[: len(singular)] = singular
        threshold = max(float(abs_tol_A), float(rel_tol) * singular_values[0])
        if not np.isfinite(threshold):
            raise ValueError("rank threshold exceeds finite precision")
        rank = int(np.sum(singular_values > threshold))
        result["svd_singular_values_A"] = [float(x) for x in singular_values]
        result["rank_threshold_A"] = float(threshold)
        result["affine_rank"] = rank

        if rank == 0:
            result["geometry_status"] = "point_0d"
            result["hull_vertex_count"] = 1
        elif rank == 1:
            result["geometry_status"] = "linear_1d"
            result["hull_vertex_count"] = 2
        elif rank == 2:
            projected = shifted @ axes[:2].T
            hull = ConvexHull(projected)
            area = float(hull.volume)  # Qhull volume is polygon area in 2D.
            if not np.isfinite(area) or area <= 0:
                raise ValueError("planar hull area must be finite and positive")
            result["geometry_status"] = "planar_2d"
            result["planar_area_A2"] = area
            result["hull_vertex_count"] = int(len(hull.vertices))
        else:
            hull = ConvexHull(shifted)
            area, volume = float(hull.area), float(hull.volume)
            if not np.isfinite(area) or not np.isfinite(volume) or min(area, volume) <= 0:
                raise ValueError("3D hull area and volume must be finite and positive")
            result["geometry_status"] = "full_3d"
            result["surface_area_A2"] = area
            result["volume_A3"] = volume
            result["hull_vertex_count"] = int(len(hull.vertices))
            result["near_planar"] = bool(singular_values[2] / singular_values[0] < 1e-4)
            if center is not None:
                center_relative = (center - anchor) - centroid
                offsets = hull.equations[:, :3] @ center_relative + hull.equations[:, 3]
                result["center_inside_hull"] = bool(np.all(offsets <= threshold))
    except (ValueError, TypeError, OverflowError, np.linalg.LinAlgError, QhullError) as exc:
        result["geometry_status"] = "hull_error"
        result["error_reason"] = f"{type(exc).__name__}: {exc}"
        result["surface_area_A2"] = None
        result["volume_A3"] = None
        result["planar_area_A2"] = None
        result["center_inside_hull"] = None
        result["near_planar"] = False
    return result


__all__ = ["cell_surface_area_A2", "describe_ligand_hull"]
