"""Physical geometry comparison after a separately declared standardization.

Phonopy symprec is not used here. The reported minimax score combines explicit
lattice and Cartesian position tolerances. Periodic shortest vectors, species
bijections, and proper rigid rotations are handled by pymatgen. There is no
volume rescaling and no anonymous or subset matching.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from scipy.cluster.hierarchy import fcluster, linkage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/screening"))
from deduplicate_cell_equivalent_structures import CellChoiceMatcher

DEFAULT_CONFIG = {
    "main": {"length_relative": 0.01, "angle_deg": 1.0,
             "max_site_A": 0.10, "rms_site_A": 0.05,
             "volume_per_atom_relative": 0.03},
    "strict": {"length_relative": 1e-5, "angle_deg": 0.001,
               "max_site_A": 1e-4, "rms_site_A": 1e-4,
               "volume_per_atom_relative": 1e-4},
    "maximum_sensitivity_multiplier": 2.0,
    "niggli_reduce_for_matching": True,
    "exact_translation_primitive_tolerance_A": 1e-6,
}


def symmetric_relative(a, b):
    return np.maximum(np.asarray(a) / np.asarray(b), np.asarray(b) / np.asarray(a)) - 1


def metric_score(metrics, thresholds):
    return max(metrics[key] / thresholds[key] for key in thresholds)


def prepare_matching_structure(structure: Structure, config: dict | None = None):
    config = DEFAULT_CONFIG if config is None else config
    prepared = structure.get_primitive_structure(
        tolerance=config.get("exact_translation_primitive_tolerance_A", 1e-6), use_site_props=False)
    if config["niggli_reduce_for_matching"]:
        prepared = prepared.get_reduced_structure(reduction_algo="niggli")
    return prepared


def compare_geometry(a: Structure, b: Structure, config: dict | None = None) -> dict:
    """Return scores <=0.5/1/2 for nested approximate pair acceptance levels.

    ``strict_score<=1`` describes strict agreement of the supplied structures,
    which may already have been idealized upstream. The position distances are
    in the average lattice metric, after removal of a mean translation; lattice
    strain is scored separately. The search enumerates candidate cell mappings
    and atom-origin translations, with species-constrained least-squares atom
    assignments, following pymatgen's algorithm. It is not a proof of a global
    arbitrary-deformation minimum. Missing scores are outside the search range.
    """
    config = DEFAULT_CONFIG if config is None else config
    main, strict = config["main"], config["strict"]
    maximum = config["maximum_sensitivity_multiplier"]
    result = {"score": None, "strict_score": None, "status": "pending"}
    if not a.is_ordered or not b.is_ordered:
        return {**result, "status": "unsupported_disordered_structure"}
    if a.composition.fractional_composition != b.composition.fractional_composition:
        return {**result, "status": "different_composition"}
    # Reduce exact translational repetitions so, for example, 2x and 3x
    # representations can be compared through their common primitive cell.
    # This tight geometric operation does not apply Phonopy's 0.5 A idealization.
    if not config.get("inputs_already_reduced", False):
        a = prepare_matching_structure(a, config)
        b = prepare_matching_structure(b, config)
    reference_is_a = len(a) >= len(b)
    reference, other = (a, b) if reference_is_a else (b, a)
    if len(reference) % len(other):
        return {**result, "status": "incompatible_site_counts"}
    fu = len(reference) // len(other)
    vdiff = float(symmetric_relative(reference.volume / len(reference), other.volume / len(other)))
    result["volume_per_atom_relative_difference"] = vdiff
    if vdiff > maximum * main["volume_per_atom_relative"] + 1e-12:
        return {**result, "status": "volume_difference_outside_search_range"}
    # stol on this object is used only to build search masks. The actual
    # acceptance below uses _cart_dists(normalization=1), explicitly in A.
    max_site_search = maximum * main["max_site_A"]
    matcher = CellChoiceMatcher(ltol=maximum * main["length_relative"],
        stol=max_site_search, angle_tol=maximum * main["angle_deg"],
        primitive_cell=False, scale=False, attempt_supercell=True, allow_subset=False)
    reference, other = matcher._process_species((reference, other))
    mask, reference_origin_indices, other_origin_index = matcher._get_mask(reference, other, fu, False)
    if mask.shape != (len(reference), len(reference)):
        raise ValueError("Species matching mask is not a complete square bijection")
    params = np.asarray(reference.lattice.parameters)
    best_main = best_strict = float("inf")
    main_evidence = strict_evidence = None
    n_lattices = n_atom_alignments = 0
    for ref_frac, other_frac, average, matrix in matcher._get_supercells(reference, other, fu, False):
        n_lattices += 1
        transformed_lattice = Lattice(np.asarray(matrix) @ other.lattice.matrix)
        other_params = np.asarray(transformed_lattice.parameters)
        length_difference = float(symmetric_relative(params[:3], other_params[:3]).max())
        angle_difference = float(np.abs(params[3:] - other_params[3:]).max())
        if (length_difference > maximum * main["length_relative"] + 1e-12
                or angle_difference > maximum * main["angle_deg"] + 1e-10):
            continue
        # Same periodic prefilter as pymatgen, but with absolute A tolerance.
        frac_tolerance = np.asarray(average.reciprocal_lattice.abc) * max_site_search / np.pi
        lll_tolerance = np.asarray(average.get_lll_reduced_lattice().reciprocal_lattice.abc) * max_site_search / np.pi
        for origin_index in reference_origin_indices:
            translation = ref_frac[origin_index] - other_frac[other_origin_index]
            moved = other_frac + translation
            if not matcher._cmp_fstruct(ref_frac, moved, frac_tolerance, mask):
                continue
            distances, correction, mapping = matcher._cart_dists(
                ref_frac, moved, average, mask, normalization=1.0, lll_frac_tol=lll_tolerance)
            n_atom_alignments += 1
            metrics = {"length_relative": length_difference, "angle_deg": angle_difference,
                       "max_site_A": float(max(distances)),
                       "rms_site_A": float(np.sqrt(np.mean(np.square(distances)))),
                       "volume_per_atom_relative": vdiff}
            if not all(np.isfinite(list(metrics.values()))):
                raise ValueError("Nonfinite geometry residual")
            score = metric_score(metrics, main)
            strict_score = metric_score(metrics, strict)
            if score < best_main or strict_score < best_strict:
                total_translation = translation + correction
                total_translation -= np.round(total_translation)
                evidence = {"reference_is_a": reference_is_a,
                    "n_matched_atoms": len(reference), "supercell_matrix_other_to_reference": np.asarray(matrix, dtype=int).tolist(),
                    "fractional_translation": total_translation.tolist(),
                    "mapping_expanded_other_to_reference": np.asarray(mapping, dtype=int).tolist(),
                    "reference_reduced_lattice_A": reference.lattice.matrix.tolist(),
                    "other_reduced_lattice_A": other.lattice.matrix.tolist(),
                    "average_lattice_A": average.matrix.tolist(), "metrics": metrics,
                    "per_atom_residual_A": np.asarray(distances).tolist()}
                if score < best_main:
                    best_main, main_evidence = score, evidence
                if strict_score < best_strict:
                    best_strict, strict_evidence = strict_score, evidence
            if best_main < 1e-10 and best_strict < 1e-5:
                break
        if best_main < 1e-10 and best_strict < 1e-5:
            break
    result.update(n_candidate_lattices=n_lattices, n_candidate_atom_alignments=n_atom_alignments)
    if not np.isfinite(best_main):
        return {**result, "status": "no_atomic_alignment_in_search_range"}
    result.update(score=float(best_main), strict_score=float(best_strict),
                  main_evidence=main_evidence, strict_evidence=strict_evidence,
                  status="aligned")
    return result


def groups_complete_link(ids, pair_scores, threshold):
    """Cut a deterministic complete-link tree; unmatched pairs cannot bridge."""
    names = sorted(ids)
    if len(names) < 2:
        return [names] if names else []
    condensed = []
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            value = pair_scores.get((first, second))
            condensed.append(float(value) if value is not None and np.isfinite(value) else 1e12)
    tree = linkage(np.asarray(condensed), method="complete")
    labels = fcluster(tree, t=threshold, criterion="distance")
    groups = {}
    for name, label in zip(names, labels):
        groups.setdefault(int(label), []).append(name)
    return sorted(groups.values(), key=lambda group: group[0])


__all__ = ["compare_geometry", "groups_complete_link", "prepare_matching_structure", "DEFAULT_CONFIG"]
