#!/usr/bin/env python3
"""Create an auditable, non-destructive structure-deduplicated candidate pool.

The canonical Materials Project inventory is never overwritten. Structures are
compared only within the same reduced formula and legacy ``dim/st1/st2/st3``
signature. StructureMatcher then removes geometrically equivalent entries while
preserving candidates with different ExtraTrees structural inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure


KEY_COLUMNS = ["formula", "dim", "st1", "st2", "st3"]
SAFE_LABEL = re.compile(r"^[A-Za-z0-9_-]+$")
DEFAULT_LARGE_BUCKET_THRESHOLD = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("structure-dedup config schema_version must be 1")
    if config.get("method_name") != "materials_project_structure_dedup_v1":
        raise ValueError("unexpected structure-dedup method_name")
    if config.get("grouping_scope") != "same_reduced_formula_only":
        raise ValueError("structures must be compared only within the same formula")

    matcher = config.get("structure_matcher")
    expected_matcher_keys = {
        "backend",
        "ltol",
        "stol",
        "angle_tol_deg",
        "primitive_cell",
        "scale",
        "attempt_supercell",
        "allow_subset",
        "anonymous",
    }
    if not isinstance(matcher, dict) or set(matcher) != expected_matcher_keys:
        raise ValueError("structure_matcher config keys differ from the v1 contract")
    if matcher["backend"] != "pymatgen.StructureMatcher":
        raise ValueError("only pymatgen.StructureMatcher is supported")
    for name in ["ltol", "stol", "angle_tol_deg"]:
        value = float(matcher[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid positive matcher parameter: {name}")
    for name in [
        "primitive_cell",
        "scale",
        "attempt_supercell",
        "allow_subset",
        "anonymous",
    ]:
        if not isinstance(matcher[name], bool):
            raise TypeError(f"matcher option must be boolean: {name}")
    if matcher["allow_subset"]:
        raise ValueError("allow_subset must remain false for structure grouping")
    if matcher["anonymous"]:
        raise ValueError("anonymous species matching is forbidden")

    guard = config.get("topology_guard")
    if not isinstance(guard, dict) or guard.get("enabled") is not True:
        raise ValueError("the v1 topology guard must be enabled")
    if guard.get("columns") != KEY_COLUMNS[1:]:
        raise ValueError(f"topology guard columns must be {KEY_COLUMNS[1:]}")
    if int(guard.get("round_decimals", -1)) != 8:
        raise ValueError("topology guard round_decimals must be 8")
    expected_policy = [
        "non_deprecated_first",
        "lowest_energy_above_hull",
        "lowest_formation_energy_per_atom",
        "lexicographically_smallest_cif_file",
    ]
    if config.get("representative_policy") != expected_policy:
        raise ValueError("representative policy differs from the v1 contract")
    return config


def _finite_or_inf(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.inf
    return numeric if math.isfinite(numeric) else math.inf


def _signature(record: dict[str, Any], config: dict[str, Any]) -> tuple[str, ...]:
    guard = config["topology_guard"]
    decimals = int(guard["round_decimals"])
    values: list[str] = []
    for column in guard["columns"]:
        numeric = _finite_or_inf(record.get(column))
        if not math.isfinite(numeric):
            values.append("NA")
            continue
        rounded = round(numeric, decimals)
        if rounded == 0.0:
            rounded = 0.0
        values.append(format(rounded, f".{decimals}f"))
    return tuple(values)


def _deprecated_rank(value: Any) -> int:
    if pd.isna(value):
        return 1
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no"}:
            return 0
        if normalized in {"true", "1", "yes"}:
            return 2
        return 1
    return 2 if bool(value) else 0


def _representative_key(record: dict[str, Any]) -> tuple[int, float, float, str]:
    return (
        _deprecated_rank(record.get("deprecated")),
        _finite_or_inf(record.get("energy_above_hull")),
        _finite_or_inf(record.get("formation_energy_per_atom")),
        str(record["cif_file"]),
    )


def _formula_from_cif_file(value: Any) -> str:
    name = Path(str(value)).name
    if "_mp-" not in name:
        raise ValueError(f"CIF filename lacks '_mp-' identity delimiter: {name}")
    raw = name.split("_mp-", 1)[0]
    try:
        return str(Composition(raw).reduced_formula)
    except Exception as exc:
        raise ValueError(f"cannot parse formula from CIF filename {name}: {exc}") from exc


def _same_reduced_composition(formula: str, structure: Structure) -> bool:
    expected = Composition(formula).fractional_composition.as_dict()
    observed = structure.composition.fractional_composition.as_dict()
    if set(expected) != set(observed):
        return False
    return all(
        math.isclose(float(expected[key]), float(observed[key]), rel_tol=0.0, abs_tol=1e-8)
        for key in expected
    )


def _matcher(config: dict[str, Any]) -> StructureMatcher:
    matcher = config["structure_matcher"]
    return StructureMatcher(
        ltol=float(matcher["ltol"]),
        stol=float(matcher["stol"]),
        angle_tol=float(matcher["angle_tol_deg"]),
        primitive_cell=matcher["primitive_cell"],
        scale=matcher["scale"],
        attempt_supercell=matcher["attempt_supercell"],
        allow_subset=matcher["allow_subset"],
    )


def _site_counts_can_match(
    matcher: StructureMatcher,
    structure_a: Structure,
    structure_b: Structure,
) -> bool:
    """Return a necessary site-count condition for this matcher.

    With ``allow_subset=False``, StructureMatcher's own ``_strict_match``
    rejects a pair whenever its species mask is not square after the selected
    supercell expansion.  Evaluating the same size condition before lattice
    enumeration is therefore a conservative short circuit, not a new
    structure criterion.
    """

    if not matcher._supercell:  # noqa: SLF001 - mirrors pinned pymatgen internals
        return len(structure_a) == len(structure_b)
    formula_units, expand_a = matcher._get_supercell_size(  # noqa: SLF001
        structure_a, structure_b
    )
    if expand_a:
        return len(structure_b) == len(structure_a) * formula_units
    return len(structure_b) * formula_units == len(structure_a)


def _fit_reduced_pair(
    matcher: StructureMatcher,
    reference: Structure,
    candidate: Structure,
) -> bool:
    """Top-level worker for an unchanged StructureMatcher comparison."""

    return bool(
        matcher.fit(reference, candidate, skip_structure_reduction=True)
    )


def _group_structures_large_bucket(
    structures: list[Structure],
    matcher: StructureMatcher,
    n_jobs: int,
) -> list[list[Structure]]:
    """Order-preserving equivalent of ``group_structures`` for a large bucket.

    Pymatgen's implementation reduces every structure once and then compares
    each reference serially with every remaining structure.  This function
    retains that reference order and the exact ``fit`` call, while pruning
    only pairs that pymatgen must reject on site-count grounds and evaluating
    the remaining independent comparisons in worker processes.
    """

    if matcher._subset:  # noqa: SLF001 - same guard as group_structures
        raise ValueError("allow_subset cannot be used with structure grouping")

    original = list(structures)
    processed = matcher._process_species(original)  # noqa: SLF001
    reduced = [
        matcher._get_reduced_structure(  # noqa: SLF001
            structure,
            matcher._primitive_cell,  # noqa: SLF001
            niggli=True,
        )
        for structure in processed
    ]

    def structure_hash(item: tuple[int, Structure]) -> Any:
        return matcher._comparator.get_hash(item[1].composition)  # noqa: SLF001

    sorted_structures = sorted(enumerate(reduced), key=structure_hash)
    all_groups: list[list[Structure]] = []
    with Parallel(n_jobs=n_jobs, prefer="processes") as parallel:
        for _, hashed_group in itertools.groupby(
            sorted_structures, key=structure_hash
        ):
            unmatched = list(hashed_group)
            while unmatched:
                source_index, reference = unmatched.pop(0)
                compatible_positions = [
                    position
                    for position, (_index, candidate) in enumerate(unmatched)
                    if _site_counts_can_match(matcher, reference, candidate)
                ]
                if n_jobs == 1:
                    fit_results = [
                        _fit_reduced_pair(
                            matcher, reference, unmatched[position][1]
                        )
                        for position in compatible_positions
                    ]
                else:
                    fit_results = parallel(
                        delayed(_fit_reduced_pair)(
                            matcher, reference, unmatched[position][1]
                        )
                        for position in compatible_positions
                    )
                matched_positions = {
                    position
                    for position, is_match in zip(
                        compatible_positions, fit_results, strict=True
                    )
                    if is_match
                }
                member_indices = [source_index]
                member_indices.extend(
                    unmatched[position][0]
                    for position in sorted(matched_positions)
                )
                all_groups.append([original[index] for index in member_indices])
                unmatched = [
                    item
                    for position, item in enumerate(unmatched)
                    if position not in matched_positions
                ]
    return all_groups


def _deduplicate_formula(
    formula: str,
    records: list[dict[str, Any]],
    cif_dir: Path,
    config: dict[str, Any],
    matcher_n_jobs: int = 1,
    large_bucket_threshold: int = DEFAULT_LARGE_BUCKET_THRESHOLD,
) -> list[dict[str, Any]]:
    """Return deterministic topology-guarded StructureMatcher assignments."""

    signatures = {
        _signature(record, config)
        for record in records
    }
    signature_number = {
        signature: number
        for number, signature in enumerate(sorted(signatures), start=1)
    }
    signature_size = Counter(_signature(record, config) for record in records)
    parsed: dict[tuple[str, ...], list[tuple[dict[str, Any], Structure]]] = {
        signature: [] for signature in signatures
    }
    matched_groups: list[
        tuple[list[dict[str, Any]], tuple[str, ...], str, str]
    ] = []
    for record in sorted(records, key=lambda row: str(row["cif_file"])):
        signature = _signature(record, config)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structure = Structure.from_file(cif_dir / str(record["cif_file"]))
            if not _same_reduced_composition(formula, structure):
                raise ValueError(
                    "CIF composition differs from the formula encoded in its filename"
                )
        except Exception as exc:
            matched_groups.append(
                (
                    [record],
                    signature,
                    "parse_error_singleton",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        parsed[signature].append((record, structure))

    # The topology signature is applied before StructureMatcher. This avoids
    # expensive comparisons between entries that must remain distinct for the
    # legacy ExtraTrees schema and makes pathological large formula groups
    # tractable without weakening the duplicate criterion.
    for signature in sorted(parsed):
        pairs = parsed[signature]
        if not pairs:
            continue
        structures = [structure for _record, structure in pairs]
        record_by_object = {id(structure): record for record, structure in pairs}
        try:
            matcher = _matcher(config)
            if len(structures) >= large_bucket_threshold:
                matched = _group_structures_large_bucket(
                    structures, matcher, matcher_n_jobs
                )
            else:
                matched = matcher.group_structures(structures, anonymous=False)
            expected_ids = {id(structure) for structure in structures}
            returned = [structure for group in matched for structure in group]
            returned_ids = {id(structure) for structure in returned}
            if len(returned) != len(structures) or returned_ids != expected_ids:
                raise RuntimeError(
                    "StructureMatcher.group_structures did not return each input exactly once"
                )
            matched_groups.extend(
                (
                    [record_by_object[id(structure)] for structure in group],
                    signature,
                    "ok",
                    "",
                )
                for group in matched
            )
        except Exception as exc:
            # Fail closed: a failed group comparison never removes a structure.
            error = f"{type(exc).__name__}: {exc}"
            matched_groups.extend(
                (
                    [record_by_object[id(structure)]],
                    signature,
                    "matcher_error_singleton",
                    error,
                )
                for structure in structures
            )
    matched_groups.sort(
        key=lambda item: min(str(row["cif_file"]) for row in item[0])
    )

    assignments: list[dict[str, Any]] = []
    for family_number, (members_raw, signature, match_status, match_error) in enumerate(
        matched_groups, start=1
    ):
        members = sorted(members_raw, key=lambda row: str(row["cif_file"]))
        representative = min(members, key=_representative_key)
        family_id = f"{formula}::sf-{family_number:04d}"
        bucket_id = f"{formula}::topo-{signature_number[signature]:04d}"
        for record in members:
            assignments.append(
                {
                    "_source_index": int(record["_source_index"]),
                    "topology_guard_bucket": bucket_id,
                    "topology_guard_bucket_size": int(signature_size[signature]),
                    "structure_family_id": family_id,
                    "structure_family_size": int(len(members)),
                    "topology_signature": "|".join(signature),
                    "structure_match_status": match_status,
                    "is_structure_representative": (
                        int(record["_source_index"])
                        == int(representative["_source_index"])
                    ),
                    "representative_cif_file": str(representative["cif_file"]),
                    "representative_material_id": str(
                        representative.get("material_id", "")
                    ),
                    "structure_match_error": match_error,
                }
            )
    return assignments


def _normalized_group_key(frame: pd.DataFrame) -> pd.Series:
    values = frame[KEY_COLUMNS].copy()
    values["formula"] = values["formula"].astype(str)
    for column in KEY_COLUMNS[1:]:
        numeric = pd.to_numeric(values[column], errors="coerce").round(8)
        values[column] = numeric.map(
            lambda value: "NA" if pd.isna(value) else format(float(value), ".8g")
        )
    return values.astype(str).agg("|".join, axis=1)


def _add_training_overlap(
    assignments: pd.DataFrame,
    training_origin_path: Path | None,
    training_groups_path: Path | None,
) -> pd.DataFrame:
    out = assignments.copy()
    out["member_seen_train_cif"] = False
    out["member_seen_train_formula"] = False
    out["member_seen_train_group_key"] = False
    if training_origin_path is not None:
        origin = pd.read_csv(training_origin_path, low_memory=False)
        required = {"cif_file", "formula"}
        if not required.issubset(origin.columns):
            raise ValueError(f"training origin lacks columns: {sorted(required - set(origin))}")
        out["member_seen_train_cif"] = out["cif_file"].astype(str).isin(
            set(origin["cif_file"].astype(str))
        )
        out["member_seen_train_formula"] = out["formula"].astype(str).isin(
            set(origin["formula"].astype(str))
        )
    if training_groups_path is not None:
        groups = pd.read_csv(training_groups_path, low_memory=False)
        missing = sorted(set(KEY_COLUMNS) - set(groups))
        if missing:
            raise ValueError(f"training groups lack columns: {missing}")
        out["member_seen_train_group_key"] = _normalized_group_key(out).isin(
            set(_normalized_group_key(groups))
        )
    grouped = out.groupby("structure_family_id", sort=False)
    out["family_contains_train_cif"] = grouped[
        "member_seen_train_cif"
    ].transform("any")
    out["family_contains_train_formula"] = grouped[
        "member_seen_train_formula"
    ].transform("any")
    out["family_contains_train_group_key"] = grouped[
        "member_seen_train_group_key"
    ].transform("any")
    out["n_train_cifs_in_family"] = grouped["member_seen_train_cif"].transform(
        "sum"
    ).astype(int)
    representative_train = (
        out["is_structure_representative"].astype(bool)
        & out["member_seen_train_cif"].astype(bool)
    )
    out["representative_is_train_cif"] = representative_train.groupby(
        out["structure_family_id"]
    ).transform("any")
    out["seen_train_structure_family"] = out["family_contains_train_cif"]
    return out


def _parse_aligned_tables(values: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    labels: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError("--aligned-table must have LABEL=PATH form")
        label, raw_path = value.split("=", 1)
        if not SAFE_LABEL.fullmatch(label):
            raise ValueError(f"unsafe aligned-table label: {label!r}")
        if label in labels:
            raise ValueError(f"duplicate aligned-table label: {label}")
        labels.add(label)
        parsed.append((label, Path(raw_path)))
    return parsed


def _family_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id, group in assignments.groupby("structure_family_id", sort=True):
        representative = group.loc[group["is_structure_representative"].astype(bool)]
        if len(representative) != 1:
            raise RuntimeError(f"family {family_id} does not have exactly one representative")
        rep = representative.iloc[0]
        rows.append(
            {
                "structure_family_id": family_id,
                "formula": str(rep["formula"]),
                "structure_family_size": int(len(group)),
                "topology_guard_bucket": str(rep["topology_guard_bucket"]),
                "topology_guard_bucket_size": int(rep["topology_guard_bucket_size"]),
                "topology_signature": str(rep["topology_signature"]),
                "structure_match_status": str(rep["structure_match_status"]),
                "representative_cif_file": str(rep["cif_file"]),
                "representative_material_id": str(rep.get("material_id", "")),
                "representative_energy_above_hull": rep.get(
                    "energy_above_hull", np.nan
                ),
                "representative_formation_energy_per_atom": rep.get(
                    "formation_energy_per_atom", np.nan
                ),
                "member_cif_files": ";".join(
                    sorted(group["cif_file"].astype(str))
                ),
                "family_contains_train_cif": bool(
                    group["family_contains_train_cif"].iloc[0]
                ),
                "family_contains_train_formula": bool(
                    group["family_contains_train_formula"].iloc[0]
                ),
                "family_contains_train_group_key": bool(
                    group["family_contains_train_group_key"].iloc[0]
                ),
                "n_train_cifs_in_family": int(
                    group["n_train_cifs_in_family"].iloc[0]
                ),
                "representative_is_train_cif": bool(
                    group["representative_is_train_cif"].iloc[0]
                ),
                "seen_train_structure_family": bool(
                    group["seen_train_structure_family"].iloc[0]
                ),
                "structure_match_error": ";".join(
                    sorted(
                        value
                        for value in group["structure_match_error"].astype(str).unique()
                        if value
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["formula", "structure_family_id"], kind="mergesort"
    )


def run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    inventory = pd.read_csv(args.inventory, low_memory=False)
    required = set(KEY_COLUMNS + ["cif_file"])
    missing = sorted(required - set(inventory))
    if missing:
        raise ValueError(f"inventory lacks columns: {missing}")
    if inventory["cif_file"].duplicated().any():
        raise ValueError("inventory cif_file values must be unique")
    inventory = inventory.copy()
    inventory["_source_index"] = np.arange(len(inventory), dtype=int)
    inventory["formula_as_loaded"] = inventory["formula"].astype(str)
    inventory["formula"] = inventory["cif_file"].map(_formula_from_cif_file)
    inventory["formula_corrected_from_loaded"] = inventory["formula"].ne(
        inventory["formula_as_loaded"]
    )

    input_hashes = {str(args.inventory): sha256_file(args.inventory)}
    if args.metadata is not None:
        metadata = pd.read_csv(args.metadata, low_memory=False)
        if "cif_file" not in metadata or metadata["cif_file"].duplicated().any():
            raise ValueError("metadata requires unique cif_file values")
        if set(metadata["cif_file"].astype(str)) != set(
            inventory["cif_file"].astype(str)
        ):
            raise ValueError("metadata and inventory cif_file sets differ")
        collisions = sorted((set(inventory) & set(metadata)) - {"cif_file"})
        metadata = metadata.drop(columns=collisions)
        inventory = inventory.merge(
            metadata, on="cif_file", how="left", validate="one_to_one", sort=False
        )
        input_hashes[str(args.metadata)] = sha256_file(args.metadata)
    if "material_id" not in inventory:
        inventory["material_id"] = inventory["cif_file"].astype(str).str.extract(
            r"_(mp-[^.]+)\.cif$", expand=False
        )

    missing_cifs = [
        value
        for value in inventory["cif_file"].astype(str)
        if not (args.cif_dir / value).is_file()
    ]
    if missing_cifs:
        raise FileNotFoundError(f"missing {len(missing_cifs)} CIF files: {missing_cifs[:5]}")

    tasks = [
        (str(formula), group.to_dict("records"))
        for formula, group in inventory.groupby("formula", sort=True)
    ]
    large_formulas = {
        formula
        for formula, records in tasks
        if max(
            Counter(_signature(record, config) for record in records).values()
        )
        >= args.large_bucket_threshold
    }
    small_tasks = [task for task in tasks if task[0] not in large_formulas]
    small_results = Parallel(
        n_jobs=args.n_jobs, prefer="processes", verbose=5
    )(
        delayed(_deduplicate_formula)(
            formula,
            records,
            args.cif_dir,
            config,
            1,
            args.large_bucket_threshold,
        )
        for formula, records in small_tasks
    )
    assignments_by_formula = {
        formula: result
        for (formula, _records), result in zip(
            small_tasks, small_results, strict=True
        )
    }
    # Large formulas run outside the formula-level process pool and receive
    # the same total worker budget internally.  This avoids nested pools and
    # removes the single-core straggler created by a pathological large bucket.
    for formula, records in tasks:
        if formula not in large_formulas:
            continue
        assignments_by_formula[formula] = _deduplicate_formula(
            formula,
            records,
            args.cif_dir,
            config,
            args.n_jobs,
            args.large_bucket_threshold,
        )
    grouped_assignments = [
        assignments_by_formula[formula] for formula, _records in tasks
    ]
    assignments_only = pd.DataFrame(
        [record for group in grouped_assignments for record in group]
    )
    if len(assignments_only) != len(inventory):
        raise RuntimeError("not every inventory row received a structure-family assignment")
    if assignments_only["_source_index"].duplicated().any():
        raise RuntimeError("a source row received multiple structure-family assignments")
    assignments = inventory.merge(
        assignments_only,
        on="_source_index",
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("_source_index", kind="mergesort")
    assignments = _add_training_overlap(
        assignments, args.training_origin, args.training_groups
    )

    representatives = assignments.loc[
        assignments["is_structure_representative"].astype(bool)
    ].copy()
    if representatives["structure_family_id"].duplicated().any():
        raise RuntimeError("representative structure_family_id values must be unique")
    family_summary = _family_summary(assignments)
    if len(family_summary) != len(representatives):
        raise RuntimeError("family summary and representative counts differ")

    args.outdir.mkdir(parents=True, exist_ok=False)
    assignment_path = args.outdir / "structure_dedup_assignments.csv"
    representatives_path = args.outdir / "representative_inventory.csv"
    families_path = args.outdir / "structure_families.csv"
    assignments.drop(columns=["_source_index"]).to_csv(assignment_path, index=False)
    representatives.drop(columns=["_source_index"]).to_csv(
        representatives_path, index=False
    )
    family_summary.to_csv(families_path, index=False)

    family_columns = [
        "cif_file",
        "topology_guard_bucket",
        "topology_guard_bucket_size",
        "structure_family_id",
        "structure_family_size",
        "topology_signature",
        "structure_match_status",
        "is_structure_representative",
        "representative_cif_file",
        "representative_material_id",
        "family_contains_train_cif",
        "family_contains_train_formula",
        "family_contains_train_group_key",
        "n_train_cifs_in_family",
        "representative_is_train_cif",
        "seen_train_structure_family",
    ]
    aligned_outputs: dict[str, dict[str, Any]] = {}
    inventory_cifs = set(assignments["cif_file"].astype(str))
    representative_cifs = set(representatives["cif_file"].astype(str))
    for label, path in _parse_aligned_tables(args.aligned_table):
        table = pd.read_csv(path, low_memory=False)
        if "cif_file" not in table or table["cif_file"].duplicated().any():
            raise ValueError(f"aligned table {label} requires unique cif_file values")
        if set(table["cif_file"].astype(str)) != inventory_cifs:
            raise ValueError(f"aligned table {label} does not cover the inventory exactly")
        selected = table.loc[table["cif_file"].astype(str).isin(representative_cifs)].copy()
        selected = representatives[["cif_file", "_source_index"]].merge(
            selected, on="cif_file", how="left", validate="one_to_one", sort=False
        ).sort_values("_source_index", kind="mergesort")
        if "formula" in selected:
            selected["formula_as_loaded"] = selected["formula"].astype(str)
            canonical_formula = representatives.set_index("cif_file")["formula"]
            selected["formula"] = selected["cif_file"].map(canonical_formula)
        additions = [
            name
            for name in family_columns
            if name != "cif_file" and name not in selected
        ]
        selected = selected.merge(
            representatives[["cif_file", *additions]],
            on="cif_file",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        selected = selected.drop(columns=["_source_index"])
        output_path = args.outdir / f"representatives__{label}.csv"
        selected.to_csv(output_path, index=False)
        input_hashes[str(path)] = sha256_file(path)
        aligned_outputs[label] = {
            "input": str(path),
            "output": str(output_path),
            "rows": int(len(selected)),
            "sha256": sha256_file(output_path),
        }

    family_sizes = Counter(family_summary["structure_family_size"].astype(int))
    topology_bucket_count = int(assignments["topology_guard_bucket"].nunique())
    parse_or_match_errors = assignments["structure_match_error"].astype(str).ne("")
    summary = {
        "schema_version": 1,
        "method_name": config["method_name"],
        "scope": "non-destructive full Materials Project candidate structure deduplication",
        "input_hashes": input_hashes,
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "implementation_sha256": sha256_file(Path(__file__)),
        "software": {
            "pymatgen": importlib.metadata.version("pymatgen"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": importlib.metadata.version("joblib"),
        },
        "matcher": config["structure_matcher"],
        "topology_guard": config["topology_guard"],
        "representative_policy": config["representative_policy"],
        "performance_policy": {
            "large_bucket_threshold": int(args.large_bucket_threshold),
            "large_formula_count": int(len(large_formulas)),
            "large_formulas": sorted(large_formulas),
            "large_bucket_matcher_n_jobs": int(args.n_jobs),
            "semantic_note": (
                "same ordered StructureMatcher.fit calls; only impossible "
                "site-count pairs are short-circuited"
            ),
        },
        "counts": {
            "input_structure_entries": int(len(assignments)),
            "input_unique_formulas": int(assignments["formula"].nunique()),
            "formula_values_corrected_from_loaded_csv": int(
                assignments["formula_corrected_from_loaded"].sum()
            ),
            "topology_guard_buckets": topology_bucket_count,
            "structurematcher_families_after_topology_guard": int(len(family_summary)),
            "representative_structures": int(len(representatives)),
            "removed_duplicate_entries": int(len(assignments) - len(representatives)),
            "duplicate_families": int(
                (family_summary["structure_family_size"] > 1).sum()
            ),
            "singleton_families": int(
                (family_summary["structure_family_size"] == 1).sum()
            ),
            "largest_structure_family": int(
                family_summary["structure_family_size"].max()
            ),
            "rows_with_parse_or_match_error": int(parse_or_match_errors.sum()),
            "representative_cifs_seen_in_training": int(
                representatives["member_seen_train_cif"].sum()
            ),
            "structure_families_containing_training_cif": int(
                family_summary["family_contains_train_cif"].sum()
            ),
            "structure_families_containing_training_formula": int(
                family_summary["family_contains_train_formula"].sum()
            ),
            "structure_families_containing_training_group_key": int(
                family_summary["family_contains_train_group_key"].sum()
            ),
        },
        "family_size_distribution": {
            str(size): int(count) for size, count in sorted(family_sizes.items())
        },
        "outputs": {
            "structure_dedup_assignments.csv": sha256_file(assignment_path),
            "representative_inventory.csv": sha256_file(representatives_path),
            "structure_families.csv": sha256_file(families_path),
        },
        "aligned_outputs": aligned_outputs,
    }
    _write_json(args.outdir / "deduplication_summary.json", summary)
    print(json.dumps(summary["counts"], indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cif-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/materials_project_structure_dedup_v1.json"),
    )
    parser.add_argument("--training-origin", type=Path)
    parser.add_argument("--training-groups", type=Path)
    parser.add_argument(
        "--aligned-table",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="subset a one-row-per-CIF table to the selected representatives",
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--large-bucket-threshold",
        type=int,
        default=DEFAULT_LARGE_BUCKET_THRESHOLD,
        help=(
            "minimum topology-bucket size that receives site-count pruning "
            "and the full --n-jobs budget"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero")
    if args.large_bucket_threshold < 2:
        raise ValueError("--large-bucket-threshold must be at least 2")
    run(args)


if __name__ == "__main__":
    main()
