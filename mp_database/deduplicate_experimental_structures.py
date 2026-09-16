#!/usr/bin/env python3
"""Filter MP experimentally observed entries, then apply the existing v1 dedup rule."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

import deduplicate_candidate_structures as dedup


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument(
        "--workdir", type=Path,
        default=Path("build/screening/materials_project_experimental_selection_v1"),
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("results/screening/materials_project_experimental_structure_dedup_v1"),
    )
    args = parser.parse_args()
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero")
    if args.outdir.exists() or args.workdir.exists():
        raise FileExistsError("Choose new output and work directories for a fresh run")

    source = Path("data/candidates/materials_project")
    config = Path("configs/materials_project_structure_dedup_v1.json")
    training_origin = Path("data/training/origin.csv")
    training_groups = Path("data/training/deduplicated.csv")
    source_paths = [source / "metadata.csv", source / "structure_metrics.csv",
                    source / "metadata.provenance.json", config,
                    training_origin, training_groups]
    source_hashes = {str(path): dedup.sha256_file(path) for path in source_paths}
    meta_columns, metadata = read_rows(source / "metadata.csv")
    inventory_columns, inventory = read_rows(source / "structure_metrics.csv")
    for label, rows in [("metadata", metadata), ("inventory", inventory)]:
        if len({row["cif_file"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate CIF identities in {label}")
    if {row["cif_file"] for row in metadata} != {row["cif_file"] for row in inventory}:
        raise ValueError("Source metadata and inventory CIF sets differ")
    values = Counter(row["theoretical"].strip().lower() for row in metadata)
    if set(values) - {"true", "false", ""}:
        raise ValueError(f"Unexpected theoretical values: {dict(values)}")
    experimental = [row for row in metadata if row["theoretical"].strip().lower() == "false"]
    if not experimental or any(row["mp_metadata_status"] != "ok" for row in experimental):
        raise ValueError("Experimental entries require valid returned MP metadata")
    selected_cifs = {row["cif_file"] for row in experimental}
    selected_inventory = [row for row in inventory if row["cif_file"] in selected_cifs]
    cif_hashes = {name: dedup.sha256_file(source / "cif" / name)
                  for name in sorted(selected_cifs)}
    args.workdir.mkdir(parents=True, exist_ok=False)
    write_rows(args.workdir / "metadata.csv", meta_columns, experimental)
    write_rows(args.workdir / "structure_metrics.csv", inventory_columns, selected_inventory)
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter": "theoretical == False (Experimentally Observed: Yes); missing excluded",
        "source_rows": len(metadata), "experimental_rows": len(experimental),
        "theoretical_rows": values["true"], "unknown_rows": values[""],
        "source_hashes": source_hashes,
        "selection_script_sha256": dedup.sha256_file(Path(__file__)),
    }
    dedup._write_json(args.workdir / "selection_provenance.json", provenance)
    print(f"Selected {len(experimental)} / {len(metadata)} experimentally observed entries", flush=True)
    dedup.run(argparse.Namespace(
        inventory=args.workdir / "structure_metrics.csv", metadata=args.workdir / "metadata.csv",
        cif_dir=source / "cif", config=config, training_origin=training_origin,
        training_groups=training_groups, aligned_table=[], outdir=args.outdir,
        n_jobs=args.n_jobs, large_bucket_threshold=32,
    ))

    assignments = pd.read_csv(args.outdir / "structure_dedup_assignments.csv")
    representatives = pd.read_csv(args.outdir / "representative_inventory.csv")
    families = pd.read_csv(args.outdir / "structure_families.csv")
    if set(assignments["cif_file"]) != selected_cifs or not assignments["cif_file"].is_unique:
        raise RuntimeError("Assignments must cover every experimental input exactly once")
    if not assignments["theoretical"].eq(False).all():
        raise RuntimeError("Non-experimental material entered the output")
    if not assignments["structure_match_status"].eq("ok").all():
        raise RuntimeError("A CIF parsing or structure matching error occurred")
    if len(families) != len(representatives) or not families["structure_family_id"].is_unique:
        raise RuntimeError("Family and representative identities are inconsistent")
    if set(representatives["cif_file"]) != set(assignments.loc[
        assignments["is_structure_representative"], "cif_file"
    ]):
        raise RuntimeError("Representative inventory differs from assignment selections")
    for family_id, group in assignments.groupby("structure_family_id"):
        selected = group.loc[group["is_structure_representative"]]
        expected = min(group.to_dict("records"), key=dedup._representative_key)
        family_row = families.loc[families["structure_family_id"].eq(family_id)]
        if (len(selected) != 1 or selected.iloc[0]["cif_file"] != expected["cif_file"]
                or not group["representative_cif_file"].eq(expected["cif_file"]).all()
                or not group["structure_family_size"].eq(len(group)).all()
                or len(family_row) != 1 or family_row.iloc[0]["structure_family_size"] != len(group)
                or group["formula"].nunique() != 1 or group["topology_signature"].nunique() != 1):
            raise RuntimeError(f"Family consistency or representative policy failure: {family_id}")
    for path, digest in source_hashes.items():
        if dedup.sha256_file(Path(path)) != digest:
            raise RuntimeError(f"Source file changed: {path}")
    for name, digest in cif_hashes.items():
        if dedup.sha256_file(source / "cif" / name) != digest:
            raise RuntimeError(f"Source CIF changed: {name}")
    pd.DataFrame([{"cif_file": name, "sha256": digest} for name, digest in cif_hashes.items()]).to_csv(
        args.outdir / "cif_content_hashes.csv", index=False
    )
    counts = []
    for halogen in ["F", "Cl", "Br", "I"]:
        before = sum(halogen in row["chemsys"].split("-") for row in experimental)
        after = int(representatives["chemsys"].map(lambda value: halogen in value.split("-")).sum())
        counts.append({"halogen": halogen, "input_entries": before,
                       "representatives": after, "removed_duplicates": before - after})
    per_halogen = pd.DataFrame(counts)
    per_halogen.to_csv(args.outdir / "counts_by_halogen.csv", index=False)
    duplicates = assignments.loc[assignments["structure_family_size"] > 1].sort_values(
        ["structure_family_id", "cif_file"]
    )
    duplicates.to_csv(args.outdir / "duplicate_family_members.csv", index=False)
    archive = args.outdir / "representative_cifs.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as handle:
        for name in sorted(representatives["cif_file"]):
            handle.write(source / "cif" / name, arcname=name)
    with ZipFile(archive) as handle:
        if set(handle.namelist()) != set(representatives["cif_file"]) or handle.testzip() is not None:
            raise RuntimeError("Representative CIF archive validation failed")
    summary_path = args.outdir / "deduplication_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["scope"] = "Independent structure deduplication of experimentally observed MP entries"
    summary["selection_provenance"] = provenance
    summary["counts_by_halogen"] = counts
    summary["family_id_scope"] = "This experimental subset only; do not join full-inventory families by ID"
    dedup._write_json(summary_path, summary)
    dedup._write_json(args.outdir / "validation_report.json", {
        "status": "pass", "experimental_inputs": len(assignments),
        "representatives": len(representatives), "removed_duplicates": len(assignments) - len(representatives),
        "parse_or_match_errors": 0, "source_hashes_unchanged": True,
        "checks": ["exact experimental input coverage", "one representative per family",
                   "same formula and topology signature within each family", "representative policy",
                   "family sizes", "original input and CIF hashes unchanged", "CIF ZIP contents and CRC"],
        "output_hashes": {path.name: dedup.sha256_file(path) for path in sorted(args.outdir.iterdir()) if path.is_file()},
    })
    print(per_halogen.to_string(index=False), flush=True)
    print(f"Validated {len(assignments)} -> {len(representatives)}; outputs: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
