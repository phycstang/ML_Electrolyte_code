#!/usr/bin/env python3
"""Standardize all local MP CIFs with the actual Phonopy --symmetry pipeline.

The versioned Phonopy check_symmetry function is executed unchanged; only its
file-writing callback is replaced to capture its final Phonopy object.  The
API pipeline is checked against real CLI PPOSCAR/BPOSCAR files before the bulk
run.  This module performs no deduplication or cross-material matching.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, ExitStack
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import warnings

import numpy as np
from pymatgen.core import Element, Structure
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.io.vasp import Poscar

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/materials_project_phonopy_symprec_0p5_standardized"
EXPECTED_PHONOPY_VERSION = "2.46.0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structure_payload(structure: Structure) -> dict:
    return {
        "lattice_matrix_A": structure.lattice.matrix.tolist(),
        "lattice_parameters_A_deg": list(structure.lattice.parameters),
        "fractional_positions": structure.frac_coords.tolist(),
        "symbols": [site.specie.symbol for site in structure],
        "n_atoms": len(structure),
        "volume_A3": float(structure.volume),
        "reduced_formula": structure.composition.reduced_formula,
    }


def load_standardized_structure(record: dict, kind: str = "primitive") -> Structure:
    """Rebuild an exact checkpoint geometry; no Phonopy import is needed.

    kind may be primitive, conventional, or original.  Failed records may retain
    original geometry but lack standardized geometry, which raises KeyError.
    """
    if kind not in {"primitive", "conventional", "original"}:
        raise ValueError("kind must be primitive, conventional, or original")
    data = record[kind]
    return Structure(data["lattice_matrix_A"], data["symbols"], data["fractional_positions"])


def _dataset_summary(atoms, tolerance: float) -> dict:
    import spglib

    dataset = spglib.get_symmetry_dataset(atoms.totuple(), symprec=tolerance)
    if dataset is None:
        return {"symprec_A": tolerance, "status": "unavailable", "number": None, "international": None}
    return {
        "symprec_A": tolerance, "status": "ok", "number": int(dataset.number),
        "international": dataset.international, "hall": dataset.hall,
        "choice": dataset.choice, "n_operations": len(dataset.rotations),
    }


def _to_structure(atoms) -> Structure:
    return Structure(atoms.cell, atoms.symbols, atoms.scaled_positions)


@contextmanager
def patched_missing_atomic_masses(symbols):
    """Temporarily supply Pu's missing reference mass for geometry-only calls.

    The installed Phonopy source and input CIFs are never modified.  This is
    reference metadata required by its input parser, not an isotope assignment
    or a mass model for phonons/dynamics.  All symmetry operations and cutoffs
    remain unchanged.  Separate processes are required for concurrent use.
    """
    from phonopy.structure.atoms import atom_data, symbol_map

    metadata = None
    index = symbol_map["Pu"]
    original = atom_data[index][3]
    if "Pu" in set(symbols) and original is None:
        mass = float(Element("Pu").atomic_mass)
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError("Pu reference mass is unavailable in pymatgen")
        metadata = {
            "element": "Pu", "atomic_number": 94, "reference_mass_u": mass,
            "original_phonopy_mass": None,
            "source": 'pymatgen.core.Element("Pu").atomic_mass',
            "source_package_version": importlib.metadata.version("pymatgen"),
            "scope": "In-memory parser metadata compatibility for geometry/symmetry only; not used for phonons, force constants, dynamics, or isotope identification.",
            "installation_files_modified": False, "restored_after_processing": False,
        }
        atom_data[index][3] = mass
    try:
        yield metadata
    finally:
        if metadata is not None:
            atom_data[index][3] = original
            metadata["restored_after_processing"] = atom_data[index][3] is None


def standardize_structure(structure: Structure, symprec: float = 0.5) -> tuple[Structure, Structure, dict]:
    """Return primitive, conventional, and details from the actual CLI pipeline.

    Thread safety: this function patches a Phonopy presentation callback during
    execution.  Use separate processes for parallel calls, never threads.
    ``primitive_poscar_text`` and ``conventional_poscar_text`` in details contain
    exact Phonopy serialization and can be compared byte-for-byte to the CLI.
    """
    import phonopy
    from phonopy import Phonopy
    from phonopy.cui.collect_cell_info import PhonopyCellInfoResult
    import phonopy.cui.show_symmetry as show_symmetry
    from phonopy.interface.vasp import get_vasp_structure_lines, read_vasp_from_strings

    if phonopy.__version__ != EXPECTED_PHONOPY_VERSION:
        raise RuntimeError(f"Expected Phonopy {EXPECTED_PHONOPY_VERSION}, got {phonopy.__version__}")
    if not structure.is_ordered:
        raise ValueError("POSCAR/Phonopy standardization requires an ordered structure")
    if not np.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be finite and positive")
    # Exact in-memory equivalent of the previous CIF -> pymatgen POSCAR ->
    # Phonopy VASP reader path, including the POSCAR numerical serialization.
    poscar_text = Poscar(structure).get_str()
    atoms = read_vasp_from_strings(poscar_text)
    if atoms.symbols != [site.specie.symbol for site in structure]:
        raise ValueError("CIF-to-POSCAR serialization changed atom ordering/species")
    if not np.allclose(atoms.cell, structure.lattice.matrix, atol=1e-12, rtol=0):
        raise ValueError("CIF-to-POSCAR serialization changed lattice")
    if not np.allclose(atoms.scaled_positions, structure.frac_coords, atol=1e-12, rtol=0):
        raise ValueError("CIF-to-POSCAR serialization changed coordinates")

    identity = np.eye(3, dtype=int)
    initial = Phonopy(atoms, identity, primitive_matrix=None, symprec=symprec,
                      calculator=None, log_level=0)
    info = PhonopyCellInfoResult(
        unitcell=atoms, optional_structure_info=("POSCAR",),
        supercell_matrix=identity, primitive_matrix=None,
        interface_mode=None, phonopy_yaml=None,
    )
    captured = {}

    def capture_output(phonon, cell_info, base_filename, standardized):
        captured["phonon"] = standardized

    # check_symmetry still calls refine_cell, sorting, guess_primitive_matrix,
    # and its final Phonopy constructor itself.  Only output rendering changes.
    with patch.object(show_symmetry, "_show_symmetry_yaml", capture_output):
        show_symmetry.check_symmetry(initial, info)
    final = captured["phonon"]
    primitive, conventional = _to_structure(final.primitive), _to_structure(final.unitcell)
    expected_comp = structure.composition.reduced_composition
    if any(s.composition.reduced_composition != expected_comp for s in (primitive, conventional)):
        raise ValueError("Phonopy symmetry standardization changed reduced composition")
    details = {
        "original_symmetry_at_requested_tolerance": _dataset_summary(atoms, symprec),
        "initial_phonopy_primitive_n_atoms": len(initial.primitive),
        "standard_primitive_symmetry_at_requested_tolerance": _dataset_summary(final.primitive, symprec),
        "standard_conventional_symmetry_at_requested_tolerance": _dataset_summary(final.unitcell, symprec),
        "standard_primitive_symmetry_at_final_constructor_tolerance": _dataset_summary(final.primitive, final.primitive_symmetry.tolerance),
        "initial_constructor_symprec_A": float(initial.primitive_symmetry.tolerance),
        "final_constructor_symprec_A": float(final.primitive_symmetry.tolerance),
        "standard_primitive_matrix": np.asarray(final.primitive_matrix).tolist(),
        "primitive_poscar_text": "\n".join(get_vasp_structure_lines(final.primitive)),
        "conventional_poscar_text": "\n".join(get_vasp_structure_lines(final.unitcell)),
    }
    return primitive, conventional, details


@contextmanager
def _capture_native_output():
    """Capture Python and native spglib messages inside a worker process."""
    with tempfile.TemporaryFile(mode="w+b") as buffer:
        sys.stdout.flush()
        sys.stderr.flush()
        original_stdout, original_stderr = os.dup(1), os.dup(2)
        os.dup2(buffer.fileno(), 1)
        os.dup2(buffer.fileno(), 2)
        captured = {}
        try:
            yield captured
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(original_stdout, 1)
            os.dup2(original_stderr, 2)
            os.close(original_stdout)
            os.close(original_stderr)
            buffer.seek(0)
            captured["text"] = buffer.read().decode("utf-8", errors="replace")


def _read_cif(path: Path) -> Structure:
    parsed = CifParser(path, frac_tolerance=0, site_tolerance=1e-8).parse_structures(primitive=False)
    if len(parsed) != 1:
        raise ValueError(f"Expected exactly one structure, received {len(parsed)}")
    return parsed[0]


def _process_one(source: dict, output_path: str, symprec: float) -> dict:
    from phonopy.interface.vasp import read_vasp_from_strings

    output = Path(output_path)
    material_id = source["material_id"]
    path = ROOT / "data/candidates/materials_project/cif" / source["cif_file"]
    record = {
        "material_id": material_id, "cif_file": source["cif_file"],
        "source_metadata": dict(source), "status": "error", "error_stage": "read_source",
        "error_reason": None, "symprec_A": symprec,
    }
    before = time.monotonic()
    with _capture_native_output() as capture, warnings.catch_warnings(record=True) as caught, ExitStack() as contexts:
        warnings.simplefilter("always")
        try:
            record["source_cif_sha256"] = sha256(path)
            record["error_stage"] = "parse_cif"
            original = _read_cif(path)
            record["original"] = structure_payload(original)
            record["formula"] = original.composition.reduced_formula
            record["error_stage"] = "phonopy_input_reader"
            record["mass_metadata_patch"] = contexts.enter_context(
                patched_missing_atomic_masses([site.specie.symbol for site in original])
            )
            original_atoms = read_vasp_from_strings(Poscar(original).get_str())
            record["original_symmetry_at_0p01_A"] = _dataset_summary(original_atoms, 0.01)
            record["original_symmetry_at_requested_tolerance"] = _dataset_summary(original_atoms, symprec)
            record["error_stage"] = "phonopy_standardization"
            primitive, conventional, details = standardize_structure(original, symprec)
            record["primitive"] = structure_payload(primitive)
            record["conventional"] = structure_payload(conventional)
            record["error_stage"] = "export_standard_geometry"
            for kind, structure in (("primitive", primitive), ("conventional", conventional)):
                filename = f"{Path(source['cif_file']).stem}_{kind}.cif"
                destination = output / kind / filename
                CifWriter(structure, symprec=None, significant_figures=16).write_file(destination)
                record[f"{kind}_cif_path"] = str(destination.relative_to(output))
                record[f"{kind}_cif_sha256"] = sha256(destination)
                poscar = output / "poscar" / material_id / ("PPOSCAR" if kind == "primitive" else "BPOSCAR")
                poscar.parent.mkdir(parents=True, exist_ok=True)
                poscar.write_text(details.pop(f"{kind}_poscar_text"), encoding="utf-8")
                record[f"{kind}_poscar_path"] = str(poscar.relative_to(output))
                record[f"{kind}_poscar_sha256"] = sha256(poscar)
            record.update(details)
            record["original_to_primitive_atom_multiplicity"] = len(original) / len(primitive)
            record["original_to_conventional_atom_multiplicity"] = len(original) / len(conventional)
            record["original_to_primitive_cell_volume_ratio"] = original.volume / primitive.volume
            record["original_volume_per_atom_A3"] = original.volume / len(original)
            record["primitive_volume_per_atom_A3"] = primitive.volume / len(primitive)
            record["volume_per_atom_relative_change"] = record["primitive_volume_per_atom_A3"] / record["original_volume_per_atom_A3"] - 1.0
            record["standardization_max_site_displacement_A"] = None
            record["standardization_displacement_note"] = "Not evaluated: requires explicit original-to-symmetrized cell correspondence; complete original and standardized geometries retained."
            if sha256(path) != record["source_cif_sha256"]:
                raise ValueError("Source CIF hash changed during processing")
            record.update(status="ok", error_stage=None, error_reason=None, source_hash_unchanged=True)
        except Exception as exc:
            record["error_reason"] = f"{type(exc).__name__}: {exc}"
    record["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))
    if capture.get("text"):
        logpath = output / "native_logs" / f"{material_id}.log"
        logpath.write_text(capture["text"], encoding="utf-8")
        record["native_log_path"] = str(logpath.relative_to(output))
    record["elapsed_seconds"] = time.monotonic() - before
    return record


def _inventory_row(record: dict) -> dict:
    source = record["source_metadata"]
    row = {**source, **{key: record.get(key) for key in [
        "status", "error_stage", "error_reason", "formula", "source_cif_sha256",
        "original_to_primitive_atom_multiplicity", "original_to_conventional_atom_multiplicity",
        "original_to_primitive_cell_volume_ratio", "original_volume_per_atom_A3",
        "primitive_volume_per_atom_A3", "volume_per_atom_relative_change",
        "primitive_cif_path", "conventional_cif_path", "primitive_poscar_path", "conventional_poscar_path",
        "primitive_cif_sha256", "conventional_cif_sha256", "native_log_path", "elapsed_seconds",
    ]}}
    row["warnings"] = " | ".join(record.get("warnings", []))
    for kind in ("original", "primitive", "conventional"):
        payload = record.get(kind, {})
        row[f"{kind}_n_atoms"] = payload.get("n_atoms")
        row[f"{kind}_volume_A3"] = payload.get("volume_A3")
        for name, value in zip(("a_A", "b_A", "c_A", "alpha_deg", "beta_deg", "gamma_deg"), payload.get("lattice_parameters_A_deg", [])):
            row[f"{kind}_{name}"] = value
    for label, key in (("original_0p01", "original_symmetry_at_0p01_A"),
                       ("original_0p5", "original_symmetry_at_requested_tolerance"),
                       ("primitive_0p5", "standard_primitive_symmetry_at_requested_tolerance"),
                       ("conventional_0p5", "standard_conventional_symmetry_at_requested_tolerance")):
        row[f"{label}_sg_number"] = record.get(key, {}).get("number")
        row[f"{label}_sg_symbol"] = record.get(key, {}).get("international")
    return row


def _write_checkpoint(output: Path, records: list[dict], provenance: dict, complete: bool) -> None:
    ordered = sorted(records, key=lambda item: item["cif_file"])
    payload = {"complete": complete, "n_records": len(ordered), "provenance": provenance, "records": ordered}
    temporary = output / "checkpoint.json.gz.tmp"
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
    os.replace(temporary, output / "checkpoint.json.gz")
    if ordered:
        rows = [_inventory_row(record) for record in ordered]
        keys = list(dict.fromkeys(key for row in rows for key in row))
        temporary_csv = output / "inventory.csv.tmp"
        with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_csv, output / "inventory.csv")


def _cli_validation(output: Path, sources: list[dict], symprec: float) -> dict:
    executable = Path(sys.executable).parent / "phonopy"
    if not executable.is_file():
        raise FileNotFoundError(f"Expected real Phonopy CLI: {executable}")
    by_id = {row["material_id"]: row for row in sources}
    checks = []
    requested = ["mp-29179", "mp-569152", "mp-25470", "mp-22865", "mp-588", "mp-23210", "mp-667324"]
    for material_id in requested:
        source = by_id[material_id]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original = _read_cif(ROOT / "data/candidates/materials_project/cif" / source["cif_file"])
            primitive, conventional, details = standardize_structure(original, symprec)
        folder = output / "cli_validation" / material_id
        folder.mkdir(parents=True, exist_ok=True)
        Poscar(original).write_file(folder / "POSCAR")
        command = [str(executable), "--symmetry", f"--tolerance={symprec:g}", "-c", "POSCAR"]
        process = subprocess.run(command, cwd=folder, text=True, capture_output=True, check=False, timeout=180)
        (folder / "stdout.log").write_text(process.stdout, encoding="utf-8")
        (folder / "stderr.log").write_text(process.stderr, encoding="utf-8")
        check = {"material_id": material_id, "command": command, "returncode": process.returncode}
        if process.returncode != 0:
            raise RuntimeError(f"CLI validation failed for {material_id}: {process.stderr}")
        for kind, filename in (("primitive", "PPOSCAR"), ("conventional", "BPOSCAR")):
            api_text = details[f"{kind}_poscar_text"]
            (folder / f"API_{filename}").write_text(api_text, encoding="utf-8")
            cli_bytes = (folder / filename).read_bytes()
            check[f"{kind}_cli_api_byte_identical"] = cli_bytes == api_text.encode("utf-8")
            check[f"{kind}_cli_sha256"] = hashlib.sha256(cli_bytes).hexdigest()
            if not check[f"{kind}_cli_api_byte_identical"]:
                raise RuntimeError(f"CLI/API {filename} differ for {material_id}")
            if material_id in {"mp-29179", "mp-569152"}:
                previous = ROOT / "results/sncl2_phonopy_comparison_0p5" / material_id / filename
                check[f"{kind}_previous_cli_byte_identical"] = previous.read_bytes() == api_text.encode("utf-8")
                if not check[f"{kind}_previous_cli_byte_identical"]:
                    raise RuntimeError(f"API differs from previous SnCl2 {filename}: {material_id}")
        checks.append(check)
        print(f"CLI/API validation: {material_id}, both POSCAR files byte-identical", flush=True)
    report = {"all_passed": True, "n_structures": len(checks), "checks": checks,
              "comparison": "Exact bytes, including atom order, lattice, fractional coordinates and Phonopy formatting"}
    (output / "cli_validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--symprec", type=float, default=0.5)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    import phonopy
    import phonopy.cui.show_symmetry as show_symmetry

    if phonopy.__version__ != EXPECTED_PHONOPY_VERSION:
        raise RuntimeError(f"Use Phonopy {EXPECTED_PHONOPY_VERSION}")
    output = args.outdir.resolve()
    if (output / "checkpoint.json.gz").exists() and not args.resume:
        raise FileExistsError("Checkpoint already exists; use --resume to preserve and continue it")
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("primitive", "conventional", "poscar", "native_logs", "cli_validation"):
        (output / directory).mkdir(exist_ok=True)
    metadata = ROOT / "data/candidates/materials_project/metadata.csv"
    with metadata.open(encoding="utf-8-sig") as handle:
        sources = list(csv.DictReader(handle))
    if len(sources) != 1702 or len({row["material_id"] for row in sources}) != 1702:
        raise ValueError("Expected all 1702 unique local MP records")
    if {r["cif_file"] for r in sources} != {p.name for p in (metadata.parent / "cif").glob("*.cif")}:
        raise ValueError("Metadata CIF inventory does not match the 1702 source files")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "symprec_A": args.symprec, "n_input_records": len(sources),
        "metadata_path": str(metadata.relative_to(ROOT)), "metadata_sha256": sha256(metadata),
        "runner_sha256": sha256(Path(__file__)),
        "versions": {name: importlib.metadata.version(name) for name in ["phonopy", "spglib", "numpy", "scipy", "pymatgen"]},
        "phonopy_check_symmetry_source_path": show_symmetry.__file__,
        "phonopy_check_symmetry_source_sha256": sha256(Path(show_symmetry.__file__)),
        "algorithm": [
            "Parse full original CIF without fractional snapping (frac_tolerance=0, site_tolerance=1e-8).",
            "Round-trip pymatgen POSCAR text through the real Phonopy VASP reader in memory.",
            "Phonopy(input, I, primitive_matrix=None, symprec=0.5, calculator=None), same as CLI.",
            "Call original phonopy.cui.show_symmetry.check_symmetry; intercept presentation callback only.",
            "That function refines phonon.primitive with spglib.refine_cell(symprec=0.5), sorts species, guesses primitive matrix at0.5, then constructs a second Phonopy object using its default symprec=1e-5, exactly as the installed CLI source.",
            "Save final ph.unitcell and ph.primitive as unchanged Phonopy BPOSCAR/PPOSCAR, exact arrays, and explicit-site CIFs.",
        ],
        "interpretation": "Symmetry standardization at 0.5 A may idealize geometry; standardized equivalence is not proof that original CIFs differ only in cell choice. No force constants or phonons are computed.",
        "failure_policy": "Keep all metadata and available original geometry; failures are unresolved, never silently removed or asserted independent structures.",
        "mass_metadata_compatibility_policy": "For Pu only when Phonopy default mass is undefined, temporarily use pymatgen Element('Pu').atomic_mass in memory, restore afterwards, and record per structure. No source/package files changed; geometry-only, not a phonon/dynamics mass assignment.",
    }
    validation = _cli_validation(output, sources, args.symprec)
    if args.validate_only:
        return
    records = []
    if args.resume and (output / "checkpoint.json.gz").exists():
        with gzip.open(output / "checkpoint.json.gz", "rt", encoding="utf-8") as handle:
            previous = json.load(handle)
        old = previous["provenance"]
        for key in ["symprec_A", "metadata_sha256", "runner_sha256", "versions", "phonopy_check_symmetry_source_sha256"]:
            if old[key] != provenance[key]:
                raise ValueError(f"Resume provenance differs: {key}")
        records = previous["records"]
        provenance = old
    completed_ids = {record["material_id"] for record in records}
    remaining = [row for row in sources if row["material_id"] not in completed_ids]
    _write_checkpoint(output, records, provenance, complete=False)
    with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(_process_one, source, str(output), args.symprec): source for source in remaining}
        for future in as_completed(futures):
            source = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {"material_id": source["material_id"], "cif_file": source["cif_file"],
                    "source_metadata": dict(source), "status": "error", "error_stage": "worker_process",
                    "error_reason": f"{type(exc).__name__}: {exc}",
                    "source_cif_sha256": sha256(metadata.parent / "cif" / source["cif_file"])}
            records.append(record)
            if len(records) % 100 == 0:
                _write_checkpoint(output, records, provenance, complete=False)
                print(f"Standardized {len(records)}/1702: {dict(Counter(r['status'] for r in records))}", flush=True)
    if {r["material_id"] for r in records} != {r["material_id"] for r in sources}:
        raise ValueError("Final checkpoint does not preserve all 1702 IDs")
    if sha256(metadata) != provenance["metadata_sha256"]:
        raise ValueError("Metadata changed during standardization")
    unchanged = all(sha256(metadata.parent / "cif" / r["cif_file"]) == r["source_cif_sha256"] for r in records)
    if not unchanged:
        raise ValueError("At least one source CIF hash changed")
    _write_checkpoint(output, records, provenance, complete=True)
    successful = [r for r in records if r["status"] == "ok"]
    summary = {
        "complete": True, "n_records": len(records), "status_counts": dict(Counter(r["status"] for r in records)),
        "error_stage_counts": dict(Counter(r.get("error_stage") for r in records if r["status"] != "ok")),
        "all_source_hashes_unchanged": unchanged, "cli_api_validation_passed": validation["all_passed"],
        "primitive_cif_count": len(list((output / "primitive").glob("*.cif"))),
        "conventional_cif_count": len(list((output / "conventional").glob("*.cif"))),
        "source_to_primitive_atom_count_reduced": sum(r["original"]["n_atoms"] > r["primitive"]["n_atoms"] for r in successful),
        "sg_changed_from_original_0p01_to_0p5": sum(r["original_symmetry_at_0p01_A"].get("number") != r["original_symmetry_at_requested_tolerance"].get("number") for r in successful),
        "maximum_absolute_volume_per_atom_relative_change": max(abs(r["volume_per_atom_relative_change"]) for r in successful),
        "mass_metadata_patch_count": sum(bool(r.get("mass_metadata_patch")) for r in successful),
        "provenance": provenance,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# MP 全1702结构：Phonopy tolerance=0.5 Å 标准化\n\n"
        f"成功 {len(successful)}，未解决 {len(records)-len(successful)}；所有1702个ID和原始元数据均保留。\n\n"
        "实际执行Phonopy 2.46.0的check_symmetry内部流程，并对7个材料的新CLI输出、两个SnCl2既有CLI输出逐字节核对PPOSCAR/BPOSCAR。第二个Phonopy构造沿用该版本CLI源码默认symprec=1e-5；未擅自改成0.5。\n\n"
        "checkpoint.json.gz含原始/标准原胞/标准常规胞完整几何及元数据；inventory.csv为扁平审查表。primitive/和conventional/存CIF，poscar/存Phonopy实际格式PPOSCAR/BPOSCAR，cli_validation/及其报告保存CLI核对证据。\n\n"
        "0.5 Å可能理想化原始结构；这是标准化输出，尚未执行去重，也不表示原始CIF仅晶胞选择不同。失败行仍需处理，不自动当作独立结构。原始到标准结构最大位移未额外匹配计算，保留了全部几何以供后续复核。\n\n"
        "加载：`from src.analysis.standardize_mp_phonopy import load_standardized_structure`；对checkpoint中的record调用`load_standardized_structure(record, kind='primitive')`。该加载函数无需Phonopy环境。\n",
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in ["complete", "n_records", "status_counts", "error_stage_counts", "all_source_hashes_unchanged"]}), flush=True)


if __name__ == "__main__":
    main()
