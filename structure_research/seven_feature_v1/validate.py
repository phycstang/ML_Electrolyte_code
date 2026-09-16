#!/usr/bin/env python3
"""Validate isolation, hashes, feature count, and post-freeze-only evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .common import MODEL_FEATURES, load_formula_boundary, load_json, sha256_file
from .pipeline import (
    FROZEN_MODEL_NAME,
    _source_hashes,
    _verify_frozen_manifest,
)


def _contains_any_formula(path: Path, formulas: set[str]) -> list[str]:
    content = path.read_bytes()
    return sorted(formula for formula in formulas if formula.encode("utf-8") in content)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _verify_frozen_manifest(
        args.frozen_manifest, args.config, args.isolation_file
    )
    targets = load_formula_boundary(args.isolation_file)
    dev_features = pd.read_csv(
        args.development_features / "seven_features.csv", low_memory=False
    )
    dev_ranked = pd.read_csv(
        args.development_results / "ranked_structures.csv", low_memory=False
    )
    evaluation_features = pd.read_csv(
        args.evaluation_features / "seven_features.csv", low_memory=False
    )
    evaluation_report = pd.read_csv(
        args.evaluation_results / "postfreeze_evaluation_report.csv", low_memory=False
    )
    evaluation_summary = load_json(
        args.evaluation_results / "postfreeze_evaluation.summary.json"
    )
    model = joblib.load(args.development_results / FROZEN_MODEL_NAME)

    development_text_files = sorted(
        path
        for path in args.development_results.iterdir()
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}
    )
    leakage = {
        path.name: _contains_any_formula(path, targets)
        for path in development_text_files
        if _contains_any_formula(path, targets)
    }
    checks = {
        "exactly_seven_model_features": tuple(manifest["model_features"])
        == MODEL_FEATURES,
        "no_pca_object_in_frozen_state": "pca" not in repr(model["transform"]).lower(),
        "development_excludes_all_evaluation_formulas": not bool(
            set(dev_features["formula"]) & targets
        ),
        "development_rankings_exclude_all_evaluation_formulas": not bool(
            set(dev_ranked["formula"]) & targets
        ),
        "development_text_outputs_contain_no_evaluation_formula_strings": not leakage,
        "evaluation_features_contain_only_evaluation_formulas": set(
            evaluation_features["formula"]
        ).issubset(targets),
        "evaluation_report_contains_exact_evaluation_formulas": set(
            evaluation_report["formula"]
        )
        == targets,
        "evaluation_was_transform_only": bool(
            evaluation_summary.get("no_refit_during_evaluation")
        ),
        "frozen_hdbscan_has_30_runs": int(manifest["n_hdbscan_runs"]) == 30,
        "frozen_source_hashes_match": manifest["source_hashes"] == _source_hashes(),
        "evaluation_evidence_is_not_relabelled_as_new_blind_discovery": (
            evaluation_summary.get("evidence_status")
            == "posthoc_mechanism_calibration_not_new_blind_discovery"
        ),
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "all_checks_passed": passed,
        "checks": checks,
        "development_text_leakage": leakage,
        "frozen_manifest_sha256": sha256_file(args.frozen_manifest),
        "evaluation_report_sha256": sha256_file(
            args.evaluation_results / "postfreeze_evaluation_report.csv"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        failures = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"seven-feature validation failed: {failures}")
    print(f"[OK] seven-feature validation passed {len(checks)}/{len(checks)} checks")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--development-results", type=Path, required=True)
    parser.add_argument("--evaluation-features", type=Path, required=True)
    parser.add_argument("--evaluation-results", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--isolation-file", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    validate(build_parser().parse_args())


if __name__ == "__main__":
    main()
