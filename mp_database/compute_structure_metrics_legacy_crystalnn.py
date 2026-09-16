#!/usr/bin/env python3
"""Explicit entry point for the archived CrystalNN structure-metrics code.

The delegated source is a byte-identical snapshot from the historical project.
It emits the legacy unprefixed ``dimension/st1/st2/st3`` schema used to explain
the historical ExtraTrees inputs.  Its source hash is verified before execution.

This wrapper does *not* claim that the full historical software environment,
command line, intermediate column-renaming step, or every original training
value can be reconstructed.  It exists to prevent the newer VESTA-v2 metric
definition from being substituted accidentally for the legacy definition.
"""

from __future__ import annotations

import hashlib
import runpy
import sys
from pathlib import Path


ARCHIVED_SOURCE_SHA256 = (
    "5cdf51a21a624881ca50cddaab681c81be9fc0e7c9bc22ae0dd6394192e7d6db"
)
ARCHIVED_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "paper"
    / "figure1_source"
    / "compute_structure_metrics.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archived_source() -> Path:
    if not ARCHIVED_SOURCE.is_file():
        raise FileNotFoundError(f"archived CrystalNN source is missing: {ARCHIVED_SOURCE}")
    actual = sha256_file(ARCHIVED_SOURCE)
    if actual != ARCHIVED_SOURCE_SHA256:
        raise RuntimeError(
            "archived CrystalNN source hash mismatch: "
            f"expected {ARCHIVED_SOURCE_SHA256}, got {actual}"
        )
    return ARCHIVED_SOURCE


def main() -> None:
    source = verify_archived_source()
    print(
        "[LEGACY] Running the hash-verified CrystalNN snapshot. Its outputs are "
        "not VESTA-v2 descriptors and the complete historical environment is not "
        "reconstructed.",
        file=sys.stderr,
    )
    runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    main()
