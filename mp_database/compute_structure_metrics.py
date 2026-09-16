#!/usr/bin/env python3
"""Retired ambiguous structure-metrics entry point.

Two incompatible metric schemas coexist in this repository:

* ``compute_structure_metrics_legacy_crystalnn.py`` delegates to the frozen
  historical CrystalNN snapshot and emits unprefixed ``dim/st1/st2/st3``;
* ``compute_structure_metrics_vesta_v2.py`` uses the frozen VESTA-2019 M-X
  graph and emits only explicitly prefixed ``vesta__*`` fields.

This filename intentionally refuses to guess between them.  In particular,
silently using VESTA-v2 values as inputs to the historical ExtraTrees model
would be a descriptor-schema error.
"""

from __future__ import annotations


MESSAGE = """Ambiguous structure-metrics entry point is retired.

Use one explicit command instead:
  python -m src.screening.compute_structure_metrics_legacy_crystalnn ...
  python -m src.screening.compute_structure_metrics_vesta_v2 ...

The two outputs have different neighbor definitions and are not interchangeable.
"""


def main() -> None:
    raise SystemExit(MESSAGE)


if __name__ == "__main__":
    main()
