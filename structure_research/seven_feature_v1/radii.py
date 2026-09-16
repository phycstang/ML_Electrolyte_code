"""Strict Pauling and Shannon-radius features for the seven-feature workflow.

This module intentionally does *not* use ``Element.ionic_radii``, atomic radii,
covalent radii, or van-der-Waals radii.  Every radius comes from the frozen
pymatgen Shannon table and every scientific absence is returned as auditable
missing data rather than silently imputed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
import math
from numbers import Integral
from typing import Any, Literal

from pymatgen.core import Element


EXPECTED_PYMATGEN_VERSION = "2025.10.7"
SHANNON_TABLE_KEY = "Shannon radii"
SHANNON_RADIUS_FIELD = "ionic_radius"
SHANNON_SOURCE = (
    "pymatgen==2025.10.7 Element.data['Shannon radii']"
    "[oxidation_state][coordination][spin]['ionic_radius']"
)

PaulingMode = Literal["absolute", "x_minus_m", "m_minus_x"]
ShannonPolicy = Literal["exact", "same_charge_nearest_cn_le_1"]

PAULING_MODES = frozenset({"absolute", "x_minus_m", "m_minus_x"})
SHANNON_POLICIES = frozenset({"exact", "same_charge_nearest_cn_le_1"})

_ROMAN_DIGITS: tuple[tuple[int, str], ...] = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


def shannon_provenance(*, verify: bool = True) -> dict[str, Any]:
    """Return, and optionally enforce, the frozen Shannon-table provenance."""

    try:
        installed = version("pymatgen")
    except PackageNotFoundError:
        installed = None
    record = {
        "pymatgen_expected_version": EXPECTED_PYMATGEN_VERSION,
        "pymatgen_installed_version": installed,
        "table_key": SHANNON_TABLE_KEY,
        "radius_field": SHANNON_RADIUS_FIELD,
        "source": SHANNON_SOURCE,
        "cross_oxidation_state_fallback": False,
        "atomic_covalent_vdw_fallback": False,
    }
    if verify and installed != EXPECTED_PYMATGEN_VERSION:
        raise RuntimeError(
            "the seven-feature Shannon contract requires pymatgen "
            f"{EXPECTED_PYMATGEN_VERSION}, found {installed!r}"
        )
    return record


def shannon_source_provenance(*, verify: bool = True) -> dict[str, Any]:
    """Compatibility name used by the seven-feature extraction entry point."""

    return shannon_provenance(verify=verify)


def _fraction_from_value(value: Any) -> Fraction | None:
    """Convert a finite scalar to a Fraction without tolerance or rounding."""

    if isinstance(value, bool):
        return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value), 1)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric):
        return None
    try:
        # str is deliberate: it preserves the caller's decimal value and avoids
        # treating a nearby binary float as an integer by tolerance.
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        return None


def _exact_integer(value: Any) -> int | None:
    fraction = _fraction_from_value(value)
    if fraction is None or fraction.denominator != 1:
        return None
    return int(fraction.numerator)


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def infer_formal_oxidation_state(n_m: Any, n_x: Any) -> dict[str, Any]:
    """Infer integer M oxidation state from exact M:X stoichiometry and X(-1).

    No rounding or tolerance is used.  Mixed-valence/non-integral compositions
    are deliberately missing under the v1 strict contract.
    """

    m_count = _fraction_from_value(n_m)
    x_count = _fraction_from_value(n_x)
    base = {
        "available": False,
        "m_count_input": str(n_m),
        "x_count_input": str(n_x),
        "m_count_fraction": _fraction_text(m_count) if m_count is not None else None,
        "x_count_fraction": _fraction_text(x_count) if x_count is not None else None,
        "x_assumed_oxidation_state": -1,
        "formal_oxidation_state": None,
        "exact_ratio_fraction": None,
        "rounding_or_tolerance_used": False,
        "missing_reason": None,
    }
    if m_count is None or x_count is None:
        base["missing_reason"] = "invalid_nonfinite_stoichiometric_count"
        return base
    if m_count <= 0 or x_count <= 0:
        base["missing_reason"] = "stoichiometric_counts_must_be_positive"
        return base
    ratio = x_count / m_count
    base["exact_ratio_fraction"] = _fraction_text(ratio)
    if ratio.denominator != 1:
        base["missing_reason"] = "non_integer_formal_oxidation_state"
        return base
    oxidation_state = int(ratio.numerator)
    if oxidation_state <= 0:
        base["missing_reason"] = "m_formal_oxidation_state_must_be_positive"
        return base
    base.update(
        available=True,
        formal_oxidation_state=oxidation_state,
        missing_reason=None,
    )
    return base


def pauling_difference_audit(
    m_symbol: str,
    x_symbol: str,
    mode: PaulingMode = "absolute",
) -> dict[str, Any]:
    """Return an audited Pauling electronegativity difference.

    ``absolute`` is the registered feature.  The signed alternatives exist only
    to make the sign convention explicit in diagnostics.
    """

    if mode not in PAULING_MODES:
        raise ValueError(
            f"unsupported Pauling difference mode {mode!r}; "
            f"expected one of {sorted(PAULING_MODES)}"
        )
    audit: dict[str, Any] = {
        "available": False,
        "m_symbol": str(m_symbol),
        "x_symbol": str(x_symbol),
        "scale": "Pauling",
        "mode": mode,
        "chi_m": None,
        "chi_x": None,
        "value": None,
        "missing_reason": None,
    }
    try:
        m_element = Element(str(m_symbol))
        x_element = Element(str(x_symbol))
    except (ValueError, KeyError) as exc:
        audit["missing_reason"] = f"invalid_element_symbol:{type(exc).__name__}"
        return audit

    chi_m = m_element.X
    chi_x = x_element.X
    if chi_m is None or chi_x is None:
        audit["missing_reason"] = "pauling_electronegativity_not_tabulated"
        return audit
    try:
        chi_m_float = float(chi_m)
        chi_x_float = float(chi_x)
    except (TypeError, ValueError, OverflowError):
        audit["missing_reason"] = "invalid_pauling_electronegativity"
        return audit
    if not (math.isfinite(chi_m_float) and math.isfinite(chi_x_float)):
        audit["missing_reason"] = "invalid_pauling_electronegativity"
        return audit

    signed_x_minus_m = chi_x_float - chi_m_float
    if mode == "absolute":
        value = abs(signed_x_minus_m)
    elif mode == "x_minus_m":
        value = signed_x_minus_m
    else:
        value = -signed_x_minus_m
    audit.update(
        available=True,
        chi_m=chi_m_float,
        chi_x=chi_x_float,
        value=value,
        missing_reason=None,
    )
    return audit


def pauling_difference(
    m_symbol: str,
    x_symbol: str,
    mode: PaulingMode = "absolute",
) -> float | None:
    """Convenience numeric view of :func:`pauling_difference_audit`."""

    return pauling_difference_audit(m_symbol, x_symbol, mode)["value"]


def _integer_to_roman(value: int) -> str | None:
    if value <= 0 or value >= 4000:
        return None
    remainder = value
    output: list[str] = []
    for number, symbol in _ROMAN_DIGITS:
        count, remainder = divmod(remainder, number)
        output.extend([symbol] * count)
    return "".join(output)


def _roman_to_integer(label: str) -> int | None:
    """Parse only canonical ordinary Roman numerals (not IVSQ/IVPY/IIIPY)."""

    text = str(label).strip().upper()
    if not text or any(character not in "IVXLCDM" for character in text):
        return None
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    previous = 0
    for character in reversed(text):
        current = values[character]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total if _integer_to_roman(total) == text else None


def _base_radius_selection(
    element_symbol: str,
    oxidation_state: Any,
    coordination_number: Any,
    policy: ShannonPolicy,
) -> dict[str, Any]:
    oxidation_integer = _exact_integer(oxidation_state)
    coordination_integer = _exact_integer(coordination_number)
    return {
        "available": False,
        "element": str(element_symbol),
        "source": SHANNON_SOURCE,
        "policy": policy,
        "requested_oxidation_state": oxidation_integer,
        "selected_oxidation_state": None,
        "exact_charge_required": True,
        "requested_cn": coordination_integer,
        "requested_cn_label": (
            _integer_to_roman(coordination_integer)
            if coordination_integer is not None
            else None
        ),
        "selected_cn": None,
        "selected_cn_label": None,
        "cn_offset": None,
        "absolute_cn_distance": None,
        "nearest_cn_tie_candidates": [],
        "nearest_cn_tie_break_rule": "lower_cn",
        "ordinary_roman_cn_only": True,
        "available_ordinary_cn": [],
        "excluded_nonordinary_cn_labels": [],
        "ionic_radius_A": None,
        "high_spin_radius_A": None,
        "low_spin_radius_A": None,
        "unspecified_spin_radius_A": None,
        "radii_by_spin_A": {},
        "spin_labels": [],
        "spin_count": 0,
        "spin_ambiguous": False,
        "spin_radius_min_A": None,
        "spin_radius_max_A": None,
        "spin_radius_span_A": None,
        "used_cross_oxidation_state_fallback": False,
        "used_atomic_covalent_vdw_fallback": False,
        "missing_reason": None,
    }


def _exact_charge_key(table: Mapping[Any, Any], charge: int) -> tuple[Any | None, list[int]]:
    matches: list[Any] = []
    available: set[int] = set()
    for raw_key in table:
        integer = _exact_integer(raw_key)
        if integer is None:
            continue
        available.add(integer)
        if integer == charge:
            matches.append(raw_key)
    if len(matches) != 1:
        return None, sorted(available)
    return matches[0], sorted(available)


def _spin_radius_summary(spin_table: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(spin_table, Mapping) or not spin_table:
        return None, "missing_or_invalid_spin_table"

    radii: dict[str, float] = {}
    for raw_spin, raw_record in spin_table.items():
        spin = str(raw_spin)
        if not isinstance(raw_record, Mapping) or SHANNON_RADIUS_FIELD not in raw_record:
            return None, "missing_ionic_radius_in_spin_entry"
        try:
            radius = float(raw_record[SHANNON_RADIUS_FIELD])
        except (TypeError, ValueError, OverflowError):
            return None, "invalid_ionic_radius_in_spin_entry"
        if not math.isfinite(radius):
            return None, "invalid_ionic_radius_in_spin_entry"
        radii[spin] = radius

    values = list(radii.values())
    radius_min = min(values)
    radius_max = max(values)
    return (
        {
            "ionic_radius_A": math.fsum(values) / len(values),
            "high_spin_radius_A": radii.get("High Spin"),
            "low_spin_radius_A": radii.get("Low Spin"),
            "unspecified_spin_radius_A": radii.get(""),
            "radii_by_spin_A": dict(sorted(radii.items())),
            "spin_labels": sorted(radii),
            "spin_count": len(radii),
            "spin_ambiguous": len(radii) > 1,
            "spin_radius_min_A": radius_min,
            "spin_radius_max_A": radius_max,
            "spin_radius_span_A": radius_max - radius_min,
        },
        None,
    )


def select_shannon_radius(
    element_symbol: str,
    oxidation_state: Any,
    coordination_number: Any,
    *,
    policy: ShannonPolicy = "exact",
    verify_version: bool = True,
) -> dict[str, Any]:
    """Select one Shannon ionic radius under a fail-closed policy.

    ``exact`` requires the exact charge and exact ordinary Roman CN.
    ``same_charge_nearest_cn_le_1`` retains exact charge and may select an
    ordinary Roman CN at distance one.  If both adjacent CNs are equidistant,
    the lower CN is chosen deterministically and all tied candidates are saved.
    """

    if policy not in SHANNON_POLICIES:
        raise ValueError(
            f"unsupported Shannon policy {policy!r}; "
            f"expected one of {sorted(SHANNON_POLICIES)}"
        )
    shannon_provenance(verify=verify_version)
    audit = _base_radius_selection(
        element_symbol, oxidation_state, coordination_number, policy
    )
    charge = audit["requested_oxidation_state"]
    requested_cn = audit["requested_cn"]
    if charge is None:
        audit["missing_reason"] = "oxidation_state_must_be_an_exact_integer"
        return audit
    if requested_cn is None or requested_cn <= 0 or requested_cn >= 4000:
        audit["missing_reason"] = "coordination_number_must_be_a_positive_exact_integer"
        return audit

    try:
        element = Element(str(element_symbol))
    except (ValueError, KeyError) as exc:
        audit["missing_reason"] = f"invalid_element_symbol:{type(exc).__name__}"
        return audit
    raw_table = element.data.get(SHANNON_TABLE_KEY)
    if not isinstance(raw_table, Mapping) or not raw_table:
        audit["missing_reason"] = "shannon_table_not_tabulated"
        return audit

    charge_key, available_charges = _exact_charge_key(raw_table, charge)
    audit["available_oxidation_states"] = available_charges
    if charge_key is None:
        audit["missing_reason"] = "exact_oxidation_state_not_tabulated"
        return audit
    raw_coordination_table = raw_table[charge_key]
    if not isinstance(raw_coordination_table, Mapping):
        audit["missing_reason"] = "invalid_coordination_table"
        return audit

    ordinary_entries: dict[int, tuple[Any, str]] = {}
    excluded_labels: list[str] = []
    for raw_label in raw_coordination_table:
        label = str(raw_label)
        cn = _roman_to_integer(label)
        if cn is None:
            excluded_labels.append(label)
        elif cn in ordinary_entries:
            audit["missing_reason"] = "duplicate_ordinary_coordination_label"
            return audit
        else:
            ordinary_entries[cn] = (raw_label, label)
    audit["available_ordinary_cn"] = sorted(ordinary_entries)
    audit["excluded_nonordinary_cn_labels"] = sorted(excluded_labels)

    if policy == "exact":
        chosen_cn = requested_cn if requested_cn in ordinary_entries else None
        tie_candidates: list[int] = [] if chosen_cn is None else [chosen_cn]
    else:
        eligible = [
            cn for cn in ordinary_entries if abs(cn - requested_cn) <= 1
        ]
        if eligible:
            minimum_distance = min(abs(cn - requested_cn) for cn in eligible)
            tie_candidates = sorted(
                cn
                for cn in eligible
                if abs(cn - requested_cn) == minimum_distance
            )
            chosen_cn = tie_candidates[0]
        else:
            tie_candidates = []
            chosen_cn = None
    audit["nearest_cn_tie_candidates"] = tie_candidates
    if chosen_cn is None:
        audit["missing_reason"] = (
            "exact_ordinary_coordination_number_not_tabulated"
            if policy == "exact"
            else "no_same_charge_ordinary_coordination_number_within_one"
        )
        return audit

    raw_label, label = ordinary_entries[chosen_cn]
    spin_summary, spin_error = _spin_radius_summary(
        raw_coordination_table[raw_label]
    )
    if spin_summary is None:
        audit["missing_reason"] = spin_error
        return audit

    audit.update(spin_summary)
    audit.update(
        available=True,
        selected_oxidation_state=charge,
        selected_cn=chosen_cn,
        selected_cn_label=label,
        cn_offset=chosen_cn - requested_cn,
        absolute_cn_distance=abs(chosen_cn - requested_cn),
        missing_reason=None,
    )
    return audit


def _normalise_site_cns(
    site_cns: Mapping[Any, Any] | Sequence[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(site_cns, Mapping):
        raw_items = list(site_cns.items())
    elif isinstance(site_cns, Sequence) and not isinstance(
        site_cns, (str, bytes, bytearray)
    ):
        raw_items = list(enumerate(site_cns))
    else:
        return [], [{"site_index": None, "cn_input": repr(site_cns)}]

    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for raw_index, raw_cn in raw_items:
        index = _exact_integer(raw_index)
        cn = _exact_integer(raw_cn)
        if index is None or index < 0 or index in seen_indices or cn is None or cn < 0:
            invalid.append(
                {"site_index": str(raw_index), "cn_input": str(raw_cn)}
            )
            continue
        seen_indices.add(index)
        valid.append({"site_index": index, "cn": cn})
    valid.sort(key=lambda row: row["site_index"])
    return valid, invalid


def _missing_scenario(
    policy: ShannonPolicy,
    *,
    records: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
    reason: str,
    x_selection: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "available": False,
        "policy": policy,
        "all_cn_ge_3_sites_required": True,
        "aggregation_rule": "mean_M_site_radius_then_derive",
        "eligible_m_site_count": len(records),
        "excluded_m_sites_cn_lt_3": excluded,
        "invalid_m_site_inputs": invalid,
        "m_site_selections": [],
        "missing_m_site_indices": [row["site_index"] for row in records],
        "m_ionic_radius_mean_A": None,
        "x_selection": x_selection,
        "x_ionic_radius_A": None if x_selection is None else x_selection["ionic_radius_A"],
        "radius_ratio_m_over_x": None,
        "field_strength_z_over_r_m2_Ainv2": None,
        "missing_reason": reason,
    }


def _radius_scenario(
    m_symbol: str,
    oxidation_state: int,
    records: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    invalid: list[dict[str, Any]],
    x_selection: dict[str, Any],
    policy: ShannonPolicy,
) -> dict[str, Any]:
    if invalid:
        return _missing_scenario(
            policy,
            records=records,
            excluded=excluded,
            invalid=invalid,
            reason="invalid_m_site_coordination_input",
            x_selection=x_selection,
        )
    if not records:
        return _missing_scenario(
            policy,
            records=records,
            excluded=excluded,
            invalid=invalid,
            reason="no_m_sites_with_vesta_cn_ge_3",
            x_selection=x_selection,
        )

    selections: list[dict[str, Any]] = []
    missing_indices: list[int] = []
    for record in records:
        selection = select_shannon_radius(
            m_symbol,
            oxidation_state,
            record["cn"],
            policy=policy,
        )
        selection["site_index"] = record["site_index"]
        selections.append(selection)
        if not selection["available"]:
            missing_indices.append(record["site_index"])

    scenario = {
        "available": False,
        "policy": policy,
        "all_cn_ge_3_sites_required": True,
        "aggregation_rule": "mean_M_site_radius_then_derive",
        "eligible_m_site_count": len(records),
        "excluded_m_sites_cn_lt_3": excluded,
        "invalid_m_site_inputs": invalid,
        "m_site_selections": selections,
        "missing_m_site_indices": missing_indices,
        "m_ionic_radius_mean_A": None,
        "x_selection": x_selection,
        "x_ionic_radius_A": x_selection["ionic_radius_A"],
        "radius_ratio_m_over_x": None,
        "field_strength_z_over_r_m2_Ainv2": None,
        "missing_reason": None,
    }
    if missing_indices:
        scenario["missing_reason"] = "at_least_one_m_site_radius_missing"
        return scenario
    if not x_selection["available"]:
        scenario["missing_reason"] = "fixed_x_minus_cn_vi_radius_missing"
        return scenario

    m_radii = [float(selection["ionic_radius_A"]) for selection in selections]
    mean_m_radius = math.fsum(m_radii) / len(m_radii)
    x_radius = float(x_selection["ionic_radius_A"])
    if mean_m_radius == 0.0:
        scenario["missing_reason"] = "mean_m_ionic_radius_is_zero"
        return scenario
    if x_radius == 0.0:
        scenario["missing_reason"] = "x_ionic_radius_is_zero"
        return scenario
    scenario.update(
        available=True,
        m_ionic_radius_mean_A=mean_m_radius,
        radius_ratio_m_over_x=mean_m_radius / x_radius,
        field_strength_z_over_r_m2_Ainv2=(
            oxidation_state / (mean_m_radius * mean_m_radius)
        ),
        missing_reason=None,
    )
    return scenario


def compute_radius_features(
    m_symbol: str,
    x_symbol: str,
    formal_oxidation_state: Any,
    site_cns: Mapping[Any, Any] | Sequence[Any],
) -> dict[str, Any]:
    """Compute strict and nearest-CN sensitivity radius features.

    Only VESTA M sites with CN >= 3 enter the crystal mean.  A missing strict
    radius at any eligible M site makes both strict derived features missing.
    X is always selected exactly as X(-1), CN VI in both scenarios.
    """

    provenance = shannon_provenance(verify=True)
    oxidation_integer = _exact_integer(formal_oxidation_state)
    records_all, invalid = _normalise_site_cns(site_cns)
    records = [record for record in records_all if record["cn"] >= 3]
    excluded = [record for record in records_all if record["cn"] < 3]

    if oxidation_integer is None or oxidation_integer <= 0:
        strict = _missing_scenario(
            "exact",
            records=records,
            excluded=excluded,
            invalid=invalid,
            reason="m_formal_oxidation_state_must_be_a_positive_exact_integer",
            x_selection=None,
        )
        sensitivity = _missing_scenario(
            "same_charge_nearest_cn_le_1",
            records=records,
            excluded=excluded,
            invalid=invalid,
            reason="m_formal_oxidation_state_must_be_a_positive_exact_integer",
            x_selection=None,
        )
        return {
            "m_symbol": str(m_symbol),
            "x_symbol": str(x_symbol),
            "formal_oxidation_state": None,
            "provenance": provenance,
            "strict": strict,
            "sensitivity_nearest_cn": sensitivity,
        }

    x_selection = select_shannon_radius(x_symbol, -1, 6, policy="exact")
    strict = _radius_scenario(
        m_symbol,
        oxidation_integer,
        records,
        excluded,
        invalid,
        x_selection,
        "exact",
    )
    sensitivity = _radius_scenario(
        m_symbol,
        oxidation_integer,
        records,
        excluded,
        invalid,
        x_selection,
        "same_charge_nearest_cn_le_1",
    )
    return {
        "m_symbol": str(m_symbol),
        "x_symbol": str(x_symbol),
        "formal_oxidation_state": oxidation_integer,
        "provenance": provenance,
        "strict": strict,
        "sensitivity_nearest_cn": sensitivity,
    }


def compute_chemical_features(
    m_symbol: str,
    x_symbol: str,
    n_m: Any,
    n_x: Any,
    site_cns: Mapping[Any, Any] | Sequence[Any],
    delta_mode: PaulingMode = "absolute",
) -> dict[str, Any]:
    """Return the three chemistry features plus complete structured audit data."""

    formal = infer_formal_oxidation_state(n_m, n_x)
    pauling = pauling_difference_audit(m_symbol, x_symbol, delta_mode)
    radius = compute_radius_features(
        m_symbol,
        x_symbol,
        formal["formal_oxidation_state"],
        site_cns,
    )
    strict = radius["strict"]
    sensitivity = radius["sensitivity_nearest_cn"]
    strict_radius_available = bool(strict["available"])
    sensitivity_radius_available = bool(sensitivity["available"])
    delta_available = bool(pauling["available"])
    return {
        "m_symbol": str(m_symbol),
        "x_symbol": str(x_symbol),
        "formal_oxidation_state": formal["formal_oxidation_state"],
        "delta_chi": pauling["value"],
        "delta_chi_mode": delta_mode,
        "strict_radius_ratio": strict["radius_ratio_m_over_x"],
        "strict_field_strength": strict["field_strength_z_over_r_m2_Ainv2"],
        "sensitivity_radius_ratio": sensitivity["radius_ratio_m_over_x"],
        "sensitivity_field_strength": sensitivity[
            "field_strength_z_over_r_m2_Ainv2"
        ],
        "delta_chi_available": delta_available,
        "strict_radius_available": strict_radius_available,
        "sensitivity_radius_available": sensitivity_radius_available,
        "strict_available": delta_available and strict_radius_available,
        "sensitivity_available": delta_available and sensitivity_radius_available,
        "audit": {
            "formal_oxidation_state": formal,
            "pauling_difference": pauling,
            "shannon_radius_features": radius,
        },
    }


__all__ = [
    "EXPECTED_PYMATGEN_VERSION",
    "PAULING_MODES",
    "SHANNON_POLICIES",
    "SHANNON_RADIUS_FIELD",
    "SHANNON_SOURCE",
    "SHANNON_TABLE_KEY",
    "compute_chemical_features",
    "compute_radius_features",
    "infer_formal_oxidation_state",
    "pauling_difference",
    "pauling_difference_audit",
    "select_shannon_radius",
    "shannon_provenance",
    "shannon_source_provenance",
]
