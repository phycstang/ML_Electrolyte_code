#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_features_revised.py

This script produces feature vectors for binary compounds comprised of a metal (M)
and a halogen (H).  It is a complete rewrite of the original `make_features.py`
to address several shortcomings:

* **Comprehensive element features** – It gathers *all* physically meaningful
  numeric properties exposed by the `mendeleev` library for each element via
  introspection.  Only numerical values (integers or floats) are retained and
  obviously internal or redundant attributes are excluded.  This ensures
  coverage of a wide range of atomic properties without having to hard‑code
  every field name.  In addition, several derived quantities are computed
  explicitly, including Mulliken electronegativity, absolute hardness, softness,
  Parr electrophilicity and an approximate field strength.  These derived
  features mirror the concept DFT metrics implemented in
  ``make_extra_features_max.py``.

* **Difference, weighted average and ratio** – For each numeric property
  gathered for the two constituent elements, the script automatically
  calculates the difference (M−H), weighted arithmetic mean (w_M·M + w_H·H)
  based on stoichiometric ratios, and the ratio (M/H).  These statistics
  constitute the T0 basic features.

* **Retention of structural (T1) and local environment (T2) features** – The
  original logic for parsing CIF files to extract lattice parameters,
  crystallographic density, coordination numbers and bond lengths is preserved.
  These optional features can be enabled or disabled via command‑line flags.

* **CSV I/O compatibility** – The script accepts an input CSV with metadata
  columns (e.g. formula, file paths) and optional explicit element columns.
  Newly generated features are appended as columns.  Missing values are
  flagged with a ``_missing`` suffix to facilitate downstream imputation.

The motivation for this redesign is to provide a robust and extensible
foundation for materials informatics where the scope of element descriptors
should be as broad as possible.  The introspective approach and derived
quantities are inspired by the ``make_extra_features_max.py`` reference,
specifically the calculation of concept DFT measures such as Mulliken
electronegativity, absolute hardness and electrophilicity, as well as
field strength (approximated here as Z/r_cov²).  These borrowed ideas are
explicitly noted below.

Usage example:

    python make_features_revised.py --csv compounds.csv --out features.csv \
        --m-col metal_symbol --h-col halogen_symbol --enable-t1 1 --enable-t2 0

"""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    from mendeleev import element as md_element
except Exception:
    md_element = None

try:
    from pymatgen.core.composition import Composition
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    PMG_AVAILABLE = True
except Exception:
    Composition = None
    Structure = None
    SpacegroupAnalyzer = None
    PMG_AVAILABLE = False

# -----------------------------------------------------------------------------
# Utility functions for element handling
# -----------------------------------------------------------------------------

def normalize_symbol(symbol: Any) -> Optional[str]:
    """Clean and standardise an element symbol.

    Removes trailing digits or whitespace, capitalises the first letter and
    lowercases the second.  Returns ``None`` if the input does not look like a
    valid symbol.
    """
    if symbol is None:
        return None
    s = str(symbol).strip()
    if not s:
        return None
    # strip trailing digits (common in some datasets, e.g. "Cl1")
    while len(s) > 1 and s[-1].isdigit():
        s = s[:-1]
    # remove non‑alpha characters
    s = ''.join(ch for ch in s if ch.isalpha())
    if not s:
        return None
    s = s[0].upper() + (s[1:].lower() if len(s) > 1 else '')
    if len(s) > 2:
        s = s[:2]
    return s


def get_elem(sym: Optional[str]):
    """Return a mendeleev Element instance given a symbol or None if invalid."""
    if md_element is None or sym is None:
        return None
    try:
        return md_element(sym)
    except Exception:
        return None


def is_number(x: Any) -> bool:
    """Check whether the input is a finite float or integer."""
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def to_float(x: Any) -> Optional[float]:
    """Attempt to convert a value to float; return None on failure."""
    try:
        f = float(x)
        return f if np.isfinite(f) else None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Gathering numeric properties from mendeleev
# -----------------------------------------------------------------------------

def get_numeric_properties(e) -> Dict[str, Optional[float]]:
    """Return a dictionary of all numeric attributes of an element.

    We iterate over attributes of the mendeleev Element object, filter out
    private names and callable attributes, and retain only those that can be
    converted to floats.  Certain internal counters and non‑physical fields are
    ignored via an explicit blacklist.  Additionally, we compute several
    derived features inspired by ``make_extra_features_max.py``:

      * **Mulliken electronegativity** `chiM` = (IE1 + EA)/2 if both values
        exist.
      * **Absolute hardness** `eta_parr` = (IE1 – EA)/2 if both exist.
      * **Softness** `soft_parr` = 1/eta if eta > 0.
      * **Parr electrophilicity** `omega_parr` = chiM²/(2·eta) if defined.
      * **Field strength** `field_strength` = Z/r_cov² approximating nuclear
        field at the covalent radius (covalent radius chosen from available
        attributes).  This corresponds to the electrostatic field strength used
        in ``make_extra_features_max.py`` for charge separation descriptors.

    Returns a dict mapping property names to float values or ``None``.
    """
    out: Dict[str, Optional[float]] = {}
    if e is None:
        return out

    # Blacklist of attribute names to ignore (non‑physical or repetitive)
    exclude = set([
        'name', 'symbol', 'atomic_number', 'atomic_weight', 'number',
        'period', 'group_id', 'group', 'block', 'series',
        'electron_configuration', 'electronic_configuration', 'ec', 'econf',
        'cas_number', 'isotopes', 'stable_isotopes', 'ionenergies',
        'bands', 'history', 'references', 'sources', 'rank',
    ])

    for attr in dir(e):
        if attr.startswith('_'):
            continue
        if attr in exclude:
            continue
        # skip callables
        try:
            v = getattr(e, attr)
        except Exception:
            continue
        if callable(v):
            continue
        # attempt to convert to float
        f = to_float(v)
        if f is not None:
            out[attr] = f

    # Derived features: Ionization energies and electron affinity
    IE1 = None
    # Ionization energies may be stored in ionenergies dict or as a property
    try:
        if hasattr(e, 'ionenergies'):
            IE1 = to_float(e.ionenergies.get(1))
    except Exception:
        IE1 = None
    if IE1 is None:
        IE1 = to_float(getattr(e, 'ionization_energy', None))
    EA = to_float(getattr(e, 'electron_affinity', None))
    # Covalent radius for field strength (choose best available)
    r_cov = None
    for rn in ['covalent_radius_pyykko', 'covalent_radius_cordero', 'covalent_radius']:
        rc = to_float(getattr(e, rn, None))
        if rc is not None and rc > 0:
            r_cov = rc
            break
    # Mulliken electronegativity
    if IE1 is not None and EA is not None:
        chiM = (IE1 + EA) / 2.0
        out['chiM'] = chiM
        # Absolute hardness
        eta = (IE1 - EA) / 2.0
        if eta is not None:
            out['eta_parr'] = eta
            if eta > 0:
                out['soft_parr'] = 1.0 / eta
                out['omega_parr'] = chiM * chiM / (2.0 * eta)
    # Field strength: approximate Z / r_cov^2
    Z = to_float(getattr(e, 'atomic_number', None))
    if Z is not None and r_cov is not None and r_cov > 0:
        out['field_strength'] = Z / (r_cov * r_cov)
    else:
        out['field_strength'] = None

    return out


# -----------------------------------------------------------------------------
# Difference/weighted average/ratio calculations
# -----------------------------------------------------------------------------

def combine_numeric_features(
    prefix: str,
    feats_M: Dict[str, Optional[float]],
    feats_H: Dict[str, Optional[float]],
    weight_M: float,
    weight_H: float,
    with_abs: bool = True,
) -> Dict[str, Any]:
    """Compute difference, weighted average and ratio for numeric features.

    Parameters
    ----------
    prefix : str
        Feature name prefix to apply (e.g. ``'T0_elem__'``).
    feats_M, feats_H : dict
        Numeric property dictionaries for the metal and halogen element.
    weight_M, weight_H : float
        Stoichiometric weights for weighted average (should sum to 1).
    with_abs : bool
        Whether to compute the absolute difference column.

    Returns
    -------
    dict
        Combined feature values keyed by ``<prefix><name>_M/H``,
        ``<prefix>d_<name>``, ``<prefix>abs_d_<name>``, ``<prefix>wa_<name>``,
        and ``<prefix>ratio_<name>`` as applicable.
    """
    out: Dict[str, Any] = {}
    keys = sorted(set(feats_M.keys()) | set(feats_H.keys()))
    # raw values
    for k in keys:
        out[f"{prefix}{k}_M"] = feats_M.get(k)
        out[f"{prefix}{k}_H"] = feats_H.get(k)
    # derived combinations
    for k in keys:
        vM = feats_M.get(k)
        vH = feats_H.get(k)
        diff = None
        ratio = None
        wa = None
        if is_number(vM) and is_number(vH):
            diff = float(vM) - float(vH)
            if float(vH) != 0:
                ratio = float(vM) / float(vH)
            wa = weight_M * float(vM) + weight_H * float(vH)
        out[f"{prefix}d_{k}"] = diff
        if with_abs:
            out[f"{prefix}abs_d_{k}"] = abs(diff) if diff is not None else None
        out[f"{prefix}wa_{k}"] = wa
        out[f"{prefix}ratio_{k}"] = ratio
    return out


# -----------------------------------------------------------------------------
# Composition features
# -----------------------------------------------------------------------------

def t0_comp_features(M: str, H: str, nM: float, nH: float, row: Dict[str, Any]) -> Dict[str, Any]:
    """Compute stoichiometry and mixed features independent of individual properties."""
    out = {}
    # Stoichiometric ratio
    if is_number(nM) and is_number(nH) and nM > 0:
        out['T0_comp__stoich_ratio_Hal_over_M'] = float(nH) / float(nM)
    else:
        out['T0_comp__stoich_ratio_Hal_over_M'] = None
    # Entropy of mixing (two species)
    tot = float(nM + nH) if is_number(nM) and is_number(nH) else None
    if tot and tot > 0:
        xM = float(nM) / tot
        xH = float(nH) / tot
        def s(x: float) -> float:
            return -x * math.log(x) if x > 0 else 0.0
        out['T0_comp__entropy'] = s(xM) + s(xH)
    else:
        out['T0_comp__entropy'] = None
    # Molar mass (via pymatgen Composition if available)
    out['T0_comp__molar_mass'] = None
    if PMG_AVAILABLE and M and H and is_number(nM) and is_number(nH):
        try:
            comp = Composition({M: nM, H: nH})
            out['T0_comp__molar_mass'] = float(comp.weight)
        except Exception:
            pass
    # Geometric mean of electronegativity across all scales (fallback)
    chi_vals_M = []
    chi_vals_H = []
    for key in row:
        if key.startswith('T0_elem__EN_') and key.endswith('_M'):
            vM = row[key]
            vH = row.get(key[:-2] + 'H')
            if is_number(vM) and is_number(vH):
                chi_vals_M.append(float(vM))
                chi_vals_H.append(float(vH))
    # Compute geomean if both lists nonempty
    if chi_vals_M and chi_vals_H:
        try:
            gm = math.sqrt(np.nanmean(chi_vals_M) * np.nanmean(chi_vals_H))
            out['T0_comp__chi_geomean'] = gm
        except Exception:
            out['T0_comp__chi_geomean'] = None
    else:
        out['T0_comp__chi_geomean'] = None
    return out


# -----------------------------------------------------------------------------
# Structural (T1) and local environment (T2) features
# -----------------------------------------------------------------------------

def build_t1_features(struct: Structure, M: str, H: str) -> Dict[str, Any]:
    """Extract global structural features from a Pymatgen Structure."""
    out = {}
    if not struct:
        return out
    try:
        lattice = struct.lattice
        out['T1_struct__volume_A3'] = float(lattice.volume)
        out['T1_struct__volume_per_atom_A3'] = float(lattice.volume) / len(struct)
        out['T1_struct__density_g_cm3'] = float(struct.density)
        out['T1_struct__a_A'] = float(lattice.a)
        out['T1_struct__b_A'] = float(lattice.b)
        out['T1_struct__c_A'] = float(lattice.c)
        out['T1_struct__alpha_deg'] = float(lattice.alpha)
        out['T1_struct__beta_deg'] = float(lattice.beta)
        out['T1_struct__gamma_deg'] = float(lattice.gamma)
        if SpacegroupAnalyzer is not None:
            try:
                sga = SpacegroupAnalyzer(struct, symprec=1e-2)
                out['T1_struct__spacegroup_number'] = float(sga.get_space_group_number())
                out['T1_struct__spacegroup_symbol'] = str(sga.get_space_group_symbol())
            except Exception:
                pass
    except Exception:
        pass
    return out


def build_t2_features(struct: Structure, M: str, H: str) -> Dict[str, Any]:
    """Compute local environment features: M–H coordination and bond length stats."""
    out = {}
    if not struct:
        return out
    try:
        # Determine covalent radii for cutoff
        eM = get_elem(M)
        eH = get_elem(H)
        rM = None
        rH = None
        if eM:
            for rn in ['covalent_radius_pyykko', 'covalent_radius_cordero', 'covalent_radius']:
                rc = to_float(getattr(eM, rn, None))
                if rc is not None:
                    rM = rc
                    break
        if eH:
            for rn in ['covalent_radius_pyykko', 'covalent_radius_cordero', 'covalent_radius']:
                rc = to_float(getattr(eH, rn, None))
                if rc is not None:
                    rH = rc
                    break
        # fallback if radii missing
        if rM is None:
            rM = 1.2
        if rH is None:
            rH = 0.8
        cutoff = 1.25 * (rM + rH)
        cn_list = []
        dist_list = []
        for i, site in enumerate(struct):
            if str(site.specie) != M:
                continue
            local_cn = 0
            for j, neigh in enumerate(struct):
                if i == j:
                    continue
                if str(neigh.specie) != H:
                    continue
                d = float(site.distance(neigh))
                if d <= cutoff:
                    local_cn += 1
                    dist_list.append(d)
            if local_cn > 0:
                cn_list.append(local_cn)
        def mean_std(arr: List[float]) -> Tuple[Optional[float], Optional[float]]:
            vals = [float(x) for x in arr if is_number(x)]
            if not vals:
                return (None, None)
            arr_np = np.array(vals)
            return (float(arr_np.mean()), float(arr_np.std(ddof=0)))
        cn_mean, cn_std = mean_std(cn_list)
        dist_mean, dist_std = mean_std(dist_list)
        out['T2_env__M_CN_Hal_mean'] = cn_mean
        out['T2_env__M_CN_Hal_std'] = cn_std
        out['T2_env__MHal_bond_length_A_mean'] = dist_mean
        out['T2_env__MHal_bond_length_A_std'] = dist_std
    except Exception:
        pass
    return out


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compositional, structural and elemental features for binary compounds.")
    parser.add_argument('--csv', required=True, help='Input CSV file path')
    parser.add_argument('--out', required=True, help='Output CSV file path')
    parser.add_argument('--m-col', default=None, help='Column name for metal element symbol')
    parser.add_argument('--h-col', default=None, help='Column name for halogen element symbol')
    parser.add_argument('--enable-t1', type=int, default=1, help='Enable structural features (default 1)')
    parser.add_argument('--enable-t2', type=int, default=1, help='Enable local environment features (default 1)')
    parser.add_argument('--with-abs-delta', type=int, default=1, help='Compute absolute differences for features (default 1)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    base_cols = [c for c in ('cif_path', 'formula', 'name', 'id', 'score', 'dim', 'st1', 'st2', 'st3') if c in df.columns]
    out_rows = []
    for _, row_orig in df.iterrows():
        row: Dict[str, Any] = {}
        for c in base_cols:
            row[c] = row_orig.get(c, None)
        formula = row_orig.get('formula', None)
        m_sym_raw = row_orig.get(args.m_col, None) if args.m_col else None
        h_sym_raw = row_orig.get(args.h_col, None) if args.h_col else None
        # Infer element symbols and stoichiometry
        M = normalize_symbol(m_sym_raw)
        H = normalize_symbol(h_sym_raw)
        nM = np.nan
        nH = np.nan
        if (M is None or H is None) and formula and PMG_AVAILABLE:
            try:
                comp = Composition(formula)
                d = comp.get_el_amt_dict()
                # Determine halogen by membership in common halogen set
                halogens = {'F', 'Cl', 'Br', 'I'}
                for el, amt in d.items():
                    if el in halogens:
                        H = el
                        nH = float(amt)
                    else:
                        M = el
                        nM = float(amt)
            except Exception:
                pass
        # Fallback if counts not set
        if not is_number(nM):
            nM = 1.0
        if not is_number(nH):
            nH = 1.0
        # Save element symbols
        row['T0_elem__symbol_M'] = M
        row['T0_elem__symbol_H'] = H
        # Retrieve numeric properties
        elem_M = get_elem(M)
        elem_H = get_elem(H)
        feats_M = get_numeric_properties(elem_M)
        feats_H = get_numeric_properties(elem_H)
        # Normalise weights
        tot = nM + nH
        wM = float(nM) / tot if is_number(tot) and tot > 0 else 0.5
        wH = 1.0 - wM
        combined = combine_numeric_features('T0_elem__', feats_M, feats_H, wM, wH, with_abs=bool(args.with_abs_delta))
        row.update(combined)
        # Composition features
        row.update(t0_comp_features(M, H, nM, nH, row))
        # Structural and environment features
        cif_path = row_orig.get('cif_path', None)
        if args.enable_t1 or args.enable_t2:
            if isinstance(cif_path, str) and cif_path and Structure is not None:
                try:
                    struct = Structure.from_file(cif_path)
                except Exception:
                    struct = None
            else:
                struct = None
        else:
            struct = None
        if args.enable_t1 and struct is not None:
            row.update(build_t1_features(struct, M, H))
        if args.enable_t2 and struct is not None:
            row.update(build_t2_features(struct, M, H))
        out_rows.append(row)
    out_df = pd.DataFrame(out_rows)
    # Add missing indicators for numeric columns
    for c in out_df.columns:
        if out_df[c].dtype != object:
            miss = out_df[c].isna()
            if miss.any():
                out_df[f"{c}_missing"] = miss.astype(float)
    # Write output
    out_df.to_csv(args.out, index=False)
    if args.verbose:
        # Print a short summary of missing rates for key derived properties
        sample_props = ['chiM', 'eta_parr', 'soft_parr', 'omega_parr', 'field_strength']
        print("Summary of derived property coverage:")
        for prop in sample_props:
            cM = f"T0_elem__{prop}_M"
            cH = f"T0_elem__{prop}_H"
            if cM in out_df.columns:
                non_null_M = out_df[cM].notna().sum()
                non_null_H = out_df[cH].notna().sum() if cH in out_df.columns else 0
                print(f"  {prop}: M non-null {non_null_M}/{len(out_df)}, H non-null {non_null_H}/{len(out_df)}")


if __name__ == '__main__':
    main()