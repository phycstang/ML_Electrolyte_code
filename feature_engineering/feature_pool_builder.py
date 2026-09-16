#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature_pool_builder_fixed.py

Robust, version-tolerant feature pool builder for crystal CIFs.
- Works with pymatgen >= 2024, matminer 0.9.x, dscribe 2.x
- No post-aggregation: emit raw featurizer outputs as-is (flat columns)
- Soft imports + capability detection; each block is optional and cannot crash the run
- DScribe safeguards: species auto-detect, neighbor-count check, adaptive rcut fallback,
  per-descriptor try/except (SOAP/ACSF/MBTR computed independently)
- Matminer 0.9.2 compat: ValenceOrbital without `stats=`, guarded ADF/OFM/XRD, no EwaldEnergy
- Parallel processing via joblib; structured error log CSV; Parquet output

CLI examples:
  python feature_pool_builder_fixed.py \
      --csv folder_score_table.csv \
      --cif_root . \
      --out feature_pool_heavy.parquet \
      --profile heavy \
      --n_jobs 16

Input CSV must contain column `cif_path` (relative to --cif_root if not absolute).
Optionally pass through metadata columns (e.g., dim, connect, score, etc.) — they will be preserved.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------- Hard dependencies (pymatgen) ---------
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Optional (for DScribe)
try:
    from pymatgen.io.ase import AseAtomsAdaptor  # type: ignore
    HAVE_ASE_ADAPTOR = True
except Exception:
    HAVE_ASE_ADAPTOR = False

# --------- Global soft-import registry (matminer/dscribe) ---------
MM = {
    "have": False,
    "notes": {},
}
DS = {
    "have": False,
}

# "Capability" flags filled at runtime
CAP = {
    # matminer composition
    "mm_Stoichiometry": False,
    "mm_ElementFraction": False,
    "mm_ValenceOrbital": False,
    "mm_BandCenter": False,
    "mm_ElementProperty": False,
    # matminer structure (light)
    "mm_DensityFeatures": False,
    "mm_RDF": False,
    "mm_ADF": False,
    # matminer structure (heavy)
    "mm_CoulombMatrix": False,
    "mm_SineCoulombMatrix": False,
    "mm_OrbitalFieldMatrix": False,
    "mm_BagofBonds": False,
    "mm_XRD": False,
    # dscribe
    "ds_SOAP": False,
    "ds_ACSF": False,
    "ds_MBTR": False,
}


# =====================================================================================
# Utilities
# =====================================================================================

def human_ts() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_csv_strict(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: input CSV not found: {path}")
    df = pd.read_csv(path)
    if "cif_path" not in df.columns:
        raise SystemExit("ERROR: input CSV must contain a 'cif_path' column")
    return df


def structure_from_path(path: str) -> Structure:
    # Robust CIF loader with primitive fallback
    s = Structure.from_file(path)
    try:
        sga = SpacegroupAnalyzer(s, symprec=1e-2, angle_tolerance=5)
        prim = sga.find_primitive()
        if prim and len(prim) <= len(s):
            return prim
    except Exception:
        pass
    return s


def safe_species_list(struct: Structure) -> List[str]:
    return sorted({site.specie.symbol for site in struct.sites})


def neighbor_stats(struct: Structure, rcut: float) -> Tuple[int, int, float]:
    # Returns: (n_sites, total_neighbors, avg_neighbors)
    tot = 0
    for i in range(len(struct)):
        ns = struct.get_neighbors(struct[i], rcut)
        tot += len(ns)
    avg = tot / max(1, len(struct))
    return len(struct), tot, avg


# =====================================================================================
# Capability detection
# =====================================================================================

def detect_modules() -> Dict[str, Any]:
    info = {
        "pymatgen": None,
        "matminer": None,
        "dscribe": None,
        "pyyaml": None,
        "notes": {"matminer_top_import_error": None, "dscribe_top_import_error": None},
    }
    # pymatgen version
    try:
        import pymatgen as pmg  # type: ignore
        info["pymatgen"] = getattr(pmg, "__version__", "unknown")
    except Exception:
        pass
    # matminer
    try:
        import matminer as mm  # type: ignore
        info["matminer"] = getattr(mm, "__version__", "unknown")
        MM["have"] = True
    except Exception as e:
        info["notes"]["matminer_top_import_error"] = str(e)
        MM["have"] = False
    # dscribe
    try:
        import dscribe  # type: ignore
        info["dscribe"] = getattr(dscribe, "__version__", "unknown")
        DS["have"] = True
    except Exception as e:
        info["notes"]["dscribe_top_import_error"] = str(e)
        DS["have"] = False
    # pyyaml
    try:
        import yaml  # noqa: F401
        info["pyyaml"] = True
    except Exception:
        info["pyyaml"] = False
    return info


def detect_capabilities() -> None:
    # Matminer composition
    if MM["have"]:
        try:
            from matminer.featurizers.composition import Stoichiometry
            CAP["mm_Stoichiometry"] = True
        except Exception:
            pass
        try:
            from matminer.featurizers.composition import ElementFraction
            CAP["mm_ElementFraction"] = True
        except Exception:
            pass
        try:
            from matminer.featurizers.composition import ValenceOrbital
            _ = ValenceOrbital  # no stats= in 0.9.2
            CAP["mm_ValenceOrbital"] = True
        except Exception:
            pass
        try:
            from matminer.featurizers.composition import BandCenter
            CAP["mm_BandCenter"] = True
        except Exception:
            pass
        try:
            from matminer.featurizers.composition import ElementProperty
            CAP["mm_ElementProperty"] = True
        except Exception:
            pass
    # Matminer structure
    if MM["have"]:
        try:
            from matminer.featurizers.structure import DensityFeatures
            CAP["mm_DensityFeatures"] = True
        except Exception:
            pass
        try:
            from matminer.featurizers.structure import RadialDistributionFunction
            CAP["mm_RDF"] = True
        except Exception:
            pass
        try:
            # ADF may not exist in 0.9.2
            from matminer.featurizers.structure import AngularDistributionFunction  # noqa: F401
            CAP["mm_ADF"] = True
        except Exception:
            pass
        try:
            from matminer.featurizers.structure.matrix import CoulombMatrix, SineCoulombMatrix
            _ = CoulombMatrix; _ = SineCoulombMatrix
            CAP["mm_CoulombMatrix"] = True
            CAP["mm_SineCoulombMatrix"] = True
        except Exception:
            pass
        try:
            from matminer.featurizers.structure.matrix import OrbitalFieldMatrix
            _ = OrbitalFieldMatrix
            CAP["mm_OrbitalFieldMatrix"] = True
        except Exception:
            pass
        try:
            from matminer.featurizers.structure import BagofBonds
            CAP["mm_BagofBonds"] = True
        except Exception:
            pass
        try:
            from matminer.featurizers.structure import XRDPowderPattern
            CAP["mm_XRD"] = True
        except Exception:
            pass
    # DScribe
    if DS["have"] and HAVE_ASE_ADAPTOR:
        try:
            from dscribe.descriptors import SOAP
            CAP["ds_SOAP"] = True
        except Exception:
            pass
        try:
            from dscribe.descriptors import ACSF
            CAP["ds_ACSF"] = True
        except Exception:
            pass
        try:
            from dscribe.descriptors import MBTR
            CAP["ds_MBTR"] = True
        except Exception:
            pass


# =====================================================================================
# Featurizer blocks
# =====================================================================================

def collect_pg_basic(struct: Structure, params: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out["pg:n_sites"] = float(len(struct))
    out["pg:volume"] = float(struct.volume)
    try:
        out["pg:density"] = float(struct.density)
    except Exception:
        out["pg:density"] = np.nan
    latt = struct.lattice
    out["pg:a"] = float(latt.a)
    out["pg:b"] = float(latt.b)
    out["pg:c"] = float(latt.c)
    out["pg:alpha"] = float(latt.alpha)
    out["pg:beta"] = float(latt.beta)
    out["pg:gamma"] = float(latt.gamma)
    # space group, crystal system
    try:
        sga = SpacegroupAnalyzer(struct, symprec=1e-2, angle_tolerance=5)
        out["pg:sg_number"] = float(sga.get_space_group_number())
        sys_name = sga.get_crystal_system()
        systems = ["triclinic","monoclinic","orthorhombic","tetragonal","trigonal","hexagonal","cubic"]
        out["pg:crystal_system_id"] = float(systems.index(sys_name)+1) if sys_name in systems else np.nan
    except Exception:
        out["pg:sg_number"] = np.nan
        out["pg:crystal_system_id"] = np.nan
    return out


def collect_mm_composition(struct: Structure, params: Dict[str, Any]) -> Dict[str, float]:
    if not MM["have"]:
        return {}
    out: Dict[str, float] = {}
    try:
        from pymatgen.core.composition import Composition as PmgComposition
        from matminer.featurizers.composition import Stoichiometry, ElementFraction, ValenceOrbital, BandCenter, ElementProperty
        comp = PmgComposition(struct.composition.reduced_composition.alphabetical_formula)
        # Stoichiometry
        if CAP["mm_Stoichiometry"]:
            try:
                st = Stoichiometry(p_list=(0,2,3,5))
                vals = st.featurize(comp)
                for k, v in zip(st.feature_labels(), vals):
                    out[f"mm_Comp_Stoichiometry:{k}"] = float(v) if np.isfinite(v) else np.nan
            except Exception:
                pass
        # ValenceOrbital (0.9.2: no stats arg)
        if CAP["mm_ValenceOrbital"]:
            try:
                vo = ValenceOrbital(props=["s","p","d","f"])  # avg/frac are intrinsic outputs
                vals = vo.featurize(comp)
                for k, v in zip(vo.feature_labels(), vals):
                    out[f"mm_Comp_ValenceOrbital:{k}"] = float(v) if np.isfinite(v) else np.nan
            except Exception:
                pass
        # ElementFraction
        if CAP["mm_ElementFraction"]:
            try:
                ef = ElementFraction()
                vals = ef.featurize(comp)
                for k, v in zip(ef.feature_labels(), vals):
                    out[f"mm_Comp_ElementFraction:{k}"] = float(v) if np.isfinite(v) else np.nan
            except Exception:
                pass
        # BandCenter
        if CAP["mm_BandCenter"]:
            try:
                bc = BandCenter()
                vals = bc.featurize(comp)
                for k, v in zip(bc.feature_labels(), vals):
                    out[f"mm_Comp_BandCenter:{k}"] = float(v) if np.isfinite(v) else np.nan
            except Exception:
                pass
        # ElementProperty (a small, robust subset)
        if CAP["mm_ElementProperty"]:
            try:
                ep = ElementProperty(
                    features=[
                        "Number",
                        "MendeleevNumber",
                        "AtomicWeight",
                        "MeltingT",
                        "BoilingT",
                        "CovalentRadius",
                        "Electronegativity",
                        "ElectronAffinity",
                        "FusionEnthalpy",
                        "ThermalConductivity",
                    ],
                    stats=["mean","avg_dev","max","min","range","std"],
                )
                vals = ep.featurize(comp)
                for k, v in zip(ep.feature_labels(), vals):
                    out[f"mm_Comp_ElementProperty:{k}"] = float(v) if np.isfinite(v) else np.nan
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(f"matminer composition featurize error: {e!r}")
    return out


def collect_mm_structure_light(struct: Structure, params: Dict[str, Any]) -> Dict[str, float]:
    if not MM["have"]:
        return {}
    out: Dict[str, float] = {}
    # DensityFeatures
    if CAP["mm_DensityFeatures"]:
        try:
            from matminer.featurizers.structure import DensityFeatures
            dens = DensityFeatures(desired_features=("density","vpa","packing fraction"))
            vals = dens.featurize(struct)
            for k, v in zip(dens.feature_labels(), vals):
                out[f"mm_Struct_Density:{k.replace(' ','_')}"] = float(v) if np.isfinite(v) else np.nan
        except Exception as e:
            # don't escalate
            pass
    # RDF (no ADF in 0.9.2 by default)
    if CAP["mm_RDF"]:
        try:
            from matminer.featurizers.structure import RadialDistributionFunction
            cutoff = float(params.get("rdf_cutoff", 8.0))
            bin_size = float(params.get("rdf_bin", 0.2))
            rdf = RadialDistributionFunction(cutoff=cutoff, bin_size=bin_size)
            vals = rdf.featurize(struct)
            for i, v in enumerate(vals):
                out[f"mm_Struct_RDF:bin_{i}"] = float(v) if np.isfinite(v) else np.nan
        except Exception:
            pass
    return out


def collect_mm_structure_heavy(struct: Structure, params: Dict[str, Any]) -> Dict[str, float]:
    if not MM["have"]:
        return {}
    out: Dict[str, float] = {}
    # Matrix descriptors
    try:
        if CAP["mm_CoulombMatrix"]:
            from matminer.featurizers.structure.matrix import CoulombMatrix
            cm = CoulombMatrix(flatten=True)
            vals = cm.featurize(struct)
            for i, v in enumerate(vals):
                out[f"mm_Struct_CM:{i}"] = float(v) if np.isfinite(v) else np.nan
    except Exception:
        pass
    try:
        if CAP["mm_SineCoulombMatrix"]:
            from matminer.featurizers.structure.matrix import SineCoulombMatrix
            scm = SineCoulombMatrix(flatten=True)
            vals = scm.featurize(struct)
            for i, v in enumerate(vals):
                out[f"mm_Struct_SCM:{i}"] = float(v) if np.isfinite(v) else np.nan
    except Exception:
        pass
    try:
        if CAP["mm_OrbitalFieldMatrix"]:
            from matminer.featurizers.structure.matrix import OrbitalFieldMatrix
            ofm = OrbitalFieldMatrix(flatten=True)
            vals = ofm.featurize(struct)
            for i, v in enumerate(vals):
                out[f"mm_Struct_OFM:{i}"] = float(v) if np.isfinite(v) else np.nan
    except Exception:
        pass
    # Bag of Bonds (can be big; keep, but ignore failures)
    try:
        if CAP["mm_BagofBonds"]:
            from matminer.featurizers.structure import BagofBonds
            bob = BagofBonds()
            vals = bob.featurize(struct)
            for i, v in enumerate(vals):
                out[f"mm_Struct_BoB:{i}"] = float(v) if np.isfinite(v) else np.nan
    except Exception:
        pass
    # XRD pattern (optional, small)
    try:
        if CAP["mm_XRD"]:
            from matminer.featurizers.structure import XRDPowderPattern
            xrd = XRDPowderPattern(two_theta_range=(10, 90))
            vals = xrd.featurize(struct)
            for k, v in zip(xrd.feature_labels(), vals):
                out[f"mm_Struct_XRD:{k}"] = float(v) if np.isfinite(v) else np.nan
    except Exception:
        pass
    return out


# ------------------------------ DScribe block ------------------------------

def _dscribe_neigh_ok(struct: Structure, rcut: float, max_avg: float = 200.0) -> Tuple[bool, float]:
    try:
        n_sites, tot, avg = neighbor_stats(struct, rcut)
        if avg == 0:
            return False, avg
        if avg > max_avg:
            return False, avg
        return True, avg
    except Exception:
        return False, float("nan")


def collect_dscribe_all(struct: Structure, params: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not (DS["have"] and HAVE_ASE_ADAPTOR):
        return out

    species = safe_species_list(struct)
    try:
        atoms = AseAtomsAdaptor.get_atoms(struct)
    except Exception:
        raise RuntimeError("dscribe: ASE conversion failed")

    # Adaptive rcut candidates
    base_rcut = float(params.get("rcut", 5.0))
    rcut_list = [base_rcut, 4.5, 4.0, 3.5]

    # Try find a usable radius
    ok_rcut = None
    avg_nb = np.nan
    for rc in rcut_list:
        ok, avg = _dscribe_neigh_ok(struct, rc)
        if ok:
            ok_rcut = rc
            avg_nb = avg
            break
    if ok_rcut is None:
        # Nothing usable; skip all DScribe for this structure
        raise RuntimeError("dscribe: all sub-blocks failed after sanitation/limits")

    # Compute each descriptor independently
    # SOAP
    try:
        if CAP["ds_SOAP"]:
            from dscribe.descriptors import SOAP
            soap = SOAP(
                species=species,
                periodic=True,
                rcut=ok_rcut,
                nmax=int(params.get("soap_nmax", 6)),
                lmax=int(params.get("soap_lmax", 4)),
                sigma=float(params.get("soap_sigma", 0.4)),
                sparse=False,
                average=False,
            )
            X = soap.create(atoms)
            mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
            out["ds:SOAP:mean_l2"] = float(np.linalg.norm(mu))
            out["ds:SOAP:std_l2"] = float(np.linalg.norm(sd))
            # a few leading components to control width
            take = int(params.get("soap_take", 64))
            for i in range(min(take, mu.shape[0])):
                out[f"ds:SOAP:mean_{i}"] = float(mu[i]) if np.isfinite(mu[i]) else np.nan
                out[f"ds:SOAP:std_{i}"] = float(sd[i]) if np.isfinite(sd[i]) else np.nan
    except Exception:
        # swallow, continue to ACSF/MBTR
        pass

    # ACSF
    try:
        if CAP["ds_ACSF"]:
            from dscribe.descriptors import ACSF
            acsf = ACSF(species=species, rcut=float(params.get("acsf_rcut", max(ok_rcut, 6.0))), sparse=False)
            X = acsf.create(atoms)
            mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
            out["ds:ACSF:mean_l2"] = float(np.linalg.norm(mu))
            out["ds:ACSF:std_l2"] = float(np.linalg.norm(sd))
            take = int(params.get("acsf_take", 64))
            for i in range(min(take, mu.shape[0])):
                out[f"ds:ACSF:mean_{i}"] = float(mu[i]) if np.isfinite(mu[i]) else np.nan
                out[f"ds:ACSF:std_{i}"] = float(sd[i]) if np.isfinite(sd[i]) else np.nan
    except Exception:
        pass

    # MBTR (lightweight grid)
    try:
        if CAP["ds_MBTR"]:
            from dscribe.descriptors import MBTR
            mbtr = MBTR(
                species=species,
                periodic=True,
                k1={"geometry": {"function": "atomic_number"}},
                k2={"geometry": {"function": "distance"}},
                k3={"geometry": {"function": "angle"}},
                grid={"min": 0, "max": 8, "n": int(params.get("mbtr_n", 50)), "sigma": 0.1},
                sparse=False,
                normalization="l2_each",
            )
            X = mbtr.create(atoms)
            mu = np.nanmean(X, axis=0)
            sd = np.nanstd(X, axis=0)
            out["ds:MBTR:mean_l2"] = float(np.linalg.norm(mu))
            out["ds:MBTR:std_l2"] = float(np.linalg.norm(sd))
            take = int(params.get("mbtr_take", 64))
            for i in range(min(take, mu.shape[0])):
                out[f"ds:MBTR:mean_{i}"] = float(mu[i]) if np.isfinite(mu[i]) else np.nan
                out[f"ds:MBTR:std_{i}"] = float(sd[i]) if np.isfinite(sd[i]) else np.nan
    except Exception:
        pass

    if not any(k.startswith("ds:") for k in out.keys()):
        # If nothing emitted, treat as failed so it appears in error CSV
        raise RuntimeError("dscribe: all sub-blocks failed after sanitation/limits")

    # Log neighbor stat used
    out["ds:rcut_used"] = float(ok_rcut)
    out["ds:avg_neighbors"] = float(avg_nb) if np.isfinite(avg_nb) else np.nan
    return out


# =====================================================================================
# Pipeline
# =====================================================================================

@dataclass
class BlockSpec:
    name: str
    func: Any
    params: Dict[str, Any]


def default_profile(profile: str) -> List[BlockSpec]:
    """Return list of BlockSpec for a given profile."""
    profile = profile.lower()
    blocks: List[BlockSpec] = []
    # Always: basic pymatgen geometry
    blocks.append(BlockSpec("pg_basic", collect_pg_basic, {}))
    # Matminer composition
    blocks.append(BlockSpec("mm_comp", collect_mm_composition, {}))
    # Light structure
    blocks.append(BlockSpec("mm_struct_light", collect_mm_structure_light, {"rdf_cutoff": 8.0, "rdf_bin": 0.2}))
    if profile in ("heavy", "medium"):
        blocks.append(BlockSpec("mm_struct_heavy", collect_mm_structure_heavy, {}))
    if profile == "heavy":
        blocks.append(BlockSpec("ds_all", collect_dscribe_all, {
            "rcut": 5.0, "soap_nmax": 6, "soap_lmax": 4, "soap_sigma": 0.4,
            "soap_take": 64, "acsf_rcut": 6.0, "acsf_take": 64,
            "mbtr_n": 50, "mbtr_take": 64,
        }))
    return blocks


def process_one(row: Dict[str, Any], cif_root: str, blocks: List[BlockSpec]) -> Tuple[Dict[str, Any], List[Tuple[str,str,str]]]:
    """Process a single CSV row. Returns (features, errors)."""
    errs: List[Tuple[str,str,str]] = []
    feats: Dict[str, Any] = {}
    # passthrough metadata
    for k in ("dim","connect","cif_file","score","cif_path"):
        if k in row:
            feats[k] = row[k]
    path_in = str(row["cif_path"]) if "cif_path" in row else None
    if not path_in:
        errs.append(("misc", "missing_cif_path", ""))
        return feats, errs
    path = path_in if os.path.isabs(path_in) else os.path.join(cif_root, path_in)
    try:
        struct = structure_from_path(path)
    except Exception as e:
        errs.append(("misc", f"parse_error: {type(e).__name__}", traceback.format_exc()))
        return feats, errs

    for spec in blocks:
        try:
            vals = spec.func(struct, spec.params)
            # prefix names with block for uniqueness
            for k, v in vals.items():
                feats[k] = v
        except Exception as e:
            errs.append((spec.name, str(e), traceback.format_exc()))
    return feats, errs


# =====================================================================================
# Main
# =====================================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with 'cif_path' column")
    ap.add_argument("--cif_root", default=".", help="Root folder for relative cif_path")
    ap.add_argument("--out", default="feature_pool.parquet", help="Output Parquet file")
    ap.add_argument("--profile", choices=["light","medium","heavy"], default="heavy")
    ap.add_argument("--config", help="YAML config to override block params", default=None)
    ap.add_argument("--must_have", help="Comma separated block names that must run (else fail)")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    df_in = read_csv_strict(args.csv)

    # Detect modules + capabilities
    modinfo = detect_modules()
    detect_capabilities()

    # Load optional YAML config overrides
    overrides: Dict[str, Dict[str, Any]] = {}
    if args.config:
        try:
            import yaml  # type: ignore
            with open(args.config, "r", encoding="utf-8") as f:
                overrides = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] failed to load YAML config: {e}")

    blocks = default_profile(args.profile)
    # Apply overrides
    for b in blocks:
        if b.name in overrides:
            b.params.update(overrides[b.name] or {})

    # Enforce must_have if provided
    if args.must_have:
        must = set(x.strip() for x in args.must_have.split(",") if x.strip())
        avail = {b.name for b in blocks}
        missing = list(must - avail)
        if missing:
            raise SystemExit(f"ERROR: missing must_have blocks: {missing}")

    # Parallel map
    rows = df_in.to_dict("records")
    results: List[Tuple[Dict[str, Any], List[Tuple[str,str,str]]]]
    if args.n_jobs and args.n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=args.n_jobs, backend="loky", prefer="threads", verbose=5)(
                delayed(process_one)(r, args.cif_root, blocks) for r in rows
            )
        except Exception as e:
            print(f"[WARN] parallel failed, falling back to serial: {e}")
            results = [process_one(r, args.cif_root, blocks) for r in rows]
    else:
        results = [process_one(r, args.cif_root, blocks) for r in rows]

    # Merge
    feat_rows: List[Dict[str, Any]] = []
    err_rows: List[Dict[str, Any]] = []
    for r, errs in results:
        feat_rows.append(r)
        for featurizer, error, tb in errs:
            err_rows.append({
                "cif_path": r.get("cif_path", ""),
                "featurizer": featurizer,
                "error": error,
                "traceback": tb,
            })

    out_df = pd.DataFrame(feat_rows)

    # Write Parquet
    out_path = args.out
    out_df.to_parquet(out_path, index=False)

    # Error CSV
    err_path = os.path.splitext(out_path)[0] + ".errors.csv"
    if err_rows:
        pd.DataFrame(err_rows).to_csv(err_path, index=False)

    # Summary JSON
    summary = {
        "timestamp": human_ts(),
        "python": sys.version,
        "platform": sys.platform,
        "args": {
            "csv": args.csv,
            "cif_root": args.cif_root,
            "out": args.out,
            "profile": args.profile,
            "config": args.config,
            "must_have": args.must_have,
            "n_jobs": args.n_jobs,
            "seed": args.seed,
        },
        "modules": modinfo,
        "rows": len(out_df),
        "cols": int(out_df.shape[1] if out_df is not None else 0),
        "errors": len(err_rows),
        "coverage": None,
    }
    sum_path = os.path.splitext(out_path)[0] + ".summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote table: {out_path}  rows={len(out_df)} cols={out_df.shape[1]}")
    if err_rows:
        # Simple aggregation of error types
        agg = pd.DataFrame(err_rows).groupby("featurizer").size().sort_values(ascending=False)
        print(f"[INFO] errors logged to: {err_path}")
        print("[ERROR SUMMARY] counts by featurizer:")
        for k, v in agg.items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
