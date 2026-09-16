#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更稳健的结构维度分类（Larsen 扫描 + 策略集成 + 共识图 + 各向异性扩胞 + 强化并行）

设计要点：
- 主判据：纯距离阈值的 Larsen 扫描（在一段 cutoff 网格上反复构图→评估维度的“稳定区间”）
- 复核：CrystalNN / EconNN / VoronoiNN / MinimumDistanceNN / CutOffDictNN / AdaptiveRadiusNN
- 平票/冲突/低置信时：构造 Majority / Union 共识图再判
- 扩胞：各向异性目标跨度（避免长轴/短轴导致的跨边界误连或断连）
- 并行：进程池 + chunksize；BLAS/OMP 线程钳制；大文件优先；单文件超时可选
- 详尽日志：ConfidenceScore / DecisionPath / 各阶段结果（JSON）

依赖：pymatgen>=2023，pandas
"""

from __future__ import annotations
import os, sys, argparse, warnings, json, shutil, math, signal
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict

import pandas as pd

from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import (
    CrystalNN, EconNN, VoronoiNN, MinimumDistanceNN, CutOffDictNN
)
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.dimensionality import get_dimensionality_larsen

# ───────────────────── 默认参数 ─────────────────────
DEFAULT_INPUT   = "/data/home/tmy/PMH/Material_project/cif_files"
DEFAULT_OUTPUT  = "triangular_lattice_dimension"
DEFAULT_SUPER   = (3, 3, 3)
DEFAULT_MAXW    = os.cpu_count() or 8
DEFAULT_SUFFIX  = (".cif", ".vasp")
EXPAND_ON_AMBIGUOUS = True

# ───────────────────── 进程级线程钳制（BLAS/OMP） ─────────────────────
PIN_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS")
def _pin_threads(n: int = 1):
    for k in PIN_ENV:
        os.environ.setdefault(k, str(n))

# ───────────────────── 数据结构 ─────────────────────
@dataclass
class DimRec:
    Material       : str
    Formula        : str
    NAtoms         : int
    Spacegroup     : str
    HalogenGroup   : str
    StructClass    : str
    Dimensionality : str
    NumericDim     : int | None
    Consistency    : str          # "强一致/多数一致/平票择优/仅单一结果"
    MethodUsed     : str
    MethodsAll     : str          # JSON
    SymprecUsed    : float
    AngleTolUsed   : float
    SupercellUsed  : str
    NewPath        : str
    Error          : str
    ConfidenceScore: float
    DecisionPath   : str

# ───────────────────── 自适应半径策略（温和保守） ─────────────────────
def _element_radius(e: Element) -> float:
    r = (e.atomic_radius or 0) * 1.0
    if not r or r <= 0:
        r = (e.van_der_waals_radius or 1.6)
    return float(r)

def _species_pair_cutoff(a: Element, b: Element, scale: float = 1.25, floor: float = 2.1, ceil: float = 4.2) -> float:
    r = _element_radius(a) + _element_radius(b)
    r *= scale
    return float(max(floor, min(ceil, r)))

class AdaptiveRadiusNN:
    """极简自适应截断：按元素对给出上限，模仿 MinimumDistanceNN 的思路。"""
    def __init__(self, scale: float = 1.25, floor: float = 2.1, ceil: float = 4.2):
        self.scale, self.floor, self.ceil = scale, floor, ceil

    def get_graph(self, structure: Structure) -> StructureGraph:
        g = StructureGraph.with_empty_graph(structure)
        sp = [site.specie for site in structure.sites]
        species = sorted(set(str(s) for s in sp))
        elems   = {s: Element(s) for s in species}
        max_cut = 0.0
        for ea in elems.values():
            for eb in elems.values():
                max_cut = max(max_cut, _species_pair_cutoff(ea, eb, self.scale, self.floor, self.ceil))
        n = len(structure)
        for i in range(n):
            ei = Element(sp[i].symbol)
            for j in range(i + 1, n):
                ej = Element(sp[j].symbol)
                cutoff_ij = _species_pair_cutoff(ei, ej, self.scale, self.floor, self.ceil)
                d = structure.get_distance(i, j)
                if d <= min(max_cut, cutoff_ij + 0.15):
                    g.add_edge(i, j)
        return g

# ───────────────────── 对称化回退 ─────────────────────
def safe_spacegroup(structure: Structure, symprec_list=(1e-2, 5e-3, 1e-3), angle_list=(5.0, 0.5)):
    last_err = None
    for sym in symprec_list:
        for ang in angle_list:
            try:
                sga = SpacegroupAnalyzer(structure, symprec=sym, angle_tolerance=ang)
                for which, fn in (("primitive", sga.get_primitive_standard_structure),
                                  ("conventional", sga.get_conventional_standard_structure)):
                    try:
                        std = fn()
                        sg  = sga.get_space_group_symbol() or "Unknown"
                        return std, sg, sym, ang, which
                    except Exception as e:
                        last_err = e
            except Exception as e:
                last_err = e
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.01, angle_tolerance=5.0)
        sg = sga.get_space_group_symbol() or "Unknown"
    except Exception:
        sg = "Unknown"
    return structure, sg, 0.0, 0.0, "as-is"

# ───────────────────── 纯距离阈值构图（主判据依赖） ─────────────────────
def graph_from_cutoff(structure: Structure, r: float) -> StructureGraph:
    """用欧氏距离 r 作为唯一标准构图；PBC 最近像由 pymatgen 处理。"""
    g = StructureGraph.with_empty_graph(structure)
    n = len(structure)
    seen = set()
    for i in range(n):
        for nb in structure.get_neighbors(i, r):
            j = nb.index
            if i == j and nb.jimage == (0, 0, 0):
                continue
            u, v = (i, j) if i < j else (j, i)
            if (u, v) in seen:
                continue
            seen.add((u, v))
            g.add_edge(u, v)
    return g

# ───────────────────── Larsen 距离扫描（主判据） ─────────────────────

def _first_shell_scale(structure: Structure, r_probe: float = 5.0) -> float:
    """估计第一近邻的典型距离用于扫描基准。"""
    import numpy as np
    d1 = []
    for i in range(len(structure)):
        neigh = structure.get_neighbors(i, r_probe)
        if not neigh:
            continue
        d1.append(min(nb.nn_distance for nb in neigh))
    if not d1:
        return 2.5
    m = float(np.median(d1))
    return max(1.6, min(4.5, m))


def larsen_dim_sweep(structure: Structure,
                     r_min_scale: float = 0.85,
                     r_max_scale: float = 1.70,
                     n_steps: int = 48) -> tuple[int, float, list[float], list[int]]:
    """
    在一段 cutoff 网格上反复构图→用 get_dimensionality_larsen 判维度。
    返回：最佳维度、该维度占比（置信度）、cutoff 网格、每个阈值的维度序列。
    """
    import numpy as np
    base = _first_shell_scale(structure)
    r_min = base * r_min_scale
    r_max = base * r_max_scale
    if r_max <= r_min:
        r_max = r_min * 1.2
    grid = np.linspace(r_min, r_max, max(8, n_steps)).tolist()

    from collections import Counter
    dim_seq = []
    for r in grid:
        try:
            g = graph_from_cutoff(structure, float(r))
            d = int(get_dimensionality_larsen(g))
        except Exception:
            d = -1
        dim_seq.append(d)

    cnt = Counter([d for d in dim_seq if d in (0, 1, 2, 3)])
    if not cnt:
        return -1, 0.0, grid, dim_seq
    best_dim, best_hits = max(cnt.items(), key=lambda kv: kv[1])
    conf = best_hits / max(1, len([x for x in dim_seq if x >= 0]))
    return best_dim, float(conf), grid, dim_seq

# ───────────────────── 自适应各向异性扩胞 ─────────────────────

def auto_supercell(structure: Structure,
                   target_span: float = 12.0,
                   hard_cap: int = 5) -> tuple[int, int, int]:
    """
    让 a/b/c 三轴长度都至少 ~target_span Å。每轴最多放大 hard_cap 倍。
    """
    a, b, c = structure.lattice.abc
    reps = []
    for L in (a, b, c):
        k = max(1, int(math.ceil(target_span / max(L, 1e-6))))
        reps.append(min(k, hard_cap))
    return tuple(reps)

# ───────────────────── 构图策略（辅判据） ─────────────────────

def build_graph(structure: Structure, strategy_name: str) -> StructureGraph:
    if strategy_name == "CrystalNN":
        strat = CrystalNN()
        return StructureGraph.with_local_env_strategy(structure, strat)
    if strategy_name == "EconNN":
        strat = EconNN(cutoff=10.0)
        return StructureGraph.with_local_env_strategy(structure, strat)
    if strategy_name == "VoronoiNN":
        strat = VoronoiNN(weight="solid_angle")
        return StructureGraph.with_local_env_strategy(structure, strat)
    if strategy_name == "MinimumDistanceNN":
        strat = MinimumDistanceNN(cutoff=0.1, get_all_sites=True, allow_radii=True)
        return StructureGraph.with_local_env_strategy(structure, strat)
    if strategy_name == "CutOffDictNN":
        base = defaultdict(lambda: 3.3)
        for el in ("F", "Cl", "Br", "I"):
            base[el] = 3.6
        strat = CutOffDictNN(dict(base))
        return StructureGraph.with_local_env_strategy(structure, strat)
    if strategy_name == "AdaptiveRadiusNN":
        return AdaptiveRadiusNN().get_graph(structure)
    raise ValueError(f"Unknown strategy: {strategy_name}")


def larsen_dim(g: StructureGraph) -> int:
    return int(get_dimensionality_larsen(g))

# ───────────────────── 共识图 ─────────────────────

def graph_union(graphs: list[StructureGraph]) -> StructureGraph:
    if not graphs:
        raise ValueError("Empty graph list")
    base = StructureGraph.with_empty_graph(graphs[0].structure)
    n = len(base.structure)
    present = [[False]*n for _ in range(n)]
    for g in graphs:
        for u in range(n):
            neigh = g.get_connected_sites(u)
            for cs in neigh:
                v = cs.index
                if v != u:
                    present[min(u,v)][max(u,v)] = True
    for i in range(n):
        for j in range(i+1, n):
            if present[i][j]:
                base.add_edge(i, j)
    return base


def graph_majority(graphs: list[StructureGraph]) -> StructureGraph:
    if not graphs:
        raise ValueError("Empty graph list")
    base = StructureGraph.with_empty_graph(graphs[0].structure)
    n = len(base.structure)
    cnt = [[0]*n for _ in range(n)]
    for g in graphs:
        for u in range(n):
            for cs in g.get_connected_sites(u):
                v = cs.index
                if v != u:
                    cnt[min(u,v)][max(u,v)] += 1
    thresh = max(1, math.floor(len(graphs)/2))  # 多数阈值：≥ floor(N/2)
    for i in range(n):
        for j in range(i+1, n):
            if cnt[i][j] >= thresh:
                base.add_edge(i, j)
    return base

# ───────────────────── 加权投票 ─────────────────────
WEIGHT = {
    "LarsenSweep": 3.2,
    "CrystalNN": 2.2,
    "EconNN": 1.6,
    "VoronoiNN": 1.2,
    "MinimumDistanceNN": 1.0,
    "CutOffDictNN": 0.9,
    "AdaptiveRadiusNN": 1.3,
    "ConsensusMajority": 2.4,
    "ConsensusUnion": 1.8,
}

PRIORITY = [
    "LarsenSweep",
    "ConsensusMajority",
    "CrystalNN",
    "EconNN",
    "VoronoiNN",
    "AdaptiveRadiusNN",
    "MinimumDistanceNN",
    "CutOffDictNN",
    "ConsensusUnion",
]


def weighted_vote(results: dict[str, int | None]) -> tuple[str, int | None, str, str, float]:
    score = defaultdict(float)
    for m, d in results.items():
        if d is None:
            continue
        score[d] += WEIGHT.get(m, 1.0)
    if not score:
        return "failed", None, "仅单一结果", "None", 0.0

    best_dim, best_score = max(score.items(), key=lambda kv: kv[1])
    valid_dims = [d for d in results.values() if d is not None]
    unique_dims = set(valid_dims)

    if len(unique_dims) == 1 and len(valid_dims) >= 2:
        consistency = "强一致"
    elif Counter(valid_dims).most_common(1)[0][1] >= max(2, math.ceil(len(valid_dims)/2)):
        consistency = "多数一致"
    elif len(valid_dims) == 1:
        consistency = "仅单一结果"
    else:
        consistency = "平票择优"

    used_method = None
    for m in PRIORITY:
        if results.get(m) == best_dim:
            used_method = m
            break
    used_method = used_method or next((m for m, d in results.items() if d == best_dim), "Unknown")
    return f"{best_dim}D", best_dim, consistency, used_method, float(best_score)

# ───────────────────── 单轮判定（主：Larsen 扫描；辅：多策略 + 共识） ─────────────────────
BASE_STRATEGIES = ["CrystalNN", "EconNN", "VoronoiNN", "MinimumDistanceNN", "CutOffDictNN", "AdaptiveRadiusNN"]


def classify_once(structure: Structure, supercell: tuple[int,int,int] | None,
                  try_consensus: bool = True) -> tuple[str, int|None, str, str, dict, float, str]:
    # === 结构准备：必要时进行各向异性扩胞 ===
    s_use = structure
    if supercell:
        s_use = structure * supercell
    else:
        reps = auto_supercell(structure)
        if reps != (1, 1, 1):
            s_use = structure * reps

    results: dict[str, int | None] = {}
    graphs: list[StructureGraph] = []
    decision_path = []

    # === ① Larsen 距离扫描（主判据） ===
    try:
        d_sweep, conf_sweep, _, _ = larsen_dim_sweep(s_use)
    except Exception:
        d_sweep, conf_sweep = None, 0.0

    results["LarsenSweep"] = d_sweep if d_sweep in (0, 1, 2, 3) else None
    decision_path.append(f"larsen_sweep(conf={conf_sweep:.2f})")

    if (d_sweep in (0, 1, 2, 3)) and (conf_sweep >= 0.65):
        label = f"{d_sweep}D"
        return label, d_sweep, "强一致", "LarsenSweep", results, float(conf_sweep * 3.0), " → ".join(decision_path)

    # === ② 备选：多策略 + 共识图 ===
    for name in BASE_STRATEGIES:
        try:
            g = build_graph(s_use, name)
            graphs.append(g)
            d = larsen_dim(g)
        except Exception:
            d = None
        results[name] = d
    decision_path.append("base_strategies")

    label, num, consistency, used, conf_vote = weighted_vote(results)

    # 平票/冲突/仅单一时引入共识图
    if try_consensus and (consistency in ("平票择优", "仅单一结果") or len(set(v for v in results.values() if v is not None)) > 1):
        try:
            gmaj = graph_majority([g for g in graphs if g is not None])
            dmaj = larsen_dim(gmaj)
        except Exception:
            dmaj = None
        results["ConsensusMajority"] = dmaj
        try:
            gunion = graph_union([g for g in graphs if g is not None])
            dunion = larsen_dim(gunion)
        except Exception:
            dunion = None
        results["ConsensusUnion"] = dunion
        decision_path.append("consensus")
        label, num, consistency, used, conf_vote = weighted_vote(results)

    # 与扫描一致时融合置信度
    if (d_sweep in (0, 1, 2, 3)) and (num == d_sweep):
        conf_final = max(conf_vote, conf_sweep * 2.0)
        used = "LarsenSweep+Ensemble"
        if consistency != "强一致":
            consistency = "多数一致"
        decision_path.append("merge_sweep_ensemble")
    else:
        conf_final = conf_vote

    decision_path_txt = " → ".join(decision_path)
    return label, num, consistency, used, results, float(conf_final), decision_path_txt

# ───────────────────── 主分类（含对称化与扩胞） ─────────────────────

def classify_dim_robust(path: Path,
                        input_root: Path,
                        output_root: Path,
                        supercell_default=(3,3,3),
                        timeout_sec: int | None = None) -> DimRec | None:
    # 读取
    try:
        rel = path.relative_to(input_root)
    except Exception:
        rel = path
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore")
        try:
            structure = Structure.from_file(path)
        except Exception as e:
            return _error_rec(path, input_root, error=f"read-failed: {e}")

    if len(structure or []) == 0:
        return _error_rec(path, input_root, error="empty-structure")

    # 对称化
    std, sg_symbol, sym_used, ang_used, which_std = safe_spacegroup(structure)

    # 单轮（不显式 supercell；内部可能做各向异性扩胞）
    label, num, consistency, used, r0, conf0, dec0 = classify_once(std, supercell=None, try_consensus=True)

    decision_chain = [f"no_super({dec0})"]

    # 扩胞触发：低维或低置信或混票
    need_expand = False
    unique0 = set(v for v in r0.values() if v is not None)
    if EXPAND_ON_AMBIGUOUS:
        if (num in (0, 1)) or (consistency in ("平票择优", "仅单一结果")) or (len(unique0) > 1):
            need_expand = True

    used_super = (1, 1, 1)
    r_all = {"no_super": r0}
    conf_final = conf0

    if need_expand:
        try:
            label2, num2, consistency2, used2, r1, conf1, dec1 = classify_once(std, supercell=supercell_default, try_consensus=True)
            r_all["with_super"] = r1
            better = False
            if (num2 is not None and num is not None and num2 > num):
                better = True
            elif _consistency_rank(consistency2) > _consistency_rank(consistency):
                better = True
            elif conf1 > conf0 + 0.5:
                better = True
            if better:
                label, num, consistency, used, conf_final = label2, num2, consistency2, used2, conf1
                used_super = supercell_default
                decision_chain.append(f"with_super({dec1})")
        except Exception as e:
            decision_chain.append(f"with_super_error:{e}")

    # 输出拷贝
    dest_dir = output_root / rel.parent / (label if label != "failed" else "failed")
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(path, dest_dir / path.name)
        dest_path = dest_dir / path.name
    except Exception:
        dest_path = Path("N/A")

    try:
        formula = std.composition.reduced_formula
    except Exception:
        formula = "Unknown"

    natoms = len(std)
    parts = rel.parts
    halogen = parts[0] if len(parts) > 0 else "unknown"
    struct_cls = parts[1] if len(parts) > 1 else "unknown"

    conf_map = {
        "强一致": "强一致",
        "多数一致": "多数一致",
        "平票择优": "平票择优",
        "仅单一结果": "仅单一结果"
    }

    return DimRec(
        Material=path.name,
        Formula=formula,
        NAtoms=natoms,
        Spacegroup=sg_symbol,
        HalogenGroup=halogen,
        StructClass=struct_cls,
        Dimensionality=label,
        NumericDim=num,
        Consistency=conf_map.get(consistency, consistency),
        MethodUsed=used,
        MethodsAll=json.dumps(r_all, ensure_ascii=False),
        SymprecUsed=sym_used,
        AngleTolUsed=ang_used,
        SupercellUsed=f"{used_super}",
        NewPath=str(dest_path),
        Error=("read-warnings" if w else ""),
        ConfidenceScore=round(conf_final, 3),
        DecisionPath="; ".join(decision_chain)
    )


def _consistency_rank(c: str) -> int:
    order = {"强一致": 3, "多数一致": 2, "平票择优": 1, "仅单一结果": 0}
    return order.get(c, 0)


def _error_rec(f: Path, input_root: Path, error: str) -> DimRec:
    try:
        rel = f.relative_to(input_root)
        parts = rel.parts
        halogen = parts[0] if len(parts) > 0 else "unknown"
        struct_cls = parts[1] if len(parts) > 1 else "unknown"
    except Exception:
        halogen, struct_cls = "unknown", "unknown"
    return DimRec(
        Material=f.name, Formula="Unknown", NAtoms=0, Spacegroup="Unknown",
        HalogenGroup=halogen, StructClass=struct_cls,
        Dimensionality="failed", NumericDim=None,
        Consistency="仅单一结果", MethodUsed="None",
        MethodsAll="{}", SymprecUsed=0.0, AngleTolUsed=0.0,
        SupercellUsed="(1,1,1)", NewPath="N/A", Error=error,
        ConfidenceScore=0.0, DecisionPath="error"
    )

# ───────────────────── 超时保护（Linux 有效） ─────────────────────
class _Timeout:
    def __init__(self, sec: int | None):
        self.sec = sec
        self._old = None
    def __enter__(self):
        if self.sec and hasattr(signal, "SIGALRM"):
            self._old = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.sec)
    def _handler(self, signum, frame):
        raise TimeoutError("per-file timeout")
    def __exit__(self, exc_type, exc, tb):
        if self.sec and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)

# ───────────────────── 工作函数（供 map 调用） ─────────────────────

def _worker(args_tuple):
    f, input_root, output_root, supercell, timeout_sec = args_tuple
    try:
        with _Timeout(timeout_sec):
            return classify_dim_robust(Path(f), Path(input_root), Path(output_root),
                                       supercell_default=supercell, timeout_sec=timeout_sec)
    except Exception as e:
        return _error_rec(Path(f), Path(input_root), error=str(e))

# ───────────────────── 主流程 ─────────────────────

def main():
    ap = argparse.ArgumentParser(description="Robust dimensionality classifier (Larsen sweep + ensemble + consensus)")
    ap.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Root folder of structures")
    ap.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output root")
    ap.add_argument("--supercell", type=str, default="3,3,3", help="Supercell like '3,3,3'")
    ap.add_argument("--max-workers", type=int, default=DEFAULT_MAXW, help="Parallel workers")
    ap.add_argument("--suffix", type=str, default=".cif,.vasp", help="File suffixes, comma-separated")
    ap.add_argument("--timeout-sec", type=int, default=0, help="Per-file timeout (0=disabled)")
    ap.add_argument("--chunksize", type=int, default=16, help="Executor map chunksize")
    args = ap.parse_args()

    _pin_threads(1)

    input_root  = Path(args.input)
    output_root = Path(args.output)
    supercell   = tuple(int(x) for x in args.supercell.split(","))
    suffixes    = tuple(s.strip().lower() for s in args.suffix.split(","))
    timeout_sec = args.timeout_sec if args.timeout_sec and args.timeout_sec > 0 else None
    chunksize   = max(1, args.chunksize)

    output_root.mkdir(parents=True, exist_ok=True)

    files = [str(f) for f in input_root.rglob("*") if f.suffix.lower() in suffixes]
    if not files:
        print("[INFO] 未发现结构文件。")
        return

    files = sorted(files, key=lambda p: os.path.getsize(p) if os.path.exists(p) else 0, reverse=True)
    print(f"[INFO] 发现 {len(files)} 个结构文件，使用 {args.max_workers} 进程。")

    task_args = [(f, str(input_root), str(output_root), supercell, timeout_sec) for f in files]

    records: list[DimRec] = []
    with ProcessPoolExecutor(max_workers=max(1, args.max_workers), initializer=_pin_threads, initargs=(1,)) as ex:
        for i, rec in enumerate(ex.map(_worker, task_args, chunksize=chunksize), 1):
            if rec:
                records.append(rec)
                print(f"[{i:>5}/{len(files)}] {rec.Material} → {rec.Dimensionality} ({rec.Consistency}, {rec.MethodUsed}, conf={rec.ConfidenceScore:.2f})")

    # ==== 写 CSV ====
    csv_detail  = output_root / "dim_layer_detail.csv"
    csv_summary = output_root / "dim_layer_summary.csv"
    df = pd.DataFrame(asdict(r) for r in records)
    df.to_csv(csv_detail, index=False)
    print(f"[✓] 详细 CSV 写入：{csv_detail}")

    ok = df[df["Dimensionality"] != "failed"].copy()
    if not ok.empty:
        summary = (
            ok
            .pivot_table(index=["HalogenGroup", "StructClass"],
                         columns="Dimensionality",
                         values="Material",
                         aggfunc="count",
                         fill_value=0)
            .reset_index()
        )
        for lbl in ["0D", "1D", "2D", "3D"]:
            if lbl not in summary.columns:
                summary[lbl] = 0
        summary["Total"] = summary[["0D", "1D", "2D", "3D"]].sum(axis=1)
        summary.to_csv(csv_summary, index=False)
        print(f"[✓] 汇总 CSV 写入：{csv_summary}")
    else:
        print("[!] 无成功项，未生成汇总 CSV。")

if __name__ == "__main__":
    main()
