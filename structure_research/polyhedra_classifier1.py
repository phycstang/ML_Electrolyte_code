import os
import json
import shutil
import argparse
from multiprocessing import Pool, cpu_count
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.transformations.standard_transformations import SupercellTransformation
import networkx as nx

# 参数配置
MAX_DISTANCE = 3.5
MIN_NEIGHBORS_FOR_CLUSTER = 3

# ---------- 多面体识别 ----------
def extract_polyhedra(structure):
    cnn = CrystalNN()
    polyhedra = []
    ligand_elements = {"Cl", "F", "Br", "I"}

    for i, site in enumerate(structure):
        if site.specie.symbol in ligand_elements:
            continue
        try:
            nns = cnn.get_nn_info(structure, i)
            neighbors = [n['site_index'] for n in nns if structure[n['site_index']].specie.symbol in ligand_elements]
            if len(neighbors) >= MIN_NEIGHBORS_FOR_CLUSTER:
                polyhedra.append({
                    "id": len(polyhedra),
                    "center_index": i,
                    "type": "unknown",
                    "neighbors": neighbors,
                    "frac_coords": list(structure[i].frac_coords),
                    "is_connected": False,
                    "component_id": None,
                    "degree": 0
                })
        except Exception:
            continue
    return polyhedra

# ---------- 多面体连通性分析 ----------
def analyze_polyhedra_connectivity(polyhedra):
    G = nx.Graph()
    for p in polyhedra:
        G.add_node(p['id'])

    for i, p1 in enumerate(polyhedra):
        for j, p2 in enumerate(polyhedra[i + 1:], i + 1):
            if set(p1["neighbors"]) & set(p2["neighbors"]):
                G.add_edge(p1['id'], p2['id'])

    components = list(nx.connected_components(G))
    for cid, comp in enumerate(components):
        for pid in comp:
            polyhedra[pid]['is_connected'] = True
            polyhedra[pid]['component_id'] = cid
            polyhedra[pid]['degree'] = len(list(G.neighbors(pid)))

    return polyhedra, G

# ---------- 分类函数 ----------
def classify_polyhedra(polyhedra, total_sites):
    if not polyhedra:
        return "no_polyhedra"
    used_atoms = set()
    else:
        return "polyhedra"

# ---------- 扩胞 ----------
def expand_structure(structure, scale_matrix=(3, 3, 3)):
    try:
        transformation = SupercellTransformation(scale_matrix)
        return transformation.apply_transformation(structure)
    except Exception:
        return structure

# ---------- 文件处理 ----------
def process_file(filepath, input_root, output_root):
    try:
        structure = Structure.from_file(filepath)
        structure = expand_structure(structure)
        polyhedra = extract_polyhedra(structure)
        polyhedra, _ = analyze_polyhedra_connectivity(polyhedra)
        category = classify_polyhedra(polyhedra, len(structure))

        rel_path = os.path.relpath(filepath, input_root)
        output_dir = os.path.join(output_root, category, os.path.dirname(rel_path))
        os.makedirs(output_dir, exist_ok=True)

        new_name = os.path.basename(filepath)
        shutil.copy(filepath, os.path.join(output_dir, new_name))
        with open(os.path.join(output_dir, new_name + ".poly.json"), "w") as f:
            json.dump({
                "structure_name": os.path.splitext(new_name)[0],
                "polyhedra": polyhedra
            }, f, indent=2)

        print(f"[✓] Processed: {rel_path} → {category}")
    except Exception as e:
        print(f"[✗] Failed: {filepath} — {e}")

# ---------- 并行处理 ----------
def process_all_files(input_root, output_root):
    all_tasks = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith((".cif", ".poscar", ".vasp", ".xyz")):
                all_tasks.append(os.path.join(root, file))

    if not all_tasks:
        print("[!] No structure files found.")
        return

    print(f"[*] Found {len(all_tasks)} structure files. Using {cpu_count()} cores.")

    args = [(path, input_root, output_root) for path in all_tasks]
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(process_file, args)

# ---------- 主函数 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input directory with structure files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for classified results")
    args = parser.parse_args()

    process_all_files(args.input, args.output)
