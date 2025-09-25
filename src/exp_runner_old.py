# exp_runner_old.py
# ACIT4610 Assignment 2 – Experiment runner
# - Discovers instances from instances.py (DATA_DIR)
# - Selects by original_name (CVRPLIB IDs)
# - Runs VEGA & NSGA-II across presets and seeds
# - Logs wall time, #evaluations
# - Computes GD, IGD, Spread (Δ), Hypervolume
# - Writes CSVs + LaTeX tables into ./results
from __future__ import annotations

import inspect
import os
import json
import time
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from instances import list_instances as list_inst_names, load_instance as load_inst, DATA_DIR as INST_DIR
import moea
import constants as C  # NOTE: used for pop/gens/pc/pm/seed ONLY. Algorithm is NOT read from here.

# =========================
# Select instances by original_name
# =========================

WANTED: Dict[str, set] = {
    "small":  {"A-n32-k5", "A-n36-k5"},
    # "medium": {"B-n45-k6", "B-n57-k7"},
    # "large":  {"E-n101-k14", "M-n151-k12"},
}


def _read_meta(name: str) -> Tuple[str, str]:
    """Return (original_name, category) for a given internal instance name."""
    path = INST_DIR / f"{name}.json"
    with open(path, "r") as f:
        lines = f.readlines()
    # allow // inline comments like your loader
    filtered = "\n".join(l for l in lines if not l.lstrip().startswith("//"))
    data = json.loads(filtered)
    return data.get("original_name", name), data.get("category", "uncategorized")


def discover_and_select() -> List[Tuple[str, str, str]]:
    """Returns a list of (internal_name, original_name, category) in stable order."""
    discovered = []
    for name in sorted(list_inst_names()):
        orig, cat = _read_meta(name)
        discovered.append((name, orig, cat))

    print("Scanning instance JSONs from:", os.path.abspath(INST_DIR))
    print("Discovered instances:", [(cat, orig, name) for (name, orig, cat) in discovered])

    # Filter by WANTED against original_name
    selected = [
        (name, orig, cat)
        for (name, orig, cat) in discovered
        if cat in WANTED and orig in WANTED[cat]
    ]

    # Order: small -> medium -> large; within each, sort by original_name
    cat_rank = {"small": 0, "medium": 1, "large": 2}
    selected.sort(key=lambda t: (cat_rank.get(t[2], 99), t[1]))
    print("Selected instances:", [(cat, orig, name) for (name, orig, cat) in selected])

    if not selected:
        raise RuntimeError(
            "No instances selected. Ensure JSON files are in src/data/instances and "
            "WANTED names match each file's 'original_name'."
        )
    return selected


# =========================
# Algorithms & presets (SOURCE OF TRUTH FOR ALGORITHM)
# =========================

ALGORITHMS = ["NSGA2", "VEGA"]

PRESETS = {
    # "fast":      dict(pop=60,  gens=150, pc=0.90, pm=0.08),
    # "balanced":  dict(pop=100, gens=250, pc=0.90, pm=0.10),
    "thorough":  dict(pop=140, gens=400, pc=0.90, pm=0.12),
}

N_RUNS_PER_CELL = 3
SEEDS = list(range(1000, 1000 + N_RUNS_PER_CELL))


# =========================
# Metrics (2D minimization)
# =========================

def nondominated(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if not keep[i]:
            continue
        dominated = np.all(points <= p, axis=1) & np.any(points < p, axis=1)
        dominated[i] = False
        keep[dominated] = False
    return keep


def generational_distance(P: np.ndarray, PF_star: np.ndarray) -> float:
    if len(P) == 0 or len(PF_star) == 0:
        return float("inf")
    d = np.min(np.linalg.norm(P[:, None, :] - PF_star[None, :, :], axis=2), axis=1)
    return float(np.mean(d))


def inverted_generational_distance(P: np.ndarray, PF_star: np.ndarray) -> float:
    if len(P) == 0 or len(PF_star) == 0:
        return float("inf")
    d = np.min(np.linalg.norm(PF_star[:, None, :] - P[None, :, :], axis=2), axis=1)
    return float(np.mean(d))


def spread_delta(P: np.ndarray, PF_star: np.ndarray) -> float:
    # Deb's Δ (simple 2D variant)
    if len(P) < 2 or len(PF_star) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    d_bar = float(np.mean(d)) if len(d) else 0.0
    f1_min = PF_star[np.argmin(PF_star[:, 0])]
    f2_min = PF_star[np.argmin(PF_star[:, 1])]
    d_f = float(np.linalg.norm(P[0] - f1_min))
    d_l = float(np.linalg.norm(P[-1] - f2_min))
    return float((d_f + d_l + np.sum(np.abs(d - d_bar))) / (d_f + d_l + (len(d) * d_bar) + 1e-12))


def hypervolume_2d(P: np.ndarray, ref: Tuple[float, float]) -> float:
    if len(P) == 0:
        return 0.0
    P = P[P[:, 0].argsort()]
    hv, prev_f1, cur = 0.0, ref[0], ref[1]
    for f1, f2 in P[::-1]:
        hv += max(0.0, prev_f1 - f1) * max(0.0, cur - f2)
        prev_f1 = f1
        cur = min(cur, f2)
    return float(hv)


# =========================
# Instrumentation (#evaluations)
# =========================

_eval_counter = {"count": 0}
_orig_eval = moea.evaluate_individual

def _counting_eval(*args, **kwargs):
    _eval_counter["count"] += 1
    return _orig_eval(*args, **kwargs)

moea.evaluate_individual = _counting_eval  # Count evaluations during a run


# =========================
# Runner internals
# =========================

def apply_preset_to_constants(preset: Dict) -> None:
    """Only size/rates come from constants; algorithm does NOT."""
    C.POP_SIZE   = preset["pop"]
    C.GENERATIONS = preset["gens"]
    C.PC         = preset["pc"]
    C.PM         = preset["pm"]


def _try_set_algorithm_on_moea(algo: str) -> None:
    """
    Best-effort hinting to moea without touching constants.
    Used only if run_moea doesn't accept an algorithm parameter.
    """
    try:
        if hasattr(moea, "set_algorithm") and callable(getattr(moea, "set_algorithm")):
            moea.set_algorithm(algo)
            return
    except Exception:
        pass
    for attr in ("MOEA_ALGORITHM", "ALGORITHM", "PREFERRED_ALGORITHM"):
        if hasattr(moea, attr):
            try:
                setattr(moea, attr, algo)
                return
            except Exception:
                continue
    # If nothing is available, we just proceed; algorithm must be passed via args.


def _call_run_moea(inst_obj, native_seed: int, algo: str):
    """
    Calls moea.run_moea while ensuring ALG comes from this file.
    Tries several signatures; prefers keyword passing of algorithm.
    Never writes constants.MOEA_ALGORITHM.
    """
    # First, try rich keyword forms
    candidates = [
        lambda: moea.run_moea(inst_obj, seed=native_seed, algorithm=algo),
        lambda: moea.run_moea(inst_obj, seed=native_seed, algo=algo),
        lambda: moea.run_moea(inst_obj, algorithm=algo, seed=native_seed),
        lambda: moea.run_moea(inst_obj, algo=algo, seed=native_seed),
        lambda: moea.run_moea(seed=native_seed, inst=inst_obj, algorithm=algo),
        lambda: moea.run_moea(seed=native_seed, inst=inst_obj, algo=algo),
        # Common positional patterns
        lambda: moea.run_moea(native_seed, inst_obj, algo),
        lambda: moea.run_moea(inst_obj, native_seed, algo),
        lambda: moea.run_moea(inst_obj, algo, native_seed),
        # Seed + inst only (we'll also hint moea with module attribute)
        lambda: moea.run_moea(inst_obj, seed=native_seed),
        lambda: moea.run_moea(native_seed, inst_obj),
        lambda: moea.run_moea(inst_obj, native_seed),
        # Last resort (rely on moea module state we set above)
        lambda: moea.run_moea(inst_obj),
    ]

    # Before last-resort attempts, try to set a hint on the moea module.
    # This doesn’t touch constants.
    _try_set_algorithm_on_moea(algo)

    last_err = None
    for i, attempt in enumerate(candidates, 1):
        try:
            return attempt()
        except TypeError as e:
            last_err = e
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Could not call moea.run_moea with algorithm='{algo}' without using constants. "
        f"Last error: {last_err}"
    )


def run_once(internal_name: str, preset_name: str, algorithm: str, seed: int) -> Tuple[np.ndarray, float, int]:
    """Load instance by internal name and run the chosen MOEA once."""
    inst_obj = load_inst(internal_name)

    # Native int seeds everywhere
    native_seed = int(seed)
    import random
    random.seed(native_seed)
    np.random.seed(native_seed % (2**32 - 1))
    C.SEED = native_seed  # seed is fine to store in constants

    # Apply preset (NOT algorithm)
    apply_preset_to_constants(PRESETS[preset_name])

    # Reset eval counter and run
    _eval_counter["count"] = 0
    t0 = time.time()

    pareto_inds, _, _ = _call_run_moea(inst_obj, native_seed, algorithm)
    wall = time.time() - t0

    # Gather (f1, f2) for THIS RUN only
    P = np.array(
        [ind.objectives for ind in pareto_inds if getattr(ind, "objectives", None) is not None],
        dtype=float
    )
    P = P[~np.any(np.isinf(P), axis=1)] if len(P) else P

    # Per-run nondominated filter
    def _nd_mask(points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros(0, dtype=bool)
        keep = np.ones(len(points), dtype=bool)
        for i, p in enumerate(points):
            if not keep[i]:
                continue
            dom = np.all(points <= p, axis=1) & np.any(points < p, axis=1)
            dom[i] = False
            keep[dom] = False
        return keep

    if len(P) > 0:
        P = P[_nd_mask(P)]

    evals = int(_eval_counter["count"])
    return P, wall, evals


def union_nondominated(fronts: List[np.ndarray]) -> np.ndarray:
    if not fronts:
        return np.zeros((0, 2))
    fronts = [f for f in fronts if len(f) > 0]
    if not fronts:
        return np.zeros((0, 2))
    allp = np.vstack(fronts)
    mask = nondominated(allp)
    return allp[mask]


# =========================
# Main experiment
# =========================

def main():
    selected = discover_and_select()

    rows_runtime: List[Dict] = []
    percell_fronts: Dict[Tuple[str, str, str], List[np.ndarray]] = {}  # (instance_display, algorithm, preset) -> [fronts]

    # 1) Execute runs
    for internal_name, instance_display, category in selected:
        for preset_name in PRESETS:
            for algo in ALGORITHMS:
                key = (instance_display, algo, preset_name)
                percell_fronts[key] = []
                for seed in SEEDS:
                    # -- run once (this enforces algorithm from ALGORITHMS) --
                    P, wall, evals = run_once(internal_name, preset_name, algo, int(seed))

                    # quick fingerprint so you can see runs differ
                    print(f"[dump] {algo}/{instance_display}/{preset_name}/seed={seed}  "
                          f"n_points={len(P)}  first_row={P[0].tolist() if len(P) else 'EMPTY'}")

                    # ----- dump this run's front -----
                    out_front_dir = os.path.join("results", "fronts", instance_display)
                    os.makedirs(out_front_dir, exist_ok=True)
                    front_path = os.path.join(out_front_dir, f"{algo.lower()}_{preset_name}_seed{int(seed)}.csv")
                    # np.savetxt(
                    #     front_path,
                    #     P.copy(),  # write a copy to avoid aliasing later
                    #     delimiter=",",
                    #     header="f1_distance,f2_balance_std",
                    #     comments=""
                    # )

                    np.save(front_path.replace(".csv", ".npy"), P)

                    # record
                    percell_fronts[key].append(P)
                    rows_runtime.append(dict(
                        instance=instance_display,
                        instance_internal=internal_name,
                        category=category,
                        algorithm=algo,
                        preset=preset_name,
                        seed=int(seed),
                        wall_clock_s=float(wall),
                        evaluations=int(evals),
                    ))

    # 2) Build reference Pareto front per instance (nondominated union over all algos/presets/seeds)
    ref_by_instance: Dict[str, np.ndarray] = {}
    for _, instance_display, _ in selected:
        all_fronts = []
        for algo in ALGORITHMS:
            for preset_name in PRESETS:
                all_fronts.extend(percell_fronts[(instance_display, algo, preset_name)])
        ref_by_instance[instance_display] = union_nondominated(all_fronts)

    # 3) Compute metrics per run against the instance-level reference front
    metrics_rows: List[Dict] = []
    for _, instance_display, _ in selected:
        PF_star = ref_by_instance[instance_display]
        if len(PF_star) == 0:
            hv_ref = (1.0, 1.0)
        else:
            worst = np.max(PF_star, axis=0)
            hv_ref = (float(worst[0] * 1.05), float(worst[1] * 1.05))
        for algo in ALGORITHMS:
            for preset_name in PRESETS:
                fronts = percell_fronts[(instance_display, algo, preset_name)]
                for seed, P in zip(SEEDS, fronts):
                    if len(P) == 0 or len(PF_star) == 0:
                        gd = igd = 1e9
                        spr = 0.0
                        hv = 0.0
                    else:
                        gd = generational_distance(P, PF_star)
                        igd = inverted_generational_distance(P, PF_star)
                        spr = spread_delta(P, PF_star)
                        hv = hypervolume_2d(P, hv_ref)
                    metrics_rows.append(dict(
                        instance=instance_display,
                        algorithm=algo,
                        preset=preset_name,
                        seed=int(seed),
                        gd=float(gd),
                        igd=float(igd),
                        spread=float(spr),
                        hv=float(hv),
                    ))

    # 4) Merge and aggregate
    df_rt = pd.DataFrame(rows_runtime)
    df_mt = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame(
        columns=["instance", "algorithm", "preset", "seed", "gd", "igd", "spread", "hv"]
    )
    if df_rt.empty:
        df_rt = pd.DataFrame(columns=[
            "instance", "instance_internal", "category",
            "algorithm", "preset", "seed",
            "wall_clock_s", "evaluations"
        ])

    df = df_rt.merge(df_mt, on=["instance", "algorithm", "preset", "seed"], how="left")

    agg = df.groupby(["instance", "algorithm", "preset"]).agg({
        "wall_clock_s": ["mean", "std"],
        "evaluations":  ["mean", "std"],
        "gd":           ["mean", "std"],
        "igd":          ["mean", "std"],
        "spread":       ["mean", "std"],
        "hv":           ["mean", "std"],
    }).reset_index()
    agg.columns = [' '.join(c).strip() if isinstance(c, tuple) else c for c in agg.columns]

    runtime = df.groupby(["instance", "algorithm", "preset"]).agg({
        "wall_clock_s": ["mean", "std"],
        "evaluations":  ["mean", "std"],
    }).reset_index()
    runtime.columns = [' '.join(c).strip() if isinstance(c, tuple) else c for c in runtime.columns]

    # 5) Save outputs
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "runs_raw.csv"), index=False)
    agg.to_csv(os.path.join(out_dir, "summary_metrics.csv"), index=False)
    runtime.to_csv(os.path.join(out_dir, "runtime_summary.csv"), index=False)

    def to_latex(df_: pd.DataFrame, caption: str, label: str) -> str:
        return df_.to_latex(index=False,
                            float_format=lambda x: f"{x:.3f}",
                            caption=caption, label=label, escape=False)

    with open(os.path.join(out_dir, "summary_metrics.tex"), "w") as f:
        f.write(to_latex(agg, "Pareto quality and runtime (mean$\\pm$std).", "tab:quality"))
    with open(os.path.join(out_dir, "runtime_summary.tex"), "w") as f:
        f.write(to_latex(runtime, "Runtime summary (mean$\\pm$std).", "tab:runtime"))

    print("Saved results to:", out_dir)


if __name__ == "__main__":
    main()
