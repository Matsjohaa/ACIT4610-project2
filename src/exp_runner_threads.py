# ACIT4610 Assignment 2 – Experiment runner
# - Discovers instances from instances.py (DATA_DIR)
# - Selects by original_name (CVRPLIB IDs)
# - Runs VEGA & NSGA-II across presets and seeds (in parallel)
# - Logs wall time, evaluations
# - Computes GD, IGD, Spread (Δ), Hypervolume
# - Writes Parquet fronts + CSV summary into OUT_ROOT

from __future__ import annotations
import os
import json
import time
import inspect
import hashlib
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from instances import (
    list_instances as list_inst_names,
    load_instance as load_inst,
    DATA_DIR as INST_DIR,
)
import moea
import constants as C  # NOTE: used for pop/gens/pc/pm/seed ONLY. Algorithm is NOT read from here.

# ---- Keep BLAS libraries to 1 thread per process BEFORE NumPy loads ----
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
del _os

ALGORITHMS = ["NSGA2", "VEGA"]

# =========================
# Select instances by original_name
# =========================

N_RUNS_PER_CELL = 20
SEEDS = list(range(1000, 1000 + N_RUNS_PER_CELL))

WANTED: Dict[str, set] = {
    "small":  {"A-n32-k5", "A-n36-k5"},
    "medium": {"B-n68-k9", "B-n78-k10"},
    "large":  {"E-n101-k14", "M-n151-k12"},
}

PRESETS = {
    "fast":      dict(pop=60,  gens=150, pc=0.90, pm=0.08),
    "balanced":  dict(pop=100, gens=250, pc=0.90, pm=0.10),
    "thorough":  dict(pop=140, gens=400, pc=0.90, pm=0.12),
}

from pathlib import Path

def _resolve_base_out_root() -> Path:
    """
    Resolve BASE_OUT in this order:
      1) ENV: EXP_OUT (points to the *parent* output dir, e.g., <repo>/exp_runner_output)
      2) <project_root>/exp_runner_output  (project_root is the folder that contains 'src')
      3) ./exp_runner_output  (CWD fallback)
    """
    env = os.environ.get("EXP_OUT")
    if env:
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p

    here = Path(__file__).resolve()
    cand = here.parent
    project_root = cand.parent if cand.name.lower() == "src" else cand
    out = (project_root / "exp_runner_output").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out

BASE_OUT = _resolve_base_out_root()
FRONTS_DIR = (BASE_OUT / "fronts").resolve()
FRONTS_DIR.mkdir(parents=True, exist_ok=True)

# Keep a readable alias for your prior OUT_ROOT name if used elsewhere:
OUT_ROOT = BASE_OUT
MAX_WORKERS = os.cpu_count() or 4
# print(f"[io] BASE_OUT={BASE_OUT}")
# print(f"[io] FRONTS_DIR={FRONTS_DIR}")


def _read_meta(name: str) -> Tuple[str, str]:
    """
    Read metadata for a given *internal* instance name.

    Returns (original_name, category).
    """
    path = INST_DIR / f"{name}.json"
    with open(path, "r") as f:
        lines = f.readlines()
    filtered = "\n".join(l for l in lines if not l.lstrip().startswith("//"))
    data = json.loads(filtered)
    return data.get("original_name", name), data.get("category", "uncategorized")


def discover_and_select() -> List[Tuple[str, str, str]]:
    """
    Discover all instances and select those requested in WANTED.

    Returns a list of (internal_name, original_name, category).
    """
    discovered = []
    for name in sorted(list_inst_names()):
        orig, cat = _read_meta(name)
        discovered.append((name, orig, cat))

    print("Scanning instance JSONs from:", os.path.abspath(INST_DIR))
    print("Discovered instances:", [(cat, orig, name) for (name, orig, cat) in discovered])

    selected = [
        (name, orig, cat)
        for (name, orig, cat) in discovered
        if cat in WANTED and orig in WANTED[cat]
    ]

    cat_rank = {"small": 0, "medium": 1, "large": 2}
    selected.sort(key=lambda t: (cat_rank.get(t[2], 99), t[1]))

    if not selected:
        raise RuntimeError(
            "No instances selected. Ensure JSON files are in src/data/instances and "
            "WANTED names match each file's 'original_name'."
        )
    return selected


# =========================
# Algorithms & metrics
# =========================

def nondominated_2d(points: np.ndarray) -> np.ndarray:
    """Faster nondominance for 2D minimization. O(n log n)."""
    if points.size == 0:
        return np.zeros(0, dtype=bool)
    idx = np.argsort(points[:, 0], kind="mergesort")  # by f1 asc
    sorted_pts = points[idx]
    best_f2 = float("inf")
    keep_sorted = np.zeros(len(sorted_pts), dtype=bool)
    for i, (_, f2) in enumerate(sorted_pts):
        if f2 < best_f2:
            keep_sorted[i] = True
            best_f2 = f2
    keep = np.zeros_like(keep_sorted)
    keep[idx] = keep_sorted
    return keep


def generational_distance(P: np.ndarray, PF: np.ndarray) -> float:
    if len(P) == 0 or len(PF) == 0:
        return float("inf")
    d = np.min(np.linalg.norm(P[:, None, :] - PF[None, :, :], axis=2), axis=1)
    return float(np.mean(d))


def inverted_generational_distance(P: np.ndarray, PF: np.ndarray) -> float:
    if len(P) == 0 or len(PF) == 0:
        return float("inf")
    d = np.min(np.linalg.norm(PF[:, None, :] - P[None, :, :], axis=2), axis=1)
    return float(np.mean(d))


def spread_delta(P: np.ndarray, PF_star: np.ndarray) -> float:
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

# Count evaluations during a run (applies in child processes too)
moea.evaluate_individual = _counting_eval


# =========================
# Runner internals
# =========================

def apply_preset_to_constants(preset: Dict) -> None:
    C.POP_SIZE    = preset["pop"]
    C.GENERATIONS = preset["gens"]
    C.PC          = preset["pc"]
    C.PM          = preset["pm"]


def _call_run_moea(inst_obj, native_seed: int, algo: str):
    """
    Enforce algorithm selection robustly across possible moea APIs,
    avoiding 'multiple values for argument' by preferring pure-keyword calls
    when names are available. Falls back through safe permutations.
    """
    algo = str(algo).upper()
    if algo not in {"NSGA2", "VEGA"}:
        raise ValueError(f"Unknown algo '{algo}'")

    # 1) Direct entry points if present
    if algo == "NSGA2":
        for name in ("run_nsga2", "nsga2", "runNSGA2"):
            if hasattr(moea, name) and callable(getattr(moea, name)):
                return getattr(moea, name)(inst_obj, seed=native_seed)
    else:
        for name in ("run_vega", "vega", "runVEGA"):
            if hasattr(moea, name) and callable(getattr(moea, name)):
                return getattr(moea, name)(inst_obj, seed=native_seed)

    # 2) Generic run_moea
    if not hasattr(moea, "run_moea") or not callable(moea.run_moea):
        raise RuntimeError("moea.run_moea is missing; can’t dispatch algorithm.")

    sig = inspect.signature(moea.run_moea)
    params = {p.name: p for p in sig.parameters.values()}
    names = set(params.keys())

    # Helper: try a list of callables in order, return first that works
    def _try(attempts):
        last_err = None
        for f in attempts:
            try:
                return f()
            except Exception as e:
                last_err = e
                continue
        if last_err:
            raise last_err
        raise RuntimeError("No viable call pattern for moea.run_moea")

    attempts = []

    # 2a) Prefer pure-keyword calls when parameter names exist (avoids duplicate binding)
    has_instance_kw = "instance" in names and params["instance"].kind != inspect.Parameter.POSITIONAL_ONLY
    has_seed_kw     = "seed"     in names and params["seed"].kind     != inspect.Parameter.POSITIONAL_ONLY
    has_alg_kw      = "algorithm" in names and params["algorithm"].kind != inspect.Parameter.POSITIONAL_ONLY
    has_algo_kw     = "algo"      in names and params["algo"].kind      != inspect.Parameter.POSITIONAL_ONLY

    if has_instance_kw and has_seed_kw and (has_alg_kw or has_algo_kw):
        if has_alg_kw:
            attempts.append(lambda: moea.run_moea(instance=inst_obj, seed=native_seed, algorithm=algo))
        if has_algo_kw:
            attempts.append(lambda: moea.run_moea(instance=inst_obj, seed=native_seed, algo=algo))

    # 2b) If only algorithm kw exists, try keyword for alg + keywords/positionals for others
    if has_alg_kw or has_algo_kw:
        alg_kw = {"algorithm": algo} if has_alg_kw else {"algo": algo}
        # try (all keywords if possible)
        if has_instance_kw and has_seed_kw:
            attempts.append(lambda: moea.run_moea(instance=inst_obj, seed=native_seed, **alg_kw))
        # try positional permutations (covers odd signatures like (seed, instance, ...))
        attempts.append(lambda: moea.run_moea(inst_obj, native_seed, **alg_kw))
        attempts.append(lambda: moea.run_moea(native_seed, inst_obj, **alg_kw))

    # 2c) If no alg kw, try module-level setter then positional/keyword for (instance, seed)
    used_setter = False
    if not (has_alg_kw or has_algo_kw):
        if hasattr(moea, "set_algorithm") and callable(moea.set_algorithm):
            moea.set_algorithm(algo); used_setter = True
        else:
            for attr in ("MOEA_ALGORITHM", "ALGORITHM", "PREFERRED_ALGORITHM"):
                if hasattr(moea, attr):
                    setattr(moea, attr, algo)
                    used_setter = True
                    break
        if used_setter:
            # prefer all-keyword if available
            if has_instance_kw and has_seed_kw:
                attempts.append(lambda: moea.run_moea(instance=inst_obj, seed=native_seed))
            # then positional variants
            attempts.append(lambda: moea.run_moea(inst_obj, native_seed))
            attempts.append(lambda: moea.run_moea(native_seed, inst_obj))

    # 2d) As a last resort, brute permutations w/ alg kw then without
    #     (covers very custom signatures)
    attempts.append(lambda: moea.run_moea(inst_obj, native_seed, algorithm=algo))
    attempts.append(lambda: moea.run_moea(inst_obj, native_seed, algo=algo))
    attempts.append(lambda: moea.run_moea(native_seed, inst_obj, algorithm=algo))
    attempts.append(lambda: moea.run_moea(native_seed, inst_obj, algo=algo))
    if used_setter:
        attempts.append(lambda: moea.run_moea(inst_obj, native_seed))
        attempts.append(lambda: moea.run_moea(native_seed, inst_obj))

    try:
        return _try(attempts)
    except TypeError as e:
        # Improve error message with the discovered signature
        raise RuntimeError(
            f"Could not call moea.run_moea with any safe pattern. Signature was: {sig}. "
            f"Last error: {e}"
        ) from e


def _stable_hash_10k(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % 10000


def run_once(internal_name: str, preset_name: str, algorithm: str, seed: int) -> Tuple[np.ndarray, float, int]:
    """
    Execute a single run for (instance, preset, algorithm, seed).
    Returns (P, wall, evals) where P is the run-local nondominated 2D front.
    """
    inst_obj = load_inst(internal_name)

    # Native int seeds everywhere
    native_seed = int(seed)
    import random
    random.seed(native_seed)
    np.random.seed(native_seed % (2**32 - 1))
    C.SEED = native_seed

    # Apply preset (NOT algorithm)
    apply_preset_to_constants(PRESETS[preset_name])

    # Reset eval counter and run
    _eval_counter["count"] = 0
    t0 = time.time()

    # Use the robust dispatcher
    pareto_inds, _, _ = _call_run_moea(inst_obj, native_seed, algorithm)
    wall = time.time() - t0

    # Gather (f1, f2) for THIS RUN only
    P = np.array(
        [ind.objectives for ind in pareto_inds if getattr(ind, "objectives", None) is not None],
        dtype=float
    )
    # Clean + nondom 2D
    if len(P):
        P = P[~np.any(np.isinf(P), axis=1)]
        if len(P):
            P = P[nondominated_2d(P)]

    evals = int(_eval_counter["count"])
    return P, wall, evals


def union_nondominated(fronts: List[np.ndarray]) -> np.ndarray:
    allp = np.vstack([f for f in fronts if len(f) > 0]) if fronts else np.zeros((0, 2))
    return allp[nondominated_2d(allp)] if len(allp) else allp


# =========================
# Main experiment
# =========================

def main():
    """
    Pipeline:
      1) Discover & select instances by original_name.
      2) For each (instance × preset × algorithm), run seeds in parallel, persist seed fronts.
      3) Build per-instance reference Pareto front (union across everything).
      4) Compute GD, IGD, Δ, HV per run vs reference.
      5) Save raw, summary under OUT_ROOT.
    """
    selected = discover_and_select()

    # rows for runtime + metrics
    rows_runtime: List[Dict] = []
    # (instance_display, algo, preset) -> { seed -> np.ndarray }
    percell_fronts: Dict[Tuple[str, str, str], Dict[int, np.ndarray]] = {}

    for internal_name, instance_display, category in selected:
        for preset_name in PRESETS:
            for algo in ALGORITHMS:
                key = (instance_display, algo, preset_name)
                percell_fronts[key] = {}

                seed_shift = _stable_hash_10k(preset_name)

                out_front_dir = (FRONTS_DIR / instance_display)
                out_front_dir.mkdir(parents=True, exist_ok=True)
                os.makedirs(out_front_dir, exist_ok=True)

                # Parallel seeds
                with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futures = {}
                    for seed in SEEDS:
                        adj_seed = int(seed + seed_shift)
                        fut = ex.submit(run_once, internal_name, preset_name, algo, adj_seed)
                        futures[fut] = seed

                    for fut in as_completed(futures):
                        seed = futures[fut]
                        P, wall, evals = fut.result()

                        # Persist per-seed front as its own Parquet (unique filename)
                        front_path = os.path.join(
                            out_front_dir, f"{algo.lower()}_{preset_name}_seed{int(seed)}.parquet"
                        )
                        if len(P):
                            pd.DataFrame(P, columns=["f1_distance", "f2_balance_std"]) \
                                .to_parquet(front_path, engine="pyarrow", index=False)
                        else:
                            pd.DataFrame(columns=["f1_distance", "f2_balance_std"]) \
                                .to_parquet(front_path, engine="pyarrow", index=False)

                        percell_fronts[key][int(seed)] = P
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

                # Guardrail: if NSGA2 and VEGA fronts are byte-for-byte identical per seed, algo kw likely ignored.
                key_nsga = (instance_display, "NSGA2", preset_name)
                key_vega = (instance_display, "VEGA", preset_name)
                if key_nsga in percell_fronts and key_vega in percell_fronts:
                    nsga_map = percell_fronts[key_nsga]
                    vega_map = percell_fronts[key_vega]
                    if nsga_map and vega_map and set(nsga_map.keys()) == set(vega_map.keys()):
                        identical_all = True
                        for s in nsga_map.keys():
                            a = nsga_map[s]
                            b = vega_map[s]
                            same = (a.shape == b.shape) and (a.size == 0 and b.size == 0 or (a == b).all())
                            if not same:
                                identical_all = False
                                break
                        if identical_all:
                            raise RuntimeError(
                                f"Identical fronts across all seeds for {instance_display}/{preset_name}. "
                                "Likely the algorithm selector is being ignored by moea.run_moea."
                            )

    # 2) Build reference Pareto front per instance (union over all algos/presets/seeds)
    ref_by_instance: Dict[str, np.ndarray] = {}
    for _, instance_display, _ in selected:
        all_fronts: List[np.ndarray] = []
        for algo in ALGORITHMS:
            for preset_name in PRESETS:
                all_fronts.extend(percell_fronts[(instance_display, algo, preset_name)].values())
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
                fronts_map = percell_fronts[(instance_display, algo, preset_name)]
                for seed in SEEDS:
                    P = fronts_map.get(int(seed), np.zeros((0, 2)))
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
    results_dir = (BASE_OUT / "src"/ "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "runs_raw.csv", index=False)
    agg.to_csv(results_dir / "summary_metrics.csv", index=False)
    runtime.to_csv(results_dir / "runtime_summary.csv", index=False)
    print("Saved results to:", results_dir)


if __name__ == "__main__":
    main()
