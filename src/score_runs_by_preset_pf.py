import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# -------- metrics (2D minimization) --------
def nondominated(P: np.ndarray) -> np.ndarray:
    if len(P) == 0: return np.zeros(0, dtype=bool)
    keep = np.ones(len(P), dtype=bool)
    for i, p in enumerate(P):
        if not keep[i]: continue
        dom = np.all(P <= p, axis=1) & np.any(P < p, axis=1)
        dom[i] = False
        keep[dom] = False
    return keep

def gd(P, PF):
    if len(P)==0 or len(PF)==0: return np.inf
    d = np.min(np.linalg.norm(P[:,None,:]-PF[None,:,:], axis=2), axis=1)
    return float(np.mean(d))

def igd(P, PF):
    if len(P)==0 or len(PF)==0: return np.inf
    d = np.min(np.linalg.norm(PF[:,None,:]-P[None,:,:], axis=2), axis=1)
    return float(np.mean(d))

def hypervolume_2d(P, ref):
    if len(P)==0: return 0.0
    P = P[P[:,0].argsort()]
    hv, prev_f1, cur = 0.0, ref[0], ref[1]
    for f1, f2 in P[::-1]:
        hv += max(0.0, prev_f1 - f1) * max(0.0, cur - f2)
        prev_f1 = f1
        cur = min(cur, f2)
    return float(hv)

def _load_csv_front(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    return arr.astype(float)

def _load_parquet_front(path: Path) -> np.ndarray:
    df = pd.read_parquet(path)
    # accept common column names; rename to canonical if needed
    cols = {c.lower(): c for c in df.columns}
    f1 = cols.get("f1_distance") or cols.get("f1") or cols.get("distance") or df.columns[0]
    f2 = cols.get("f2_balance_std") or cols.get("f2") or cols.get("balance") or df.columns[1]
    A = df[[f1, f2]].to_numpy(dtype=float)
    if A.ndim == 1: A = A.reshape(1, -1)
    return A

def load_front(path: Path) -> np.ndarray:
    try:
        if path.suffix.lower() == ".parquet":
            return _load_parquet_front(path)
        # fallback to CSV
        return _load_csv_front(path)
    except Exception:
        return np.zeros((0,2), dtype=float)

def parse_name(stem: str):
    # expected: <algo>_<preset>_seed<NNNN>
    parts = stem.split("_")
    if len(parts) < 3 or "seed" not in parts[-1]:
        return None, None, None
    algo = parts[0].upper()
    preset = "_".join(parts[1:-1]).lower()  # supports names like "very_thorough"
    try:
        seed = int(parts[-1].replace("seed",""))
    except ValueError:
        seed = None
    return algo, preset, seed

# -------- main --------
def main():
    import sys, re

    def slugify(s: str) -> str:
        s = re.sub(r"\s+", "-", s.strip())
        s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("._-")

    fronts_base = PROJECT_ROOT / "results" / "fronts"
    if not fronts_base.exists():
        sys.exit(f"No fronts found in {fronts_base}. Run the experiment runner first.")

    # ---- discover all presets present in results/fronts/<instance>/*.parquet|csv
    discovered = {}  # instance -> {preset -> [Path,...]}
    for inst_dir in sorted([p for p in fronts_base.iterdir() if p.is_dir()]):
        for fp in sorted(list(inst_dir.glob("*_seed*.parquet")) + list(inst_dir.glob("*_seed*.csv"))):
            algo, preset, seed = parse_name(fp.stem)
            if algo is None or preset is None:
                continue
            discovered.setdefault(inst_dir.name, {}).setdefault(preset, []).append(fp)

    if not discovered:
        sys.exit("No matching <algo>_<preset>_seed*.parquet|csv files found under results/fronts.")

    # ---- build PF★ per (instance, preset) using ALL algos for that preset
    ref_by_inst_preset = {}  # (instance, preset) -> PF★ ndarray
    for inst, by_preset in discovered.items():
        for preset, files in by_preset.items():
            pts = [load_front(p) for p in files]
            if any(len(a) > 0 for a in pts):
                U = np.vstack([a for a in pts if len(a) > 0])
                PF = U[nondominated(U)]
            else:
                PF = np.zeros((0, 2), dtype=float)
            ref_by_inst_preset[(inst, preset)] = PF

    # ---- score every run vs its preset PF★
    rows = []
    for inst, by_preset in discovered.items():
        for preset, files in by_preset.items():
            PF = ref_by_inst_preset.get((inst, preset), np.zeros((0, 2), dtype=float))
            if len(PF) == 0:
                # nothing to compare against for this (inst,preset); skip scoring
                continue
            worst = np.max(PF, axis=0)
            hv_ref = (float(worst[0] * 1.05), float(worst[1] * 1.05))

            for fp in files:
                algo, preset_name, seed = parse_name(fp.stem)
                P = load_front(fp)
                g = gd(P, PF)
                ig = igd(P, PF)
                hv = hypervolume_2d(P, hv_ref)
                spr = 0.0
                if len(P) >= 2:
                    Psort = P[P[:, 0].argsort()]
                    gaps = np.linalg.norm(np.diff(Psort, axis=0), axis=1)
                    spr = float(np.std(gaps))
                rows.append(dict(
                    instance=inst,
                    algorithm=algo,
                    preset=preset_name,
                    seed=seed,
                    gd_fixed=g,
                    igd_fixed=ig,
                    hv_fixed=hv,
                    spread_proxy=spr,
                ))

    df = pd.DataFrame(rows)

    # ---- write outputs grouped by preset
    out_root = PROJECT_ROOT / "results" / "grouped"
    out_root.mkdir(parents=True, exist_ok=True)

    if df.empty:
        # produce empty groups for traceability
        # also handy when some presets had no PF★ at all
        all_presets = sorted({p for _, p in ref_by_inst_preset.keys()})
        for preset in all_presets:
            out_dir = out_root / slugify(preset)
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["instance","algorithm","preset","seed","gd_fixed","igd_fixed","hv_fixed","spread_proxy"]) \
              .to_csv(out_dir / "metrics_auto.csv", index=False)
            pd.DataFrame(columns=[
                "instance algorithm preset",
                "gd_fixed mean","gd_fixed std",
                "igd_fixed mean","igd_fixed std",
                "hv_fixed mean","hv_fixed std",
                "spread_proxy mean","spread_proxy std",
            ]).to_csv(out_dir / "summary_auto.csv", index=False)
        print("No runs scored (df was empty). Wrote empty CSVs per preset.")
        return

    for preset, dfp in df.groupby("preset"):
        out_dir = out_root / slugify(preset)
        out_dir.mkdir(parents=True, exist_ok=True)
        dfp.to_csv(out_dir / "metrics_auto.csv", index=False)

        agg = (
            dfp.groupby(["instance", "algorithm", "preset"])[["gd_fixed", "igd_fixed", "hv_fixed", "spread_proxy"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg.columns = [" ".join(c).strip() if isinstance(c, tuple) else c for c in agg.columns]
        agg.to_csv(out_dir / "summary_auto.csv", index=False)

    print(f"Wrote grouped outputs under: {out_root}")

if __name__ == "__main__":
    main()
