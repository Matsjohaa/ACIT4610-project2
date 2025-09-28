import sys, re
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# -------- metrics (2D minimization) --------
def nondominated(P: np.ndarray) -> np.ndarray:
    if len(P) == 0:
        return np.zeros(0, dtype=bool)
    keep = np.ones(len(P), dtype=bool)
    for i, p in enumerate(P):
        if not keep[i]:
            continue
        dom = np.all(P <= p, axis=1) & np.any(P < p, axis=1)
        dom[i] = False
        keep[dom] = False
    return keep

def gd(P, PF):
    if len(P) == 0 or len(PF) == 0: return float("inf")
    d = np.min(np.linalg.norm(P[:, None, :] - PF[None, :, :], axis=2), axis=1)
    return float(np.mean(d))

def igd(P, PF):
    if len(P) == 0 or len(PF) == 0: return float("inf")
    d = np.min(np.linalg.norm(PF[:, None, :] - P[None, :, :], axis=2), axis=1)
    return float(np.mean(d))

def hypervolume_2d(P, ref):
    if len(P) == 0: return 0.0
    P = P[P[:, 0].argsort()]
    hv, prev_f1, cur = 0.0, ref[0], ref[1]
    for f1, f2 in P[::-1]:
        hv += max(0.0, prev_f1 - f1) * max(0.0, cur - f2)
        prev_f1 = f1
        cur = min(cur, f2)
    return float(hv)

def slugify(s: str) -> str:
    s = re.sub(r"\s+", "-", str(s).strip())
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("._-").lower()

def main():
    # Inputs
    data_csv = PROJECT_ROOT / "src" / "results" / "fronts_long.csv"
    data_pq  = PROJECT_ROOT / "src" / "results" / "fronts_long.parquet"

    if data_pq.exists():
        df = pd.read_parquet(data_pq)
        print(f"[io] Loaded: {data_pq}")
    elif data_csv.exists():
        df = pd.read_csv(data_csv)
        print(f"[io] Loaded: {data_csv}")
    else:
        sys.exit("No fronts_long.csv or fronts_long.parquet found in src/results.")

    # Column mapping (robust to variations)
    cols = {c.lower(): c for c in df.columns}
    inst_col   = cols.get("instance") or cols.get("instance_name") or cols.get("instanceid") or cols.get("instance id")
    preset_col = cols.get("preset")
    algo_col   = cols.get("algorithm") or cols.get("algo")
    seed_col   = cols.get("seed")
    f1_col     = cols.get("f1_distance") or cols.get("f1") or list(df.columns)[0]
    f2_col     = cols.get("f2_balance_std") or cols.get("f2") or list(df.columns)[1]

    if inst_col is None:   sys.exit("Missing instance column (expected: instance / instance_name).")
    if preset_col is None: sys.exit("Missing 'preset' column.")
    if algo_col is None:   sys.exit("Missing algorithm column (expected: algorithm).")
    if seed_col is None:
        df["seed"] = pd.NA
        seed_col = "seed"

    # --- CLEANING STEP: robust removal of incorrect / extreme rows (all presets) ---
    # 1) Coerce objectives to numeric
    df[f1_col] = pd.to_numeric(df[f1_col], errors="coerce")
    df[f2_col] = pd.to_numeric(df[f2_col], errors="coerce")

    # 2) Drop rows with missing key fields
    total_before = len(df)
    must_have = [inst_col, preset_col, f1_col, f2_col]
    df = df.dropna(subset=must_have)

    # 3) Drop non-finite values (NaN/inf) and impossible negatives
    is_finite = np.isfinite(df[f1_col]) & np.isfinite(df[f2_col])
    nonfinite_dropped = int((~is_finite).sum())
    df = df[is_finite]
    neg_mask = (df[f1_col] < 0) | (df[f2_col] < 0)
    negatives_dropped = int(neg_mask.sum())
    if negatives_dropped:
        df = df[~neg_mask]

    # 4) Trim huge outliers per (instance, preset) group using conservative fences
    def _trim_group(g: pd.DataFrame) -> pd.DataFrame:
        if g.empty:
            return g
        q1_f1, q3_f1 = g[f1_col].quantile([0.25, 0.75])
        q1_f2, q3_f2 = g[f2_col].quantile([0.25, 0.75])
        iqr_f1 = q3_f1 - q1_f1
        iqr_f2 = q3_f2 - q1_f2

        hi_f1 = q3_f1 + 3.0 * (iqr_f1 if iqr_f1 > 0 else 0.0)
        hi_f2 = q3_f2 + 3.0 * (iqr_f2 if iqr_f2 > 0 else 0.0)
        if iqr_f1 == 0:
            hi_f1 = min(hi_f1, g[f1_col].quantile(0.999))
        if iqr_f2 == 0:
            hi_f2 = min(hi_f2, g[f2_col].quantile(0.999))
        return g[(g[f1_col] <= hi_f1) & (g[f2_col] <= hi_f2)]

    df_trimmed = (
        df.groupby([inst_col, preset_col], group_keys=False)
          .apply(_trim_group)
          .reset_index(drop=True)
    )
    outliers_dropped = len(df) - len(df_trimmed)
    df = df_trimmed

    total_dropped = total_before - len(df)
    if total_dropped > 0:
        print(
            f"[clean] Dropped rows: total={total_dropped} "
            f"(nonfinite={nonfinite_dropped}, negatives={negatives_dropped}, outliers={outliers_dropped})."
        )

    # 5) Normalize seed to int where possible (avoids crashes on strings)
    #    Keep None when missing.
    def _seed_to_int(x):
        if pd.isna(x): return None
        try:
            return int(pd.to_numeric(x))
        except Exception:
            return None
    df[seed_col] = df[seed_col].map(_seed_to_int)

    # Build reference PF★ per (instance, preset)
    ref_by_inst_preset = {}
    for (inst, preset), g in df.groupby([inst_col, preset_col]):
        A = g[[f1_col, f2_col]].to_numpy(dtype=float)
        PF = A[nondominated(A)] if len(A) else np.zeros((0, 2), dtype=float)
        ref_by_inst_preset[(inst, preset)] = PF

    # Score each run (instance,preset,algorithm,seed) vs PF★ of its (instance,preset)
    rows = []
    for (inst, preset, algo, seed), g in df.groupby([inst_col, preset_col, algo_col, seed_col]):
        PF = ref_by_inst_preset.get((inst, preset), np.zeros((0, 2), dtype=float))
        if len(PF) == 0:
            continue
        worst = np.max(PF, axis=0)
        hv_ref = (float(worst[0] * 1.05), float(worst[1] * 1.05))
        P = g[[f1_col, f2_col]].to_numpy(dtype=float)
        gdm = gd(P, PF)
        igdm = igd(P, PF)
        hv   = hypervolume_2d(P, hv_ref)
        spr = 0.0
        if len(P) >= 2:
            Psort = P[P[:, 0].argsort()]
            gaps = np.linalg.norm(np.diff(Psort, axis=0), axis=1)
            spr = float(np.std(gaps))
        rows.append(dict(
            instance=str(inst),
            algorithm=str(algo).upper(),
            preset=str(preset),
            seed=seed,
            gd_fixed=gdm,
            igd_fixed=igdm,
            hv_fixed=hv,
            spread_proxy=spr,
        ))

    df_scores = pd.DataFrame(rows)

    metric_cols = ["gd_fixed", "igd_fixed", "hv_fixed", "spread_proxy"]

    # keep only rows where all metrics are finite, >0, and <= 1e9
    finite_mask = np.logical_and.reduce([np.isfinite(df_scores[c]) for c in metric_cols])
    gt0_mask    = np.logical_and.reduce([df_scores[c] > 0 for c in metric_cols])
    cap_mask    = np.logical_and.reduce([df_scores[c] <= 1_000_000_000 for c in metric_cols])

    before = len(df_scores)
    df_scores = df_scores[finite_mask & gt0_mask & cap_mask].reset_index(drop=True)
    removed = before - len(df_scores)
    if removed:
        print(f"[filter] Removed {removed} rows with zero/inf or >1e9 metrics before writing.")


    # Outputs: one CSV per preset, named metrics_auto_<preset>.csv in src/results/
    out_dir = PROJECT_ROOT / "src" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if df_scores.empty:
        # still emit empty markers for traceability
        seen_presets = sorted({p for _, p in ref_by_inst_preset.keys()})
        for preset in seen_presets:
            fname = out_dir / f"metrics_auto_{slugify(preset)}.csv"
            fname.write_text(
                "instance,algorithm,preset,seed,gd_fixed,igd_fixed,hv_fixed,spread_proxy\n",
                encoding="utf-8",
            )
            print(f"[write-empty] {fname}")
        print(f"No runs scored. Wrote empty headers in {out_dir}.")
        return

    # Write per-preset files and echo absolute paths
    for preset, dfp in df_scores.groupby("preset"):
        fname = out_dir / f"metrics_auto_{slugify(preset)}.csv"
        dfp.to_csv(fname, index=False)
        print(f"[write] {fname.resolve()}  rows={len(dfp)}")

    print(f"[done] Metrics per preset written under: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
