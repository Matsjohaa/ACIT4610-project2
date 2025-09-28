from pathlib import Path
import pandas as pd
import re

# --------- CONFIG ---------

BASE_DIR = Path("/Users/kristina/Dropbox/Mac/Desktop/ACIT4610-project2/ACIT4610-project2/visualisations/fronts")



INSTANCE_TO_SIZE = {
    # small
    "A-n32-k5": "small",
    "A-n36-k5": "small",
    # medium
    "B-n45-k6": "medium",
    "B-n57-k7": "medium",
    # large
    "E-n101-k14": "large",
    "M-n151-k12": "large",
}

ALGORITHMS = {"nsga2", "vega"}  # as given
PRESETS = {"fast", "balanced", "thorough"}

# filename pattern: <algorithm>_<preset>_seed<seed>.parquet
FNAME_RE = re.compile(r"^(?P<algorithm>[^_]+)_(?P<preset>[^_]+)_seed(?P<seed>\d{1,})\.parquet$", re.IGNORECASE)
# --------------------------

def parse_meta_from_filename(fname: str):
    m = FNAME_RE.match(fname)
    if not m:
        raise ValueError(f"Unexpected filename format: {fname}")
    alg = m.group("algorithm").lower()
    preset = m.group("preset").lower()
    seed = int(m.group("seed"))
    if alg not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{alg}' in {fname}")
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}' in {fname}")
    return alg, preset, seed

def load_front_file(fp: Path, instance_name: str, instance_size: str) -> pd.DataFrame:
    alg, preset, seed = parse_meta_from_filename(fp.name)
    df = pd.read_parquet(fp)

    # Normalize expected columns: drop '#' if present; ensure correct names and dtypes
    # Expected inside each file: columns '#', 'f1_distance', 'f2_balance_std'
    cols = {c.lower(): c for c in df.columns}
    # Some parquet writers might preserve case; standardize access:
    f1_col = next((c for c in df.columns if c.lower() == "f1_distance"), None)
    f2_col = next((c for c in df.columns if c.lower() == "f2_balance_std"), None)
    if f1_col is None or f2_col is None:
        raise ValueError(f"Missing f1/f2 columns in {fp}: found {list(df.columns)}")

    # Build long-form rows (one per point in the front)
    out = pd.DataFrame({
        "algorithm": alg,
        "Instance_size": instance_size,
        "instance_name": instance_name,
        "preset": preset,
        "seed": seed,
        "f1_distance": pd.to_numeric(df[f1_col], errors="coerce"),
        "f2_balance_std": pd.to_numeric(df[f2_col], errors="coerce"),
    })

    # Optional: drop NaN rows just in case
    out = out.dropna(subset=["f1_distance", "f2_balance_std"]).reset_index(drop=True)
    return out

def build_long_table(base_dir: Path = BASE_DIR) -> pd.DataFrame:
    base_dir = Path(base_dir)
    all_rows = []
    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")
    print("[INFO] Using base_dir:", base_dir.resolve())
    for inst_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        instance_name = inst_dir.name
        if instance_name not in INSTANCE_TO_SIZE:
            # skip unknown instance folders (or raise if you prefer strictness)
            continue
        instance_size = INSTANCE_TO_SIZE[instance_name]

        for fp in sorted(inst_dir.glob("*.parquet")):
            try:
                all_rows.append(load_front_file(fp, instance_name, instance_size))
            except Exception as e:
                # If you prefer to fail hard, replace with: raise
                print(f"[WARN] Skipping {fp}: {e}")

    if not all_rows:
        raise RuntimeError(f"No parquet files found under {base_dir}")

    #df_long = pd.concat(all_rows, ignore_index=True)
    df_long = pd.concat(all_rows, ignore_index=True).drop_duplicates()
    # Ensure column order
    df_long = df_long[[
        "algorithm", "Instance_size", "instance_name",
        "preset", "seed", "f1_distance", "f2_balance_std"
    ]]
    return df_long

if __name__ == "__main__":
    df = build_long_table(BASE_DIR)# root containing instance subfolders


    # Save for downstream analysis
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "fronts_long.parquet", index=False)
    df.to_csv(out_dir / "fronts_long.csv", index=False)

    # Quick sanity print
    print(df.head())
    print("Rows:", len(df), "| Unique files:", df.groupby(["instance_name","algorithm","preset","seed"]).ngroups)
