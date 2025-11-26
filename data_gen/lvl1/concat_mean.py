import pandas as pd
import glob
import os

# ==========================================
# CONFIG
# ==========================================
INPUT_DIR = "test/lvl1_mean/test2"   # change if needed
INPUT_PATTERN = "experiment_results-*.csv"
OUTPUT_FILE = "combined_psychometric_data_mean.csv"

# ==========================================
# FIND AND LOAD ALL EXPERIMENT FILES
# ==========================================
pattern = os.path.join(INPUT_DIR, INPUT_PATTERN)
files = glob.glob(pattern)

if len(files) == 0:
    raise FileNotFoundError(f"No files matching {pattern!r} found.")

df_list = []
for f in files:
    try:
        tmp = pd.read_csv(f)
    except Exception as e:
        print(f"[WARN] Could not read {f}: {e}")
        continue

    # Basic sanity: must have at least the core columns
    required_cols = {"S1_val", "S1_std", "S2_val", "S2_std"}
    if not required_cols.issubset(tmp.columns):
        print(f"[WARN] Skipping {f}: missing some of {required_cols}")
        continue

    # Robustly handle decision column naming
    decision_col = None
    for cand in ["Decision (S1>S2)", "Decision", "Choice", "Result"]:
        if cand in tmp.columns:
            decision_col = cand
            break
    if decision_col is None:
        print(f"[WARN] Skipping {f}: no decision column found.")
        continue

    # Keep only what we need, rename decision
    tmp = tmp[["S1_val", "S1_std", "S2_val", "S2_std", decision_col]].copy()
    tmp = tmp.rename(columns={decision_col: "Decision"})

    # Cast decisions to int 0/1 (drop rows that fail)
    tmp["Decision"] = pd.to_numeric(tmp["Decision"], errors="coerce")
    before = len(tmp)
    tmp = tmp.dropna(subset=["Decision"])
    tmp["Decision"] = tmp["Decision"].astype(int)
    if len(tmp) < before:
        print(f"[INFO] Dropped {before - len(tmp)} invalid decision rows in {f}")

    df_list.append(tmp)

if not df_list:
    raise RuntimeError("No valid data frames were loaded; check warnings above.")

df = pd.concat(df_list, ignore_index=True)

print(f"[INFO] Loaded {len(df)} total trials from {len(df_list)} files.")

# ==========================================
# OPTIONAL SANITY CHECKS FOR THIS SETUP
# ==========================================
print("\n[INFO] Unique S2_val:", df["S2_val"].unique())
print("[INFO] Unique S2_std:", df["S2_std"].unique())
print("[INFO] Example S1_std values:", sorted(df["S1_std"].unique())[:10], "...")

# ==========================================
# COMPUTE PSYCHOMETRIC POINTS
# For mean estimation, we care about P(choose S1)
# per (S1_std, S1_val). We also keep S2_* for completeness.
# ==========================================
grouped = (
    df.groupby(["S1_std", "S2_std", "S2_val", "S1_val"])["Decision"]
      .agg(P_choose1="mean", N_trials="count")
      .reset_index()
)

# Sort nicely
grouped = grouped.sort_values(["S1_std", "S1_val"]).reset_index(drop=True)

# ==========================================
# SAVE OUTPUT
# ==========================================
grouped.to_csv(OUTPUT_FILE, index=False)

print("\n[INFO] Saved combined psychometric data to:", OUTPUT_FILE)
print(grouped.head())
print("\n[INFO] Unique S1_std in combined data:", sorted(grouped["S1_std"].unique()))