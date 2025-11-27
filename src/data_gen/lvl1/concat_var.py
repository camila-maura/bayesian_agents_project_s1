import pandas as pd
import glob
import os

# ==========================================
# CONFIG
# ==========================================
INPUT_DIR = "test/lvl1_var/test8"
INPUT_PATTERN = "experiment_results-*.csv"

OUTPUT_TRIALS = "combined_trials_var.csv"          # all raw trials
OUTPUT_PSYCHO = "combined_psychometric_data_var_08.csv"    # binned psychometric

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

    # Core columns required
    required_cols = {"S1_val", "S1_std", "S2_val", "S2_std"}
    if not required_cols.issubset(tmp.columns):
        print(f"[WARN] Skipping {f}: missing required columns.")
        continue

    # detect decision column name
    decision_col = None
    for cand in ["Decision (S1>S2)", "Decision", "Choice", "Result"]:
        if cand in tmp.columns:
            decision_col = cand
            break
    if decision_col is None:
        print(f"[WARN] Skipping {f}: no decision column found.")
        continue

    # keep only necessary columns
    tmp = tmp[["S1_val", "S1_std", "S2_val", "S2_std", decision_col]].copy()
    tmp = tmp.rename(columns={decision_col: "Decision"})

    # clean decision data
    tmp["Decision"] = pd.to_numeric(tmp["Decision"], errors="coerce")
    before = len(tmp)
    tmp = tmp.dropna(subset=["Decision"])
    tmp["Decision"] = tmp["Decision"].astype(int)

    if len(tmp) < before:
        print(f"[INFO] Dropped {before - len(tmp)} invalid decisions in {f}")

    df_list.append(tmp)

if not df_list:
    raise RuntimeError("No valid data frames loaded.")

df = pd.concat(df_list, ignore_index=True)

print(f"[INFO] Loaded {len(df)} total trials from {len(df_list)} files.")

# ==========================================
# SAVE RAW TRIALS
# ==========================================
df.to_csv(OUTPUT_TRIALS, index=False)
print(f"[INFO] Saved all combined raw trials -> {OUTPUT_TRIALS}")

# ==========================================
# SANITY CHECKS
# ==========================================
print("\n[INFO] Unique S2_val (should be a single value near μ0):", df["S2_val"].unique())
print("[INFO] Unique S2_std (should be a single tiny value):", df["S2_std"].unique())
print("[INFO] Example S1_std values:", sorted(df["S1_std"].unique())[:10], "...")

# HARD CHECKS
if len(df["S2_std"].unique()) != 1:
    print("\n[WARNING] Multiple S2_std values found. "
          "This breaks the slope-based variance design.")
if len(df["S2_val"].unique()) != 1:
    print("\n[WARNING] Multiple S2_val values found. "
          "S2_val should be fixed at prior mean.")

# ==========================================
# COMPUTE PSYCHOMETRIC POINTS
# For slope-based design:
# We only need P(choose S1) vs (S1_std, S1_val)
# ==========================================
grouped = (
    df.groupby(["S1_std", "S1_val"])["Decision"]
      .agg(P_choose1="mean", N_trials="count")
      .reset_index()
)

# sort for plotting and fitting
grouped = grouped.sort_values(["S1_std", "S1_val"]).reset_index(drop=True)

# ==========================================
# SAVE PSYCHOMETRIC DATA
# ==========================================
grouped.to_csv(OUTPUT_PSYCHO, index=False)

print("\n[INFO] Saved psychometric data ->", OUTPUT_PSYCHO)
print(grouped.head())
print("\n[INFO] Unique S1_std levels:", sorted(grouped["S1_std"].unique()))