import pandas as pd
import glob
import os


# To run this, the output of 01_data_gen_meanPrior.py must be available. It has to be uploaded to the website and the results downloaded. Results should be saved in the folder corresponding to the variable. In this case, the variable is mean
"data/experiment_1/website_output/raw/mean"

# add test number with following nomenclature:
# test = f"test{n}"

# were n is the test number
"data/experiment_1/website_output/raw/mean/test{n}"

# For example, for test 10, the path is
"data/experiment_1/website_output/raw/mean/test10"

# ==========================================
# CONFIG
# ==========================================
data_dir = "data"
experiment_name = "experiment_1"
source_of_data = "website_output"

type_of_data_input = "raw"
type_of_var = "mean"
test_n = 2
test = f"test{test_n}"
input_pattern = "experiment_results-*.csv"

type_of_data_output = "processed"
output_file_name = "mean_outputs_experiment_1.csv"

# ==========================================
# FIND AND LOAD ALL EXPERIMENT FILES
# ==========================================

folder_input = os.path.join(
    data_dir, 
    experiment_name, 
    source_of_data, 
    type_of_data_input, 
    type_of_var,
    test
)

pattern = os.path.join(folder_input, input_pattern)
files = glob.glob(pattern)

folder_output = os.path.join(
    data_dir, 
    experiment_name, 
    source_of_data, 
    type_of_data_output, 
    type_of_var,
    test
)

output_file = os.path.join(folder_output, output_file_name)

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
grouped.to_csv(output_file, index=False)

print("\n[INFO] Saved combined psychometric data to:", output_file)
print(grouped.head())
print("\n[INFO] Unique S1_std in combined data:", sorted(grouped["S1_std"].unique()))

# output combined psychometric data for mean experiment
# with that, we run level 1 to get mean estimation
# with thant we go to var prior -> input for website
# then concat var with website output