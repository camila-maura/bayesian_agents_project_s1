import numpy as np
import pandas as pd
import os

## Get the thing to input to the agents website - input data for variance
## mu_0_est comes from estimate from 03_get_mean_estimate.py and its located in mean_estimate_experiment_1.npy

# ==========================================
# LOAD DATA
# ==========================================
# Source for prior mean estimate
data_dir = "data"
experiment_name = "experiment_1"
source_of_data = "website_output"   
type_of_data_input = "processed"
type_of_var = "mean"
test_n = 2
test = f"test{test_n}"
input_file_name = "mean_outputs_experiment_1.csv"

folder = os.path.join(
    data_dir,
    experiment_name,
    source_of_data,
    type_of_data_input,
    type_of_var,
    test
)

file_name_input = "mean_estimate_experiment_1.npy"

input_filename = os.path.join(folder, file_name_input)

# Source of output of variance experiment data generation
# Output of this will be used to upload to the website
type_of_var = "variance"
source_of_data = "website_input"
folder = os.path.join(
    data_dir,
    experiment_name,
    source_of_data,
    type_of_var,
)
file_name_output = "var_inputs_experiment_1.csv"
output_filename = os.path.join(folder, file_name_output)

# =====================================
# CONFIG
# =====================================

# Plug in your current prior-mean estimate from the mean experiment
mu0_est = np.load(input_filename)
print("Loaded prior mean estimate (μ0):", mu0_est)
mu0_est = -0.8161      # <-- update this if you re-estimate μ0

# S1 sweep around prior mean
s1_min_offset = -4.0
s1_max_offset =  4.0
s1_step        =  0.25

s1_min = mu0_est + s1_min_offset
s1_max = mu0_est + s1_max_offset

# Reference stimulus: fixed at prior mean, almost noiseless
s2_val = mu0_est
s2_std = 0.05          # very precise ref → posterior var ~ 0

# ---- S1 noise levels ----
# C: drop extremely low-noise blocks → no std < 0.7
# A: double density in [1.5, 4.0]

# a few moderate levels (still informative, but not ultra-precise)
low_mid = np.geomspace(0.7, 1.4, num=4)

# dense sampling in prior-dominant regime
dense_prior = np.geomspace(1.5, 3.3, num=20)

# a few very noisy levels to see asymptote more clearly
high = np.geomspace(4.5, 8.0, num=5)

s1_std_series = np.unique(np.concatenate([low_mid, dense_prior, high]))
s1_std_series.sort()

# B: increase trials per S1/S1_std condition
trials_per_s1 = 120        # more trials → tighter slope estimates

shuffle_within_block = True
RNG = np.random.default_rng(42)

# =====================================
# BUILD DESIGN
# =====================================

s1_values = np.arange(s1_min, s1_max + 1e-9, s1_step)

rows = []
for s1_std in s1_std_series:
    for s1_val in s1_values:
        for _ in range(trials_per_s1):
            rows.append({
                "S1_val": float(s1_val),
                "S1_std": float(s1_std),
                "S2_val": float(s2_val),
                "S2_std": float(s2_std),
            })

df = pd.DataFrame(rows)

# Optional: shuffle within each S1_std block
if shuffle_within_block:
    df = (
        df.groupby("S1_std", group_keys=False)
          .apply(lambda g: g.sample(frac=1, random_state=42))
          .reset_index(drop=True)
    )

# Add trial index
df.insert(0, "Trial", np.arange(1, len(df) + 1))

# =====================================
# SAVE
# =====================================

df.to_csv(output_filename, index=False)
print(df.head())
print("\nSaved design to:", output_filename)
print("S1_std levels:", s1_std_series)
print("Number of S1_std levels:", len(s1_std_series))
print("Total trials:", len(df))

#TODO
#problem -> inputting too high and low variances. we need to find "sweet spots"