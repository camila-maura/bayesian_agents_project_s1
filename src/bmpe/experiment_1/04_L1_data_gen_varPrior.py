import numpy as np
import pandas as pd
import os
import argparse

## Get the thing to input to the agents website - input data for variance
## mu_0_est comes from estimate from 03_get_mean_estimate.py and its located in mean_estimate_experiment_1.npy

# ==========================================
# LOAD DATA
# ==========================================
# Source for prior mean estimate
def make_data_var(experiment_n,
                  mean_prio = -0.81,
                  s1vl_start = -4.0, 
                  s1vl_end = 4.0, 
                  s1vl_inc = 0.2, 
                  s1std_start = 0.7, 
                  s1std_end = 1.4, 
                  s2std = 0.005, 
                  simetric = True ):
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
    file_name_output = f"var_inputs_experiment_{experiment_n}.csv"
    output_filename = os.path.join(folder, file_name_output)

    # =====================================
    # CONFIG
    # =====================================

    # Plug in your current prior-mean estimate from the mean experiment
    mu0_est = mean_prio#np.load(input_filename)
    print("Loaded prior mean estimate (μ0):", mu0_est)

    # S1 sweep around prior mean
    s1_min_offset = s1vl_start
    s1_max_offset =  s1vl_end
    s1_step        = s1vl_inc

    s1_min = mu0_est + s1_min_offset
    s1_max = mu0_est + s1_max_offset

    # Reference stimulus: fixed at prior mean, almost noiseless
    s2_val = mu0_est
    s2_std = s2std          # very precise ref → posterior var ~ 0

    # ---- S1 noise levels ----
    # C: drop extremely low-noise blocks → no std < 0.7
    # A: double density in [1.5, 4.0]

    # a few moderate levels (still informative, but not ultra-precise)
    #low_mid = np.geomspace(s1std_start - 0.4 , s1std_start - 0.1, num=4)

    # dense sampling in prior-dominant regime
    dense_prior = np.geomspace(s1std_start, s1std_end, num=30)

    # a few very noisy levels to see asymptote more clearly
    #high = np.geomspace(s1std_end + 0.1, s1std_end + 3, num=4)

    s1_std_series = np.unique(dense_prior)
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
                    "S2_std": float(s1_std if simetric else s2_std),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ex_n", type=int, default=None)
    parser.add_argument("--mean_pri", type=float, default=-0.81)
    parser.add_argument("--starts1", type=float, default=-4.0)
    parser.add_argument("--ends1", type=float, default=4.0)
    parser.add_argument("--incs1", type=float, default=0.1)
    parser.add_argument("--startstds1", type=float, default=0.7)
    parser.add_argument("--endstds1", type=float, default=1.3)
    parser.add_argument("--s2std", type=float, default=0.005)
    parser.add_argument("--simetric", type=bool, default=True)
    args = parser.parse_args()
    if args.ex_n:
        make_data_var(args.ex_n,
                      args.mean_pri, 
                      args.starts1, 
                      args.ends1, 
                      args.incs1, 
                      args.startstds1,
                      args.endstds1,
                      args.s2std,
                      args.simetric)
    else:
        ValueError("Experiment number needs to be defined!")
