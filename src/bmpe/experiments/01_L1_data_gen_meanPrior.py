import pandas as pd
import numpy as np
import os
import argparse
# LEVEL 1
# This script computes the input data for the mean prior experiment on the agents website

# ==========================================
# CONFIG
# ==========================================


def make_data_experiment_4(
    # We will fix the reference cue (S2) to be very noisy.
    # This will ensure that the agent relies mostly on its prior.
    s2_vals_set = 0.0, 
    s1_vals_min=-4.0,
    s1_vals_max=4.0,
    s1_vals_step = 0.1,
    trials_per_s1_value = 50,
    s2_std_vals_set = 8,
    s1_std_vals_set = 2 
    
):
    '''
    Prior parameter estimation experiment
    '''
    s1_vals_set = np.arange(
        s1_vals_min,
        s1_vals_max,
        s1_vals_step
    )
    n_trials_total = (
        (s1_vals_max - s1_vals_min)/0.1
        ) * trials_per_s1_value
    
    s1_vals = np.repeat(
        s1_vals_set,
        trials_per_s1_value
    )
    s2_vals = np.repeat(
        s2_vals_set,
        n_trials_total
    )
    s1_std_vals = np.repeat(
        s1_std_vals_set,
        n_trials_total
    )
    s2_std_vals = np.repeat(
        s2_std_vals_set,
        n_trials_total
    )
    return s1_vals, s2_vals, s1_std_vals, s2_std_vals

def concat_data(s1_vals, s2_vals, s1_std_vals, s2_std_vals):
    # Store in dataframe
    trial_ids = np.arange(0, len(s1_vals))
    exp_dic = {
        "Trial": trial_ids,
        "S1_val": s1_vals,
        "S2_val": s2_vals,
        "S1_std": s1_std_vals,
        "S2_std": s2_std_vals,
    }
    
    exp_df = pd.DataFrame(exp_dic)
    return exp_df
    

def make_data(
    experiment_n, 
    start=-4.0, 
    end=4.0, 
    inc=0.1
):
    # === REFERENCE ===
    # We will fix the reference cue (S2) to be very noisy.
    # This will ensure that the agent relies mostly on its prior.
    S2_val = 0.0            # reference
    S2_std = 8.0            # very noisy cue (≈ prior-only)

    # === GENERATE S1 VALUES ===
    trials_per_s1 = 50
    s1_min, s1_max, s1_step = start, end, inc
    s1_vals = np.arange(s1_min, s1_max + 1e-9, s1_step)

    s1_std_vals = np.arange(0.1,0.9,0.1)
    data_dir = "data"
    experiment_name = f"experiment_{experiment_n}"
    type_of_data = "website_input"
    type_of_var = "mean"
    output_file_name = f"mean_inputs_{experiment_name}.csv"

    folder = os.path.join(data_dir, experiment_name, type_of_data, type_of_var)

    rows = []
    trial_id = 1
    for val_std in s1_std_vals:
        for val_obs in s1_vals:
            for _ in range(trials_per_s1):
                rows.append({
                    "Trial": trial_id,
                    "S1_val": float(val_obs),
                    "S1_std": float(val_std),
                    "S2_val": float(S2_val),
                    "S2_std": float(S2_std),
                })
                trial_id += 1

    df = pd.DataFrame(rows)

    print(df.head())
    print(df.dtypes)

    df.to_csv(folder + "/" + output_file_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ex_n", type=int, default=None)
    parser.add_argument("--start", type=float, default=-4.0)
    parser.add_argument("--end", type=float, default=4.0)
    parser.add_argument("--inc", type=float, default=0.1)
    args = parser.parse_args()
    if args.ex_n:
        make_data(args.ex_n, args.start, args.end, args.inc)
    else:
        ValueError("Experiment number needs to be defined!")