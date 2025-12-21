import pandas as pd
import numpy as np
import os
import argparse

# This script generates all experiments' data

# ==========================================
# CONFIG
# ==========================================

def make_data_experiment_4(
    # We will fix the reference cue (S2) to be very noisy.
    # This will ensure that the agent relies mostly on its prior.
    s2_vals_set=0.0, 
    s1_vals_min=-4.0,
    s1_vals_max=4.0,
    s1_vals_step=0.1,
    trials_per_s1_value=50,
    s2_std_vals_set=8,
    s1_std_vals_set=2 
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
    return (s1_vals, s2_vals, s1_std_vals, s2_std_vals)

def concat_data(exp_data):
    # Store in dataframe
    s1_vals, s2_vals, s1_std_vals, s2_std_vals = exp_data
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
   
def save_experiment(
    exp_df,
    experiment_n,
    type_of_data,
    type_of_var,
    data_dir="data/experiments"
):
    experiment_name = f"experiment_{experiment_n}"
    output_file_name = f"{type_of_var}_{type_of_data}_{experiment_name}.csv"

    exp_df.to_csv(
        (data_dir + "/" + 
         type_of_data + "/" + 
         type_of_var + "/" + 
         output_file_name
         ), index=False)
    return None

def run_and_store_exp(
    exp_gen_func,
    experiment_n,
    type_of_data,
    type_of_var 
):
    exp_data = exp_gen_func()
    exp_df = concat_data(exp_data)
    save_experiment(exp_df, experiment_n, type_of_data, type_of_var)
    return None

if __name__ == "__main__":
    run_and_store_exp(
       make_data_experiment_4,
       4,
       "website_input",
       "mean" 
    )
    