import pandas as pd
import glob
import os



def concat_var_mle(pattern):
  df = pd.read_csv(pattern)

  grouped = (
      df.groupby(["S1_std", "S2_std", "S2_val", "S1_val"])["Decision (S1>S2)"]
        .agg(P_choose1="mean", N_trials="count")
        .reset_index()
  )

  # Sort nicely
  grouped = grouped.sort_values(["S1_std", "S1_val"]).reset_index(drop=True)

  return grouped