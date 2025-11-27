import pandas as pd
import numpy as np

# === PARAMETERS ===
s1_min, s1_max, s1_step = -4.0, 4.0, 0.1
S1_std = 0.5            # precise cue
S2_val = 0.0            # reference
S2_std = 8.0            # very noisy cue (≈ prior-only)
trials_per_s1 = 50

# === GENERATE S1 VALUES ===
S1_vals = np.arange(s1_min, s1_max + 1e-9, s1_step)
S1_std_vals = np.arange(0.1,0.9,0.1)
rows = []
trial_id = 1
for s1_std in S1_std_vals:
    for s1 in S1_vals:
        for _ in range(trials_per_s1):
            rows.append({
                "Trial": trial_id,
                "S1_val": float(s1),
                "S1_std": float(s1_std),
                "S2_val": float(S2_val),
                "S2_std": float(S2_std),
            })
            trial_id += 1

df = pd.DataFrame(rows)

print(df.head())
print(df.dtypes)

df.to_csv("mean_bayesian_experiment1_conditions.csv", index=False)