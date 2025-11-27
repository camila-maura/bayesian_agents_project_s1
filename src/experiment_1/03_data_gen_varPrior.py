import numpy as np
import pandas as pd

# =====================================
# CONFIG
# =====================================
## Get the thing to input to the agents website - input data for variance
## mu_0_est comes from estimate from notebook experiment

# Plug in your current prior-mean estimate from the mean experiment
mu0_est = -0.8161      # <-- update this if you re-estimate μ0

# S1 sweep around prior mean
S1_MIN_OFFSET = -4.0
S1_MAX_OFFSET =  4.0
S1_STEP        =  0.25

s1_min = mu0_est + S1_MIN_OFFSET
s1_max = mu0_est + S1_MAX_OFFSET

# Reference stimulus: fixed at prior mean, almost noiseless
S2_VAL = mu0_est
S2_STD = 0.05          # very precise ref → posterior var ~ 0

# ---- S1 noise levels ----
# C: drop extremely low-noise blocks → no std < 0.7
# A: double density in [1.5, 4.0]

# a few moderate levels (still informative, but not ultra-precise)
low_mid = np.geomspace(0.7, 1.4, num=4)

# dense sampling in prior-dominant regime
dense_prior = np.geomspace(1.5, 3.3, num=20)

# a few very noisy levels to see asymptote more clearly
high = np.geomspace(4.5, 8.0, num=5)

S1_STD_SERIES = np.unique(np.concatenate([low_mid, dense_prior, high]))
S1_STD_SERIES.sort()

# B: increase trials per S1/S1_std condition
TRIALS_PER_S1 = 120        # more trials → tighter slope estimates

SHUFFLE_WITHIN_BLOCK = True
RNG = np.random.default_rng(42)

OUT_PATH = "ex_data/lvl1/lvl1_var/var_ex7.csv"

# =====================================
# BUILD DESIGN
# =====================================

s1_values = np.arange(s1_min, s1_max + 1e-9, S1_STEP)

rows = []
for s1_std in S1_STD_SERIES:
    for s1_val in s1_values:
        for _ in range(TRIALS_PER_S1):
            rows.append({
                "S1_val": float(s1_val),
                "S1_std": float(s1_std),
                "S2_val": float(S2_VAL),
                "S2_std": float(S2_STD),
            })

df = pd.DataFrame(rows)

# Optional: shuffle within each S1_std block
if SHUFFLE_WITHIN_BLOCK:
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

df.to_csv(OUT_PATH, index=False)
print(df.head())
print("\nSaved design to:", OUT_PATH)
print("S1_std levels:", S1_STD_SERIES)
print("Number of S1_std levels:", len(S1_STD_SERIES))
print("Total trials:", len(df))

#problem -> inputting too high and low variances. we need to find "sweet spots"