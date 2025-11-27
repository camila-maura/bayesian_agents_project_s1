import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
import os

# This script uses the concatenated mean obtained from the website to generate the mean estimate for experiment 1.

"data/experiment_1/website_output/processed/mean/test2/mean_outputs_experiment_1.csv"

# ==========================================
# LOAD DATA
# ==========================================
data_dir = "data"
experiment_name = "experiment_1"
source_of_data = "website_output"   
type_of_data_input = "processed"
type_of_var = "mean"
test_n = 2
test = f"test{test_n}"
input_file_name = "mean_outputs_experiment_1.csv"

output_file_name = "mean_estimate_experiment_1.csv"

folder_input = os.path.join(
    data_dir,
    experiment_name,
    source_of_data,
    type_of_data_input,
    type_of_var,
    test
)
input_file = os.path.join(folder_input, input_file_name)

df = pd.read_csv(input_file)
required_cols = {"S1_std", "S2_std", "S1_val", "S2_val", "P_choose1", "N_trials"}

# ==========================================
# FUNCTIONS FOR MEAN ESTIMATE
# ==========================================


def binomial_loglik(k, n, p):
    # avoid log(0)
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)
    return np.sum(k * np.log(p) + (n - k) * np.log(1 - p))


def fit_model(x, p, n, model, p0, bounds):
    # fit on probabilities (least squares)
    params, pcov = curve_fit(model, x, p, p0=p0, bounds=bounds, maxfev=20000)
    p_hat = model(x, *params)
    ll = binomial_loglik(np.round(p * n).astype(int), n, p_hat)
    k_params = len(params)
    aic = 2 * k_params - 2 * ll
    return params, aic, p_hat


def cum_gauss(x, mu, sigma):
    return norm.cdf((x - mu) / sigma)

# df is the combined psychometric data:
# columns expected: S1_val, S1_std, P_choose1, N_trials

# Pool across S1_std: weighted by number of trials
group_pool = (
    df.groupby("S1_val")
      .apply(lambda g: pd.Series({
          "P_choose1": np.average(g["P_choose1"], weights=g["N_trials"]),
          "N_trials":  g["N_trials"].sum()
      }))
      .reset_index()
)

x_data = group_pool["S1_val"].values.astype(float)
p_data = group_pool["P_choose1"].values.astype(float)
n_data = group_pool["N_trials"].values.astype(int)
k_data = np.round(p_data * n_data).astype(int)   # approximate counts


# sanity check: monotonic-ish
print(group_pool.head())

# initial guesses: threshold ~ where p ~ 0.5
idx50 = np.argmin(np.abs(p_data - 0.5))
mu_guess = x_data[idx50]

probit_params, probit_aic, probit_p = fit_model(
    x_data, p_data, n_data,
    model=cum_gauss,
    p0=[mu_guess, 1.0],
    bounds=([-np.inf, 1e-4], [np.inf, np.inf])
)

mu_probit, sigma_probit = probit_params

params = np.array(probit_params)
np.save("params.npy", params)

print(f"Probit:   mu={mu_probit:.4f}, sigma={sigma_probit:.4f}")