import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from concat_mean import concat_mean

# ============================================================
# Bayesian observer model: posterior mean and variance
# ============================================================

def cum_gauss(x, mu, sigma):
    return norm.cdf((x - mu) / sigma)

def model_p_choose_1(s1, s2, sigma1, sigma2, mu_prior, sigma_prior, sigma_internal):
    v1 = sigma1**2 + sigma_internal**2
    v2 = sigma2**2 + sigma_internal**2
    w1 = sigma_prior**2 / (sigma_prior**2 + v1)
    w2 = sigma_prior**2 / (sigma_prior**2 + v2)
    m1 = w1 * s1 + (1 - w1) * mu_prior
    m2 = w2 * s2 + (1 - w2) * mu_prior
    pv1 = 1 / (1/sigma_prior**2 + 1/v1)
    pv2 = 1 / (1/sigma_prior**2 + 1/v2)
    dmu = m1 - m2
    dsigma = np.sqrt(pv1 + pv2)
    return norm.cdf(dmu / dsigma)

# ============================================================
# Negative log-likelihood
# ============================================================

def neg_log_likelihood(params, s1, s2, sigma1, sigma2, P_obs, n_trials, mu_prior):
    sigma_prior, sigma_internal = params
    if sigma_prior <= 0 or sigma_internal <= 0:
        return np.inf
    P_model = model_p_choose_1(s1, s2, sigma1, sigma2, mu_prior, sigma_prior, sigma_internal)
    P_model = np.clip(P_model, 1e-8, 1 - 1e-8)
    k = P_obs * n_trials
    n = n_trials
    ll = k * np.log(P_model) + (n - k) * np.log(1 - P_model)
    return -np.sum(ll)

# ============================================================
# Fit function with confidence intervals
# ============================================================

def fit_prior_MLE_CI(s1, s2, sigma1, sigma2, P_obs, n_trials, mu_prior, alpha=0.05):
    init = np.array([1.0, 0.3])
    res = minimize(
        neg_log_likelihood,
        init,
        args=(s1, s2, sigma1, sigma2, P_obs, n_trials, mu_prior),
        method='L-BFGS-B',
        bounds=[(1e-6, np.inf), (1e-6, np.inf)]
    )
    
    # Approximate covariance from inverse Hessian
    if res.hess_inv is not None:
        # L-BFGS-B returns an LbfgsInvHessProduct object
        try:
            cov = res.hess_inv.todense()  # convert to full matrix
        except:
            cov = res.hess_inv  # fallback
        se = np.sqrt(np.diag(cov))
    else:
        se = np.full_like(res.x, np.nan)
    
    # Confidence intervals
    z = norm.ppf(1 - alpha/2)
    ci_lower = res.x - z * se
    ci_upper = res.x + z * se
    
    return res.x, se, (ci_lower, ci_upper), res.fun, res


def estimate_agent_var(data: pd.DataFrame, mu_prior: float):
    s1 = data["S1_val"].values.astype(float)
    s2 = data["S2_val"].values.astype(float)
    sigma1 = data["S1_std"].values.astype(float)
    sigma2 = data["S2_std"].values.astype(float)
    P_obs = data["P_choose1"].values.astype(float)
    n_trials = data["N_trials"].values.astype(int)

    params, se, ci, nll, res = fit_prior_MLE_CI(s1, s2, sigma1, sigma2, P_obs, n_trials, mu_prior)

    sigma_prior_mle, sigma_internal_mle = params

    return sigma_prior_mle, sigma_internal_mle

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

def find_mu(file):
    df = concat_mean(file)

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
    #print(group_pool.head())

    # initial guesses: threshold ~ where p ~ 0.5
    idx50 = np.argmin(np.abs(p_data - 0.5))
    mu_guess = x_data[idx50]
    # --- Probit ---
    probit_params, probit_aic, probit_p = fit_model(
        x_data, p_data, n_data,
        model=cum_gauss,
        p0=[mu_guess, 1.0],
        bounds=([-np.inf, 1e-4], [np.inf, np.inf])
    )
    mu_probit, sigma_probit = probit_params


    return mu_probit


def estimate_agent(data: pd.DataFrame, mu_prior: float = None):

    sigma_prior_mle, sigma_internal_mle = estimate_agent_var(data, mu_prior)

    return mu_prior, sigma_prior_mle, sigma_internal_mle
