import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
import os

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


def logistic(x, mu, s):
    # classic logistic psychometric
    return 1.0 / (1.0 + np.exp(-(x - mu) / s))