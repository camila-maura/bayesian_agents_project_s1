
from math import sqrt

def calculate_prior_variance(sigma_d: float, sigma_s1: float, sigma_s2: float, noise_std: float = 0.35) -> float:

    S = sigma_d**2
    S1 = 1 / ((sigma_s1 + noise_std) ** 2)
    S2 = 1 / ((sigma_s2 + noise_std) ** 2)

    prior_variance = (2 * S) / (2 - (S * (S1 + S2)) + (sqrt((S ** 2) * ((S1 - S2) ** 2) + 4)))

    return sqrt(prior_variance)

print(calculate_prior_variance(sigma_d=5.9, sigma_s1=2, sigma_s2=8))
