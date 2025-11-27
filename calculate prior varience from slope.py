
from math import sqrt


def calculate_prior_variance(sigma_d: float, sigma_s1: float, sigma_s2: float) -> float:

    S = sigma_d ** 2
    S1 = 1 / (sigma_s1 ** 2)
    S2 = 1 / (sigma_s2 ** 2)

    prior_variance = (2 * S) / (2 - (S * (S1 + S2)) + (sqrt((S ** 2) * ((S1 - S2) ** 2) + 4)))

    return 1 / prior_variance

print(calculate_prior_variance(sigma_d=5.9, sigma_s1=2, sigma_s2=8))
