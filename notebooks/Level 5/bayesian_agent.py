import numpy as np
from scipy.stats import norm
import math

def double_factorial(n: int) -> int:
    """Compute n!! = 1*3*5*...*n for odd n, or 2*4*6*...*n for even n"""
    if n <= 0:
        return 1
    else:
        return n * double_factorial(n - 2)

def var_Xn_zero_mean(sigma: float, n: int) -> float:
    """
    Variance of X^n for X ~ N(0, sigma^2)
    """
    if n == 0:
        return 0.0
    # nth raw moment
    if n % 2 == 0:  # even n
        E_Xn = double_factorial(n - 1) * sigma**n
    else:           # odd n
        E_Xn = 0.0
    # 2n-th raw moment
    E_X2n = double_factorial(2*n - 1) * sigma**(2*n)
    return E_X2n - E_Xn**2

class BayesianReplica():
    def __init__(self, mu_prior, sigma_prior, sigma_internal, eks_s = 0, P_init=1.0):
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior
        self.sigma_internal = sigma_internal
        self.P = P_init
        self.safety_mu = mu_prior
        self.safety_sigma = sigma_prior
        self.safety_P = P_init
        self.eks_s = eks_s
        self.P_sigma = P_init**2

    def decision_task(self, s1, s2, sigma1, sigma2):
        v1 = sigma1**2 + self.sigma_internal**2
        v2 = sigma2**2 + self.sigma_internal**2

        w1 = self.sigma_prior**2 / (self.sigma_prior**2 + v1)
        w2 = self.sigma_prior**2 / (self.sigma_prior**2 + v2)

        m1 = w1 * s1 + (1 - w1) * self.mu_prior
        m2 = w2 * s2 + (1 - w2) * self.mu_prior

        pv1 = 1 / (1/self.sigma_prior**2 + 1/v1)
        pv2 = 1 / (1/self.sigma_prior**2 + 1/v2)

        dmu = m1 - m2
        dsigma = np.sqrt(pv1 + pv2)

        P_choose_s1 = norm.cdf(dmu / dsigma)
        return P_choose_s1
    

    def train_agent_kalmann_m_noise(self, n_trials, final_mean, final_std, measurement_noise):
        sample = np.random.normal(final_mean, final_std, n_trials)
        measurement = sample + np.random.normal(0, measurement_noise, n_trials)
        mean_list = []
        var_list = []
        for obs in measurement:
            obs_var = (self.sigma_internal ** 2 + measurement_noise ** 2)
            kalman_gain = self.P / (self.P + obs_var)

            self.mu_prior = self.mu_prior + kalman_gain * (obs - self.mu_prior)
            self.sigma_prior = np.sqrt((1 - kalman_gain) * (self.sigma_prior ** 2))
            self.P = (1 - kalman_gain) * self.P
            mean_list.append(self.mu_prior)
            var_list.append(self.sigma_prior)
        
        return mean_list, var_list 
    
    def train_agent_kalmann_pred_error(self, n_trials, final_mean, final_std, measurement_noise):
        sample = np.random.normal(final_mean, final_std, n_trials)
        measurement = sample + np.random.normal(0, measurement_noise, n_trials)
        mean_list = []
        var_list = []
        for obs in measurement:
            obs_var = (self.sigma_internal ** 2 + measurement_noise ** 2)
            kalman_gain = self.P / (self.P + obs_var)

            self.mu_prior = self.mu_prior + kalman_gain * (obs - self.mu_prior)
            self.P = (1 - kalman_gain) * self.P
            mean_list.append(self.mu_prior)

            z = obs - self.mu_prior
            self.sigma_prior = np.sqrt(self.sigma_prior**2 + kalman_gain * (z**2 - self.sigma_prior**2))
            var_list.append(self.sigma_prior)
        
        return mean_list, var_list
    
        
        return mean_list, var_list
    
    
    def train_agent_EKF_sq_residuals(self, n_trials, final_mean, final_std, measurement_noise):
        
        sample = np.random.normal(final_mean, final_std, n_trials)
        measurement = sample + np.random.normal(0, measurement_noise, n_trials)

        mean_list = []
        var_list = []
        for obs in measurement:

            obs_noise = measurement_noise**2 + self.sigma_internal**2
            sigma = np.exp(self.eks_s)
            h_s = sigma**2 + obs_noise

            z = (obs - self.mu_prior)**2 

            H = 2 * sigma * sigma

            R = 2 * h_s**2
            S = H * self.P * H + R
            K = self.P * H / S

            self.eks_s = self.eks_s + K * (z - h_s)
            self.sigma_prior = np.exp(self.eks_s)
            self.P_sigma = (1.0 - K * H) * self.P_sigma  # process noise

            mean_list.append(self.mu_prior)
            var_list.append(self.P_sigma)
        return mean_list, var_list

    
    
    def train_agent_EKF_absolute(self, n_trials, final_mean, final_std, measurement_noise):
        sample = np.random.normal(final_mean, final_std, n_trials)
        measurement = sample + np.random.normal(0, measurement_noise, n_trials)
        mean_list = []
        var_list = []
        for obs in measurement:
            obs_var = (self.sigma_internal ** 2 + measurement_noise ** 2)
            z = abs(obs - self.mu_prior)

            sigma = np.exp(self.eks_s)
            c = np.sqrt(2 / np.pi)
            h_s = c * sigma 

            H = c * sigma

            R = h_s**2 * (np.pi / 2 - 1)
            S = H * self.P_sigma * H + R
            K = self.P_sigma * H / S

            self.eks_s = self.eks_s + K * (z - h_s)
            self.sigma_prior = np.exp(self.eks_s)

            self.P_sigma = (1 - K * H) * self.P_sigma

            self.sigma_prior = np.sqrt(self.P)
            mean_list.append(self.mu_prior)
            var_list.append(self.sigma_prior)

    
    def train_together_EKF(self, n_trials, final_mean, final_std, measurement_noise):
        sample = np.random.normal(final_mean, final_std, n_trials)
        measurement = sample + np.random.normal(0, measurement_noise, n_trials)
        mean_list = []
        var_list = []
        for obs in measurement:
            obs_var = (self.sigma_internal ** 2 + measurement_noise ** 2)
            kalman_gain = self.P / (self.P + obs_var)

            # self.mu_prior = self.mu_prior + kalman_gain * (obs - self.mu_prior)
            self.P = (1 - kalman_gain) * self.P
            mean_list.append(self.mu_prior)
            sigma = np.exp(self.eks_s)
            h_s = sigma**2 + obs_var

            z = (obs - self.mu_prior)**2 

            H = 2 * sigma * sigma

            R = 2 * h_s**2
            S = H * self.P * H + R
            K = self.P * H / S
            self.mu_prior = self.mu_prior + K * (obs - self.mu_prior)
            self.eks_s = self.eks_s + K * (z - h_s)
            self.sigma_prior = np.exp(self.eks_s)
            self.P_sigma = (1.0 - K * H) * self.P_sigma


        return mean_list, var_list 
    
    def train_together_final(self, n_trials, final_mean, final_std, measurement_noise):
        sample = np.random.normal(final_mean, final_std, n_trials)
        measurement = sample + np.random.normal(0, measurement_noise, n_trials)
        mean_list = []
        var_list = []
        for obs in measurement:
            obs_var = (self.sigma_internal ** 2 + measurement_noise ** 2)
            kalman_gain = self.P / (self.P + obs_var)

            self.mu_prior = self.mu_prior + kalman_gain * (obs - self.mu_prior)
            self.P = (1 - kalman_gain) * self.P
            mean_list.append(self.mu_prior)
            sigma = np.exp(self.eks_s)
            h_s = sigma**2 + obs_var

            z = (obs - self.mu_prior)**2 

            H = 2 * sigma * sigma

            R = 2 * h_s**2
            S = H * self.P * H + R
            K = self.P * H / S
            # self.mu_prior = self.mu_prior + K * (obs - self.mu_prior)
            self.eks_s = self.eks_s + K * (z - h_s)
            self.sigma_prior = np.exp(self.eks_s)
            self.P_sigma = (1.0 - K * H) * self.P_sigma


        return mean_list, var_list 

    
    
    def train_ddm(self, n_trials, final_mean, final_std, measurement_noise,
              alpha_mu=0.1, alpha_sigma=0.05,
              noise_mu=0.01, noise_sigma=0.01):

        # generate data
        sample = np.random.normal(final_mean, final_std, n_trials)
        measurement = sample + np.random.normal(0, measurement_noise, n_trials)

        mean_list = []
        var_list = []

        for obs in measurement:

            # ----- MEAN DDM -----
            residual = obs - self.mu_prior
            self.mu_prior += alpha_mu * residual \
                            + noise_mu * np.random.randn()

            mean_list.append(self.mu_prior)

            # ----- VARIANCE DDM -----
            z = residual**2
            sigma2_pred = np.exp(self.eks_s)

            delta_sigma = z - sigma2_pred

            self.eks_s += alpha_sigma * delta_sigma \
                        + noise_sigma * np.random.randn()

            sigma2 = np.exp(self.eks_s)
            self.sigma_prior = sigma2
            var_list.append(sigma2)

        return mean_list, var_list

    

    def reset_agent(self):
        self.mu_prior = self.safety_mu
        self.sigma_prior = self.safety_sigma
        self.P = self.safety_P
        self.P_sigma = self.safety_P**2

    def get_prior(self):
        return self.mu_prior, self.sigma_prior
        

    


