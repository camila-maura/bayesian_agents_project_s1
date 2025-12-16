import numpy as np
from scipy.stats import norm
import math


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
        "Decision task: replicate original website agent"
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
        "Simple Kalman filter updating both mean and variance"
        sample = np.random.normal(final_mean, final_std, n_trials)
        measurement = sample + np.random.normal(0, measurement_noise, n_trials)
        mean_list = []
        var_list = []
        for obs in measurement:
            obs_var = (self.sigma_internal ** 2 + measurement_noise ** 2)
            kalman_gain = self.P / (self.P + obs_var) 

            self.mu_prior = self.mu_prior + kalman_gain * (obs - self.mu_prior)
            self.sigma_prior = np.sqrt((1 - kalman_gain) * (self.sigma_prior ** 2)) # 1-Kalman Gain * P
            self.P = (1 - kalman_gain) * self.P
            mean_list.append(self.mu_prior)
            var_list.append(self.sigma_prior)

        return mean_list, var_list 
    
    def train_agent_kalmann_pred_error(self, n_trials, final_mean, final_std, measurement_noise):
        "Kalman filter updating mean and variance updated based on prediction error"
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
            self.sigma_prior = np.sqrt(self.sigma_prior**2 + kalman_gain * (z**2 - self.sigma_prior**2)) # Euclidean distance between predicted and observed
            var_list.append(self.sigma_prior)
        return mean_list, var_list
    
    
    def train_agent_EKF_sq_residuals(self, n_trials, final_mean, final_std, measurement_noise):
        '''
        Correct Extended Kalman Filter updating variance based on squared residuals and
        exponential function for sigma
        '''
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

    def train_agent_EKF_absolute(self, n_trials, final_mean, 
                                 final_std, measurement_noise):
        sample = np.random.normal(final_mean, final_std, n_trials)
        measurement = sample + np.random.normal(0, measurement_noise, n_trials)
        mean_list = []
        var_list = []
        for obs in measurement:
            obs_var = (self.sigma_internal * 2 + measurement_noise * 2)
            z = abs(obs - self.mu_prior)

            sigma = np.exp(self.eks_s)
            c = np.sqrt(2 / np.pi)
            h_s = np.sqrt(sigma**2 + obs_var) * c

            H = c * 2 * sigma * sigma / (2 * np.sqrt(sigma**2 + obs_var))

            R = (sigma**2 + obs_var) * (1 - 2/np.pi)
            S = H * self.P_sigma * H + R
            K = self.P_sigma * H / S

            self.eks_s = self.eks_s + K * (z - h_s)
            self.sigma_prior = np.exp(self.eks_s)

            self.P_sigma = (1 - K * H) * self.P_sigma

            self.sigma_prior = np.exp(self.eks_s)
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


    def train_variational_bayes(
        self, n_trials, final_mean, final_std, measurement_noise, alpha_mu=0.1, alpha_sigma=0.05):
        
        sample = np.random.normal(final_mean, final_std, n_trials)
        measurement = sample + np.random.normal(0, measurement_noise, n_trials)
        mean_list = []
        var_list = []
        count = 0
        for obs in measurement:
            if count%500 == 0 and count != 0:
                alpha_mu = alpha_mu / 10
            count = count + 1
            sigma = np.sqrt(self.sigma_internal**2 + measurement_noise**2 + self.sigma_prior**2)
            sigma_obs = np.sqrt(self.sigma_internal**2 + measurement_noise**2)
            mu_pred = self.mu_prior
            sigma_pred = self.sigma_prior**2 + self.sigma_internal**2 + measurement_noise**2
            for _ in range(5):
                grad_mu = (obs - self.mu_prior) / sigma_obs**2
                grad_sigma = ((obs - self.mu_prior)**2 - (sigma_obs**2 + self.sigma_prior**2)) / (self.sigma_prior**3)

                grad_mu_KL = - (self.mu_prior - mu_pred) / sigma_pred
                grad_sigma_KL = 1/self.sigma_prior - self.sigma_prior / sigma_pred

                grad_mu += grad_mu_KL
                grad_sigma += grad_sigma_KL

                self.mu_prior += alpha_mu * grad_mu

                self.sigma_prior += alpha_sigma * grad_sigma    
                self.sigma_prior = max(self.sigma_prior, 1e-3)

            var_list.append(self.sigma_prior)
            mean_list.append(self.mu_prior)

        return mean_list, var_list
