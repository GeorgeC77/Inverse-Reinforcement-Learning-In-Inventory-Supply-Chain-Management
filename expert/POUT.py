import numpy as np
from scipy.stats import norm

class POUT_policy:
    def __init__(self, env):
        self.env = env
        self.alpha = 0
        self.l = self.env.l
        self.rho = self.env.rho
        self.sigma = self.env.sigma
        self.mu = self.env.mu

        # get the optimal alpha

        # get TNS
        varns = ((self.l * (1 - self.rho ** 2) + self.rho * (1 - self.rho ** self.l) * (self.rho ** (self.l + 1) - self.rho - 2)) / (1 - self.rho) ** 2 / (
                1 - self.rho ** 2) + (self.rho ** self.l - 1) ** 2 * self.alpha ** 2 / (self.rho - 1) ** 2 / (
                          1 - self.alpha ** 2)) * self.sigma ** 2
        z = norm.ppf(9 / (9 + 1))
        self.TNS = z * varns ** 0.5
        # create suboptimal expert trajectory
        if self.rho == 0:
            self.TNS *= 1


    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = obs.squeeze()
        if self.env.l == 1:
            current_NS = obs[0]
            current_D = obs[-1]
            order = (1 - self.rho) * self.mu + self.env.rho * current_D + (1 - self.alpha) * (self.TNS - current_NS)
        else:
            current_WIP = obs[:self.env.l - 1]
            current_NS = obs[self.env.l - 1]
            current_D = obs[-1]
            order = (1 - self.rho ** self.l) * self.mu + self.env.rho ** self.env.l * current_D + (1 - self.alpha) * (
                        self.TNS + (self.env.rho - self.env.rho ** self.env.l) / (1 - self.env.rho) * current_D + self.mu * (self.l - 1 - (self.env.rho - self.env.rho ** self.env.l) / (1 - self.env.rho)) - current_NS - np.sum(current_WIP))
        return np.array([[order]]).astype(np.float32), {}

        # return order.reshape(-1, 1).astype(np.float32), {}






    def __call__(self, obs, state=None, dones=None):
        action, _ = self.predict(obs, deterministic=True)
        return action, state  # 第二个返回值必须是 state（即使是 None）