import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.utils import Action_adapter, Action_adapter_reverse

class InventoryEnv(gym.Env):
    def __init__(self, mu=20, rho=0, sigma=2, seq_len=100, Lead_time=1, is_linear=0, metric=1, is_train=True, two_phase=False):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.sigma = sigma
        self.seq_len = seq_len
        self.Lead_time = Lead_time
        self.is_linear = is_linear
        self.metric = metric
        self.is_train = is_train
        self.l = self.Lead_time

        self.two_phase = two_phase
        self.k = None


        self.seq_length = self.seq_len + 10  # Prevent bug in step


        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.l - 1 + 1 + 6,) if self.l > 1 else (1 + 6,),
            dtype=np.float32
        )

        # 修改 action space: 第一维为 capacity, 第二维为 order
        if self.two_phase:
            self.action_space = spaces.Box(low=np.array([0.0, -self.mu]), high=np.array([self.mu * 2, self.mu * 3]),
                                       shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-self.mu, high=self.mu * 3, shape=(1,), dtype=np.float32)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.two_phase:
            self.k = None
            self.set_capacity()

        self.t = 0
        self.r = np.zeros(self.seq_length)
        self.a = np.zeros(self.seq_length)
        self.act = np.zeros(self.seq_length)
        self.dw = False

        self.h = 1
        self.p = 9

        self.u = 4
        self.m = 1.5

        self.NS = np.zeros(self.seq_length)
        if self.l > 1:
            self.WIP = np.zeros([self.seq_length, self.l - 1])
        self.h_cost = np.zeros(self.seq_length)
        self.b_cost = np.zeros(self.seq_length)
        self.demand = generate_demand(self.mu, self.rho, self.sigma, self.seq_length + 5)
        self.order = np.zeros(self.seq_length)

        self.NS[self.t] = self.mu - self.demand[self.t + 5]
        if self.l > 1:
            for i in range(self.l - 1):
                self.WIP[self.t, i] = self.mu + np.random.normal(0, self.sigma)

        self._update_costs()
        self._apply_linearity_constraints()

        obs = self._get_obs()
        obs = obs.astype(np.float32)
        info = {}  # 可以根据需要填写
        return obs, info

    def step(self, action):
        info = {}  # 可以根据需要填写

        if self.two_phase:
            if self.k is None:
                self.k = action[0]
            order_action = action[1]
        else:
            order_action = action[0]

        self.act[self.t] = order_action

        # self.act[self.t] = action[0]
        self.a[self.t] = Action_adapter_reverse(action[0], self.mu)
        self.order[self.t] = self.act[self.t]
        if self.is_linear == 0:
            self.order[self.t] = max(self.order[self.t], 0)

        self.t += 1

        if self.l == 1:
            self.NS[self.t] = self.NS[self.t - 1] + self.order[self.t - 1] - self.demand[self.t + 5]
        else:
            self.NS[self.t] = self.NS[self.t - 1] + self.WIP[self.t - 1, 0] - self.demand[self.t + 5]
            self.WIP[self.t, :] = np.concatenate((self.WIP[self.t - 1, 1:], [self.order[self.t - 1]]))

        self._update_costs()

        self.r[self.t - 1] = - (self.b_cost[self.t - 1] + self.h_cost[self.t - 1])
        if self.metric in [2, 4]:
            self.r[self.t - 1] += -abs(self.order[self.t - 1] - self.mu) * 2.1816

        self.dw = (self.t == self.seq_length - 10)


        self._apply_linearity_constraints()



        terminated = 0

        obs = self._get_obs()
        obs = obs.astype(np.float32)

        if self.dw:
            info["TimeLimit.truncated"] = True
            info["terminal_observation"] = obs

        return obs, self.r[self.t - 1], terminated, self.dw, info

    def set_capacity(self, capacity_action):
        # 仅在 reset 后调用一次
        self.k = capacity_action

    def _update_costs(self):
        self.h_cost[self.t] = max(0, self.NS[self.t]) * self.h  # self.h assumed = 1
        self.b_cost[self.t] = max(0, -self.NS[self.t]) * self.p  # self.b assumed = 9

    def _apply_linearity_constraints(self):
        if self.is_linear == 0:
            self.NS[self.t] = max(self.NS[self.t], 0)
            if self.l > 1:
                for i in range(self.l - 1):
                    self.WIP[self.t, i] = max(self.WIP[self.t, i], 0)

    def _get_obs(self):
        if self.l > 1:
            return np.concatenate((self.WIP[self.t, :], [self.NS[self.t]], self.demand[self.t:self.t + 6]))
        else:
            return np.concatenate(([self.NS[self.t]], self.demand[self.t:self.t + 6]))


def generate_demand(mu, rho, sigma, length):
    d = np.zeros([length + 5])
    d[0] = mu
    for i in range(1, len(d)):
        d[i] = mu + rho * (d[i - 1] - mu) + (np.random.normal(0, sigma))
    return d