import numpy as np
import gymnasium as gym
from expert.POUT import POUT_policy
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.data.rollout import generate_trajectories
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from gymnasium.envs.registration import register
import sys
import os
import argparse


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()


parser.add_argument('--Lead_time', type=int, default=4, help='Lead time')  # Taka lead time, equals Steve lead time+1
parser.add_argument('--rho', type=float, default=0, help='AR coefficient')  # auto correlation parameter
parser.add_argument('--mu', type=float, default=20, help='AR average')  # mean
parser.add_argument('--sigma', type=float, default=4, help='AR std')  # std of the white noise
parser.add_argument('--seq_len', type=int, default=100, help='Sequence length')  # sequence length for a single simulation run

opt = parser.parse_args()

print(opt)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

register(
    id='InventoryEnv-v0',
    entry_point='envs.env:InventoryEnv',
    kwargs={
        'mu': 20,
        'rho': 0,
        'sigma': 4,
        'seq_len': 100,
        'Lead_time': 4,
        'is_linear': 1,
        'metric': 1,
        'is_train': True
    }
)

print('InventoryEnv-v0' in gym.envs.registry)

SEED = 42


env = make_vec_env("InventoryEnv-v0", n_envs=1, rng=np.random.default_rng(SEED), env_make_kwargs={
        'mu': opt.mu,
        'rho': opt.rho,
        'sigma': opt.sigma,
        'seq_len': opt.seq_len,
        'Lead_time': 4,
        'is_linear': 1,
        'metric': 1,
        'is_train': True
    })

expert = POUT_policy(env.envs[0])

# checking the expert
mean_reward, std_reward = evaluate_policy(
    expert,
    env,
    n_eval_episodes=100,
    deterministic=True,
    return_episode_rewards=False  # 设置为 True 可返回每轮的 reward 列表
)

print(f"Expert mean reward over 100 episodes: {mean_reward:.2f} ± {std_reward:.2f}")