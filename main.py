import numpy as np
import gymnasium as gym
from envs.env import POUT_policy
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


parser.add_argument('--Lead_time', type=int, default=1, help='Lead time')  # Taka lead time, equals Steve lead time+1
parser.add_argument('--rho', type=float, default=0.9, help='AR coefficient')  # auto correlation parameter
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
        'rho': 0.9,
        'sigma': 4,
        'seq_len': 100,
        'Lead_time': 1,
        'is_linear': 0,
        'metric': 1,
        'is_train': True
    }
)

print('InventoryEnv-v0' in gym.envs.registry)

SEED = 42


env = make_vec_env("InventoryEnv-v0", n_envs=8, rng=np.random.default_rng(SEED), env_make_kwargs={
        'mu': opt.mu,
        'rho': opt.rho,
        'sigma': opt.sigma,
        'seq_len': opt.seq_len,
        'Lead_time': 1,
        'is_linear': 0,
        'metric': 1,
        'is_train': True
    })

expert = POUT_policy(env.envs[0])


rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_episodes=60),
    rng=np.random.default_rng(SEED),
)

learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0005,
    gamma=0.95,
    clip_range=0.1,
    vf_coef=0.1,
    n_epochs=5,
    seed=SEED,
)
reward_net = BasicShapedRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=2048,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=16,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
)

env.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)
airl_trainer.train(20000)  # Train for 2_000_000 steps to match expert.
env.seed(SEED)
learner_rewards_after_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))