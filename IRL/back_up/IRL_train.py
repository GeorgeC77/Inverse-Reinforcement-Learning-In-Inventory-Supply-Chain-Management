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



# obs = env.envs[0].reset()
# done = False
# total_reward = 0
# while not done:
#     action, _ = expert.predict(obs)
#     obs, reward, done, truncated, info = env.envs[0].step(action)
#     total_reward += reward
# print("Manual run total reward:", total_reward)


print(type(expert))
# checking the expert
mean_reward, std_reward = evaluate_policy(
    expert,
    env,
    n_eval_episodes=100,
    deterministic=True,
    return_episode_rewards=False  # 设置为 True 可返回每轮的 reward 列表
)

print(f"Expert mean reward over 100 episodes: {mean_reward:.2f} ± {std_reward:.2f}")

rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_episodes=500),
    rng=np.random.default_rng(SEED),
    unwrap=False,
)

learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.05,
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
    allow_variable_horizon=True
)



env.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

train_steps = 20_000
log_interval = 10_000
num_iters = train_steps // log_interval

rewards_during_training = []

for i in range(num_iters):
    airl_trainer.train(log_interval)
    mean_rew, _ = evaluate_policy(learner, env, n_eval_episodes=50)
    rewards_during_training.append(mean_rew)
    print(f"[{(i+1)*log_interval}] Mean reward: {mean_rew:.2f}")

env.seed(SEED)
learner_rewards_after_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

print(rewards_during_training)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))