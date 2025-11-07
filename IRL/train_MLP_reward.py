# pip install stable-baselines3 imitation gymnasium torch
import gymnasium as gym
import numpy as np
import torch
from expert.POUT import POUT_policy
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit
from imitation.data.types import TrajectoryWithRew
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from imitation.data import rollout
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.util import make_vec_env
from gymnasium.envs.registration import register
from reward.piece_wise_linear import ParamInventoryReward
import pandas as pd

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

# 1) 环境

venv = make_vec_env("InventoryEnv-v0", n_envs=1, rng=np.random.default_rng(SEED), env_make_kwargs={
    'mu': 20,
    'rho': 0,
    'sigma': 4,
    'seq_len': 100,
    'Lead_time': 4,
    'is_linear': 1,
    'metric': 1,
    'is_train': True
})

venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                    gamma=0.99, clip_obs=10.0, clip_reward=10.0)


# 2) 载入/生成专家演示（示例：用你已实现的OUT/POUT策略生成）
def collect_expert_trajectories(env, expert_policy, n_traj=100, traj_len=200):
    trajs = []
    for _ in range(n_traj):
        obs, _ = env.reset()
        expert = expert_policy(env)
        obs_list, acts_list, rews_list, dones_list, info_list = [obs], [], [], [], []
        for t in range(traj_len):

            act = expert.predict(obs)
            next_obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            acts_list.append(act[0])
            rews_list.append(reward)  # IRL里奖励可以置0
            dones_list.append(done)
            info_list.append(info)

            obs_list.append(next_obs)  # 注意这里：追加 next_obs
            obs = next_obs
            if done:
                break

        trajs.append(
            TrajectoryWithRew(
                obs=np.array(obs_list),
                acts=np.array(acts_list).reshape(-1, 1),
                rews=np.array(rews_list),
                terminal=dones_list[-1],
                infos=info_list,
            )
        )
    return trajs


# 这里请换成你现有的专家策略
expert_policy = POUT_policy

expert_trajs = collect_expert_trajectories(venv.envs[0], expert_policy, n_traj=200, traj_len=100)

# 3) 奖励网络：同时吃 obs 和 act（推荐）
reward_net = BasicRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    # 将 state 和 action 一起输入
    use_action=True,
    hid_sizes=(16,),
)

# 4) 生成器（策略）——在学得奖励下训练
gen_algo = SAC("MlpPolicy", venv, verbose=1, tensorboard_log="./tb_irlinv/", gamma=0.99, learning_rate=3e-4,
               ent_coef="auto", batch_size=256, train_freq=64, gradient_steps=64, buffer_size=100000)

# 5) AIRL 训练器
airl = AIRL(
    venv=venv,
    demonstrations=expert_trajs,
    gen_algo=gen_algo,
    reward_net=reward_net,
    demo_batch_size=128,
    n_disc_updates_per_round=1,  # 判别器每轮更新步数
    disc_opt_cls=torch.optim.AdamW,
    disc_opt_kwargs=dict(lr=3e-3, weight_decay=1e-3)
    # 其他可调：bc_train_kwargs, disc_opt_cls/kwargs 等
)





# 6) 训练循环（例如 1e6 steps，按你算力调整）

# airl.train(total_timesteps=300_000)

opt = -np.inf

train_steps = 20000
log_interval = 200
num_iters = train_steps // log_interval

rewards_during_training = []

for i in range(num_iters):
    airl.train(log_interval)
    mean_rew, _ = evaluate_policy(gen_algo, venv, n_eval_episodes=100)
    rewards_during_training.append(mean_rew)
    print(f"[{(i+1)*log_interval}] Mean reward: {mean_rew:.2f}")

    # check improvement and save the optimal model
    if mean_rew > opt:
        opt = mean_rew
        gen_algo.save("model/sac_gen_policy_airl")
        venv.save("model/vec_normalize.pkl")

print(rewards_during_training)

df = pd.DataFrame(rewards_during_training, columns=["reward"])
df.to_csv("data/output.csv", index=False)