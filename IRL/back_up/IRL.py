# pip install stable-baselines3 imitation gymnasium torch
import gymnasium as gym
import numpy as np
from expert.POUT import POUT_policy
from stable_baselines3 import SAC
import torch
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


register(
    id='InventoryEnv-v0',
    entry_point='envs.env:InventoryEnv',
    kwargs={
        'mu': 20,
        'rho': 0.5,
        'sigma': 1,
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
        'rho': 0.5,
        'sigma': 1,
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
            rews_list.append(reward)   # IRL里奖励可以置0
            dones_list.append(done)
            info_list.append(info)

            obs_list.append(next_obs)   # 注意这里：追加 next_obs
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
gen_algo = SAC("MlpPolicy", venv, verbose=1, tensorboard_log="./tb_irlinv/", gamma=0.999, learning_rate=3e-4)

# 5) AIRL 训练器
airl_trainer = AIRL(
    venv=venv,
    demonstrations=expert_trajs,
    demo_batch_size=128,
    gen_algo=gen_algo,
    reward_net=reward_net,
    n_disc_updates_per_round=1,   # 判别器每轮更新步数
    disc_opt_cls=torch.optim.AdamW,
    disc_opt_kwargs=dict(lr=3e-4, weight_decay=1e-4)
    # 其他可调：bc_train_kwargs, disc_opt_cls/kwargs 等
)


venv.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    gen_algo, venv, 100, return_episode_rewards=True,
)


# 6) 训练循环（例如 1e6 steps，按你算力调整）

def lr_schedule(progress_remaining):   # progress_remaining 1→0
    base = 3e-4
    return base * (0.1 + 0.9 * progress_remaining)

# gen = SAC("MlpPolicy", venv,
#           gamma=0.99, learning_rate=lr_schedule, tau=0.005, ent_coef='auto', verbose=1,
#           buffer_size=100000, batch_size=256, train_freq=64, gradient_steps=64)
# # 注意：这里环境 step() 要返回真实奖励 = -真实成本
# gen.learn(300_000)

# airl_trainer.train(total_timesteps=50000)


train_steps = 20000
log_interval = 100
num_iters = train_steps // log_interval

rewards_during_training = []

for i in range(num_iters):
    airl_trainer.train(log_interval)
    mean_rew, _ = evaluate_policy(gen_algo, venv, n_eval_episodes=100)
    rewards_during_training.append(mean_rew)
    print(f"[{(i+1)*log_interval}] Mean reward: {mean_rew:.2f}")


print(rewards_during_training)


venv.seed(SEED)
learner_rewards_after_training, _ = evaluate_policy(
    gen_algo, venv, 100, return_episode_rewards=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))

# 7) 导出学得的奖励函数
# learned_reward = airl_trainer.reward_net
# 用法：r = learned_reward(obs_tensor, act_tensor, None, None)
# 你可以在评估脚本中网格化地查看 r(s,a) 的变化，或替换业务仿真的成本函数
