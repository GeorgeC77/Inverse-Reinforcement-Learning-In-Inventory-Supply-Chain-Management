import torch as th
import torch.nn as nn
from imitation.rewards.reward_nets import RewardNet

class ParamInventoryReward(RewardNet):
    """
    r_theta(s, a, s') = - [ h * pos(I') + p * pos(-I') ] - cΔ * |a - a_prev| - cCap * relu(a - C)
    其中参数通过 Softplus 保证非负，并用 p = h + softplus(raw_p_excess) 实现 p >= h。
    """
    def __init__(
        self,
        observation_space,
        action_space,
        idx_inv_next: int,                # next_obs 中库存 I' 的索引

        scale: float = 1.0,               # 训练时的整体尺度（可把 reward 再缩放 0.1 等）
        device: str = "cpu",
    ):
        # 兼容新/老版本的 RewardNet 构造函数
        try:
            # 新版接口：支持 use_action / use_next_obs
            super().__init__(observation_space, action_space,
                             use_action=True, use_next_obs=True)
        except TypeError:
            # 老版接口：不支持这两个参数
            super().__init__(observation_space, action_space)
            # 手动挂上标志，部分组件会读取
            self.use_action = True
            self.use_next_obs = True

        self.idx_inv_next = int(idx_inv_next)

        self.register_buffer("scale", th.tensor(float(scale)))
        self.softplus = nn.Softplus()

        # 原始可训练参数（未经约束）
        self.raw_h = nn.Parameter(th.tensor(0.1))          # -> h = softplus(raw_h)
        self.raw_p_excess = nn.Parameter(th.tensor(0.2))   # -> p = h + softplus(raw_p_excess) 保障 p>=h


        # 可选：L2 正则时用到（给 AIRL 的优化器传 weight_decay 即可）

        self.to(device)

    def _params(self):
        h = self.softplus(self.raw_h)
        p = h + self.softplus(self.raw_p_excess)   # p >= h
        return h, p

    def forward(self, obs, acts, next_obs, dones):
        """
        obs:  [batch, obs_dim]
        acts: [batch, act_dim]
        next_obs: [batch, obs_dim]
        dones: [batch, 1]  (不会用到，但接口需要)
        """
        h, p = self._params()

        # 取 I'（下一期库存，允许为负=欠货）
        I_next = obs[:, self.idx_inv_next]
        pos_inv = th.relu(I_next)       # [I']_+
        neg_inv = th.relu(-I_next)      # [-I']_+






        # 组合奖励（注意是负成本）
        r = -(h * pos_inv + p * neg_inv)

        # 训练期的整体尺度（可设为 0.1）
        return (self.scale * r)

    # 便捷方法：拿到当前参数的标量值
    def get_params(self):
        h, p = self._params()
        return {
            "h": float(h.detach().cpu()),
            "p": float(p.detach().cpu()),
            "scale": float(self.scale.detach().cpu()),
        }
