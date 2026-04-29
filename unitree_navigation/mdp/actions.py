# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""低层封装 ActionTerm：上层 policy 输出 (vx, vy, wz) -> 低层 ckpt 推理 -> 关节 target。

工作流：
  - 上层 policy 决策频率：env.dt = sim.dt * decimation = 0.005 * 20 = 0.1s（10Hz）
  - 一个上层 step 内会调用 apply_actions() decimation 次（200/20=10 次？不，apply_actions 每个 sim step 都会调用）
  - 每次 apply_actions：用最新的速度指令 + Go2 本体观测 喂低层 ckpt -> 关节 pos target

低层 obs 顺序遵循 Isaac Lab `Isaac-Velocity-*-Unitree-Go2-v0` 默认 (48 维)：
  base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + velocity_commands(3)
  + joint_pos_rel(12) + joint_vel_rel(12) + last_actions(12)
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse

from config import CFG
from locomotion.policy_loader import LocomotionPolicy

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class LowLevelVelocityAction(ActionTerm):
    """3 维速度指令 -> 低层 ckpt -> 12 维关节 target。"""

    cfg: "LowLevelVelocityActionCfg"

    def __init__(self, cfg: "LowLevelVelocityActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # 上层动作缓存
        self._raw_vel = torch.zeros(self.num_envs, 3, device=self.device)        # tanh 输出
        self._scaled_vel = torch.zeros(self.num_envs, 3, device=self.device)     # 物理速度
        self._smoothed_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_low_action = torch.zeros(self.num_envs, 12, device=self.device)

        # 低层 policy
        loc = CFG.locomotion
        self.policy = LocomotionPolicy(
            ckpt_path=str(loc.ckpt_path) if loc.ckpt_path else None,
            obs_dim=int(loc.obs_dim),
            action_dim=int(loc.action_dim),
            hidden_dims=list(loc.hidden_dims),
            activation=str(loc.activation),
            empirical_normalization=bool(loc.empirical_normalization),
            device=str(self.device),
        )

        # 默认关节位置（用于把低层输出 + default 得到 target）
        self._default_joint_pos = self.robot.data.default_joint_pos.clone()
        self._action_scale = 0.25  # Isaac Lab Go2 任务默认动作 scale

    # -------------------------------------------------------------------------
    # ActionTerm 接口
    # -------------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_vel

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._smoothed_vel

    def process_actions(self, actions: torch.Tensor) -> None:
        """上层每个 env step 调用一次，缓存速度指令。"""
        a = torch.tanh(actions)
        self._raw_vel = a
        scale = torch.tensor(
            [self.cfg.vx_scale, self.cfg.vy_scale, self.cfg.wz_scale],
            device=self.device,
        )
        self._scaled_vel = a * scale
        alpha = float(self.cfg.smoothing_alpha)
        self._smoothed_vel = alpha * self._scaled_vel + (1.0 - alpha) * self._smoothed_vel

    def apply_actions(self) -> None:
        """每个 sim step 调用：把速度指令喂低层，写关节 target。"""
        low_obs = self._build_low_obs()
        delta = self.policy.act(low_obs)              # (N, 12)
        joint_target = self._default_joint_pos + self._action_scale * delta
        self.robot.set_joint_position_target(joint_target)
        self._last_low_action = delta

    # -------------------------------------------------------------------------
    def _build_low_obs(self) -> torch.Tensor:
        """组装低层 48 维观测，顺序与 Isaac Lab Go2 velocity 任务一致。"""
        r = self.robot.data
        base_lin_vel = r.root_lin_vel_b                                # (N,3)
        base_ang_vel = r.root_ang_vel_b                                # (N,3)
        proj_g = r.projected_gravity_b                                 # (N,3)
        vel_cmd = self._smoothed_vel                                   # (N,3)
        joint_pos_rel = r.joint_pos - r.default_joint_pos              # (N,12)
        joint_vel_rel = r.joint_vel - r.default_joint_vel              # (N,12)
        last_act = self._last_low_action                               # (N,12)
        return torch.cat(
            [base_lin_vel, base_ang_vel, proj_g, vel_cmd, joint_pos_rel, joint_vel_rel, last_act],
            dim=1,
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._raw_vel.zero_()
            self._scaled_vel.zero_()
            self._smoothed_vel.zero_()
            self._last_low_action.zero_()
        else:
            self._raw_vel[env_ids] = 0.0
            self._scaled_vel[env_ids] = 0.0
            self._smoothed_vel[env_ids] = 0.0
            self._last_low_action[env_ids] = 0.0


@configclass
class LowLevelVelocityActionCfg(ActionTermCfg):
    class_type: type = LowLevelVelocityAction
    asset_name: str = MISSING
    vx_scale: float = 1.0
    vy_scale: float = 0.5
    wz_scale: float = 1.0
    smoothing_alpha: float = 0.7
