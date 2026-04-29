# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
抗干扰行走任务 —— 奖励函数。

所有函数签名均符合 Isaac Lab 2.3 Manager-based 框架：
    fn(env: ManagerBasedRLEnv, ...) -> torch.Tensor   # shape: (num_envs,)

底层张量全部走 GPU pipeline (`omni.physics.tensors`)，避免任何 .cpu() / .item()。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# 1. 关节力矩惩罚
# -----------------------------------------------------------------------------
def joint_torques_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """对施加力矩的 L2 范数平方求和。值越大代表越费力，应作为负向奖励。"""
    asset: Articulation = env.scene[asset_cfg.name]
    # applied_torque shape: (num_envs, num_joints)
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torque.square(), dim=1)


# -----------------------------------------------------------------------------
# 2. 姿态稳定奖励（base 直立 + 角速度阻尼）
# -----------------------------------------------------------------------------
def base_upright(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """投影重力的 z 分量。机器人完全直立时该值 = -1。

    返回 `1 - |gx|^2 - |gy|^2`，越接近 1 越直立；姿态崩了 -> 趋近 0。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    proj_g = asset.data.projected_gravity_b  # (num_envs, 3) base 系
    return 1.0 - proj_g[:, 0].square() - proj_g[:, 1].square()


def base_ang_vel_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base 角速度（roll/pitch）的 L2，作为姿态稳定的辅助负奖励。"""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_b = asset.data.root_ang_vel_b  # (num_envs, 3)
    return ang_vel_b[:, :2].square().sum(dim=1)


def base_height_target(
    env: "ManagerBasedRLEnv",
    target_height: float = 1.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """对偏离目标高度做平方惩罚（返回正值，权重给负即可）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    return (h - target_height).square()


# -----------------------------------------------------------------------------
# 3. 目标速度追踪（线速度 xy + 朝向角速度 yaw）
# -----------------------------------------------------------------------------
def track_lin_vel_xy_exp(
    env: "ManagerBasedRLEnv",
    std: float = 0.25,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    使用指数核追踪 base 系线速度 (vx, vy)。

    奖励 = exp(-||v_cmd - v_meas||^2 / std^2)，范围 (0, 1]。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)  # (num_envs, 3): vx, vy, wz
    # base 系下的水平线速度
    lin_vel_b = asset.data.root_lin_vel_b[:, :2]
    err = torch.sum((cmd[:, :2] - lin_vel_b).square(), dim=1)
    return torch.exp(-err / (std ** 2))


def track_ang_vel_z_exp(
    env: "ManagerBasedRLEnv",
    std: float = 0.25,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """指数核追踪 yaw 角速度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    ang_vel_z_b = asset.data.root_ang_vel_b[:, 2]
    err = (cmd[:, 2] - ang_vel_z_b).square()
    return torch.exp(-err / (std ** 2))


# -----------------------------------------------------------------------------
# 4. 抗干扰附加项（鼓励 base 在世界 xy 上别被推飞）
# -----------------------------------------------------------------------------
def base_lin_vel_z_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚竖直方向速度（被推时不能起飞 / 急坠）。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b[:, 2].square()


def survival_bonus(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """每步给一个常数奖励，鼓励"活着"，配合 termination 用。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.ones(env.num_envs, device=asset.device)


# -----------------------------------------------------------------------------
# 工具：把全局速度旋到 yaw frame（备用，给可能的 heading 奖励项使用）
# -----------------------------------------------------------------------------
def lin_vel_in_heading_frame(asset: Articulation) -> torch.Tensor:
    quat_yaw = yaw_quat(asset.data.root_quat_w)
    return quat_rotate_inverse(quat_yaw, asset.data.root_lin_vel_w)
