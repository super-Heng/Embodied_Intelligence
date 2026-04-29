# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""上层 navigation policy 的奖励项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

from .observations import lidar_min_distance, target_distance, target_heading_err

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# 距离/到达
# -----------------------------------------------------------------------------
def progress_to_waypoint(env: "ManagerBasedRLEnv", command_name: str = "waypoint") -> torch.Tensor:
    """距离每步的减少量（dense）。"""
    cmd = env.command_manager.get_term(command_name)
    return cmd.progress  # (N,)


def reach_waypoint(env: "ManagerBasedRLEnv", command_name: str = "waypoint") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    return cmd.just_reached.float()


def finish_all_waypoints(env: "ManagerBasedRLEnv", command_name: str = "waypoint") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    return cmd.just_finished_all.float()


def heading_alignment(
    env: "ManagerBasedRLEnv",
    std: float = 0.5,
    command_name: str = "waypoint",
) -> torch.Tensor:
    err = target_heading_err(env, command_name).squeeze(-1)
    return torch.exp(-err.square() / (std ** 2))


# -----------------------------------------------------------------------------
# 安全
# -----------------------------------------------------------------------------
def lidar_proximity_penalty(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    decay: float = 0.3,
    max_range: float = 5.0,
) -> torch.Tensor:
    """exp(-min_dist / decay)：越靠近障碍惩罚越大。"""
    d = lidar_min_distance(env, sensor_cfg, max_range).squeeze(-1) * max_range  # 还原物理距离
    return torch.exp(-d / max(decay, 1e-3))


def collision_with_obstacle(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """contact_forces sensor 上任一 body 法向力超过阈值即记一次碰撞（per-step）。"""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w_history shape: (N, history, B, 3)
    forces = sensor.data.net_forces_w_history
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    f_norm = torch.linalg.norm(forces, dim=-1)            # (N, history, B)
    max_f = f_norm.amax(dim=(1, 2))                       # (N,)
    return (max_f > threshold).float()


# -----------------------------------------------------------------------------
# 平滑/效率
# -----------------------------------------------------------------------------
def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """上层动作变化率 L2（来自 action_manager 的内置接口）。"""
    actions = env.action_manager.action
    prev = env.action_manager.prev_action
    return (actions - prev).square().sum(dim=1)


def time_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


# -----------------------------------------------------------------------------
# 终止惩罚（与 termination 信号联动）
# -----------------------------------------------------------------------------
def fall_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fall_height: float = 0.15,
    fall_tilt_deg: float = 60.0,
) -> torch.Tensor:
    a: Articulation = env.scene[asset_cfg.name]
    z = a.data.root_pos_w[:, 2]
    proj_g = a.data.projected_gravity_b
    tilt_cos = -proj_g[:, 2]                               # 直立时 = 1
    tilt_thresh = float(torch.cos(torch.tensor(fall_tilt_deg * 3.14159 / 180.0)).item())
    fell = (z < fall_height) | (tilt_cos < tilt_thresh)
    return fell.float()
