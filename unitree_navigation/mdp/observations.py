# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""上层 navigation policy 的观测项。

所有函数返回 (num_envs, dim) 的 GPU tensor。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_rotate_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# -----------------------------------------------------------------------------
# 本体（base 自感）
# -----------------------------------------------------------------------------
def base_lin_vel_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    a: Articulation = env.scene[asset_cfg.name]
    return a.data.root_lin_vel_b


def base_ang_vel_b(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    a: Articulation = env.scene[asset_cfg.name]
    return a.data.root_ang_vel_b


def projected_gravity(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    a: Articulation = env.scene[asset_cfg.name]
    return a.data.projected_gravity_b


# -----------------------------------------------------------------------------
# 目标 waypoint（base 系）
# -----------------------------------------------------------------------------
def target_pos_b(env: "ManagerBasedRLEnv", command_name: str = "waypoint") -> torch.Tensor:
    """当前目标在 base yaw 系下的 (dx, dy)。"""
    asset: Articulation = env.scene["robot"]
    target_w_xy = env.command_manager.get_command(command_name)  # (N, 2) world XY
    base_pos_w = asset.data.root_pos_w[:, :3]
    diff_w = torch.zeros_like(base_pos_w)
    diff_w[:, :2] = target_w_xy - base_pos_w[:, :2]
    qyaw = yaw_quat(asset.data.root_quat_w)
    diff_b = quat_rotate_inverse(qyaw, diff_w)
    return diff_b[:, :2]


def target_distance(env: "ManagerBasedRLEnv", command_name: str = "waypoint") -> torch.Tensor:
    d = target_pos_b(env, command_name)
    return torch.linalg.norm(d, dim=1, keepdim=True)


def target_heading_err(env: "ManagerBasedRLEnv", command_name: str = "waypoint") -> torch.Tensor:
    d = target_pos_b(env, command_name)  # (N, 2) base 系
    heading = torch.atan2(d[:, 1], d[:, 0])
    return wrap_to_pi(heading).unsqueeze(-1)


# -----------------------------------------------------------------------------
# 感知
# -----------------------------------------------------------------------------
def lidar_distances(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_range: float = 5.0,
) -> torch.Tensor:
    """归一化 lidar 距离 (N, num_rays) ∈ [0, 1]。"""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # ray_hits_w 为击中点世界坐标；用 distance = ray hit - sensor pos 的范数
    pos = sensor.data.pos_w.unsqueeze(1)             # (N, 1, 3)
    hits = sensor.data.ray_hits_w                    # (N, R, 3)
    dist = torch.linalg.norm(hits - pos, dim=-1)
    dist = torch.nan_to_num(dist, nan=max_range, posinf=max_range, neginf=max_range)
    return torch.clamp(dist, max=max_range) / max_range


def lidar_min_distance(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_range: float = 5.0,
) -> torch.Tensor:
    d = lidar_distances(env, sensor_cfg, max_range)
    return d.min(dim=1, keepdim=True).values


def heightmap_relative(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("heightmap"),
    max_height: float = 1.0,
) -> torch.Tensor:
    """前向高度图：相对 base z 的高度差，clip 后归一化到 [-1, 1]。"""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene["robot"]
    hits_z = sensor.data.ray_hits_w[..., 2]          # (N, K)
    base_z = asset.data.root_pos_w[:, 2:3]           # (N, 1)
    rel = hits_z - base_z
    rel = torch.nan_to_num(rel, nan=0.0, posinf=max_height, neginf=-max_height)
    return torch.clamp(rel, min=-max_height, max=max_height) / max_height
