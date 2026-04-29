# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""上层任务的终止项。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def all_waypoints_reached(env: "ManagerBasedRLEnv", command_name: str = "waypoint") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    return cmd.num_reached >= cmd.cfg.num_waypoints


def fell_over(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fall_height: float = 0.15,
    fall_tilt_deg: float = 60.0,
) -> torch.Tensor:
    a: Articulation = env.scene[asset_cfg.name]
    z = a.data.root_pos_w[:, 2]
    proj_g = a.data.projected_gravity_b
    tilt_cos = -proj_g[:, 2]
    thresh = math.cos(math.radians(fall_tilt_deg))
    return (z < fall_height) | (tilt_cos < thresh)


def severe_collision(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 50.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history
    if forces is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    f_norm = torch.linalg.norm(forces, dim=-1)
    return f_norm.amax(dim=(1, 2)) > threshold
