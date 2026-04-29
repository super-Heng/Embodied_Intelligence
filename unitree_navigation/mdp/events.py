# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""上层任务的事件项：障碍随机化、reset 时姿态扰动等。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from config import CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_stage1_obstacles(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
):
    """每次 reset 给本批 env 重新摆放障碍物。

    - 在以 env 原点为中心、半径 spawn_area_radius 的圆内均匀采样
    - 与机器人初始位置保持 min_dist_to_robot 安全距离
    """
    o = CFG.stage1_obstacles
    n_per = int(o.num_per_env)
    radius = float(o.spawn_area_radius)
    safe = float(o.min_dist_to_robot)
    n_envs = env_ids.numel()
    device = env.device

    env_origins = env.scene.env_origins[env_ids]   # (n, 3)

    for i in range(n_per):
        name = f"obstacle_{i}"
        if name not in env.scene.rigid_objects:
            continue
        obj: RigidObject = env.scene[name]

        # 极坐标采样，遮蔽机器人正中心 safe 半径内
        for _ in range(8):  # 重采样兜底
            r = torch.empty(n_envs, device=device).uniform_(safe, radius)
            theta = torch.empty(n_envs, device=device).uniform_(-math.pi, math.pi)
            offset = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
            if (offset.norm(dim=1) >= safe).all():
                break
        z = torch.full((n_envs, 1), 0.4, device=device)  # 障碍中心高度（近似）
        pos = env_origins + torch.cat([offset, z], dim=1)

        # 随机 yaw
        yaw = torch.empty(n_envs, device=device).uniform_(-math.pi, math.pi)
        zero = torch.zeros_like(yaw)
        quat = torch.stack(
            [torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)],
            dim=1,
        )
        root_state = torch.zeros(n_envs, 13, device=device)
        root_state[:, :3] = pos
        root_state[:, 3:7] = quat
        obj.write_root_state_to_sim(root_state, env_ids=env_ids)
