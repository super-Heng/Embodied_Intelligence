# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
事件项：周期性给 base link 一个随机水平推力 / 速度扰动，模拟外部干扰。
GPU 张量原地写回 root_state，避免 CPU 同步。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def push_by_setting_velocity(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    通过覆写 base 线速度 / 角速度的方式施加冲击式扰动。

    velocity_range: {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-0.5, 0.5)}
    """
    asset: Articulation = env.scene[asset_cfg.name]
    device = asset.device

    # 取当前 root state（GPU 张量），按 env_ids slice
    root_state = asset.data.root_state_w[env_ids].clone()  # (N, 13)

    def _rand(key: str, n: int) -> torch.Tensor:
        lo, hi = velocity_range.get(key, (0.0, 0.0))
        return torch.empty(n, device=device).uniform_(lo, hi)

    n = env_ids.numel()
    # 世界系线速度 [7:10]
    root_state[:, 7] += _rand("x", n)
    root_state[:, 8] += _rand("y", n)
    root_state[:, 9] += _rand("z", n)
    # 角速度 [10:13]
    root_state[:, 12] += _rand("yaw", n)

    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
