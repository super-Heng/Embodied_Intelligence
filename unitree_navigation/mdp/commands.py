# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""WaypointCommand：每 env 维护一个 waypoint 队列，发布"当前目标"。

实现为 Isaac Lab 的 CommandTerm —— 上层 policy 通过观测 `target_pos_b` 看到目标。
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class WaypointCommand(CommandTerm):
    """维护 (num_envs, N, 2) 的 waypoint 队列与当前 idx。"""

    cfg: "WaypointCommandCfg"

    def __init__(self, cfg: "WaypointCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        device = self.device
        N = int(cfg.num_waypoints)

        # waypoints in world XY (env-local origin already accounted by env_origins)
        self.waypoints = torch.zeros(self.num_envs, N, 2, device=device)
        self.current_idx = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        # 上一次到目标的距离，给 progress reward 用
        self.last_distance = torch.zeros(self.num_envs, device=device)
        # 累积已到达数（每个 ep）
        self.num_reached = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        # 当前 step 是否刚到达一个 wp（脉冲信号，给 reward 用）
        self.just_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        # 是否所有 wp 完成（脉冲）
        self.just_finished_all = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        # 距离每步减少量（dense reward 用），首步=0
        self.progress = torch.zeros(self.num_envs, device=device)

    # -------------------------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        """返回当前 waypoint 在世界系下的 XY (num_envs, 2)。

        通常使用方会再转到 base 系，见 observations.target_pos_b。
        """
        idx = self.current_idx.clamp(max=self.cfg.num_waypoints - 1)
        return self.waypoints.gather(1, idx.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)

    # -------------------------------------------------------------------------
    def _update_metrics(self) -> None:
        # base 当前 XY（世界系）
        base_xy = self.robot.data.root_pos_w[:, :2]
        target = self.command
        dist = torch.linalg.norm(target - base_xy, dim=1)

        # 进展量 = 上一次距离 - 当前距离
        # 注意首次需要初始化 last_distance
        self.metrics["target_distance"] = dist
        self.metrics["num_reached"] = self.num_reached.float()

        # 到达判定
        reach_mask = (dist < self.cfg.reach_radius) & (self.current_idx < self.cfg.num_waypoints)
        self.just_reached = reach_mask
        self.num_reached = self.num_reached + reach_mask.long()
        # 推进 idx（最多到 N，不溢出）
        self.current_idx = torch.where(
            reach_mask,
            torch.clamp(self.current_idx + 1, max=self.cfg.num_waypoints),
            self.current_idx,
        )
        self.just_finished_all = (self.num_reached == self.cfg.num_waypoints) & reach_mask

        # 进展量基于"新当前目标"重算
        new_target = self.command
        new_dist = torch.linalg.norm(new_target - base_xy, dim=1)
        progress = self.last_distance - new_dist
        # ⚠️ 到达瞬间 idx 已切换到下一个 wp → new_dist 会跳变到与新 wp 的距离，
        # 此时把 progress 钳为 0，避免污染 dense reward；reach_waypoint 脉冲奖励另算。
        progress = torch.where(reach_mask, torch.zeros_like(progress), progress)
        self.progress = progress
        self.last_distance = new_dist

    # -------------------------------------------------------------------------
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """在 ep reset 时调用：重新生成该批 env 的 waypoint 队列。"""
        device = self.device
        n = env_ids.numel()
        N = int(self.cfg.num_waypoints)
        rmin = float(self.cfg.spawn_radius_min)
        rmax = float(self.cfg.spawn_radius_max)

        # 以 env 原点为参考，依次采样 N 个 waypoint
        # 每个 wp 距上一个点距离 ∈ [rmin, rmax]，方向均匀
        env_origins = self._env.scene.env_origins[env_ids, :2]  # (n, 2)
        prev = env_origins.clone()
        wps = torch.zeros(n, N, 2, device=device)
        for i in range(N):
            r = torch.empty(n, device=device).uniform_(rmin, rmax)
            theta = torch.empty(n, device=device).uniform_(-3.14159, 3.14159)
            offset = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
            wps[:, i, :] = prev + offset
            prev = wps[:, i, :]

        self.waypoints[env_ids] = wps
        self.current_idx[env_ids] = 0
        self.num_reached[env_ids] = 0
        # 初始化 last_distance（用第一个 waypoint 与机器人当前位置）
        base_xy = self.robot.data.root_pos_w[env_ids, :2]
        self.last_distance[env_ids] = torch.linalg.norm(wps[:, 0, :] - base_xy, dim=1)

    def _update_command(self) -> None:
        # CommandTerm 父类需要这个钩子；本类没有"漂移式"指令更新，留空
        pass

    # 可视化（debug_vis）——在当前目标 wp 处画一个绿球
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_goal_marker"):
                from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
                import isaaclab.sim as sim_utils
                cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/waypoint_goal",
                    markers={
                        "goal": sim_utils.SphereCfg(
                            radius=float(self.cfg.reach_radius),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.1, 0.9, 0.2), opacity=0.5,
                            ),
                        ),
                    },
                )
                self._goal_marker = VisualizationMarkers(cfg)
            self._goal_marker.set_visibility(True)
        elif hasattr(self, "_goal_marker"):
            self._goal_marker.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_goal_marker"):
            return
        # 当前目标 XY (world)， z 取机器人高度 + 0.5m
        target_xy = self.command
        z = self.robot.data.root_pos_w[:, 2:3] + 0.5
        pos = torch.cat([target_xy, z], dim=1)
        self._goal_marker.visualize(translations=pos)


@configclass
class WaypointCommandCfg(CommandTermCfg):
    class_type: type = WaypointCommand
    asset_name: str = MISSING
    num_waypoints: int = 4
    reach_radius: float = 0.5
    spawn_radius_min: float = 2.0
    spawn_radius_max: float = 8.0
    # 父类要求字段（resampling_time_range/debug_vis），这里设为占位
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # 永不主动重采样
    debug_vis: bool = False
