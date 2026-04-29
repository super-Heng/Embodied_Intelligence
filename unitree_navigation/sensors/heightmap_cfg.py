# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""前向高度图：在 base 前方一个矩形网格内向下射线，得到地形/障碍相对高度。"""

from __future__ import annotations

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from config import CFG


def build_heightmap_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot/base",
    mesh_prim_paths: list[str] | None = None,
) -> RayCasterCfg:
    h = CFG.sensors.heightmap
    size_x, size_y = float(h.size_m[0]), float(h.size_m[1])
    res = float(h.resolution)
    return RayCasterCfg(
        prim_path=prim_path,
        offset=RayCasterCfg.OffsetCfg(pos=(float(h.forward_offset), 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=res,
            size=(size_x, size_y),
        ),
        max_distance=10.0,
        update_period=float(h.update_period_s),
        debug_vis=False,
        # 只对静态地形射线 —— 详见 lidar_cfg.py 中的同样限制说明。
        mesh_prim_paths=list(mesh_prim_paths) if mesh_prim_paths else ["/World/ground"],
    )


def heightmap_dim() -> int:
    """计算 heightmap 的输出维度（用于 obs_dim 推算）。"""
    h = CFG.sensors.heightmap
    nx = int(round(h.size_m[0] / h.resolution)) + 1
    ny = int(round(h.size_m[1] / h.resolution)) + 1
    return nx * ny
