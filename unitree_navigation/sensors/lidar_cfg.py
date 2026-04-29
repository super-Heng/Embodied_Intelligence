# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""2D Lidar：水平 360° 单层扫描，N 条射线。

实现方式：用 isaaclab.sensors.RayCasterCfg + LidarPatternCfg。
RayCaster 的击中距离结果是 GPU tensor，零 CPU 同步。
"""

from __future__ import annotations

from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from config import CFG


def build_lidar_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot/base") -> RayCasterCfg:
    s = CFG.sensors.lidar
    return RayCasterCfg(
        prim_path=prim_path,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, float(s.height_offset))),
        attach_yaw_only=True,         # lidar 跟随机器人 yaw，不随 roll/pitch 倾斜
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),     # 单层水平
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=360.0 / int(s.num_rays),
        ),
        max_distance=float(s.max_range),
        update_period=float(s.update_period_s),
        debug_vis=False,
        # ⚠️ Isaac Lab RayCaster 用 warp BVH，只能击中 *启动期* 注册的 *静态 mesh*：
        #   - stage2/3 的地形 mesh、stage3 的占位楼（kinematic）会被命中
        #   - stage1 的 RigidObject 障碍 (dynamic) 不会被 lidar 看到 → 该阶段避障靠
        #     (a) collision_with_obstacle (ContactSensor) 反馈，(b) 课程后期切到 stage2
        # 想真正动态扫障，请改用 sensors.RayCasterCamera 或 RTX Lidar。
        mesh_prim_paths=["/World/ground"],
    )
