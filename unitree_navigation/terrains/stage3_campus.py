# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""Stage 3：自定义园区 USD 场景（占位实现）。

真实园区 USD 由用户后续替换 CAMPUS_USD 路径。
占位场景 = 平地 + 几栋固定矩形"楼"。
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg

# 用户替换为真实园区 USD 绝对路径；环境变量优先
CAMPUS_USD = os.environ.get("CAMPUS_USD", "")


def build_terrain_cfg() -> TerrainImporterCfg:
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


def build_campus_assets() -> dict[str, AssetBaseCfg]:
    """返回 stage3 额外 prim：要么是 USD，要么是几栋占位楼。"""
    if CAMPUS_USD and os.path.isfile(CAMPUS_USD):
        return {
            "campus": AssetBaseCfg(
                prim_path="/World/Campus",
                spawn=sim_utils.UsdFileCfg(usd_path=CAMPUS_USD),
            )
        }
    # 占位：4 栋 5×5×4m 楼围在原点周围，留出十字通道
    bldgs: dict[str, AssetBaseCfg] = {}
    layouts = [(8.0, 8.0), (-8.0, 8.0), (8.0, -8.0), (-8.0, -8.0)]
    for i, (x, y) in enumerate(layouts):
        bldgs[f"bldg_{i}"] = AssetBaseCfg(
            prim_path=f"/World/Bldg_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(5.0, 5.0, 4.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.6)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, 2.0)),
        )
    return bldgs
