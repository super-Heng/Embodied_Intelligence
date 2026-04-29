# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""Stage 1：平地 + 程序化散布的盒子/圆柱障碍。

实现思路：
- 地面用 GroundPlaneCfg 走 TerrainImporterCfg
- 障碍用 RigidObjectCollectionCfg：每 env N 个障碍，初始位姿在 reset 事件里随机化
  （此处只声明 collection；随机化逻辑在 mdp/events.py 的 randomize_obstacles 里）
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.terrains import TerrainImporterCfg

from config import CFG


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


def build_obstacle_cfgs() -> dict[str, RigidObjectCfg]:
    """返回 env-local 障碍物 RigidObjectCfg 字典。

    spawn 处用一个统一尺寸的胶囊/盒子 placeholder；reset 事件里再
    通过 `write_root_pose_to_sim` + 物理材质改变实际半径不行 —— 因此这里
    简化为：所有障碍统一用 size_range_box 中位值的盒子，仅随机位姿。
    若需变尺寸障碍，可扩展为多个 collection。
    """
    o = CFG.stage1_obstacles
    n = int(o.num_per_env)
    mid_size = 0.5 * (float(o.size_range_box[0]) + float(o.size_range_box[1]))
    mid_h = 0.5 * (float(o.height_range[0]) + float(o.height_range[1]))

    cfgs: dict[str, RigidObjectCfg] = {}
    for i in range(n):
        cfgs[f"obstacle_{i}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Obstacles/Box_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(mid_size, mid_size, mid_h),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,           # 静态障碍：kinematic，不被推动但参与碰撞
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.3, 0.3)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(20.0 + i, 20.0, mid_h * 0.5)),
        )
    return cfgs
