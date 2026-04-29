# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""Stage 2：Isaac Lab 内置 TerrainGenerator —— 平地 / 障碍 / 缓坡 / 矮台阶混合。"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg


def build_terrain_cfg() -> TerrainImporterCfg:
    sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "obstacles": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.4,
            grid_width=0.45,
            grid_height_range=(0.05, 0.25),
            platform_width=2.0,
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.25),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.25),
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
        ),
    }
    gen = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=2.0,
        num_rows=10,
        num_cols=10,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        sub_terrains=sub_terrains,
        use_cache=True,
    )
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=gen,
        max_init_terrain_level=5,
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
