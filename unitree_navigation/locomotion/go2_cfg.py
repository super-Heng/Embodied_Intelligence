# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""Go2 ArticulationCfg —— 直接复用 isaaclab_assets 自带配置。

isaaclab_assets.UNITREE_GO2_CFG 已经把 USD/PD/joint 配好；
我们只在 prim_path / init_state 上做最小覆盖。
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


def build_go2_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    cfg = UNITREE_GO2_CFG.replace(prim_path=prim_path)
    # Go2 站立约 0.40m
    cfg.init_state = cfg.init_state.replace(pos=(0.0, 0.0, 0.40))
    return cfg
