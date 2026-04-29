# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""Unitree humanoid 机器人配置：URDF -> USD 自动转换 + ArticulationCfg。

所有可调项均来自 config.yaml -> config.CFG。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from config import CFG


def _resolve_urdf_path() -> str:
    p = CFG.robot.urdf_path or os.environ.get("UNITREE_URDF")
    if not p:
        raise FileNotFoundError(
            "未指定 URDF 路径：在 config.yaml 设置 robot.urdf_path，"
            "或导出环境变量 UNITREE_URDF=/abs/path/to/xxx.urdf"
        )
    p = os.path.expanduser(p)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"找不到 URDF: {p}")
    return p


def convert_urdf_to_usd(urdf_path: str | None = None, force: bool | None = None) -> str:
    """把 URDF 转为 USD 并缓存。返回 USD 绝对路径。"""
    urdf_path = urdf_path or _resolve_urdf_path()
    force = CFG.robot.force_reconvert if force is None else force
    cache_dir = Path(os.path.expanduser(CFG.robot.usd_cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)

    h = hashlib.md5()
    with open(urdf_path, "rb") as f:
        h.update(f.read())
    tag = h.hexdigest()[:10]
    stem = Path(urdf_path).stem
    usd_path = cache_dir / f"{stem}_{tag}.usd"

    if usd_path.is_file() and not force:
        return str(usd_path)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=str(cache_dir),
        usd_file_name=usd_path.name,
        fix_base=False,
        merge_fixed_joints=True,
        convert_mimic_joints_to_normal_joints=True,
        replace_cylinders_with_capsules=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
            target_type="position",
            drive_type="force",
        ),
    )
    UrdfConverter(cfg)
    if not usd_path.is_file():
        candidates = list(cache_dir.glob(f"{stem}*.usd"))
        if not candidates:
            raise RuntimeError(f"URDF 转换失败：{urdf_path}")
        usd_path = candidates[-1]
    return str(usd_path)


def build_unitree_articulation_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    usd_path = convert_urdf_to_usd()

    # 把 yaml 中的 actuator_groups 动态展开成 ImplicitActuatorCfg
    actuators: dict[str, ImplicitActuatorCfg] = {}
    for name, g in vars(CFG.robot.actuator_groups).items():
        actuators[name] = ImplicitActuatorCfg(
            joint_names_expr=list(g.joint_names_expr),
            stiffness=float(g.stiffness),
            damping=float(g.damping),
            effort_limit=float(g.effort_limit),
            velocity_limit=float(g.velocity_limit),
        )

    init_pos = tuple(CFG.robot.init_pos)

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=int(CFG.sim.physx.max_position_iteration_count),
                solver_velocity_iteration_count=int(CFG.sim.physx.max_velocity_iteration_count),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators=actuators,
    )
