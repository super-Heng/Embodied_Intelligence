# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""园区巡逻+避障 ManagerBasedRLEnvCfg。

通过 build_env_cfg(stage=N) 返回对应阶段的配置实例。
"""

from __future__ import annotations

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CommandTermCfg as _CmdCfg,  # noqa: F401  保留以便扩展
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from config import CFG
from locomotion.go2_cfg import build_go2_cfg
from mdp import events as custom_events
from mdp import observations as obs
from mdp import rewards as custom_rewards
from mdp import terminations as custom_dones
from mdp.actions import LowLevelVelocityActionCfg
from mdp.commands import WaypointCommandCfg
from sensors.heightmap_cfg import build_heightmap_cfg
from sensors.lidar_cfg import build_lidar_cfg
from terrains import stage1_obstacles, stage2_terrain, stage3_campus


# =============================================================================
# Scene（按 stage 切换）
# =============================================================================
def _build_scene_cls(stage: int):
    """返回一个 InteractiveSceneCfg 子类（按 stage 装配）。"""

    # 静态 mesh 路径只可能被 raycaster 击中：
    #   - stage1：地面为一（障碍 dynamic，看不到）
    #   - stage2：地形 mesh为一
    #   - stage3：地面 + 占位楼 / 园区 USD。为避免击中机器人自身，
    #            不用 "/World" 根路径，而是列出占位楼 或 /World/Campus。
    if stage == 1:
        terrain_cfg = stage1_obstacles.build_terrain_cfg()
        obstacles = stage1_obstacles.build_obstacle_cfgs()
        extras: dict = {}
        sensor_meshes = ["/World/ground"]
    elif stage == 2:
        terrain_cfg = stage2_terrain.build_terrain_cfg()
        obstacles, extras = {}, {}
        sensor_meshes = ["/World/ground"]
    elif stage == 3:
        terrain_cfg = stage3_campus.build_terrain_cfg()
        obstacles = {}
        extras = stage3_campus.build_campus_assets()
        # 直接从 AssetBaseCfg 拿 prim_path（去掉末尾 /*），保证与场景实际 prim 一致
        sensor_meshes = ["/World/ground"]
        for c in extras.values():
            pp = getattr(c, "prim_path", None)
            if pp:
                sensor_meshes.append(pp)
        sensor_meshes = sorted(set(sensor_meshes))
    else:
        raise ValueError(f"unknown stage: {stage}")

    @configclass
    class _SceneCfg(InteractiveSceneCfg):
        terrain = terrain_cfg
        robot = build_go2_cfg(prim_path="{ENV_REGEX_NS}/Robot")
        # 只挂在 base/torso/trunk 上，避免脚正常触地误触发
        contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*(base|trunk).*",
            history_length=3,
            update_period=0.0,
            track_air_time=False,
        )
        lidar = build_lidar_cfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            mesh_prim_paths=sensor_meshes,
        )
        heightmap = build_heightmap_cfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            mesh_prim_paths=sensor_meshes,
        )
        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)),
        )

    # 把动态生成的障碍/园区作为类属性注入
    for name, c in {**obstacles, **extras}.items():
        setattr(_SceneCfg, name, c)
    return _SceneCfg


# =============================================================================
# Commands
# =============================================================================
@configclass
class CommandsCfg:
    waypoint = WaypointCommandCfg(
        asset_name="robot",
        num_waypoints=int(CFG.waypoint.num_waypoints),
        reach_radius=float(CFG.waypoint.reach_radius),
        spawn_radius_min=float(CFG.waypoint.spawn_radius_min),
        spawn_radius_max=float(CFG.waypoint.spawn_radius_max),
    )


# =============================================================================
# Actions
# =============================================================================
@configclass
class ActionsCfg:
    low_level_velocity = LowLevelVelocityActionCfg(
        asset_name="robot",
        vx_scale=float(CFG.action.vx_scale),
        vy_scale=float(CFG.action.vy_scale),
        wz_scale=float(CFG.action.wz_scale),
        smoothing_alpha=float(CFG.action.smoothing_alpha),
    )


# =============================================================================
# Observations
# =============================================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 本体
        base_lin_vel = ObsTerm(func=obs.base_lin_vel_b, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=obs.base_ang_vel_b, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=obs.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        last_action = ObsTerm(func=base_mdp.last_action)

        # 目标
        target_pos = ObsTerm(func=obs.target_pos_b, params={"command_name": "waypoint"})
        target_dist = ObsTerm(func=obs.target_distance, params={"command_name": "waypoint"})
        target_yaw_err = ObsTerm(func=obs.target_heading_err, params={"command_name": "waypoint"})

        # 感知
        lidar = ObsTerm(
            func=obs.lidar_distances,
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),
                "max_range": float(CFG.sensors.lidar.max_range),
            },
        )
        heightmap = ObsTerm(
            func=obs.heightmap_relative,
            params={
                "sensor_cfg": SceneEntityCfg("heightmap"),
                "max_height": float(CFG.sensors.heightmap.max_height),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Events（按 stage 启用障碍随机化）
# =============================================================================
def _build_events_cls(stage: int):
    @configclass
    class _EventsCfg:
        # ⚠️ startup 期的 randomize_rigid_body_material / _mass 在当前 IsaacLab 5.1 上
        # 用我们这套配置触发 "got an unexpected keyword argument 'asset_cfg'" 报错
        # （疑似 ManagerTermBase 实例化路径与函数调用路径选择问题）。这两项不影响导航
        # 任务首轮训练（域随机化对 sim2real 才关键），先注释掉，等 pipeline 通了再补。
        #
        # physics_material = EventTerm(
        #     func=base_mdp.randomize_rigid_body_material, mode="startup", params={...},
        # )
        # add_base_mass = EventTerm(
        #     func=base_mdp.randomize_rigid_body_mass, mode="startup", params={...},
        # )

        reset_base = EventTerm(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {k: tuple(v) for k, v in vars(CFG.domain_rand.reset_pose_range).items()},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            },
        )
        reset_joints = EventTerm(
            func=base_mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": tuple(CFG.domain_rand.reset_joint_pos_scale),
                "velocity_range": tuple(CFG.domain_rand.reset_joint_vel_range),
            },
        )

    if stage == 1:
        # 障碍随机化（reset 时摆放）
        _EventsCfg.randomize_obstacles = EventTerm(
            func=custom_events.randomize_stage1_obstacles,
            mode="reset",
        )
    return _EventsCfg


# =============================================================================
# Rewards
# =============================================================================
_W = CFG.rewards.weights


@configclass
class RewardsCfg:
    progress = RewTerm(
        func=custom_rewards.progress_to_waypoint,
        weight=float(_W.progress_to_waypoint),
        params={"command_name": "waypoint"},
    )
    reach = RewTerm(
        func=custom_rewards.reach_waypoint,
        weight=float(_W.reach_waypoint),
        params={"command_name": "waypoint"},
    )
    finish_all = RewTerm(
        func=custom_rewards.finish_all_waypoints,
        weight=float(_W.finish_all_waypoints),
        params={"command_name": "waypoint"},
    )
    heading = RewTerm(
        func=custom_rewards.heading_alignment,
        weight=float(_W.heading_alignment),
        params={"std": float(CFG.rewards.heading_std), "command_name": "waypoint"},
    )
    proximity = RewTerm(
        func=custom_rewards.lidar_proximity_penalty,
        weight=float(_W.lidar_proximity),
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "decay": float(CFG.rewards.proximity_decay),
            "max_range": float(CFG.sensors.lidar.max_range),
        },
    )
    collision = RewTerm(
        func=custom_rewards.collision_with_obstacle,
        weight=float(_W.collision),
        params={"sensor_cfg": SceneEntityCfg("contact_forces"), "threshold": 1.0},
    )
    action_rate = RewTerm(func=custom_rewards.action_rate_l2, weight=float(_W.action_rate))
    time = RewTerm(func=custom_rewards.time_penalty, weight=float(_W.time_penalty))
    fall = RewTerm(
        func=custom_rewards.fall_penalty,
        weight=float(_W.fall),
        params={
            "fall_height": float(CFG.terminations.fall_height),
            "fall_tilt_deg": float(CFG.terminations.fall_tilt_deg),
        },
    )


# =============================================================================
# Terminations
# =============================================================================
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=custom_dones.all_waypoints_reached, params={"command_name": "waypoint"})
    fall = DoneTerm(
        func=custom_dones.fell_over,
        params={
            "fall_height": float(CFG.terminations.fall_height),
            "fall_tilt_deg": float(CFG.terminations.fall_tilt_deg),
        },
    )
    collision = DoneTerm(
        func=custom_dones.severe_collision,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": float(CFG.terminations.collision_force_threshold),
        },
    )


# =============================================================================
# Env
# =============================================================================
def build_env_cfg(stage: int | None = None) -> ManagerBasedRLEnvCfg:
    stage = int(stage if stage is not None else CFG.curriculum.stage)
    SceneCls = _build_scene_cls(stage)
    EventsCls = _build_events_cls(stage)

    @configclass
    class NavigationEnvCfg(ManagerBasedRLEnvCfg):
        scene: SceneCls = SceneCls(
            num_envs=int(CFG.scene.num_envs),
            env_spacing=float(CFG.scene.env_spacing),
        )
        observations: ObservationsCfg = ObservationsCfg()
        actions: ActionsCfg = ActionsCfg()
        commands: CommandsCfg = CommandsCfg()
        rewards: RewardsCfg = RewardsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        events: EventsCls = EventsCls()

        def __post_init__(self):
            self.decimation = int(CFG.sim.decimation)
            self.episode_length_s = float(CFG.sim.episode_length_s)
            px = CFG.sim.physx
            self.sim: SimulationCfg = SimulationCfg(
                dt=float(CFG.sim.dt),
                render_interval=self.decimation,
                device=str(CFG.sim.device),
                physx=PhysxCfg(
                    solver_type=int(px.solver_type),
                    max_position_iteration_count=int(px.max_position_iteration_count),
                    max_velocity_iteration_count=int(px.max_velocity_iteration_count),
                    bounce_threshold_velocity=float(px.bounce_threshold_velocity),
                    gpu_max_rigid_contact_count=int(px.gpu_max_rigid_contact_count),
                    gpu_max_rigid_patch_count=int(px.gpu_max_rigid_patch_count),
                    gpu_found_lost_pairs_capacity=int(px.gpu_found_lost_pairs_capacity),
                    gpu_found_lost_aggregate_pairs_capacity=int(
                        px.gpu_found_lost_aggregate_pairs_capacity
                    ),
                    gpu_total_aggregate_pairs_capacity=int(px.gpu_total_aggregate_pairs_capacity),
                ),
            )
            self.viewer.eye = (10.0, 10.0, 6.0)

    return NavigationEnvCfg()
