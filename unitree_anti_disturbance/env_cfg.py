# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""抗干扰行走 —— Manager-based RL Env 配置（参数全部来自 config.yaml）。"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from config import CFG
from mdp import events as custom_events
from mdp import rewards as custom_rewards
from robot_cfg import build_unitree_articulation_cfg


# =============================================================================
# Scene
# =============================================================================
@configclass
class UnitreeSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
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
    robot = build_unitree_articulation_cfg(prim_path="{ENV_REGEX_NS}/Robot")
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)),
    )


# =============================================================================
# Commands
# =============================================================================
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=tuple(CFG.commands.resampling_time_range),
        rel_standing_envs=float(CFG.commands.rel_standing_envs),
        rel_heading_envs=1.0,
        heading_command=bool(CFG.commands.heading_command),
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=tuple(CFG.commands.lin_vel_x),
            lin_vel_y=tuple(CFG.commands.lin_vel_y),
            ang_vel_z=tuple(CFG.commands.ang_vel_z),
            heading=(-math.pi, math.pi),
        ),
    )


# =============================================================================
# Actions
# =============================================================================
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=float(CFG.action.scale),
        use_default_offset=bool(CFG.action.use_default_offset),
    )


# =============================================================================
# Observations
# =============================================================================
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Events
# =============================================================================
def _dr_dict(ns) -> dict:
    """yaml 里 {x:[a,b], y:[c,d]} -> {"x":(a,b), "y":(c,d)}"""
    return {k: tuple(v) for k, v in vars(ns).items()}


@configclass
class EventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": tuple(CFG.domain_rand.friction_static_range),
            "dynamic_friction_range": tuple(CFG.domain_rand.friction_dynamic_range),
            "restitution_range": tuple(CFG.domain_rand.restitution_range),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(CFG.robot.base_body_names)),
            "mass_distribution_params": tuple(CFG.domain_rand.added_base_mass_range),
            "operation": "add",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": _dr_dict(CFG.domain_rand.reset_pose_range),
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": tuple(CFG.domain_rand.reset_joint_pos_scale),
            "velocity_range": tuple(CFG.domain_rand.reset_joint_vel_range),
        },
    )
    push_robot = EventTerm(
        func=custom_events.push_by_setting_velocity,
        mode="interval",
        interval_range_s=tuple(CFG.disturbance.interval_range_s),
        params={"velocity_range": _dr_dict(CFG.disturbance.velocity_range)},
    )


# =============================================================================
# Rewards
# =============================================================================
_W = CFG.rewards.weights


@configclass
class RewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=custom_rewards.track_lin_vel_xy_exp,
        weight=float(_W.track_lin_vel_xy),
        params={"std": float(CFG.rewards.tracking_std), "command_name": "base_velocity"},
    )
    track_ang_vel_z = RewTerm(
        func=custom_rewards.track_ang_vel_z_exp,
        weight=float(_W.track_ang_vel_z),
        params={"std": float(CFG.rewards.tracking_std), "command_name": "base_velocity"},
    )
    upright = RewTerm(func=custom_rewards.base_upright, weight=float(_W.upright))
    base_height = RewTerm(
        func=custom_rewards.base_height_target,
        weight=float(_W.base_height),
        params={"target_height": float(CFG.rewards.target_height)},
    )
    ang_vel_xy = RewTerm(func=custom_rewards.base_ang_vel_l2, weight=float(_W.ang_vel_xy))
    lin_vel_z = RewTerm(func=custom_rewards.base_lin_vel_z_l2, weight=float(_W.lin_vel_z))
    joint_torques = RewTerm(func=custom_rewards.joint_torques_l2, weight=float(_W.joint_torques))
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=float(_W.action_rate))
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=float(_W.joint_acc))
    alive = RewTerm(func=custom_rewards.survival_bonus, weight=float(_W.alive))
    termination = RewTerm(func=mdp.is_terminated, weight=float(_W.termination))


# =============================================================================
# Terminations
# =============================================================================
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_fell = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": float(CFG.terminations.min_base_height)},
    )


# =============================================================================
# Env
# =============================================================================
@configclass
class UnitreeAntiDisturbanceEnvCfg(ManagerBasedRLEnvCfg):
    scene: UnitreeSceneCfg = UnitreeSceneCfg(
        num_envs=int(CFG.scene.num_envs),
        env_spacing=float(CFG.scene.env_spacing),
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

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
                gpu_found_lost_aggregate_pairs_capacity=int(px.gpu_found_lost_aggregate_pairs_capacity),
                gpu_total_aggregate_pairs_capacity=int(px.gpu_total_aggregate_pairs_capacity),
            ),
        )
        self.viewer.eye = (8.0, 8.0, 5.0)
