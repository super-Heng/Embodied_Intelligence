# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""仅可视化：N 个机器人并行物理 + 周期推力。参数读 config.yaml。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Unitree parallel sim demo.")
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

import config as _cfg_mod  # noqa: E402

if args.config:
    _cfg_mod.CFG = _cfg_mod.load_config(args.config)
if args.num_envs is not None:
    _cfg_mod.CFG.scene.num_envs = args.num_envs

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from config import CFG  # noqa: E402
from robot_cfg import build_unitree_articulation_cfg  # noqa: E402


@configclass
class _DemoSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(500.0, 500.0)),
    )
    sky = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)),
    )
    robot = build_unitree_articulation_cfg(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=float(CFG.sim.dt), device=str(CFG.sim.device))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[12.0, 12.0, 6.0], target=[0.0, 0.0, 1.0])

    scene = InteractiveScene(_DemoSceneCfg(
        num_envs=int(CFG.scene.num_envs),
        env_spacing=float(CFG.scene.env_spacing),
    ))
    sim.reset()
    print(f"[INFO] Spawned {scene.num_envs} robots on {sim.device}.")

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    push_every = int(3.0 / sim_dt)
    vx_lo, vx_hi = CFG.disturbance.velocity_range.x
    vy_lo, vy_hi = CFG.disturbance.velocity_range.y

    step = 0
    while simulation_app.is_running():
        if step % push_every == 0 and step > 0:
            root_state = robot.data.root_state_w.clone()
            n = scene.num_envs
            root_state[:, 7] += torch.empty(n, device=sim.device).uniform_(vx_lo, vx_hi)
            root_state[:, 8] += torch.empty(n, device=sim.device).uniform_(vy_lo, vy_hi)
            robot.write_root_state_to_sim(root_state)

        robot.set_joint_effort_target(torch.zeros_like(robot.data.joint_pos))
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
