# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""单 env GUI 可视化：加载训练好的 navigation ckpt，让 Go2 在场景里巡逻。"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play navigation policy.")
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--stage", type=int, default=None)
parser.add_argument("--resume", type=str, required=True, help="navigation policy ckpt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

import config as _cfg_mod  # noqa: E402

if args.config:
    _cfg_mod.CFG = _cfg_mod.load_config(args.config)
_cfg_mod.CFG.scene.num_envs = args.num_envs
if args.stage is not None:
    _cfg_mod.CFG.curriculum.stage = args.stage

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from config import CFG  # noqa: E402
from env_cfg import build_env_cfg  # noqa: E402


def main():
    env_cfg = build_env_cfg(stage=int(CFG.curriculum.stage))
    # 打开传感器 debug_vis（lidar 射线 / heightmap 网格）
    env_cfg.scene.lidar.debug_vis = True
    env_cfg.scene.heightmap.debug_vis = True

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    t = CFG.train
    runner_cfg = RslRlOnPolicyRunnerCfg(
        seed=int(t.seed),
        num_steps_per_env=int(t.num_steps_per_env),
        max_iterations=1,
        save_interval=10000,
        experiment_name="play",
        empirical_normalization=bool(t.empirical_normalization),
        policy=dict(
            class_name="ActorCritic",
            init_noise_std=float(t.policy.init_noise_std),
            actor_hidden_dims=list(t.policy.actor_hidden_dims),
            critic_hidden_dims=list(t.policy.critic_hidden_dims),
            activation=str(t.policy.activation),
        ),
        algorithm=dict(class_name="PPO", value_loss_coef=1.0, use_clipped_value_loss=True,
                       clip_param=0.2, entropy_coef=0.0, num_learning_epochs=1,
                       num_mini_batches=1, learning_rate=1e-4, schedule="fixed",
                       gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    )
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=str(env.device))
    runner.load(args.resume)
    print(f"[INFO] loaded navigation ckpt: {args.resume}")

    policy = runner.get_inference_policy(device=env.device)
    obs, _ = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
