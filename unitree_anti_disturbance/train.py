# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""PPO 训练入口（rsl_rl）—— 超参全部来自 config.yaml。

运行：
    ${ISAACLAB_PATH}/isaaclab.sh -p train.py --headless
    ${ISAACLAB_PATH}/isaaclab.sh -p train.py --headless --config my.yaml --num_envs 200
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Unitree anti-disturbance walking with PPO.")
parser.add_argument("--config", type=str, default=None, help="自定义 yaml；缺省读 ./config.yaml")
parser.add_argument("--num_envs", type=int, default=None, help="覆盖 yaml 中 scene.num_envs")
parser.add_argument("--max_iterations", type=int, default=None, help="覆盖 yaml 中 train.max_iterations")
parser.add_argument("--logdir", type=str, default="./logs/anti_disturbance")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# —— 在 AppLauncher 之前重新加载 yaml（如果用户指定）——
import config as _cfg_mod  # noqa: E402

if args.config:
    _cfg_mod.CFG = _cfg_mod.load_config(args.config)
if args.num_envs is not None:
    _cfg_mod.CFG.scene.num_envs = args.num_envs
if args.max_iterations is not None:
    _cfg_mod.CFG.train.max_iterations = args.max_iterations

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
import torch  # noqa: E402,F401

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from config import CFG  # noqa: E402
from env_cfg import UnitreeAntiDisturbanceEnvCfg  # noqa: E402


def build_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    t = CFG.train
    return RslRlOnPolicyRunnerCfg(
        seed=int(t.seed),
        num_steps_per_env=int(t.num_steps_per_env),
        max_iterations=int(t.max_iterations),
        save_interval=int(t.save_interval),
        experiment_name=str(t.experiment_name),
        run_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        empirical_normalization=bool(t.empirical_normalization),
        policy=dict(
            class_name="ActorCritic",
            init_noise_std=float(t.policy.init_noise_std),
            actor_hidden_dims=list(t.policy.actor_hidden_dims),
            critic_hidden_dims=list(t.policy.critic_hidden_dims),
            activation=str(t.policy.activation),
        ),
        algorithm=dict(
            class_name="PPO",
            value_loss_coef=float(t.algorithm.value_loss_coef),
            use_clipped_value_loss=bool(t.algorithm.use_clipped_value_loss),
            clip_param=float(t.algorithm.clip_param),
            entropy_coef=float(t.algorithm.entropy_coef),
            num_learning_epochs=int(t.algorithm.num_learning_epochs),
            num_mini_batches=int(t.algorithm.num_mini_batches),
            learning_rate=float(t.algorithm.learning_rate),
            schedule=str(t.algorithm.schedule),
            gamma=float(t.algorithm.gamma),
            lam=float(t.algorithm.lam),
            desired_kl=float(t.algorithm.desired_kl),
            max_grad_norm=float(t.algorithm.max_grad_norm),
        ),
    )


def main():
    env_cfg = UnitreeAntiDisturbanceEnvCfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner_cfg = build_runner_cfg()
    log_dir = os.path.join(args.logdir, runner_cfg.experiment_name, runner_cfg.run_name)
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=str(env.device))
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
