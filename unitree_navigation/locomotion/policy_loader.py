# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""加载 Isaac Lab / rsl_rl 训出的 Go2 velocity policy ckpt，封装为可调用对象。

约定：
- ckpt 是 `OnPolicyRunner.save()` 产物，结构形如 {"model_state_dict": ..., "obs_norm_state_dict": ...}
- forward 输入：(num_envs, obs_dim)  GPU tensor
- forward 输出：(num_envs, action_dim) GPU tensor，是关节位置 target 的偏移量
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def _build_mlp(in_dim: int, hidden: list[int], out_dim: int, activation: str = "elu") -> nn.Sequential:
    act_cls = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act_cls()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class _EmpiricalNormalizer(nn.Module):
    """rsl_rl 的 EmpiricalNormalization 推理等价实现：(x - mean) / sqrt(var + eps)。"""

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class LocomotionPolicy(nn.Module):
    """加载 ckpt 并提供 act(obs) 接口。"""

    def __init__(
        self,
        ckpt_path: Optional[str],
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int],
        activation: str = "elu",
        empirical_normalization: bool = True,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._loaded = False

        self.actor = _build_mlp(obs_dim, hidden_dims, action_dim, activation).to(self.device)
        self.normalizer = (
            _EmpiricalNormalizer(obs_dim).to(self.device) if empirical_normalization else nn.Identity()
        )

        if ckpt_path:
            p = Path(ckpt_path)
            if not p.is_file():
                print(f"[WARN] locomotion ckpt 不存在: {p}；将输出零动作（仅供调试）")
            else:
                self._load(str(p))
                self._loaded = True
                print(f"[INFO] 已加载低层 locomotion ckpt: {p}")
        else:
            print("[WARN] 未指定 locomotion ckpt，将输出零动作")

        self.eval()
        for p_ in self.parameters():
            p_.requires_grad_(False)

    # -------------------------------------------------------------------------
    def _load(self, path: str) -> None:
        sd = torch.load(path, map_location=self.device, weights_only=False)
        # rsl_rl 不同版本字段名略有差异，做兼容
        model_sd = sd.get("model_state_dict") or sd.get("model") or sd
        # actor 权重命名通常 "actor.0.weight" 等
        actor_sd = {k[len("actor."):]: v for k, v in model_sd.items() if k.startswith("actor.")}
        if actor_sd:
            self.actor.load_state_dict(actor_sd, strict=False)
        # normalizer
        norm_sd = sd.get("obs_norm_state_dict") or sd.get("normalizer")
        if norm_sd and isinstance(self.normalizer, _EmpiricalNormalizer):
            mean = norm_sd.get("running_mean", norm_sd.get("mean"))
            var = norm_sd.get("running_var", norm_sd.get("var"))
            if mean is not None:
                self.normalizer.mean.copy_(mean.to(self.device))
            if var is not None:
                self.normalizer.var.copy_(var.to(self.device))

    # -------------------------------------------------------------------------
    @torch.inference_mode()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (N, obs_dim) -> (N, action_dim)"""
        if not self._loaded:
            return torch.zeros(obs.shape[0], self.action_dim, device=self.device)
        x = self.normalizer(obs)
        return self.actor(x)
