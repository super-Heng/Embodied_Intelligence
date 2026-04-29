# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""yaml -> SimpleNamespace loader（与 unitree_anti_disturbance 同款）。"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _ns_to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_ns_to_dict(v) for v in obj]
    return obj


def load_config(path: str | os.PathLike | None = None) -> SimpleNamespace:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "需要 PyYAML：${ISAACLAB_PATH}/isaaclab.sh -p -m pip install pyyaml"
        ) from e

    p = Path(path) if path else DEFAULT_CONFIG
    if not p.is_file():
        raise FileNotFoundError(f"找不到配置文件：{p}")
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _to_ns(raw)


CFG: SimpleNamespace = load_config()


def cfg_as_dict() -> dict:
    return _ns_to_dict(CFG)
