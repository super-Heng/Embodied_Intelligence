# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause
"""
读取 config.yaml 的统一入口。

使用：
    from config import load_config, CFG
    CFG = load_config()           # 默认读取 ./config.yaml
    print(CFG.scene.num_envs)

CLI 也可以通过 --config 指定其他 yaml；命令行覆盖 yaml。
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def _to_ns(obj: Any) -> Any:
    """递归把 dict 转成 SimpleNamespace，方便点访问；list/tuple/原生类型保持。"""
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
    """加载 yaml -> SimpleNamespace。未安装 PyYAML 时抛清晰错误。"""
    try:
        import yaml  # PyYAML, Isaac Sim Python 自带
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


# 模块级单例：首次 import 即加载。脚本里若要换 yaml，调 load_config(path) 后赋给 CFG 即可。
CFG: SimpleNamespace = load_config()


def cfg_as_dict() -> dict:
    """rsl_rl runner 需要的是 dict，提供一个反向转换。"""
    return _ns_to_dict(CFG)
