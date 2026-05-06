# Navigation 训练 Debug 接力笔记

> 给在服务器上接力 debug 的 agent 看。本文件只记录**当前未解决问题的最小上下文**，不重复 README。
> 本机：Windows 改代码 → push；服务器：Ubuntu 22.04 + L40 (48GB) + Isaac Sim 5.1 + Isaac Lab 5.1，仅 pull + 跑。

---

## 1. 目标

让 `unitree_navigation/train.py` 在 L40 上至少跑通 **一次** PPO 迭代：

```bash
cd ~/work_su/Embodied_Intelligence && git pull
cd unitree_navigation
HYDRA_FULL_ERROR=1 PYTHONFAULTHANDLER=1 \
  ~/isaaclab/IsaacLab/isaaclab.sh -p train.py \
  --headless --num_envs 4 --max_iterations 1 \
  2>&1 | tee /tmp/nav_train.log
grep "\[NAV-TRAIN\]" /tmp/nav_train.log
grep -E "Traceback|Error|FAILED|raise " /tmp/nav_train.log | tail -40
```

成功标准：日志里出现 `Actor Model: MLPModel(...)` + `Critic Model: MLPModel(...)` + `Learning iteration 0/1`，并正常退出。

---

## 2. 环境关键路径（服务器）

| 项 | 路径 |
| --- | --- |
| 项目 | `/home/LiXinYu/work_su/Embodied_Intelligence/` |
| Isaac Lab | `/home/LiXinYu/isaaclab/IsaacLab/` |
| Isaac Sim venv | `/home/LiXinYu/isaacsim/.venv/` (Python 3.11) |
| Asset root | `/home/LiXinYu/isaacsim_assets/Assets/Isaac/5.1/` |
| Go2 USD | `.../Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd` (8KB 入口 + Props/instanceable_meshes.usd) |

关键库版本（已踩过的坑都跟这个有关）：
- **rsl_rl ≥ 5.0**：用 `MLPModel` + `TensorDict`，cfg 顶层是 `actor` / `critic` / `algorithm` / `obs_groups`，不再是 `policy=`。
- **Isaac Lab 5.1**：`randomize_rigid_body_mass` / `randomize_rigid_body_material` 是 `ManagerTermBase` 子类（callable class），不是函数。
- 启动入口在 `IsaacLab/scripts/reinforcement_learning/rsl_rl/{train,play,cli_args}.py`（不是老的 `source/standalone/...`）。

---

## 3. 当前架构（一定要看 train.py / env_cfg.py）

- **分层 RL**：上层 PPO 输出 `(vx, vy, wz)` 3 维；底层冻结 Go2 velocity policy（48 维 obs → 12 维 joint target）在 `mdp/actions.py:LowLevelVelocityAction.apply_actions` 里 200Hz 跑，上层 decimation=20（10Hz）。
- **obs**：单一 group `"policy"`，shape `(N, 267)`（lidar 64 + heightmap 16×10=160 + 当前 base 状态等 + waypoint rel pose）。
- **act**：3 维连续。
- **stage 1 only**：平地 + 8 个 kinematic box 障碍/env，prim_path = `{ENV_REGEX_NS}/Obstacle_i`（已改扁平）。

---

## 4. 已解决的报错（按顺序，全是 commit）

| commit | 症状 | 修法 |
| --- | --- | --- |
| `561c6b6` | 文档老路径 | README 路径修正 |
| `315f06e` `bc452aa` | RayCaster mesh 找不到 / ckpt normalizer key 不匹配 / waypoint progress 抖动 | 多处补丁 |
| `19315e5` | 训练静默退出看不到错误 | 在 `train.py:main()` 加 `[NAV-TRAIN]` step prints + try/except 包裹 traceback |
| `9046460` | `RuntimeError: Unable to find source prim path: '/World/envs/env_.*/Obstacles'` | 障碍物 prim_path 扁平到 `{ENV_REGEX_NS}/Obstacle_i` |
| `b702194` | `TypeError: randomize_rigid_body_mass.__init__() got unexpected keyword 'asset_cfg'` 在 `event_manager.py:276` | 注释掉 startup 期 `physics_material` + `add_base_mass` 两个 EventTerm |
| `865ee4f` ⭐ 最新 | `KeyError: 'class_name'` 在 `rsl_rl/algorithms/ppo.py:477` `cfg["actor"].pop("class_name")` | `train.py:build_runner_cfg()` 由 `policy=RslRlPpoActorCriticCfg(...)` 迁移到 `actor=`/`critic=RslRlMLPModelCfg(...)` + `obs_groups={"actor":["policy"], "critic":["policy"]}` |

---

## 5. 当前未解决：跑 `865ee4f` 后用户回报"还是有问题"

⚠️ **缺 traceback**：服务器上 agent 第一步必须重新跑上面那条命令，把 `[NAV-TRAIN]` 最后一行 + `Traceback` 完整贴到上下文。

可能的疑点（按概率从高到低，给后续 agent 节省探路时间）：

### 疑点 A：`RslRlMLPModelCfg` 我塞了过时字段
我在 actor/critic cfg 里同时设了新字段（`distribution_cfg`）和**已 deprecated 的兼容字段**（`stochastic`、`init_noise_std`），目的是双保险。但 `@configclass` 可能把这些字段直接塞进 `**cfg["actor"]` 传给 `MLPModel(...)`，导致 `TypeError: __init__() got an unexpected keyword argument 'stochastic'`。

**预案**：删掉 `actor_cfg` / `critic_cfg` 里的 `stochastic=...` 和 `init_noise_std=...`，只保留 `distribution_cfg`（critic 的 `distribution_cfg=None` 表示确定性）。

### 疑点 B：`obs_groups` 里 critic 也只读 `"policy"` 是否合法
`resolve_obs_groups` 默认要求 `"actor"` 和 `"critic"` 两个 set 都存在。我们都映射到 `["policy"]`。理论上没问题，因为 obs 里就是 `policy` 这一组。

### 疑点 C：`RslRlMLPModelCfg.GaussianDistributionCfg` 的命名空间
我用了 `RslRlMLPModelCfg.GaussianDistributionCfg(...)`，但 IsaacLab 源码里是 `RslRlMLPModelCfg` 类内部 `class DistributionCfg` / `class GaussianDistributionCfg`。访问路径可能是 `RslRlMLPModelCfg.GaussianDistributionCfg` ✓ 也可能要从模块 import。如果报 `AttributeError`，改 import：

```python
# 翻 IsaacLab 源 source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py 看实际暴露
```

### 疑点 D：rsl_rl 5.x 的 `OnPolicyRunner` 构造签名
我的调用：`OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=...)`。如果 5.x 改了签名（比如必须 `train_cfg=`），会立刻抛 `TypeError`。

### 疑点 E：env.observation_space 是 dict 而 wrapper 期望 TensorDict
日志里看到 `obs space: {'policy': (4, 267)}`，wrapper 进去之后 PPO `construct_algorithm` 拿到的是 TensorDict。如果哪里出现 `KeyError: 'policy'` 在 `resolve_obs_groups` 内部 → `obs_groups` 写错。

---

## 6. 如何继续 debug（推荐流程）

1. **先复现拿 traceback**：
   ```bash
   cd ~/work_su/Embodied_Intelligence && git pull
   cd unitree_navigation
   HYDRA_FULL_ERROR=1 PYTHONFAULTHANDLER=1 \
     ~/isaaclab/IsaacLab/isaaclab.sh -p train.py \
     --headless --num_envs 4 --max_iterations 1 2>&1 | tee /tmp/nav_train.log
   tail -120 /tmp/nav_train.log
   ```

2. **比对官方 5.1 + rsl_rl 5.x 的最简 PPO cfg**：
   ```bash
   # IsaacLab 自带 Go2 velocity 任务的 agents cfg 是黄金参考
   find ~/isaaclab/IsaacLab/source/isaaclab_tasks -path '*go2*' -name '*ppo*.py'
   find ~/isaaclab/IsaacLab/source/isaaclab_tasks -path '*velocity*' -name 'rsl_rl_ppo_cfg.py'
   ```
   照着它的 `RslRlOnPolicyRunnerCfg(...)` 字段对齐 `train.py:build_runner_cfg()`。**官方怎么写就怎么抄**，不要发明字段。

3. **看 rsl_rl 5.x 的 MLPModel 构造**：
   ```bash
   python -c "from rsl_rl.models import MLPModel; help(MLPModel.__init__)"
   ```
   然后核对 `train.py` 里 actor/critic cfg 的字段名是否完全匹配。

4. **改完别再加新的兼容字段**。当前留的 `stochastic` / `init_noise_std` 那两行可以直接删（疑点 A）。删之前先确认 traceback 不是在那报。

5. 修完照常 commit：
   ```bash
   git -c user.email=you@example.com -c user.name=you commit -am "fix(navigation): <one-liner>"
   git push
   ```

---

## 7. 下一阶段（pipeline 通了之后再做，别现在动）

- 重新加回 startup 期域随机化：用 `body_names="base"`（Go2 没有 `trunk`），或者改 `mode="reset"`
- stage 2 / stage 3 的 USD 园区
- RayCaster 看不见动态障碍物（warp BVH 只支持 static mesh）→ 用 contact sensor 兜底或换成 RTX lidar

---

## 8. 关键文件锚点

- [unitree_navigation/train.py](unitree_navigation/train.py)：`build_runner_cfg()` 在第 ~60 行；`main()` 在第 ~90 行带 `[NAV-TRAIN]` print
- [unitree_navigation/env_cfg.py](unitree_navigation/env_cfg.py)：`_build_events_cls(stage)` 已注释掉两个 startup event
- [unitree_navigation/config.yaml](unitree_navigation/config.yaml)：所有数值
- [unitree_navigation/mdp/actions.py](unitree_navigation/mdp/actions.py)：底层 velocity policy 在 `apply_actions` 里被调用
- [unitree_navigation/locomotion/policy_loader.py](unitree_navigation/locomotion/policy_loader.py)：兼容 rsl_rl 5.x 的 normalizer key
