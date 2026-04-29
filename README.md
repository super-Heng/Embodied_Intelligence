# Unitree Humanoid · Anti-Disturbance Walking (Isaac Sim 6.0 / Isaac Lab 2.3)

并行 100 个宇树人形机器人实例，做"抗干扰行走"任务的 RL 训练骨架。

## 环境要求

- Ubuntu 22.04
- NVIDIA Driver ≥ 535（你当前 580.126.20 ✅）
- NVIDIA Isaac Sim **6.0**
- NVIDIA Isaac Lab **2.3**（已安装 `isaaclab` / `isaaclab_assets` / `isaaclab_tasks` / `isaaclab_rl`）
- GPU：L40（48GB，跑 100 envs 完全够用，建议 sim dt = 1/200, decimation = 4）

> Isaac Lab 2.3 已经把命名空间从 `omni.isaac.lab` 重命名为 `isaaclab`。本项目全部使用新命名空间。

## 目录结构

```
unitree_anti_disturbance/
├── README.md
├── config.yaml         # ⭐ 所有可调参数都在这里
├── config.py           # yaml -> SimpleNamespace loader
├── robot_cfg.py        # URDF -> USD 自动转换 + ArticulationCfg
├── env_cfg.py          # Scene / Observation / Action / Reward / Event / Termination
├── mdp/
│   ├── __init__.py
│   ├── rewards.py      # 关节力矩惩罚 / 姿态稳定 / 目标速度追踪
│   └── events.py       # 推力扰动
├── run_sim.py          # 仅可视化：N 个机器人 + 周期推力
└── train.py            # rsl_rl PPO 训练入口
```

## 配置入口（重要）

所有可调项集中在 [config.yaml](config.yaml)，分组：

| 分组 | 内容 |
|---|---|
| `robot` | URDF 路径、初始高度、根 link 名、腿/上身 PD 参数 |
| `scene` | num_envs、env_spacing |
| `sim` | dt、decimation、device、PhysX GPU 容量、求解器迭代 |
| `action` | 关节位置动作 scale / offset |
| `commands` | 速度指令采样区间、重采样间隔 |
| `disturbance` | 推力间隔 + xyz/yaw 速度强度 |
| `domain_rand` | 摩擦/质量随机化、reset 姿态/关节扰动 |
| `rewards` | 11 个奖励项权重 + 跟踪指数核宽 + 目标高度 |
| `terminations` | 摔倒高度阈值 |
| `train` | PPO 全部超参、policy/critic 网络结构 |

### URDF 路径

`config.yaml` 中 `robot.urdf_path: null` 时回退到环境变量 `UNITREE_URDF`：
```bash
export UNITREE_URDF=/abs/path/to/h1.urdf      # 或 g1.urdf / go2_description.urdf
```
首次运行会自动调用 `UrdfConverter` 转 USD，按 URDF 内容 hash 缓存到 `robot.usd_cache_dir`，后续启动秒开。

### 切换机器人

H1 → G1 只需要改 `config.yaml`：
```yaml
robot:
  urdf_path: /abs/path/to/g1.urdf
  init_pos: [0.0, 0.0, 0.78]
  base_body_names: ["pelvis"]
rewards:
  target_height: 0.78
terminations:
  min_base_height: 0.4
```

## 跑起来

```bash
# 1) 默认 config.yaml
${ISAACLAB_PATH}/isaaclab.sh -p run_sim.py --headless
${ISAACLAB_PATH}/isaaclab.sh -p train.py   --headless

# 2) 临时覆盖单项参数（无需改 yaml）
${ISAACLAB_PATH}/isaaclab.sh -p train.py --headless --num_envs 200 --max_iterations 5000

# 3) 加载自定义 yaml（多套实验配置切换）
${ISAACLAB_PATH}/isaaclab.sh -p train.py --headless --config configs/g1_hard.yaml
```

## 备注

- 若用 G1（23/29 dof）请相应调整 `robot_cfg.py` 中 `actuators` 的 `joint_names_expr` 与默认刚度阻尼。
- L40 上 100 envs 推荐 `physx.gpu_max_rigid_contact_count` 给到 2^23；本项目已设。
- Windows 上请勿运行——`omni.physics.tensors` GPU pipeline 只在 Linux + RTX 上稳定。
