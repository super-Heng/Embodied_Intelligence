# Unitree Go2 园区巡逻 + 避障 (Isaac Sim 6.0 / Isaac Lab 2.3)

分层 RL 框架：**低层** Go2 velocity policy（冻结） + **上层** PPO navigation policy（输出 `vx, vy, wz`）。
感知：64 线 2D Lidar + 1.6×1.0m 前向高度图 + 当前 waypoint 相对位姿。
课程化三阶段：散布障碍 → 内置粗糙地形 → 自定义园区 USD。

## 目录

```
unitree_navigation/
├── README.md
├── config.yaml             ⭐ 所有可调项
├── config.py               yaml -> SimpleNamespace loader
├── locomotion/
│   ├── __init__.py
│   ├── go2_cfg.py          Go2 ArticulationCfg
│   └── policy_loader.py    加载 .pt ckpt -> callable(obs)->joint_target
├── sensors/
│   ├── __init__.py
│   ├── lidar_cfg.py        RayCasterCfg (64 线水平扫描)
│   └── heightmap_cfg.py    RayCasterCfg (前向高度图网格)
├── terrains/
│   ├── __init__.py
│   ├── stage1_obstacles.py 平地 + 程序化散布盒子/圆柱
│   ├── stage2_terrain.py   TerrainGeneratorCfg 课程地形
│   └── stage3_campus.py    自定义园区 USD（占位）
├── mdp/
│   ├── __init__.py
│   ├── commands.py         WaypointCommand
│   ├── actions.py          LowLevelVelocityActionCfg（封装低层 ckpt）
│   ├── observations.py     目标位姿 / lidar / heightmap 归一化
│   ├── rewards.py          progress / reach / collision / proximity / fall
│   ├── events.py           waypoint 重生、障碍随机化
│   └── terminations.py     success / fall / severe_collision / timeout
├── env_cfg.py              NavigationEnvCfg + Stage1/2/3
├── train.py                上层 PPO 训练入口
└── play.py                 单 env GUI 可视化
```

## 训练流程（Ubuntu / L40）

### Phase 0：先准备低层 Go2 velocity ckpt

复用 Isaac Lab 自带任务（不用我写代码）：
```bash
cd ${ISAACLAB_PATH}
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --num_envs 4096 --headless --max_iterations 1500
```
约 1~2h。训练日志目录里找到 `model_*.pt`，拷过来：
```bash
mkdir -p unitree_navigation/locomotion/checkpoints
cp ${ISAACLAB_PATH}/logs/rsl_rl/unitree_go2_rough/<run>/model_1500.pt \
   unitree_navigation/locomotion/checkpoints/go2_velocity.pt
```

### Phase 1+：训上层 navigation

```bash
# Stage 1：平地 + 散布障碍
${ISAACLAB_PATH}/isaaclab.sh -p train.py --headless --num_envs 1024

# Stage 2：从 stage1 ckpt warm start，切换到课程地形
${ISAACLAB_PATH}/isaaclab.sh -p train.py --headless --num_envs 1024 \
    --stage 2 --resume logs/.../stage1/model_3000.pt

# Stage 3：园区 USD
${ISAACLAB_PATH}/isaaclab.sh -p train.py --headless --num_envs 256 \
    --stage 3 --resume logs/.../stage2/model_3000.pt
```

### 可视化

```bash
${ISAACLAB_PATH}/isaaclab.sh -p play.py --num_envs 1 --resume <ckpt>
```

## 实物部署

上层 policy 输出 `(vx, vy, wz)` → 直接喂给宇树 Go2 SDK 的 velocity 接口。
**locomotion 这层 sim2real gap 由宇树官方控制器消化**，无需迁移低层 ckpt。
