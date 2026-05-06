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

复用 Isaac Lab 自带任务（不用我写代码）。**注意**：Isaac Lab 已把脚本目录从
`source/standalone/workflows/...` 迁到 `scripts/reinforcement_learning/...`，
旧文档里的路径已失效。

```bash
cd ${ISAACLAB_PATH}
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --num_envs 4096 --headless --max_iterations 1500
```
约 1~2h。训练日志目录里找到 `model_*.pt`，拷过来：
```bash
mkdir -p unitree_navigation/locomotion/checkpoints
cp ${ISAACLAB_PATH}/logs/rsl_rl/unitree_go2_rough/<run>/model_1500.pt \
   unitree_navigation/locomotion/checkpoints/go2_velocity.pt
```

> 兼容性提示：rsl_rl 5.0 起 ckpt 内字段从 `model_state_dict` / `obs_norm_state_dict`
> 切到了 `model_state_dict` 内嵌 `obs_normalizer.*` 前缀。我已经在
> `locomotion/policy_loader.py` 里同时兼容两种格式，无需手动改 ckpt。

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

## ⚠️ 已知限制 / 风险

1. **stage1 的 lidar 看不到动态障碍**
   Isaac Lab 的 `RayCaster` 走 warp BVH，启动时一次性烘焙静态 mesh。stage1 的盒子是
   `RigidObject`（reset 时 `write_root_state_to_sim`），warp BVH **不会跟着更新**，所以：
   - lidar 观测里 stage1 障碍是"透明"的 → 上层只能靠 `target_pos_b` + 撞上后的 `collision`
     反馈学避障。
   - **建议**：stage1 当作热身（学走 + 学奔向 wp），真正学避障从 stage2（地形烘焙静态）开始。
   - 想要真正的动态扫描，把 lidar 改成 `RayCasterCameraCfg`（CPU/RTX）或 IsaacSim 的
     `RtxLidarCfg`（需要 RTX，性能下降明显）。

2. **`base_lin_vel` 是 sim-only 观测**
   真机 IMU 给不出，部署前需要把这一项替换为 odom/legged-state estimator 估计值，
   或者去掉重训。这是宇树官方 locomotion policy 的同款限制。

3. **低层 ckpt 关节顺序必须与 `isaaclab_assets.UNITREE_GO2_CFG` 一致**
   如果你用第三方 ckpt（比如 legged_gym 训的），关节排列可能不同，需在
   `policy_loader.act` 里加置换矩阵。

4. **`waypoint` 采样不会避开障碍**
   极坐标随机采样可能落在障碍内。短期影响：episode 内 `unreachable wp → time_out`，
   但 dense progress 信号仍可学习。要严格避障可在 `_resample_command` 里加
   "采样 N 次 + lidar pre-check" 重试。

5. **`stage3_campus` 占位实现**
   默认是 4 栋 5×5×4m 楼。要换真实园区 USD：
   `export CAMPUS_USD=/abs/path/campus.usd` 后再训。注意 USD 内必须有命名为
   `/World/Campus/.../ground` 的可碰撞地面，否则要相应改 `mesh_prim_paths`。

6. **PhysX GPU buffer 与 num_envs 强相关**
   `config.yaml` 里默认按 1024 envs 配 `gpu_max_rigid_contact_count=2^24`。
   如果你显存紧把 num_envs 降到 256，可以把这些 buffer 减到 2^22 节省显存；
   反过来，envs > 2048 必须再加大，否则 sim 会 crash。
