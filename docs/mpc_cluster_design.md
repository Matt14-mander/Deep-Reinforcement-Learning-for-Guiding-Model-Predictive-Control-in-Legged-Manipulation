# MPC Cluster 优化方案设计（EigenIPC 版）
## —— 参考 AugMPC/MPCHive (RA-L 2026) 架构解决 GPU 等待 CPU 问题

> 参考论文: "RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion" (IEEE RA-L, 2026-03)
> IPC 库: EigenIPC (AndrePatri, GPLv2)，本地克隆 `E:\Robot\hw毕业设计\EigenIPC`
> 开发环境: Windows（写代码，串行路径调试）；训练环境: Ubuntu（EigenIPC + MPC Cluster）

## 实施状态（2026-07-18）

- ✅ **Phase 0/1 代码完成**：`source/RL_Bezier_MPC/RL_Bezier_MPC/mpc_cluster/`
  （defs / backend / worker / launcher / client），env 集成
  （cfg 开关、`_pre_physics_step` 拆分、`_prepare_env_mpc_inputs` 共用、
  cluster guard、reset 转发、close、P0 计时每 500 步打印）
- ✅ 协议测试通过：`scripts/test_mpc_cluster_local.py`（local 线程后端，5 项断言：
  路由/RESET 一次性/异常隔离/solve_mask/关停），Windows 无 EigenIPC 可跑
- ⬜ Ubuntu 集成待做：装 EigenIPC → `use_mpc_cluster=True` 跑通 →
  串行 vs 集群 A/B（语义一致性 + steps/s）
- ⬜ Phase 2 流水线、Phase 3 扩容未开始

---

## 1. 问题定位

`quadruped_mpc_env.py::_pre_physics_step`（约 408 行起）：

```python
for env_idx in range(self.num_envs):          # 串行!
    solution = self.mpc_controllers[env_idx].solve(...)   # 5~30ms FDDP
```

- **并行度 = 1**：32 个 Crocoddyl FDDP 求解在 Isaac 主进程逐个执行
- 每控制步 CPU 耗时 ≈ 32 × (5~30ms) = 0.16~1.0s，GPU（PhysX+策略网络）全程空闲
- 扩大 num_envs 线性变慢 → 训练吞吐被单核锁死

**论文解法**：CPU 端 MPC Cluster Server 多进程并行求解全部实例，EigenIPC 共享内存与
仿真进程零拷贝交换数据，GPU 仿真与 CPU 求解流水线重叠 → 800 envs 闭环训练。

---

## 2. 目标架构

**关键设计决策：求解集群与 Isaac 是相互独立的进程，只通过命名共享内存耦合。**
这是 EigenIPC 带来的架构红利——workers 不是 Isaac 的子进程，彻底绕开
CUDA 进程 fork/spawn 的全部雷区，集群可独立启停/重启/调试。

```
┌────────────── Isaac 训练进程 (GPU) ── EigenIPC Server 侧 ──────────────┐
│ QuadrupedMPCEnv.__init__:                                              │
│   创建全部共享张量 (is_server=True) + Producer("mpc_go")               │
│   subprocess.Popen 启动 cluster launcher（或手动另开终端，便于调试）    │
│ _pre_physics_step:                                                     │
│   ① Bezier 轨迹生成/切片/padding (numpy, 原逻辑不动)                   │
│   ② states/com_ref/foot_pos/gait/cmd 写入共享张量                      │
│   ③ producer.trigger() → wait_ack_from(W, timeout)   (Phase 2: 不等)   │
│   ④ 读 out_* 张量 → 原 guard/提取/_apply_control 逻辑不动             │
└──────────────────────────┬─────────────────────────────────────────────┘
                    命名共享内存 (/dev/shm, namespace=run_id)
┌──────────────────────────┴─────────────────────────────────────────────┐
│              MPC Cluster Launcher（独立 python 进程）                   │
│   spawn W 个 worker（W ≈ 物理核-2；集群内部无 CUDA，fork 随意）         │
│  ┌─────────────────┐ ┌─────────────────┐     ┌─────────────────┐       │
│  │ Worker 0        │ │ Worker 1        │ ... │ Worker W-1      │       │
│  │ envs[0..E/W)    │ │ envs[E/W..)     │     │                 │       │
│  │ Client 侧张量 + │ │ Consumer.wait() │     │ OMP_THREADS=1   │       │
│  │ 专属控制器实例  │ │ →solve→ack()    │     │ taskset 绑核    │       │
│  └─────────────────┘ └─────────────────┘     └─────────────────┘       │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 EigenIPC API 映射（已对照仓库源码确认）

| 用途 | EigenIPC 原语 | 说明 |
|---|---|---|
| 数据张量 | `SharedTWrapper(namespace, basename, is_server, n_rows, n_cols, dtype, safe, force_reconnection)` | 2D 张量；env 侧 `is_server=True` 创建，worker 侧 attach；`run()` 后 `write(data,r,c)/read(r,c)`，带 `*_retry` 变体 |
| 触发 | `Producer(basename, namespace, ...)` | env 侧：`run()` → 每步 `trigger()` → `wait_ack_from(W, timeout)` |
| 应答 | `Consumer(basename, namespace, ...)` | worker 侧：`run()` → `wait()` 阻塞 → 求解 → `ack()` |
| GPU 镜像 | `with_torch_view=True, with_gpu_mirror=True` | 可选优化：env 侧张量直连 torch/GPU，省 torch→numpy 拷贝链 |

注意：SharedTWrapper 仅支持 2D → 所有布局摊平成 (E, k)。

### 2.2 共享张量布局（E = num_envs，dtype 对齐 crocoddyl 用 Double）

| basename | 形状 | dtype | 方向 | 内容 |
|---|---|---|---|---|
| `mpc_states` | (E, 37) | Double | env→wkr | q(19)+v(18)，Pinocchio 顺序 |
| `mpc_com_ref` | (E, 75) | Double | env→wkr | 25×3 已切片+padding，row-major 摊平 |
| `mpc_foot_pos` | (E, 12) | Double | env→wkr | 4×3 当前足端位置 |
| `mpc_foot_vel` | (E, 12) | Double | env→wkr | 4×3 世界系足端线速度 |
| `mpc_foot_contact` | (E, 4) | Int | env→wkr | LF/RF/LH/RH 物理接触状态 |
| `mpc_foot_force` | (E, 12) | Double | env→wkr | 4×3 世界系净接触力 |
| `mpc_gait` | (E, 3) | Double | env→wkr | step_length/height/frequency mod |
| `mpc_state_time` | (E, 1) | Double | env→wkr | 单调时钟采样时间戳（秒） |
| `mpc_protocol` | (E, 1) | Int | env→wkr | wire protocol version，当前为 3 |
| `mpc_state_ids` | (E, 2) | Int | env→wkr | physics_step_id, reset_generation |
| `mpc_cmd` | (E, 1) | Int | env→wkr | 位标志: SOLVE=1, RESET=2, IDLE=0 |
| `mpc_out_ctrl` | (E, 36) | Double | wkr→env | tau_ff(12), q_ref(12), dq_ref(12) |
| `mpc_out_meta` | (E, 8) | Double | wkr→env | cost, converged, status, solve_time, iterations, dynamics_gap, constraint_violation, source_timestamp |
| `mpc_out_ids` | (E, 3) | Int | wkr→env | source_state_id, solution_id, reset_generation |

namespace 统一为 `rlbmpc_<run_id>`，防多训练任务共存冲突。

**语义搬迁**：`solution` 对象不再跨边界。guard 回退逻辑（`_last_good_solutions`）
改为 env 侧持有 `(E,24)+(E,)` 的 last-good 数组，判据与行为不变（bug #12/#17 语义保持）。

### 2.3 Worker 主循环

```python
# mpc_cluster/worker.py —— 只 import numpy/crocoddyl/pinocchio + 本包 gait/controllers
def worker_main(namespace, worker_id, env_slice, cfg_dict):
    os.environ["OMP_NUM_THREADS"] = "1"                  # 防 Eigen 超订
    ctrls = {i: CrocoddylQuadrupedMPC(**cfg_dict) for i in env_slice}
    T = attach_tensors(namespace)                        # is_server=False
    consumer = Consumer("mpc_go", namespace); consumer.run()
    while not shutdown:
        consumer.wait()
        for i in env_slice:
            cmd = T.cmd.read(i, 0)
            if cmd & RESET: ctrls[i].reset()
            if cmd & SOLVE:
                try:
                    sol = ctrls[i].solve(...)            # 从 T.* 读输入
                    T.out_ctrl.write([sol.control, sol.predicted_states[1][7:19]], i, 0)
                    T.out_meta.write([sol.cost, sol.converged, OK], i, 0)
                except Exception:
                    T.out_meta.write([1e6, 0, EXC], i, 0)   # guard 在 env 侧处理
        consumer.ack()
```

### 2.4 硬约束

- **env→worker 静态亲和**：warm-start（`_prev_xs/us`）与 `_gait_clock` 驻留 worker 内
  控制器实例，同一 env 永远由同一 worker 求解，禁止动态负载均衡
- **reset 转发**：`_reset_idx` 置 RESET 位，worker 下次触发时先 `ctrl.reset()`
- **worker 异常必须 ack**：所有异常捕获后写 status 上报，绝不跳过 `ack()`（防屏障挂死）；
  env 侧 `wait_ack_from` 带超时 + 告警

---

## 3. 环境侧改动面

| 位置 | 改动 |
|---|---|
| cfg | `use_mpc_cluster: bool = False`（Windows 本地调试走原串行路径）、`cluster_workers`、`cluster_namespace` |
| `__init__` | cluster 模式：创建 server 张量 + Producer + Popen 启动 launcher；不创建进程内控制器 |
| `_pre_physics_step` | 循环拆三段：打包写入（原轨迹逻辑照搬）→ trigger/wait → 读出走原 guard/apply |
| `_reset_idx` | 追加 RESET 位写入 |
| `close` | trigger SHUTDOWN + 张量 close（server 侧负责 unlink） |

原串行路径完整保留：Windows 上开发调试逻辑、Ubuntu 上 A/B 验证语义一致性。

---

## 4. 分阶段实施

### Phase 0 — 基准测量（半天，可在 Windows 串行路径先做）
`_pre_physics_step` 加计时（MPC 循环 / 仿真步 / 总步长），500 步基线：
steps/s、MPC 占比。**这是论文"优化前后对比"的基线数据。**

### Phase 1 — 同步 Cluster（2~3 天，核心收益）
- 新模块 `source/RL_Bezier_MPC/RL_Bezier_MPC/mpc_cluster/`：
  `tensors.py`（张量表定义，server/client 两用）、`worker.py`、`launcher.py`、`client.py`（env 侧 API）
- Windows 上写代码 + 单测协议逻辑（EigenIPC import 失败时 mock）；Ubuntu 上集成
- 同步屏障语义与串行版完全一致 → 先验证训练统计一致，再谈提速
- 验收：steps/s ≈ ×min(W, E)/串行；两条训练曲线统计一致

### Phase 2 — 流水线重叠（1~2 天，吞吐再 ×2）
- 一步延迟控制：第 k 步应用第 k-1 步的解，`trigger()` 后立即返回不等 ack，
  下一步先 `wait_ack_from` 再读 → GPU 仿真与 CPU 求解完全重叠（论文异步模式）
- 代价：控制延迟 +20ms（warm-start + PD 可吸收）；reward 的 MPC cost 信号滞后一步
- reset 后首步：同步求解一次或应用站立姿态
- 独立开关 `mpc_pipeline: bool`，与 Phase 1 A/B 对比后再默认开启

### Phase 3 — 扩容与调优（1 天）
- num_envs 32→64→128；观察 straggler（最慢 env 拖屏障）
- 缓解：`mpc_max_iterations` 50→30 实验；worker `taskset` 绑核；
  可选 `with_gpu_mirror` 直写省拷贝
- 服务器核数决定 W 上限；论文口径 64 核跑 800 envs 可作外推参照

### Phase 4 — 实机复用（毕业设计后期）
- 同一套张量表 + Producer/Consumer 直接迁到 ros2_deploy：
  估计器/MPC/PD 多进程 200Hz 通信，EigenIPC rt 特性（低抖动、rt-safe 信号量）的主场
- EigenIPC 自带 ros_bridge / zmq_bridge 扩展，可桥接监控与遥测

---

## 5. Ubuntu 部署清单

```bash
# 方式一：conda（py3.7-3.11，IsaacLab 的 py3.10 兼容）
conda install -c AndrePatri eigenipc
# 方式二：源码
sudo apt install libeigen3-dev libboost-all-dev
cd EigenIPC && cmake -B build -DWITH_PYTHON=ON && cmake --build build -j && cmake --install build

# 训练（cluster 由 env 自动 Popen，或手动起便于看日志）：
python -m RL_Bezier_MPC.mpc_cluster.launcher --namespace rlbmpc_run1 --workers 30 &
python scripts/train_quadruped_mpc.py --num_envs 128  # cfg.use_mpc_cluster=True
```

## 6. 风险清单

| 风险 | 应对 |
|---|---|
| /dev/shm 残留（异常退出） | `force_reconnection=True` + server 侧 close/unlink + launcher 启动时清理同 namespace 旧段 |
| namespace 冲突（多任务共存） | namespace 含 run_id/PID |
| worker 异常挂死屏障 | 异常全捕获→status 上报→必 ack；env wait 超时告警并按 EXC 走 guard |
| straggler 拖屏障 | maxiter 上限、绑核、Phase 2 流水线天然缓解 |
| 流水线延迟致训练不稳 | 独立开关 A/B 对比 |
| EigenIPC 版本兼容 | conda 包滞后 main 分支——按论文复现口径优先源码构建并记录 commit |
| GPLv2 传染 | 集群为独立进程通过 IPC 通信；论文/仓库注明引用 |
