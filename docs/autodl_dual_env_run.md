# AutoDL 双环境 MPC Cluster 运行指南

当前架构只需要一个训练终端。Isaac 进程由 `rlbmpc45` 的 Python 启动，
随后自动使用 `rlbmpc_mpc` 的 Python 启动 launcher 和全部 Crocoddyl worker；
两侧只通过 EigenIPC 共享内存交换数值张量。

## 1. 每次开机设置路径

```bash
export RLBMPC_ROOT=/root/autodl-tmp/rlbmpc_workspace
export ISAAC_ENV="$RLBMPC_ROOT/envs/rlbmpc45"
export MPC_ENV="$RLBMPC_ROOT/envs/rlbmpc_mpc"
export PROJECT_ROOT="$RLBMPC_ROOT/RL_Bezier_MPC"

cd "$PROJECT_ROOT"
git pull
```

两个环境都应以 editable 方式安装项目。若此前已经执行过，`git pull` 后无需重装；
否则分别执行：

```bash
"$ISAAC_ENV/bin/python" -m pip install --no-deps -e "$PROJECT_ROOT/source/RL_Bezier_MPC"
"$MPC_ENV/bin/python" -m pip install --no-deps -e "$PROJECT_ROOT/source/RL_Bezier_MPC"
```

## 2. 找到 Go2 URDF

项目训练协议要求自由浮动 Go2 模型为 `nq=19, nv=18`。必须传入真实 URDF，
不要让 worker 猜测模型来源。

```bash
export GO2_URDF="$(find "$RLBMPC_ROOT" -type f \( -name 'go2_description.urdf' -o -name 'go2.urdf' \) -print -quit)"
test -n "$GO2_URDF" || { echo '未找到 Go2 URDF，请先把 unitree_ros/go2_description 放到工作区'; exit 1; }
echo "$GO2_URDF"
```

验证 MPC 环境能够生成正确的自由浮动模型：

```bash
GO2_URDF="$GO2_URDF" "$MPC_ENV/bin/python" - <<'PY'
import os
from RL_Bezier_MPC.robots.quadruped_cfg import load_pinocchio_model

model, path = load_pinocchio_model(os.environ["GO2_URDF"], "go2", floating_base=True)
print("URDF:", path)
print("nq/nv:", model.nq, model.nv)
assert (model.nq, model.nv) == (19, 18)
PY
```

## 3. 先做小规模集成测试

```bash
cd "$PROJECT_ROOT"
OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_ENV/bin/python" scripts/train_quadruped_mpc.py \
  --headless \
  --num_envs 4 \
  --max_iterations 2 \
  --use_mpc_cluster \
  --cluster_workers 2 \
  --mpc_python "$MPC_ENV/bin/python" \
  --robot_urdf "$GO2_URDF"
```

日志中应同时看到：

- `[MPCClusterClient] launched cluster ... python=.../rlbmpc_mpc/bin/python`
- `[MPCWorker 0] python=.../rlbmpc_mpc/bin/python`
- 所有 worker 输出 `ready`
- 训练进入 PPO iteration，且没有 barrier timeout

## 4. 正式 Stage 1 训练

小规模测试通过后再扩大。4090/约 16 个 CPU 配额建议先从 16 个环境、8 个 worker 开始：

```bash
OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_ENV/bin/python" scripts/train_quadruped_mpc.py \
  --headless \
  --num_envs 16 \
  --max_iterations 500 \
  --use_mpc_cluster \
  --cluster_workers 8 \
  --mpc_python "$MPC_ENV/bin/python" \
  --robot_urdf "$GO2_URDF"
```

当前配置默认 `fix_gait_params=True`，运行的是四足 Go2 Stage 1。Stage 2 swing/gait
调制和 humanoid 训练属于后续代码开发，不应在本次 IPC 集成验证之前同时开启。
