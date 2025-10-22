import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio as pinocchio

import crocoddyl
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution

# -----------------------------------------------------------------------------
# 运行开关
# -----------------------------------------------------------------------------
WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# -----------------------------------------------------------------------------
# Bézier helpers
# -----------------------------------------------------------------------------
def _bezier_cubic(P0, P1, P2, P3, t):
    """返回三次 Bézier 曲线的位置 p(t) 和一阶导 dp/dt（未按实际时间缩放）。"""
    t = np.asarray(t)
    B0 = (1 - t) ** 3
    B1 = 3 * (1 - t) ** 2 * t
    B2 = 3 * (1 - t) * t ** 2
    B3 = t ** 3
    p = (B0[:, None] * P0 + B1[:, None] * P1 + B2[:, None] * P2 + B3[:, None] * P3)

    dB0 = -3 * (1 - t) ** 2
    dB1 = 3 * (1 - t) ** 2 - 6 * (1 - t) * t
    dB2 = 6 * (1 - t) * t - 3 * t ** 2
    dB3 = 3 * t ** 2
    dp = (dB0[:, None] * P0 + dB1[:, None] * P1 + dB2[:, None] * P2 + dB3[:, None] * P3)
    return p, dp


def sample_xy_yaw_bezier(P0, P1, P2, P3, N, yaw_mode="tangent"):
    """采样 N 个点，返回 (x, y, yaw)。yaw 为切向角或常数朝向。"""
    t = np.linspace(0.0, 1.0, N)
    pos, dpos = _bezier_cubic(np.array(P0), np.array(P1), np.array(P2), np.array(P3), t)
    x, y = pos[:, 0], pos[:, 1]
    if yaw_mode == "tangent":
        yaw = np.arctan2(dpos[:, 1], dpos[:, 0] + 1e-9)
    else:
        yaw_const = np.arctan2(P3[1] - P0[1], P3[0] - P0[0])
        yaw = np.full_like(x, yaw_const)
    return x, y, yaw


def build_frames_from_xy_yaw(x, y, yaw, base_height=0.42):
    """把 (x, y, yaw) 转成每个 knot 的 pinocchio.SE3（仅 z 轴旋转）。"""
    frames = []
    for k in range(len(x)):
        c, s = np.cos(yaw[k]), np.sin(yaw[k])
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        p = np.array([x[k], y[k], base_height])
        frames.append(pinocchio.SE3(R, p))
    return frames


def inject_bezier_refs(problem, model, base_frame_id, frames_SE3, w_run=1e2, w_term=5e2):
    """
    向整个 shooting problem 注入 Bézier 基座位姿跟踪成本（Residual-based API）。
    - running 节点：ResidualModelFramePlacement + CostModelResidual
    - terminal 节点：同上，权重更大
    """
    # 运行期
    for k in range(problem.T):
        running = problem.runningModels[k]
        dmodel = running.differential
        state = dmodel.state

        target = frames_SE3[min(k, len(frames_SE3) - 1)]
        res = crocoddyl.ResidualModelFramePlacement(state, base_frame_id, target)
        act = crocoddyl.ActivationModelWeightedQuad(np.ones(6))
        cost = crocoddyl.CostModelResidual(state, act, res)
        dmodel.costs.addCost("bezier_place", cost, float(w_run))

    # 终端
    target_t = frames_SE3[-1]
    term = problem.terminalModel
    dmodel_t = term.differential if hasattr(term, "differential") else term
    state_t = dmodel_t.state
    res_t = crocoddyl.ResidualModelFramePlacement(state_t, base_frame_id, target_t)
    act_t = crocoddyl.ActivationModelWeightedQuad(np.ones(6))
    cost_t = crocoddyl.CostModelResidual(state_t, act_t, res_t)
    dmodel_t.costs.addCost("bezier_place_term", cost_t, float(w_term))


# -----------------------------------------------------------------------------
# 加载模型 & 初始状态
# -----------------------------------------------------------------------------
anymal = example_robot_data.load("anymal")
q0 = anymal.model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(anymal.model.nv)
x0 = np.concatenate([q0, v0])

# -----------------------------------------------------------------------------
# 构建步态问题器（默认前向动力学；如需逆动力学可设 fwddyn=False）
# -----------------------------------------------------------------------------
lfFoot, rfFoot, lhFoot, rhFoot = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
gait = SimpleQuadrupedalGaitProblem(anymal.model, lfFoot, rfFoot, lhFoot, rhFoot)

# -----------------------------------------------------------------------------
# 任务配置（与你原脚本一致）
# -----------------------------------------------------------------------------
GAITPHASES = [
    {
        "walking": {
            "stepLength": 0.25,
            "stepHeight": 0.15,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 2,
        }
    },
    {
        "trotting": {
            "stepLength": 0.15,
            "stepHeight": 0.1,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 2,
        }
    },
    {
        "pacing": {
            "stepLength": 0.15,
            "stepHeight": 0.1,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 5,
        }
    },
    {
        "bounding": {
            "stepLength": 0.15,
            "stepHeight": 0.1,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 5,
        }
    },
    {
        "jumping": {
            "jumpHeight": 0.15,
            "jumpLength": [0.0, 0.3, 0.0],
            "timeStep": 1e-2,
            "groundKnots": 10,
            "flyingKnots": 20,
        }
    },
]

# -----------------------------------------------------------------------------
# 选择基座 frame（自动尝试常见名字）
# -----------------------------------------------------------------------------
base_frame_name_try = ["base", "base_link", "trunk", "floating_base"]
for _name in base_frame_name_try:
    if anymal.model.existFrame(_name):
        BASE_FRAME = _name
        break
else:
    raise RuntimeError(
        f"Cannot find a base frame among {base_frame_name_try}. "
        f"Some frames: {[f.name for f in anymal.model.frames[:20]]}"
    )
BASE_FID = anymal.model.getFrameId(BASE_FRAME)

# -----------------------------------------------------------------------------
# 主循环：逐阶段创建问题 -> 注入 Bézier 参考 -> 求解 -> 串接
# -----------------------------------------------------------------------------
solver = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "walking":
            problem = gait.createWalkingProblem(
                x0,
                value["stepLength"],
                value["stepHeight"],
                value["timeStep"],
                value["stepKnots"],
                value["supportKnots"],
            )
        elif key == "trotting":
            problem = gait.createTrottingProblem(
                x0,
                value["stepLength"],
                value["stepHeight"],
                value["timeStep"],
                value["stepKnots"],
                value["supportKnots"],
            )
        elif key == "pacing":
            problem = gait.createPacingProblem(
                x0,
                value["stepLength"],
                value["stepHeight"],
                value["timeStep"],
                value["stepKnots"],
                value["supportKnots"],
            )
        elif key == "bounding":
            problem = gait.createBoundingProblem(
                x0,
                value["stepLength"],
                value["stepHeight"],
                value["timeStep"],
                value["stepKnots"],
                value["supportKnots"],
            )
        elif key == "jumping":
            problem = gait.createJumpingProblem(
                x0,
                value["jumpHeight"],
                value["jumpLength"],
                value["timeStep"],
                value["groundKnots"],
                value["flyingKnots"],
            )
        else:
            raise ValueError(f"Unknown gait key: {key}")

        # === 新增：为该阶段注入 Bézier 基座参考 ===
        T = problem.T
        # 你可以按阶段改不同的 A/B；这里先统一写一条轨迹
        A = np.array([0.0, 0.0])
        B = np.array([0.6, 0.3])  # 终点（可按需要调整）
        P1 = A + np.array([0.20, 0.00])
        P2 = B + np.array([-0.20, 0.10])
        x_ref, y_ref, yaw_ref = sample_xy_yaw_bezier(A, P1, P2, B, T + 1, yaw_mode="tangent")
        frames_SE3 = build_frames_from_xy_yaw(x_ref, y_ref, yaw_ref, base_height=0.42)

        # 注入成本（权重可调；若不易收敛可先把 w_run 降到 1e1 ）
        inject_bezier_refs(problem, anymal.model, BASE_FID, frames_SE3, w_run=1e2, w_term=5e2)

        # 创建并保存求解器
        solver[i] = crocoddyl.SolverFDDP(problem)

    # 回调
    print("*** SOLVE " + key + " ***")
    if WITHPLOT:
        solver[i].setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
    else:
        solver[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # 初始化并求解
    xs = [x0] * (solver[i].problem.T + 1)
    us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
    solver[i].solve(xs, us, 100, False)

    # 串接阶段：末状态作为下一阶段初态
    x0 = solver[i].xs[-1]

# -----------------------------------------------------------------------------
# 显示动画
# -----------------------------------------------------------------------------
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        cameraTF = [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
        display = crocoddyl.GepettoDisplay(anymal, 4, 4, cameraTF)
    except Exception:
        display = crocoddyl.MeshcatDisplay(anymal)
    display.rate = -1
    display.freq = 1
    while True:
        for i, phase in enumerate(GAITPHASES):
            display.displayFromSolver(solver[i])
        time.sleep(1.0)

# -----------------------------------------------------------------------------
# 绘制收敛曲线
# -----------------------------------------------------------------------------
if WITHPLOT:
    plotSolution(solver, figIndex=1, show=False)
    for i, phase in enumerate(GAITPHASES):
        title = next(iter(phase.keys())) + " (phase " + str(i) + ")"
        log = solver[i].getCallbacks()[1]
        crocoddyl.plotConvergence(
            log.costs,
            log.pregs,
            log.dregs,
            log.grads,
            log.stops,
            log.steps,
            figTitle=title,
            figIndex=i + 3,
            show=True if i == len(GAITPHASES) - 1 else False,
        )
