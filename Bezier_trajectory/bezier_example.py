import os
import signal
import sys
import time

import numpy as np
import pinocchio
import example_robot_data

import crocoddyl
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution

# 运行开关
WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT    = "plot"    in sys.argv or "CROCODYL_PLOT"    in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Bézier 辅助函数
def _bezier_cubic(P0, P1, P2, P3, t):
    t = np.asarray(t)
    B0 = (1 - t) ** 3
    B1 = 3 * (1 - t) ** 2 * t
    B2 = 3 * (1 - t) * t ** 2
    B3 = t ** 3
    p = (B0[:, None] * P0 + B1[:, None] * P1 + B2[:, None] * P2 + B3[:, None] * P3)

    dB0 = -3 * (1 - t) ** 2
    dB1 =  3 * (1 - t) ** 2 - 6 * (1 - t) * t
    dB2 =  6 * (1 - t) * t - 3 * t ** 2
    dB3 =  3 * t ** 2
    dp = (dB0[:, None] * P0 + dB1[:, None] * P1 + dB2[:, None] * P2 + dB3[:, None] * P3)
    return p, dp

def sample_xy_yaw_bezier(P0, P1, P2, P3, N, yaw_mode="tangent"):
    t = np.linspace(0.0, 1.0, N)
    pos, dpos = _bezier_cubic(np.array(P0), np.array(P1), np.array(P2), np.array(P3), t)
    x, y = pos[:,0], pos[:,1]
    if yaw_mode == "tangent":
        yaw = np.arctan2(dpos[:,1], dpos[:,0] + 1e-9)
    else:
        yaw_const = np.arctan2(P3[1]-P0[1], P3[0]-P0[0])
        yaw = np.full_like(x, yaw_const)
    return x, y, yaw

def build_frames_from_xy_yaw(x, y, yaw, base_height=0.42):
    frames = []
    for k in range(len(x)):
        c, s = np.cos(yaw[k]), np.sin(yaw[k])
        R = np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0,0.0,1.0]])
        p = np.array([x[k], y[k], base_height])
        frames.append(pinocchio.SE3(R, p))
    return frames

def inject_bezier_refs(problem, model, base_frame_id, frames_SE3, w_run=1e2, w_term=5e2):
    """向 shooting problem 注入 Bézier 基座位姿跟踪成本。"""
    for k in range(problem.T):
        running = problem.runningModels[k]
        if hasattr(running, "differential"):
            dmodel = running.differential
        else:
            dmodel = running
        state = dmodel.state
        nu    = getattr(dmodel, "nu", 0)
        target = frames_SE3[min(k, len(frames_SE3)-1)]

        res = crocoddyl.ResidualModelFramePlacement(state, base_frame_id, target, nu)
        act = crocoddyl.ActivationModelWeightedQuad(np.ones(6))
        cost = crocoddyl.CostModelResidual(state, act, res)
        dmodel.costs.addCost("bezier_place", cost, float(w_run))

    term = problem.terminalModel
    if hasattr(term, "differential"):
        dmodel_t = term.differential
    else:
        dmodel_t = term
    state_t = dmodel_t.state
    nu_t   = getattr(dmodel_t, "nu", 0)
    target_t = frames_SE3[-1]

    res_t = crocoddyl.ResidualModelFramePlacement(state_t, base_frame_id, target_t, nu_t)
    act_t = crocoddyl.ActivationModelWeightedQuad(np.ones(6))
    cost_t= crocoddyl.CostModelResidual(state_t, act_t, res_t)
    dmodel_t.costs.addCost("bezier_place_term", cost_t, float(w_term))


# 加载模型 & 初始状态
anymal = example_robot_data.load("anymal")
q0     = anymal.model.referenceConfigurations["standing"].copy()
v0     = pinocchio.utils.zero(anymal.model.nv)
x0     = np.concatenate([q0, v0])

# 设置步态选择（walk）
lfFoot, rfFoot, lhFoot, rhFoot = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
gait = SimpleQuadrupedalGaitProblem(anymal.model, lfFoot, rfFoot, lhFoot, rhFoot)

# 只运行 “walking” 步态
value = {
    "stepLength": 0.25,
    "stepHeight": 0.15,
    "timeStep": 1e-2,
    "stepKnots": 50,     # 可按需调整
    "supportKnots": 2,
}
problem = gait.createWalkingProblem(
    x0,
    value["stepLength"],
    value["stepHeight"],
    value["timeStep"],
    value["stepKnots"],
    value["supportKnots"],
)

# 生成 Bézier 曲线参考
T = problem.T
A  = np.array([0.0, 0.0])
B  = np.array([1.0, 0.5])   # 终点位置 (x, y)
P1 = A + np.array([0.3, 0.0])
P2 = B + np.array([-0.3, 0.1])
x_ref, y_ref, yaw_ref = sample_xy_yaw_bezier(A, P1, P2, B, T+1, yaw_mode="tangent")
frames_SE3 = build_frames_from_xy_yaw(x_ref, y_ref, yaw_ref, base_height=0.42)

# 注入 Bézier 参考成本
inject_bezier_refs(problem, anymal.model, anymal.model.getFrameId("base"), frames_SE3, w_run=1e2, w_term=1e3)

# 创建求解器
solver = crocoddyl.SolverFDDP(problem)
solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])

xs = [x0 for _ in range(solver.problem.T + 1)]
us = [u for u in solver.problem.quasiStatic(xs[:-1])]

# 求解
solver.solve(xs, us, 500, False)

# 显示动画
if WITHDISPLAY:
    try:
        import gepetto
        gepetto.corbaserver.Client()
        display = crocoddyl.GepettoDisplay(anymal, 4, 4, [2.0,2.68,0.84,0.2,0.62,0.72,0.22])
    except Exception:
        display = crocoddyl.MeshcatDisplay(anymal)
    display.rate = -1
    display.freq = 1
    display.displayFromSolver(solver)

# 绘制收敛曲线
if WITHPLOT:
    log = solver.getCallbacks()[1]
    crocoddyl.plotOCSolution(solver.xs, solver.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2)
