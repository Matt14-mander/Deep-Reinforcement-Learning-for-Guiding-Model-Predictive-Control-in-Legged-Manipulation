import os, sys, time, signal
import numpy as np
import pinocchio as pin
import example_robot_data
import crocoddyl
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem

# 运行开关
WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT    = "plot"    in sys.argv or "CROCODDYL_PLOT"    in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# ----------------- 小工具：直线轨迹 → SE3 -----------------
def straight_xy_yaw(A, B, N, base_h=0.42, yaw_mode="tangent"):
    """
    生成 N 个 SE3 位姿：在 XY 上 A→B 直线；yaw 朝向行进方向；z=base_h
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    ts = np.linspace(0.0, 1.0, N)
    xy = (1 - ts)[:,None]*A[None,:] + ts[:,None]*B[None,:]
    if yaw_mode == "tangent":
        theta = np.arctan2((B - A)[1], (B - A)[0] + 1e-9)
        yaw = np.full(N, theta)
    else:
        yaw = np.zeros(N)
    frames = []
    for k in range(N):
        c, s = np.cos(yaw[k]), np.sin(yaw[k])
        R = np.array([[c, -s, 0.0],[s, c, 0.0],[0.0,0.0,1.0]])
        p = np.array([xy[k,0], xy[k,1], base_h])
        frames.append(pin.SE3(R, p))
    return frames

def choose_base_frame(model):
    for name in ["base","base_link","trunk","floating_base"]:
        if model.existFrame(name):
            return model.getFrameId(name), name
    raise RuntimeError(f"No suitable base frame found. Sample frames: {[f.name for f in model.frames[:20]]}")

def strip_conflicting_base_costs(problem):
    """删除 gait 内部可能已有的 base/CoM 相关代价，避免与我们自定义直线参考冲突。"""
    def rm(model_like):
        d = model_like.differential if hasattr(model_like, "differential") else model_like
        for key in list(d.costs.costs.keys()):
            low = key.lower()
            if any(k in low for k in ["base", "com", "pelvis"]):
                d.costs.removeCost(key)
    for k in range(problem.T):
        rm(problem.runningModels[k])
    rm(problem.terminalModel)

def inject_line_refs(problem, base_fid, frames_SE3, w_run=1e4, w_term=1e6):
    """在 running/terminal 节点添加 FramePlacement 残差代价，跟踪直线参考。"""
    # running
    for k in range(problem.T):
        running = problem.runningModels[k]
        d = running.differential if hasattr(running,"differential") else running
        state = d.state
        nu    = getattr(d, "nu", 0)
        target = frames_SE3[min(k, len(frames_SE3)-1)]
        res = crocoddyl.ResidualModelFramePlacement(state, base_fid, target, nu)
        act = crocoddyl.ActivationModelWeightedQuad(np.ones(6))
        d.costs.addCost("line_place", crocoddyl.CostModelResidual(state, act, res), w_run)
    # terminal
    term = problem.terminalModel
    dT = term.differential if hasattr(term,"differential") else term
    stateT = dT.state
    nuT = getattr(dT, "nu", 0)
    targetT = frames_SE3[-1]
    resT = crocoddyl.ResidualModelFramePlacement(stateT, base_fid, targetT, nuT)
    actT = crocoddyl.ActivationModelWeightedQuad(np.ones(6))
    dT.costs.addCost("line_place_T", crocoddyl.CostModelResidual(stateT, actT, resT), w_term)

def warmstart_with_line(xs, frames_SE3, model):
    """用直线参考给浮动基位姿一个更接近目标的初值（加快收敛）。"""
    for k in range(len(xs)):
        q = xs[k][:model.nq].copy()
        M = frames_SE3[min(k, len(frames_SE3)-1)]
        q[:3] = M.translation
        qz = pin.Quaternion(M.rotation)
        q[3:7] = np.array([qz.w, qz.x, qz.y, qz.z])
        xs[k][:model.nq] = q

# ----------------- 主流程 -----------------
def main():
    # 1) 载入机器人与初始状态
    robot = example_robot_data.load("anymal")
    model = robot.model
    q0 = model.referenceConfigurations["standing"].copy()
    v0 = pin.utils.zero(model.nv)
    x0 = np.concatenate([q0, v0])

    # 2) 构建步态（行走）。先用逆动力学版更稳：fwddyn=False
    lf, rf, lh, rh = "LF_FOOT","RF_FOOT","LH_FOOT","RH_FOOT"
    gait = SimpleQuadrupedalGaitProblem(model, lf, rf, lh, rh, fwddyn=False)

    # 步态参数（保守一些，容易收敛）
    timeStep     = 1e-2
    stepKnots    = 60
    supportKnots = 4
    stepLength   = 0.15
    stepHeight   = 0.08
    problem = gait.createWalkingProblem(x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots)

    # 3) 直线参考（A→B）
    A = np.array([0.0, 0.0])
    B = np.array([0.4, 0.10])  # 可以先短一点&小侧移，确认能动再加大
    frames_SE3 = straight_xy_yaw(A, B, problem.T+1, base_h=0.42, yaw_mode="tangent")

    # 4) 选择基座 frame，去掉冲突 cost，注入直线参考
    base_fid, base_name = choose_base_frame(model)
    print("Using base frame:", base_name)
    strip_conflicting_base_costs(problem)
    inject_line_refs(problem, base_fid, frames_SE3, w_run=1e4, w_term=1e6)

    # 5) 求解器 + 初值（暖启动）
    solver = crocoddyl.SolverFDDP(problem)
    logger = crocoddyl.CallbackLogger()
    solver.setCallbacks([crocoddyl.CallbackVerbose(), logger])
    xs = [x0.copy() for _ in range(problem.T+1)]
    warmstart_with_line(xs, frames_SE3, model)
    us = problem.quasiStatic(xs[:-1])

    solver.solve(xs, us, maxiter=200, isFeasible=False)

    # 6) 可视化
    if WITHDISPLAY:
        try:
            import gepetto
            gepetto.corbaserver.Client()
            display = crocoddyl.GepettoDisplay(robot, 4, 4, [2.0,2.68,0.84,0.2,0.62,0.72,0.22])
        except Exception:
            display = crocoddyl.MeshcatDisplay(robot)
        display.rate = -1
        display.freq = 1
        display.displayFromSolver(solver)

    # 7) 可选：绘制收敛
    if WITHPLOT:
        crocoddyl.plotConvergence(logger.costs, logger.pregs, logger.dregs, logger.grads, logger.stops, logger.steps)

if __name__ == "__main__":
    main()
