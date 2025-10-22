# sim_demo/run_bezier_anymal.py
import os, sys, time, numpy as np, pinocchio as pin, crocoddyl
import example_robot_data
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem

from bezier.generator import sample_xy_yaw
from crocoddyl_iface.base_refs import build_base_refs
from crocoddyl_iface.mpc_problem import inject_bezier_refs

WITHDISPLAY = True

def main():
    # 1) 加载机器人
    anymal = example_robot_data.load("anymal")
    model = anymal.model
    data  = anymal.data
    q0 = model.referenceConfigurations["standing"].copy()
    v0 = pin.utils.zero(model.nv)
    x0 = np.concatenate([q0, v0])

    # 2) 步态模板（可先用行走）
    lf, rf, lh, rh = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
    gait = SimpleQuadrupedalGaitProblem(model, lf, rf, lh, rh, fwddyn=False)

    timeStep, stepKnots, supportKnots = 0.01, 40, 4  # 总长度约 0.44s
    stepLength, stepHeight = 0.20, 0.10

    problem = gait.createWalkingProblem(
        x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots
    )

    # 3) 生成 Bézier 曲线参考（A->B）
    A = np.array([0.0, 0.0])
    B = np.array([0.6, 0.3])
    # 两个中间控制点，可手动调/之后让RL输出
    P1 = A + np.array([0.2, 0.0])
    P2 = B + np.array([-0.2, 0.1])

    N = problem.T + 1
    x_ref, y_ref, yaw_ref = sample_xy_yaw(A, P1, P2, B, N, yaw_mode="tangent")
    frames_SE3, _ = build_base_refs(x_ref, y_ref, yaw_ref, h=0.42)

    # 4) 将 Bézier 参考注入 Crocoddyl 问题
    base_frame_id = model.getFrameId("base") if model.existFrame("base") else model.getFrameId(model.frames[1].name)  # 按你的URDF改
    inject_bezier_refs(problem, base_frame_id, frames_SE3, w_pos=1e3, w_rot=1e2)

    # 5) 求解（FDDP 更快；Intro 也能用）
    solver = crocoddyl.SolverFDDP(problem)
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    xs = [x0.copy() for _ in range(problem.T + 1)]
    us = problem.quasiStatic([x0] * problem.T)
    solver.solve(xs, us, maxiter=100, isFeasible=False)

    # 6) 可视化
    if WITHDISPLAY:
        try:
            import gepetto, crocoddyl
            gepetto.corbaserver.Client()
            display = crocoddyl.GepettoDisplay(anymal, 4, 4, [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22])
        except Exception:
            display = crocoddyl.MeshcatDisplay(anymal)
        display.rate = -1
        display.freq = 1
        display.displayFromSolver(solver)

if __name__ == "__main__":
    main()
