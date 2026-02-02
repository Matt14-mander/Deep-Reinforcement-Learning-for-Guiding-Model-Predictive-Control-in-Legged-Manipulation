import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution

def yaw_to_quat(yaw):
    R = pinocchio.rpy.rpyToMatrix(0, 0, yaw)
    q = pinocchio.Quaternion(R).coeffs()
    return np.array(q)

def set_base_xy_yaw(q, x, y, yaw):
    q = q.copy()
    quat = yaw_to_quat(yaw)
    q[0:4] = quat
    q[4:7] = np.array([x, y, q[6] if len(q) > 6 else 0.0])
    return q

def get_base_xy(q):
    return float(q[0]), float(q[1])

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the anymal model
anymal = example_robot_data.load("anymal")
robot_model = anymal.model
lims = robot_model.effortLimit
lims *= 0.5  # reduced artificially the torque limits
robot_model.effortLimit = lims

# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
gait = SimpleQuadrupedalGaitProblem(robot_model, lfFoot, rfFoot, lhFoot, rhFoot)

# Defining the initial state of the robot
q0 = robot_model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(robot_model.nv)
x0 = np.concatenate([q0, v0])

# Define the point A and B
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
stop_tol = 0.05

# dir_vec = B - A
# yaw_des = float(np.arctan2(dir_vec[1], dir_vec[0]))
# x0[:anymal.model.nq] = set_base_xy_yaw(x0[:anymal.model.nq], A[0], A[1], yaw_des)

base_step_len = 0.20
# Defining the walking gait parameters
walking_gait = {
        "stepLength": 0.15,
        "stepHeight": 0.1,
        "timeStep": 1e-2,
        "stepKnots": 25,
        "supportKnots": 5,
}

# Setting up the control-limited DDP solver
def build_walking_solver(x_int, step_len):
    robot_model.referenceConfigurations["standing"] = x_int[:anymal.model.nq].copy()
    solver = crocoddyl.SolverBoxDDP(
        gait.createWalkingProblem(
            x_int,
            step_len,
            walking_gait["stepHeight"],
            walking_gait["timeStep"],
            walking_gait["stepKnots"],
            walking_gait["supportKnots"],
        )
    )
    return solver

display = None
# Display the entire motion
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


prev_xs, prev_us = None, None
phase_idx = 0
while True:
    q_cur = x0[:anymal.model.nq]
    x_cur, y_cur = get_base_xy(q_cur)
    print("x_cur = ", x_cur)
    rem = float(B[0] - x_cur)
    # if x_cur > B[0]:
    #     print("Reached B, Done")
    #     break
    if rem < stop_tol:
        print(f"Reached B within {stop_tol} m, Done")
        break
    step_len = min(base_step_len, rem)
    solver = build_walking_solver(x0, step_len)
    print("rem = ", rem)

    if prev_us is not None and prev_xs is not None:
        T = solver.problem.T
        xs0 = list(prev_xs[-(T+1):])
        if len(xs0) < T+1:
            xs0 += [xs0[-1]] * ((T+1) - len(xs0))
        us0 = list(prev_us[-T:]) if len(prev_us) > T else solver.problem.quasiStatic([x0] * T)
    else:
        xs0 = [x0] * (solver.problem.T + 1)
        us0 = solver.problem.quasiStatic([x0] * solver.problem.T)

    print(f"*** SOLVED walking (phase {phase_idx}), rem={rem:.3f} m, step_len={step_len:.3f} m ***")
    if WITHPLOT:
        solver.setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
    else:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])
    solver.solve(xs0, us0, 100, False, 0.1)

    x0 = solver.xs[-1]
    prev_xs = solver.xs
    prev_us = solver.us
    phase_idx += 1

    # # Solve the DDP problem
    # xs = [x0] * (solver.problem.T + 1)
    # us = solver.problem.quasiStatic([x0] * solver.problem.T)
    # solver.solve(xs, us, 100, False, 0.1)

    if WITHDISPLAY and display is not None:
        display.displayFromSolver(solver)

    time.sleep(0.05)


