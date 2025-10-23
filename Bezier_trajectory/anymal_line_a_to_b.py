import numpy as np
import pinocchio
import example_robot_data
import crocoddyl

# -----------------------------
# 机器人模型 & 初始状态
# -----------------------------
anymal = example_robot_data.load("anymal")
q0 = anymal.model.referenceConfigurations["standing"].copy()
v0 = np.zeros(anymal.model.nv)
x0 = np.concatenate([q0, v0])

# -----------------------------
# 简单平面移动：A -> B
# -----------------------------
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.5])
base_height = 0.42
T = 50  # 时间步数

# 生成线性轨迹 SE3
frames_SE3 = []
for k in range(T+1):
    alpha = k / T
    pos = (1-alpha)*A + alpha*B
    R = np.eye(3)  # 保持朝向不变
    p = np.array([pos[0], pos[1], base_height])
    frames_SE3.append(pinocchio.SE3(R, p))

# -----------------------------
# 构建 Crocoddyl Shooting Problem
# -----------------------------
state = crocoddyl.StateMultibody(anymal.model)
act_model = crocoddyl.ActuationModelFull(state)
runningModels = []

for k in range(T):
    dmodel = crocoddyl.DifferentialActionModelAbstract(state, act_model)
    # 位姿跟踪成本
    frame_id = anymal.model.getFrameId("base")
    target = frames_SE3[k]
    res = crocoddyl.ResidualModelFramePlacement(state, frame_id, target)
    act = crocoddyl.ActivationModelWeightedQuad(np.ones(6))
    cost = crocoddyl.CostModelResidual(state, act, res)
    dmodel.costs.addCost("base_track", cost, 1e3)
    runningModels.append(crocoddyl.IntegratedActionModelEuler(dmodel, dt=0.02))

# 终端模型
dmodel_term = crocoddyl.DifferentialActionModelFullyActuated(state, act_model)
res_term = crocoddyl.ResidualModelFramePlacement(state, anymal.model.getFrameId("base"), frames_SE3[-1])
cost_term = crocoddyl.CostModelResidual(state, act, res_term)
dmodel_term.costs.addCost("base_track_term", cost_term, 5e3)
terminalModel = crocoddyl.IntegratedActionModelEuler(dmodel_term, dt=0.02)

# Shooting problem
problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)

# -----------------------------
# 创建求解器
# -----------------------------
solver = crocoddyl.SolverFDDP(problem)
solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])

# 初始 guess
xs = [x0.copy() for _ in range(T+1)]
us = [np.zeros(anymal.model.nv) for _ in range(T)]

# 求解
solver.solve(xs, us, maxiter=100)

# -----------------------------
# 可视化
# -----------------------------
try:
    import gepetto
    gepetto.corbaserver.Client()
    display = crocoddyl.GepettoDisplay(anymal, 4, 4)
except Exception:
    display = crocoddyl.MeshcatDisplay(anymal)

display.displayFromSolver(solver)
