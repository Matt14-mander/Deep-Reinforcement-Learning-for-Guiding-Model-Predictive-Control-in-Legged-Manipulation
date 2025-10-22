# crocoddyl_iface/mpc_problem.py
import crocoddyl
import pinocchio as pin

def add_base_tracking_to_running_model(running_model, base_frame_id, target_SE3, w_placement=1e2):
    """
    用 ResidualModelFramePlacement + CostModelResidual 同时跟踪平移+旋转
    """
    dmodel = running_model.differential
    state  = dmodel.state

    # 残差：基座frame相对于target_SE3的位姿误差 [pos(3), rot(3)]
    res_place = crocoddyl.ResidualModelFramePlacement(state, base_frame_id, target_SE3)

    # 激活：加权二次范数（6维）
    act6 = crocoddyl.ActivationModelWeightedQuad(np.ones(6))

    # 成本：Residual + Activation 组合
    cost_place = crocoddyl.CostModelResidual(state, act6, res_place)

    # 加到 running model 的成本和里
    dmodel.costs.addCost("base_place", cost_place, w_placement)


def inject_bezier_refs(problem, base_frame_id, frames_SE3, w_pos=1e2, w_rot=5e1):
    """
    注意：这里直接用 placement 统一跟踪，所以只用一个权重 w_place。
    为了兼容你原来的调用，这里仍保留 w_pos/w_rot 形参，但内部当成同一个尺度用。
    """
    w_place = float(w_pos)  # 你也可以写成 0.5*w_pos + 0.5*w_rot

    # running 节点
    for k in range(problem.T):
        running = problem.runningModels[k]
        target  = frames_SE3[min(k, len(frames_SE3)-1)]
        add_base_tracking_to_running_model(running, base_frame_id, target, w_place)

    # terminal 节点
    target_last = frames_SE3[-1]
    term = problem.terminalModel
    # 兼容不同 Crocoddyl 版本（有的终端就是 Differential，有的是 Integrated）
    dmodel = term.differential if hasattr(term, "differential") else term
    state  = dmodel.state

    res_last = crocoddyl.ResidualModelFramePlacement(state, base_frame_id, target_last)
    act6     = crocoddyl.ActivationModelWeightedQuad(np.ones(6))
    cost_last= crocoddyl.CostModelResidual(state, act6, res_last)
    dmodel.costs.addCost("term_base_place", cost_last, 5.0*w_place)
