# crocoddyl_iface/mpc_problem.py
import crocoddyl
import pinocchio as pin

def add_base_tracking_to_running_model(running_model, base_frame_id, target_SE3, w_pos=1e2, w_rot=5e1):
    """
    向一个 running IntegratedActionModel 里添加/更新“基座位姿跟踪”的代价。
    """
    # 取 differential model
    dmodel = running_model.differential
    state = dmodel.state
    # 平移+旋转分别加代价，防止权重耦合
    activation_xyz = crocoddyl.ActivationModelWeightedQuad(np.array([1.,1.,1.])*1.0)
    act_rot = crocoddyl.ActivationModelWeightedQuad(np.array([1.,1.,1.])*1.0)

    # 平移
    Mref = pin.SE3.Identity()
    Mref.translation = target_SE3.translation
    frame_tr_cost = crocoddyl.CostModelFrameTranslation(
        state, crocoddyl.ActivationModelWeightedQuad(np.array([1.,1.,1.])*1.0),
        pin.FrameTranslation(base_frame_id, Mref.translation)
    )
    # 旋转（用FrameRotation或完整FramePlacement）
    frame_rot_cost = crocoddyl.CostModelFrameRotation(
        state, act_rot,
        pin.FrameRotation(base_frame_id, target_SE3.rotation)
    )

    dmodel.costs.addCost("base_tr", frame_tr_cost, w_pos)
    dmodel.costs.addCost("base_rot", frame_rot_cost, w_rot)

def inject_bezier_refs(problem, base_frame_id, frames_SE3, w_pos=1e2, w_rot=5e1):
    """
    把每个 knot 的目标位姿设置为 Bézier 采样得到的基座位姿。
    """
    assert len(frames_SE3) == problem.T+1 or len(frames_SE3) == problem.T, \
        "参考长度需与 shooting nodes 对齐（可传T或T+1）"

    for k in range(problem.T):
        running = problem.runningModels[k]
        add_base_tracking_to_running_model(running, base_frame_id, frames_SE3[min(k, len(frames_SE3)-1)], w_pos, w_rot)

    # 终端 cost（可选）：让终点状态更贴近 Bézier 末端
    term = problem.terminalModel
    # 终端也加一点位置/姿态跟踪（权重更大）
    dmodel = term.differential
    state = dmodel.state if hasattr(term, "differential") else term.state  # 兼容不同版本
    target_last = frames_SE3[-1]
    term.costs.addCost(
        "term_base_tr",
        crocoddyl.CostModelFrameTranslation(
            state, crocoddyl.ActivationModelWeightedQuad(np.ones(3)),
            pin.FrameTranslation(base_frame_id, target_last.translation)
        ),
        w_pos*5.0
    )
    term.costs.addCost(
        "term_base_rot",
        crocoddyl.CostModelFrameRotation(
            state, crocoddyl.ActivationModelWeightedQuad(np.ones(3)),
            pin.FrameRotation(base_frame_id, target_last.rotation)
        ),
        w_rot*5.0
    )
