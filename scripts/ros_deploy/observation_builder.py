"""
observation_builder.py — 将 ROS 传感器数据组装为 45D 观测向量

严格对应 quadruped_mpc_env.py 的 _get_observations() 方法，保证与
IsaacLab 训练时的观测分布完全一致。

观测向量布局 (45D):
    [0:3]   root_pos_w       — 世界坐标系位置 (x, y, z)
    [3:7]   root_quat_w      — 四元数 (w, x, y, z)  IsaacLab 约定
    [7:10]  root_lin_vel_w   — 世界系线速度 (vx, vy, vz)
    [10:13] root_ang_vel_w   — 世界系角速度 (ωx, ωy, ωz)
    [13:25] joint_pos        — 12 关节角度 (Isaac Lab 顺序)
    [25:37] joint_vel        — 12 关节速度 (Isaac Lab 顺序)
    [37:41] foot_contacts    — 4 脚接触二值 (LF, RF, LH, RH)
    [41:44] target_pos       — 目标位置 (x, y, z)
    [44]    gait_phase       — 步态相位 [0, 1]

关节顺序说明:
    URDF / Pinocchio / Gazebo /joint_states 顺序 (按腿分组):
        0:FL_hip  1:FL_thigh  2:FL_calf
        3:FR_hip  4:FR_thigh  5:FR_calf
        6:RL_hip  7:RL_thigh  8:RL_calf
        9:RR_hip 10:RR_thigh 11:RR_calf

    IsaacLab 顺序 (按关节类型分组):
        0:FL_hip  1:FR_hip  2:RL_hip  3:RR_hip     ← 所有髋关节
        4:FL_thigh 5:FR_thigh 6:RL_thigh 7:RR_thigh ← 所有大腿
        8:FL_calf  9:FR_calf 10:RL_calf 11:RR_calf  ← 所有小腿

    变换: PIN_TO_ISAAC[pin_idx] = isaac_idx
"""

import numpy as np

# ── 关节顺序映射 ─────────────────────────────────────────────────────────────
# Pinocchio (URDF) → IsaacLab 的索引变换
PIN_TO_ISAAC = np.array([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=np.int32)
# ISAAC_TO_PIN[isaac_idx] = pin_idx
ISAAC_TO_PIN = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=np.int32)

# 训练中使用的站立高度参数（与 env_cfg.py 一致）
STANDING_HEIGHT = 0.4  # meters


def reorder_pin_to_isaac(joint_array: np.ndarray) -> np.ndarray:
    """将 Pinocchio/URDF 关节顺序转换为 IsaacLab 顺序。

    Args:
        joint_array: shape (12,)，Pinocchio 顺序

    Returns:
        shape (12,)，IsaacLab 顺序
    """
    out = np.empty_like(joint_array)
    for pin_idx in range(12):
        out[PIN_TO_ISAAC[pin_idx]] = joint_array[pin_idx]
    return out


def estimate_foot_contacts(root_pos_z: float) -> np.ndarray:
    """基于机身高度估计脚部接触状态（与 IsaacLab 训练保持一致）。

    IsaacLab 训练中使用：
        foot_contacts = (position[:, 2] < standing_height * 0.5).expand(-1, 4)
    即：机身高度 < 0.2m 时，认为全部脚接触地面。

    这是一个粗略估计。如果 Gazebo 配置了接触传感器，应优先使用传感器数据。

    Args:
        root_pos_z: 机身 z 坐标 (world frame)

    Returns:
        shape (4,) float，值为 0.0 或 1.0
    """
    contact = float(root_pos_z < STANDING_HEIGHT * 0.5)
    return np.array([contact, contact, contact, contact], dtype=np.float32)


def build_observation(
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    root_lin_vel_w: np.ndarray,
    root_ang_vel_w: np.ndarray,
    joint_pos_pin: np.ndarray,
    joint_vel_pin: np.ndarray,
    target_pos: np.ndarray,
    gait_phase: float,
    foot_contacts: np.ndarray | None = None,
) -> np.ndarray:
    """组装 45D 观测向量。

    Args:
        root_pos_w:      (3,) 世界系位置，来自 /gazebo/model_states
        root_quat_wxyz:  (4,) 四元数 [w, x, y, z]，来自 /imu 或 model_states
                         注意：ROS IMU 发布 [x,y,z,w]，需在调用前转换为 [w,x,y,z]
        root_lin_vel_w:  (3,) 世界系线速度，来自 /gazebo/model_states 或差分
        root_ang_vel_w:  (3,) 世界系角速度，来自 /imu (body frame) 或 model_states
                         注意：IMU 的 angular_velocity 是体坐标系；
                         如果使用 IMU，需先转换为世界系：v_world = R @ v_body
        joint_pos_pin:   (12,) Pinocchio/URDF 关节顺序的关节角度
        joint_vel_pin:   (12,) Pinocchio/URDF 关节顺序的关节速度
        target_pos:      (3,) 目标位置 (x, y, z)
        gait_phase:      步态相位 [0, 1]，由 gait_clock / cycle_duration 计算
        foot_contacts:   (4,) 可选，各脚接触状态 [LF, RF, LH, RH]
                         为 None 时自动用机身高度估计

    Returns:
        obs: shape (45,)，float32 numpy 数组
    """
    # 将 Pinocchio 关节顺序 → IsaacLab 顺序（RL 策略用此顺序训练）
    joint_pos_isaac = reorder_pin_to_isaac(joint_pos_pin)
    joint_vel_isaac = reorder_pin_to_isaac(joint_vel_pin)

    # 脚部接触：优先使用传入值，否则用高度估计
    if foot_contacts is None:
        foot_contacts = estimate_foot_contacts(root_pos_w[2])

    obs = np.concatenate([
        root_pos_w.astype(np.float32),         # [0:3]   位置
        root_quat_wxyz.astype(np.float32),     # [3:7]   四元数 (w,x,y,z)
        root_lin_vel_w.astype(np.float32),     # [7:10]  线速度 (世界系)
        root_ang_vel_w.astype(np.float32),     # [10:13] 角速度 (世界系)
        joint_pos_isaac.astype(np.float32),    # [13:25] 关节位置
        joint_vel_isaac.astype(np.float32),    # [25:37] 关节速度
        foot_contacts.astype(np.float32),      # [37:41] 脚部接触
        target_pos.astype(np.float32),         # [41:44] 目标位置
        np.array([gait_phase], dtype=np.float32),  # [44]    步态相位
    ])

    assert obs.shape == (45,), f"观测向量维度错误: {obs.shape}"
    return obs


def build_mpc_state(
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    root_lin_vel_b: np.ndarray,
    root_ang_vel_b: np.ndarray,
    joint_pos_pin: np.ndarray,
    joint_vel_pin: np.ndarray,
    z_vel_filter_state: np.ndarray,
    alpha: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """构建 37D Pinocchio/Crocoddyl 状态向量。

    状态格式（nx = nq + nv = 19 + 18 = 37）：
        state[0:3]   = position (x, y, z)
        state[3:7]   = quaternion (qx, qy, qz, qw)  ← Pinocchio 约定，w 在末尾
        state[7:19]  = joint_pos (12DOF，Pinocchio 顺序)
        state[19:22] = lin_vel_body (vx, vy, vz)     ← 体坐标系！
        state[22:25] = ang_vel_body (ωx, ωy, ωz)
        state[25:37] = joint_vel (12DOF，Pinocchio 顺序)

    Args:
        root_pos_w:         (3,) 世界系位置
        root_quat_wxyz:     (4,) 四元数 [w, x, y, z]（IsaacLab/ROS 约定）
        root_lin_vel_b:     (3,) 体坐标系线速度（由 lin_vel_w 转换而来）
        root_ang_vel_b:     (3,) 体坐标系角速度（直接来自 IMU）
        joint_pos_pin:      (12,) Pinocchio 顺序关节角度
        joint_vel_pin:      (12,) Pinocchio 顺序关节速度
        z_vel_filter_state: (1,) EMA 滤波器状态，在外部维护，此函数会更新并返回
        alpha:              EMA 滤波系数（0.3 与训练一致）

    Returns:
        state:              (37,) MPC 状态向量
        z_vel_filter_state: 更新后的滤波器状态
    """
    state = np.zeros(37, dtype=np.float64)

    # 位置
    state[0:3] = root_pos_w

    # 四元数：IsaacLab [w,x,y,z] → Pinocchio [x,y,z,w]
    state[3] = root_quat_wxyz[1]  # qx
    state[4] = root_quat_wxyz[2]  # qy
    state[5] = root_quat_wxyz[3]  # qz
    state[6] = root_quat_wxyz[0]  # qw

    # 关节角度（Pinocchio 顺序，直接使用）
    state[7:19] = joint_pos_pin

    # 体坐标系线速度（含 Z 轴 EMA 低通滤波，与训练一致）
    z_vel_filter_state = alpha * root_lin_vel_b[2] + (1.0 - alpha) * z_vel_filter_state
    lin_vel_filtered = root_lin_vel_b.copy()
    lin_vel_filtered[2] = float(np.clip(z_vel_filter_state, -0.15, 0.15))
    state[19:22] = lin_vel_filtered

    # 体坐标系角速度（钳制与训练一致）
    state[22:25] = np.clip(root_ang_vel_b, -1.0, 1.0)

    # 关节速度（Pinocchio 顺序）
    state[25:37] = joint_vel_pin

    return state, z_vel_filter_state


def quat_wxyz_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """四元数 [w,x,y,z] → 旋转矩阵 R (3×3)，世界→体坐标系转换用 R.T @ v_world。"""
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def world_vel_to_body(vel_world: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """将世界系速度转换为体坐标系速度。

    v_body = R^T @ v_world，其中 R 是世界→体的旋转矩阵。
    """
    R = quat_wxyz_to_rotation_matrix(quat_wxyz)
    return R.T @ vel_world
