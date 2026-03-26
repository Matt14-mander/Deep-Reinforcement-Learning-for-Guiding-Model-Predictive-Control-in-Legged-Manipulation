"""
Observation builder for ROS2 deployment.

Constructs the 45-dimensional observation vector that exactly matches
what _get_observations() produces inside quadruped_mpc_env.py.

Observation layout (must be IDENTICAL to training):
    [0:3]   position          (3D)  — world frame
    [3:7]   quaternion        (4D)  — (w, x, y, z) Isaac Lab convention
    [7:10]  linear_vel        (3D)  — world frame
    [10:13] angular_vel       (3D)  — world frame
    [13:25] joint_pos         (12D) — Isaac Lab joint order
    [25:37] joint_vel         (12D) — Isaac Lab joint order
    [37:41] foot_contacts     (4D)  — binary per foot [FL, FR, RL, RR]
    [41:44] target_pos        (3D)  — goal position
    [44]    gait_phase        (1D)  — normalised [0, 1]

Joint order expected by the trained policy (Isaac Lab USD order):
    Index  0-3:  FL_hip, FR_hip, RL_hip, RR_hip      (all 4 hips)
    Index  4-7:  FL_thigh, FR_thigh, RL_thigh, RR_thigh
    Index  8-11: FL_calf, FR_calf, RL_calf, RR_calf

This is different from Pinocchio / URDF order (per-leg grouping).
The reordering is handled by build_ros_to_isaac_mapping().
"""

from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Joint name constants
# ---------------------------------------------------------------------------

# Go2 joint names in Isaac Lab (USD) training order
ISAAC_JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",   # hips
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",  # thighs
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",      # calves
]

# Go2 joint names in Pinocchio / URDF order (per-leg grouping)
PINOCCHIO_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# Foot contact height threshold (meters)
FOOT_CONTACT_HEIGHT = 0.04


# ---------------------------------------------------------------------------
# Joint ordering helpers
# ---------------------------------------------------------------------------

def build_ros_to_isaac_mapping(ros_joint_names: list[str]) -> np.ndarray:
    """Build index map: ros_joint_names order → Isaac Lab training order.

    Args:
        ros_joint_names: Joint names as published on /joint_states.

    Returns:
        ros_to_isaac: int array of shape (12,).
            isaac_joints[i] = ros_joints[ros_to_isaac[i]]

    Call once in node __init__ when first /joint_states arrives.
    """
    ros_name_to_idx = {name: i for i, name in enumerate(ros_joint_names)}
    mapping = np.zeros(12, dtype=np.int32)
    for isaac_idx, name in enumerate(ISAAC_JOINT_ORDER):
        if name not in ros_name_to_idx:
            raise ValueError(
                f"Joint '{name}' not found in /joint_states. "
                f"Available: {ros_joint_names}"
            )
        mapping[isaac_idx] = ros_name_to_idx[name]
    return mapping


def build_ros_to_pinocchio_mapping(ros_joint_names: list[str]) -> np.ndarray:
    """Build index map: ros_joint_names order → Pinocchio URDF order.

    Used for MPC state vector construction (not for RL observation).
    """
    ros_name_to_idx = {name: i for i, name in enumerate(ros_joint_names)}
    mapping = np.zeros(12, dtype=np.int32)
    for pin_idx, name in enumerate(PINOCCHIO_JOINT_ORDER):
        if name not in ros_name_to_idx:
            raise ValueError(
                f"Joint '{name}' not found in /joint_states. "
                f"Available: {ros_joint_names}"
            )
        mapping[pin_idx] = ros_name_to_idx[name]
    return mapping


# ---------------------------------------------------------------------------
# Quaternion / rotation utilities
# ---------------------------------------------------------------------------

def quat_wxyz_to_rotation_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 3×3 rotation matrix."""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def world_to_body_velocity(
    lin_vel_world: np.ndarray,
    ang_vel_world: np.ndarray,
    q_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate world-frame velocities to body frame.

    Used for building the 37D Pinocchio MPC state (not the RL observation).
    The RL observation uses world-frame velocities.

    Args:
        lin_vel_world: (3,) linear velocity in world frame.
        ang_vel_world: (3,) angular velocity in world frame.
        q_wxyz: (4,) quaternion (w, x, y, z).

    Returns:
        lin_vel_body, ang_vel_body — both (3,).
    """
    R = quat_wxyz_to_rotation_matrix(q_wxyz)
    lin_vel_body = R.T @ lin_vel_world
    ang_vel_body = R.T @ ang_vel_world
    return lin_vel_body.astype(np.float32), ang_vel_body.astype(np.float32)


# ---------------------------------------------------------------------------
# MPC state builder (37D Pinocchio state)
# ---------------------------------------------------------------------------

def build_mpc_state(
    root_pos_w: np.ndarray,       # (3,)  world position
    q_wxyz: np.ndarray,           # (4,)  quaternion w,x,y,z
    joint_pos_pin: np.ndarray,    # (12,) in Pinocchio order
    lin_vel_body: np.ndarray,     # (3,)  body-frame linear velocity
    ang_vel_body: np.ndarray,     # (3,)  body-frame angular velocity
    joint_vel_pin: np.ndarray,    # (12,) in Pinocchio order
    z_vel_filter_state: Optional[float] = None,
    z_vel_alpha: float = 0.3,
) -> tuple[np.ndarray, float]:
    """Build the 37D Pinocchio state vector for Crocoddyl.

    Pinocchio convention:
        q  = [x, y, z, qx, qy, qz, qw, joint1..12]  (nq = 19)
        v  = [vx_b, vy_b, vz_b, wx_b, wy_b, wz_b, dq1..12]  (nv = 18)
        state = q ∥ v   (37D)

    Note: quaternion order FLIPS to (x,y,z,w) in Pinocchio.

    Returns:
        state_37d, new_z_filter_state
    """
    state = np.zeros(37, dtype=np.float32)

    # q part
    state[0:3] = root_pos_w
    state[3] = q_wxyz[1]   # qx
    state[4] = q_wxyz[2]   # qy
    state[5] = q_wxyz[3]   # qz
    state[6] = q_wxyz[0]   # qw (note: Pinocchio stores w last)
    state[7:19] = joint_pos_pin

    # v part — apply EMA low-pass filter on vz to suppress impact spikes
    vz_raw = float(lin_vel_body[2])
    if z_vel_filter_state is None:
        z_vel_filter_state = vz_raw
    z_vel_filtered = z_vel_alpha * vz_raw + (1.0 - z_vel_alpha) * z_vel_filter_state
    z_vel_filtered = float(np.clip(z_vel_filtered, -0.15, 0.15))

    lin_vel_body_filtered = lin_vel_body.copy()
    lin_vel_body_filtered[2] = z_vel_filtered

    ang_vel_body_clipped = np.clip(ang_vel_body, -1.0, 1.0)

    state[19:22] = lin_vel_body_filtered
    state[22:25] = ang_vel_body_clipped
    state[25:37] = joint_vel_pin

    return state, z_vel_filtered


# ---------------------------------------------------------------------------
# RL observation builder (45D)
# ---------------------------------------------------------------------------

def build_observation(
    root_pos_w: np.ndarray,           # (3,)
    q_wxyz: np.ndarray,               # (4,)  w,x,y,z  (Isaac Lab convention)
    lin_vel_world: np.ndarray,        # (3,)
    ang_vel_world: np.ndarray,        # (3,)
    joint_pos_isaac: np.ndarray,      # (12,) in Isaac Lab order
    joint_vel_isaac: np.ndarray,      # (12,) in Isaac Lab order
    foot_heights: np.ndarray,         # (4,)  foot z positions, [FL,FR,RL,RR]
    target_pos: np.ndarray,           # (3,)
    gait_phase: float,                # normalised [0, 1]
) -> np.ndarray:
    """Construct the 45D RL observation vector.

    The layout MUST match _get_observations() in quadruped_mpc_env.py.
    Changes to the training env require corresponding updates here.

    Args:
        foot_heights: Foot centre heights above ground (from Pinocchio FK).
                      Contact is detected when height < FOOT_CONTACT_HEIGHT.

    Returns:
        obs: (45,) float32 numpy array.
    """
    obs = np.zeros(45, dtype=np.float32)

    # [0:3]   position
    obs[0:3] = root_pos_w

    # [3:7]   quaternion (w, x, y, z) — Isaac Lab order
    obs[3:7] = q_wxyz

    # [7:10]  linear velocity (world frame)
    obs[7:10] = lin_vel_world

    # [10:13] angular velocity (world frame)
    obs[10:13] = ang_vel_world

    # [13:25] joint positions (Isaac Lab order)
    obs[13:25] = joint_pos_isaac

    # [25:37] joint velocities (Isaac Lab order)
    obs[25:37] = joint_vel_isaac

    # [37:41] foot contacts  [FL, FR, RL, RR]
    obs[37:41] = (foot_heights < FOOT_CONTACT_HEIGHT).astype(np.float32)

    # [41:44] target position
    obs[41:44] = target_pos

    # [44]    gait phase
    obs[44] = float(gait_phase)

    return obs
