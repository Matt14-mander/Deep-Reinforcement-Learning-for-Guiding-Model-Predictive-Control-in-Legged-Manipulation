# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mathematical utility functions for quadruped MPC.

This module provides common mathematical operations for:
- Quaternion and rotation matrix conversions
- Bezier curve evaluation and derivatives
- Angle normalization and wrapping
- Trajectory blending
"""

from typing import Tuple, Union

import numpy as np


# =============================================================================
# Quaternion Operations
# =============================================================================

def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix.

    Args:
        quat: Quaternion in (w, x, y, z) format. Shape: (4,) or (N, 4)

    Returns:
        Rotation matrix. Shape: (3, 3) or (N, 3, 3)
    """
    quat = np.asarray(quat)
    single = quat.ndim == 1

    if single:
        quat = quat[np.newaxis, :]

    # Normalize quaternions
    quat = quat / np.linalg.norm(quat, axis=1, keepdims=True)

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Build rotation matrices
    R = np.zeros((len(quat), 3, 3))

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)

    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)

    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    if single:
        return R[0]
    return R


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion.

    Uses Shepperd's method for numerical stability.

    Args:
        R: Rotation matrix. Shape: (3, 3) or (N, 3, 3)

    Returns:
        Quaternion in (w, x, y, z) format. Shape: (4,) or (N, 4)
    """
    R = np.asarray(R)
    single = R.ndim == 2

    if single:
        R = R[np.newaxis, :, :]

    N = len(R)
    quat = np.zeros((N, 4))

    for i in range(N):
        trace = np.trace(R[i])

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            quat[i, 0] = 0.25 / s
            quat[i, 1] = (R[i, 2, 1] - R[i, 1, 2]) * s
            quat[i, 2] = (R[i, 0, 2] - R[i, 2, 0]) * s
            quat[i, 3] = (R[i, 1, 0] - R[i, 0, 1]) * s
        elif R[i, 0, 0] > R[i, 1, 1] and R[i, 0, 0] > R[i, 2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[i, 0, 0] - R[i, 1, 1] - R[i, 2, 2])
            quat[i, 0] = (R[i, 2, 1] - R[i, 1, 2]) / s
            quat[i, 1] = 0.25 * s
            quat[i, 2] = (R[i, 0, 1] + R[i, 1, 0]) / s
            quat[i, 3] = (R[i, 0, 2] + R[i, 2, 0]) / s
        elif R[i, 1, 1] > R[i, 2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[i, 1, 1] - R[i, 0, 0] - R[i, 2, 2])
            quat[i, 0] = (R[i, 0, 2] - R[i, 2, 0]) / s
            quat[i, 1] = (R[i, 0, 1] + R[i, 1, 0]) / s
            quat[i, 2] = 0.25 * s
            quat[i, 3] = (R[i, 1, 2] + R[i, 2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[i, 2, 2] - R[i, 0, 0] - R[i, 1, 1])
            quat[i, 0] = (R[i, 1, 0] - R[i, 0, 1]) / s
            quat[i, 1] = (R[i, 0, 2] + R[i, 2, 0]) / s
            quat[i, 2] = (R[i, 1, 2] + R[i, 2, 1]) / s
            quat[i, 3] = 0.25 * s

    # Ensure w is positive (canonical form)
    mask = quat[:, 0] < 0
    quat[mask] = -quat[mask]

    if single:
        return quat[0]
    return quat


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Uses ZYX (yaw-pitch-roll) convention.

    Args:
        quat: Quaternion in (w, x, y, z) format. Shape: (4,) or (N, 4)

    Returns:
        Euler angles (roll, pitch, yaw) in radians. Shape: (3,) or (N, 3)
    """
    quat = np.asarray(quat)
    single = quat.ndim == 1

    if single:
        quat = quat[np.newaxis, :]

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(
        np.abs(sinp) >= 1,
        np.sign(sinp) * np.pi / 2,
        np.arcsin(sinp)
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    euler = np.stack([roll, pitch, yaw], axis=-1)

    if single:
        return euler[0]
    return euler


def euler_to_quat(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to quaternion.

    Uses ZYX (yaw-pitch-roll) convention.

    Args:
        euler: Euler angles (roll, pitch, yaw) in radians. Shape: (3,) or (N, 3)

    Returns:
        Quaternion in (w, x, y, z) format. Shape: (4,) or (N, 4)
    """
    euler = np.asarray(euler)
    single = euler.ndim == 1

    if single:
        euler = euler[np.newaxis, :]

    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quat = np.stack([w, x, y, z], axis=-1)

    if single:
        return quat[0]
    return quat


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions.

    Args:
        q1: First quaternion (w, x, y, z). Shape: (4,) or (N, 4)
        q2: Second quaternion (w, x, y, z). Shape: (4,) or (N, 4)

    Returns:
        Product quaternion q1 * q2. Shape: (4,) or (N, 4)
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)

    single = q1.ndim == 1 and q2.ndim == 1

    if q1.ndim == 1:
        q1 = q1[np.newaxis, :]
    if q2.ndim == 1:
        q2 = q2[np.newaxis, :]

    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = np.stack([w, x, y, z], axis=-1)

    if single:
        return result[0]
    return result


def quat_inverse(quat: np.ndarray) -> np.ndarray:
    """Compute quaternion inverse (conjugate for unit quaternion).

    Args:
        quat: Quaternion (w, x, y, z). Shape: (4,) or (N, 4)

    Returns:
        Inverse quaternion. Shape: (4,) or (N, 4)
    """
    quat = np.asarray(quat)
    result = quat.copy()
    if result.ndim == 1:
        result[1:] = -result[1:]
    else:
        result[:, 1:] = -result[:, 1:]
    return result


# =============================================================================
# Angle Operations
# =============================================================================

def normalize_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Normalize angle to [-pi, pi] range."""
    return wrap_angle(angle)


def wrap_angle(angle: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap angle to [-pi, pi] range."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# Bezier Curve Operations
# =============================================================================

def bezier_curve(control_points: np.ndarray, num_samples: int) -> np.ndarray:
    """Evaluate cubic Bezier curve at uniform parameter values.

    Args:
        control_points: Control points with shape (4, D) for cubic Bezier.
        num_samples: Number of points to sample along the curve.

    Returns:
        Sampled curve points with shape (num_samples, D).
    """
    control_points = np.asarray(control_points)
    t = np.linspace(0, 1, num_samples)

    # Cubic Bezier: B(t) = (1-t)³P₀ + 3(1-t)²tP₁ + 3(1-t)t²P₂ + t³P₃
    one_minus_t = 1 - t

    b0 = one_minus_t ** 3
    b1 = 3 * (one_minus_t ** 2) * t
    b2 = 3 * one_minus_t * (t ** 2)
    b3 = t ** 3

    result = (
        np.outer(b0, control_points[0])
        + np.outer(b1, control_points[1])
        + np.outer(b2, control_points[2])
        + np.outer(b3, control_points[3])
    )

    return result


def bezier_tangent(control_points: np.ndarray, t: float) -> np.ndarray:
    """Compute tangent vector of cubic Bezier at parameter t.

    The derivative of a cubic Bezier curve is:
        B'(t) = 3(1-t)²(P₁-P₀) + 6(1-t)t(P₂-P₁) + 3t²(P₃-P₂)

    Args:
        control_points: Control points with shape (4, D) for cubic Bezier.
        t: Parameter value in [0, 1].

    Returns:
        Tangent vector with shape (D,).
    """
    control_points = np.asarray(control_points)
    t = np.clip(t, 0, 1)

    one_minus_t = 1 - t

    q0 = control_points[1] - control_points[0]
    q1 = control_points[2] - control_points[1]
    q2 = control_points[3] - control_points[2]

    tangent = 3 * (
        (one_minus_t ** 2) * q0
        + 2 * one_minus_t * t * q1
        + (t ** 2) * q2
    )

    return tangent


def bezier_tangent_at_samples(
    control_points: np.ndarray, num_samples: int
) -> np.ndarray:
    """Compute tangent vectors at uniform sample points along a cubic Bezier.

    Args:
        control_points: Control points with shape (4, D).
        num_samples: Number of sample points.

    Returns:
        Tangent vectors with shape (num_samples, D).
    """
    t_values = np.linspace(0, 1, num_samples)
    tangents = np.array([bezier_tangent(control_points, t) for t in t_values])
    return tangents


def heading_from_tangent(tangent_xy: np.ndarray) -> float:
    """Compute yaw angle from 2D tangent vector.

    Args:
        tangent_xy: 2D tangent vector (x, y) or 3D vector (uses first 2 dims).

    Returns:
        Yaw angle in radians, computed as atan2(ty, tx).
    """
    tangent_xy = np.asarray(tangent_xy)
    tx = tangent_xy[0] if tangent_xy.ndim == 1 else tangent_xy[:, 0]
    ty = tangent_xy[1] if tangent_xy.ndim == 1 else tangent_xy[:, 1]

    if np.isscalar(tx):
        if np.abs(tx) < 1e-6 and np.abs(ty) < 1e-6:
            return 0.0
        return float(np.arctan2(ty, tx))
    else:
        heading = np.arctan2(ty, tx)
        near_zero = (np.abs(tx) < 1e-6) & (np.abs(ty) < 1e-6)
        heading[near_zero] = 0.0
        return heading


def rotation_matrix_z(yaw: float) -> np.ndarray:
    """Create 3x3 rotation matrix around z-axis.

    Args:
        yaw: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def rotation_matrix_z_batch(yaw: np.ndarray) -> np.ndarray:
    """Create batch of 3x3 rotation matrices around z-axis.

    Args:
        yaw: Rotation angles in radians, shape (N,).

    Returns:
        Rotation matrices with shape (N, 3, 3).
    """
    yaw = np.asarray(yaw)
    c, s = np.cos(yaw), np.sin(yaw)
    N = len(yaw)
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def yaw_from_quaternion(quat: np.ndarray) -> float:
    """Extract yaw angle from quaternion.

    Args:
        quat: Quaternion in (w, x, y, z) format.

    Returns:
        Yaw angle in radians.
    """
    euler = quat_to_euler(quat)
    return euler[2]


def quaternion_from_yaw(yaw: float) -> np.ndarray:
    """Create quaternion representing pure yaw rotation.

    Args:
        yaw: Yaw angle in radians.

    Returns:
        Quaternion in (w, x, y, z) format.
    """
    return euler_to_quat(np.array([0.0, 0.0, yaw]))


# =============================================================================
# Trajectory Operations
# =============================================================================

def blend_trajectories(
    old_trajectory: np.ndarray,
    new_trajectory: np.ndarray,
    blend_steps: int = 3,
    blend_type: str = "linear",
) -> np.ndarray:
    """Smoothly blend between old and new trajectories.

    Args:
        old_trajectory: Previous trajectory, shape (T_old, state_dim).
        new_trajectory: New trajectory, shape (T_new, state_dim).
        blend_steps: Number of steps over which to blend.
        blend_type: Blending method ("linear", "cubic", "cosine").

    Returns:
        Blended trajectory, shape (T_new, state_dim).
    """
    if blend_steps <= 0 or len(old_trajectory) == 0:
        return new_trajectory.copy()

    blend_steps = min(blend_steps, len(old_trajectory), len(new_trajectory))
    result = new_trajectory.copy()

    if blend_type == "linear":
        weights = np.linspace(0, 1, blend_steps)
    elif blend_type == "cubic":
        t = np.linspace(0, 1, blend_steps)
        weights = 3 * t**2 - 2 * t**3
    elif blend_type == "cosine":
        t = np.linspace(0, 1, blend_steps)
        weights = 0.5 * (1 - np.cos(np.pi * t))
    else:
        raise ValueError(f"Unknown blend type: {blend_type}")

    for i in range(blend_steps):
        w = weights[i]
        result[i] = (1 - w) * old_trajectory[i] + w * new_trajectory[i]

    return result
