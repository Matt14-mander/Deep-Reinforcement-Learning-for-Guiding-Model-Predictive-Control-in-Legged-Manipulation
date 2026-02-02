# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mathematical utilities for quadruped MPC."""

from .math_utils import (
    quat_to_rotation_matrix,
    rotation_matrix_to_quat,
    quat_to_euler,
    euler_to_quat,
    quat_multiply,
    quat_inverse,
    normalize_angle,
    wrap_angle,
    bezier_curve,
    bezier_tangent,
    bezier_tangent_at_samples,
    heading_from_tangent,
    rotation_matrix_z,
    rotation_matrix_z_batch,
    yaw_from_quaternion,
    quaternion_from_yaw,
    blend_trajectories,
)

__all__ = [
    "quat_to_rotation_matrix",
    "rotation_matrix_to_quat",
    "quat_to_euler",
    "euler_to_quat",
    "quat_multiply",
    "quat_inverse",
    "normalize_angle",
    "wrap_angle",
    "bezier_curve",
    "bezier_tangent",
    "bezier_tangent_at_samples",
    "heading_from_tangent",
    "rotation_matrix_z",
    "rotation_matrix_z_batch",
    "yaw_from_quaternion",
    "quaternion_from_yaw",
    "blend_trajectories",
]
