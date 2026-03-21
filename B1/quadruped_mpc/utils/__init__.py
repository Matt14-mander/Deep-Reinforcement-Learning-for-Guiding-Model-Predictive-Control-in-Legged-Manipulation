# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mathematical utilities and Meshcat visualization helpers for quadruped MPC."""

from .meshcat_viz import (
    hex_to_int,
    rgb_to_hex,
    mc_sphere,
    mc_line,
    mc_cylinder,
    mc_cone,
    mc_delete,
    draw_friction_cone,
    draw_grf_arrow,
    draw_contact_viz,
)

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
    # meshcat_viz
    "hex_to_int",
    "rgb_to_hex",
    "mc_sphere",
    "mc_line",
    "mc_cylinder",
    "mc_cone",
    "mc_delete",
    "draw_friction_cone",
    "draw_grf_arrow",
    "draw_contact_viz",
    # math_utils
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
