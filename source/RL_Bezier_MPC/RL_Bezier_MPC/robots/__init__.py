# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot configurations for RL+MPC Bezier control system."""

from .quadrotor_cfg import QUADROTOR_CFG, QuadrotorCfg
from .quadruped_cfg import (
    QuadrupedCfg,
    QuadrupedPhysicsCfg,
    QuadrupedJointsCfg,
    QuadrupedFramesCfg,
    DEFAULT_QUADRUPED_CFG,
    ANYMAL_C_CFG,
    GO1_CFG,
    SOLO12_CFG,
    load_pinocchio_model,
    get_foot_frame_ids,
)

__all__ = [
    # Quadrotor
    "QUADROTOR_CFG",
    "QuadrotorCfg",
    # Quadruped
    "QuadrupedCfg",
    "QuadrupedPhysicsCfg",
    "QuadrupedJointsCfg",
    "QuadrupedFramesCfg",
    "DEFAULT_QUADRUPED_CFG",
    "ANYMAL_C_CFG",
    "GO1_CFG",
    "SOLO12_CFG",
    "load_pinocchio_model",
    "get_foot_frame_ids",
]
