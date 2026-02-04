# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Robot configurations for RL+MPC Bezier control system."""

# Quadrotor configurations (always available - core config doesn't need IsaacLab)
from .quadrotor_cfg import QuadrotorCfg, CRAZYFLIE_CFG, ISAACLAB_AVAILABLE

# Conditionally import IsaacLab-specific quadrotor configs
if ISAACLAB_AVAILABLE:
    from .quadrotor_cfg import QUADROTOR_CFG, QuadrotorSpawnCfg
else:
    QUADROTOR_CFG = None
    QuadrotorSpawnCfg = None

# Quadruped configurations (always available - core configs don't need IsaacLab)
from .quadruped_cfg import (
    QuadrupedCfg,
    QuadrupedPhysicsCfg,
    QuadrupedJointsCfg,
    QuadrupedFramesCfg,
    DEFAULT_QUADRUPED_CFG,
    ANYMAL_C_CFG,
    B1_CFG,
    GO1_CFG,
    SOLO12_CFG,
    load_pinocchio_model,
    get_foot_frame_ids,
)

__all__ = [
    # Availability flag
    "ISAACLAB_AVAILABLE",
    # Quadrotor
    "QuadrotorCfg",
    "CRAZYFLIE_CFG",
    "QUADROTOR_CFG",
    "QuadrotorSpawnCfg",
    # Quadruped
    "QuadrupedCfg",
    "QuadrupedPhysicsCfg",
    "QuadrupedJointsCfg",
    "QuadrupedFramesCfg",
    "DEFAULT_QUADRUPED_CFG",
    "ANYMAL_C_CFG",
    "B1_CFG",
    "GO1_CFG",
    "SOLO12_CFG",
    "load_pinocchio_model",
    "get_foot_frame_ids",
]
