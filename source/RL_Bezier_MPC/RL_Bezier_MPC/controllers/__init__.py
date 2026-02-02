# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPC controller module for RL+MPC Bezier control system."""

from .base_mpc import BaseMPC, MPCSolution
from .crocoddyl_quadrotor_mpc import CROCODDYL_AVAILABLE, CrocoddylQuadrotorMPC
from .crocoddyl_quadruped_mpc import CrocoddylQuadrupedMPC

__all__ = [
    "BaseMPC",
    "MPCSolution",
    "CrocoddylQuadrotorMPC",
    "CrocoddylQuadrupedMPC",
    "CROCODDYL_AVAILABLE",
]
