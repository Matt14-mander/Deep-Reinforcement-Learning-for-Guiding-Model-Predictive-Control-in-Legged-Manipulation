# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPC controllers for quadruped locomotion."""

from .base_mpc import BaseMPC, MPCSolution
from .crocoddyl_quadruped_mpc import CrocoddylQuadrupedMPC

__all__ = [
    "BaseMPC",
    "MPCSolution",
    "CrocoddylQuadrupedMPC",
]
