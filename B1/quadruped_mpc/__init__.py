# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quadruped MPC Module for RL+MPC Bezier Trajectory Control.

This standalone module implements quadruped locomotion control using:
1. Bezier curves for CoM trajectory parameterization (from RL policy)
2. Gait scheduling for contact timing (trot, walk, pace, bound)
3. Foothold planning for curve walking
4. Crocoddyl MPC for trajectory optimization

This module has minimal dependencies and can run without IsaacLab:
- numpy, scipy (core math)
- crocoddyl, pinocchio (for MPC)
- example-robot-data (for robot models in testing)
- matplotlib, meshcat (for visualization)

Usage:
    # Import gait components
    from quadruped_mpc.gait import GaitScheduler, FootholdPlanner, ContactSequence

    # Import trajectory generators
    from quadruped_mpc.trajectory import BezierTrajectoryGenerator, BezierFootTrajectory

    # Import MPC controller
    from quadruped_mpc.controllers import CrocoddylQuadrupedMPC
"""

__version__ = "0.1.0"
__author__ = "Isaac Lab Project Developers"

# Import core modules
from . import utils
from . import trajectory
from . import gait

# Convenience imports from trajectory
from .trajectory import BezierTrajectoryGenerator, BezierFootTrajectory

# Convenience imports from gait
from .gait import (
    ContactPhase,
    ContactSequence,
    GaitScheduler,
    FootholdPlanner,
    FootholdPlan,
    OCPFactory,
)

# Try to import Crocoddyl-dependent modules
try:
    from . import controllers
    from .controllers import BaseMPC, MPCSolution, CrocoddylQuadrupedMPC
    CROCODDYL_AVAILABLE = True
except ImportError:
    CROCODDYL_AVAILABLE = False
    controllers = None
    BaseMPC = None
    MPCSolution = None
    CrocoddylQuadrupedMPC = None

__all__ = [
    # Modules
    "utils",
    "trajectory",
    "gait",
    "controllers",
    # Trajectory classes
    "BezierTrajectoryGenerator",
    "BezierFootTrajectory",
    # Gait classes
    "ContactPhase",
    "ContactSequence",
    "GaitScheduler",
    "FootholdPlanner",
    "FootholdPlan",
    "OCPFactory",
    # Controller classes (may be None if Crocoddyl unavailable)
    "BaseMPC",
    "MPCSolution",
    "CrocoddylQuadrupedMPC",
    # Availability flags
    "CROCODDYL_AVAILABLE",
]
