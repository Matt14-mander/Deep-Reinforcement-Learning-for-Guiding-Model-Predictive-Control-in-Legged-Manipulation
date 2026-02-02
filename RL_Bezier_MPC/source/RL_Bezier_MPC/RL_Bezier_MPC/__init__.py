# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
RL+MPC Bezier Trajectory Control System.

This package implements a hierarchical control system that combines:
1. Reinforcement Learning (RL) for high-level trajectory planning
2. Model Predictive Control (MPC) for low-level trajectory tracking
3. Bezier curves as the trajectory parameterization

The RL policy outputs Bezier control points at 5-10 Hz, which are converted
to dense waypoints tracked by an MPC controller at 50 Hz.
"""

# Version info
__version__ = "0.1.0"
__author__ = "Isaac Lab Project Developers"

# Import core modules that don't depend on IsaacLab
from . import controllers
from . import trajectory
from . import utils

# Try to import IsaacLab-dependent modules
# These will fail if IsaacLab/Isaac Sim is not available
try:
    from . import envs
    from . import robots

    # Register Gym environments
    from .tasks import *

    # Register UI extensions
    from .ui_extension_example import *

    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False
    # These modules are not available without IsaacLab
    envs = None
    robots = None
