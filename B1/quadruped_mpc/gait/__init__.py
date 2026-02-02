# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gait management module for quadruped locomotion.

This module provides:
- Contact sequence data structures for timing contact phases
- Gait scheduler for generating standard gait patterns (trot, walk, pace, bound)
- Foothold planner for computing foot landing positions from CoM trajectory
- OCP factory for building Crocoddyl optimal control problem nodes
"""

from .contact_sequence import ContactPhase, ContactSequence
from .foothold_planner import FootholdPlan, FootholdPlanner
from .gait_scheduler import GaitScheduler
from .ocp_factory import OCPFactory

__all__ = [
    "ContactPhase",
    "ContactSequence",
    "GaitScheduler",
    "FootholdPlanner",
    "FootholdPlan",
    "OCPFactory",
]
