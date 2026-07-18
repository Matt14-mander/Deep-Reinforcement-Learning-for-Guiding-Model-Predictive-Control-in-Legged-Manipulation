# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPC solver cluster: parallel Crocoddyl solving in worker processes,
coupled to the IsaacLab env via EigenIPC shared memory (Linux) or an
in-process thread backend (protocol tests).

Design doc: docs/mpc_cluster_design.md
"""

from .client import MPCClusterClient
from .defs import mpc_cfg_to_dict, partition_envs

__all__ = ["MPCClusterClient", "mpc_cfg_to_dict", "partition_envs"]
