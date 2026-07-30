# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared definitions for the MPC cluster: tensor table, command bits, status codes.

This is the single source of truth for the shared-memory layout used by both
the environment side (server) and the worker side (client). Any change here
changes the wire format — bump ``PROTOCOL_VERSION`` when editing.

Layout reference: docs/mpc_cluster_design.md §2.2
"""

from typing import Dict, List, NamedTuple

import numpy as np

PROTOCOL_VERSION = 1

# --- Command bits (mpc_cmd tensor, int32) -----------------------------------
CMD_IDLE = 0
CMD_SOLVE = 1
CMD_RESET = 2  # worker must call controller.reset() before solving
CMD_SHUTDOWN = 4  # worker acks, then exits its loop

# --- Status codes (mpc_out_meta[:, META_STATUS]) -----------------------------
STATUS_OK = 0.0
STATUS_EXCEPTION = 1.0  # solve raised; outputs invalid, env guard takes over

# --- Column indices ----------------------------------------------------------
STATE_DIM = 37          # q (19) + v (18), Pinocchio ordering
FOOT_POS_DIM = 12       # 4 feet x 3, ordered by FOOT_ORDER

# Wire order of feet inside mpc_foot_pos. The controller API takes a dict
# {foot_name: (3,)}; both sides serialize through this fixed order.
FOOT_ORDER = ("LF", "RF", "LH", "RH")
GAIT_DIM = 3            # step_length / step_height / step_frequency modulation
OUT_CTRL_DIM = 24       # [0:12] feedforward torques, [12:24] predicted joint positions
CTRL_TORQUE = slice(0, 12)
CTRL_QPOS = slice(12, 24)

OUT_META_DIM = 3
META_COST = 0
META_CONVERGED = 1
META_STATUS = 2

# Fallback standing joint positions (matches env fallback pose)
STANDING_JOINTS = np.array(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 0.8, -1.5]
)


class TensorSpec(NamedTuple):
    basename: str
    n_cols: int
    dtype: str  # "double" | "int"


def tensor_specs(horizon_steps: int) -> List[TensorSpec]:
    """Full tensor table. n_rows is always num_envs (decided at runtime)."""
    return [
        TensorSpec("mpc_states", STATE_DIM, "double"),
        TensorSpec("mpc_com_ref", horizon_steps * 3, "double"),
        TensorSpec("mpc_foot_pos", FOOT_POS_DIM, "double"),
        TensorSpec("mpc_gait", GAIT_DIM, "double"),
        TensorSpec("mpc_cmd", 1, "int"),
        TensorSpec("mpc_out_ctrl", OUT_CTRL_DIM, "double"),
        TensorSpec("mpc_out_meta", OUT_META_DIM, "double"),
    ]


INPUT_TENSORS = ("mpc_states", "mpc_com_ref", "mpc_foot_pos", "mpc_gait", "mpc_cmd")
OUTPUT_TENSORS = ("mpc_out_ctrl", "mpc_out_meta")

# Producer/Consumer channel used for solve triggering
TRIGGER_BASENAME = "mpc_go"


def partition_envs(num_envs: int, num_workers: int) -> List[range]:
    """Contiguous static partition of env indices across workers.

    Static affinity is a hard requirement: warm-start state and the gait clock
    live inside the controller instance owned by one worker.
    """
    num_workers = min(num_workers, num_envs)
    base, extra = divmod(num_envs, num_workers)
    parts, start = [], 0
    for w in range(num_workers):
        n = base + (1 if w < extra else 0)
        parts.append(range(start, start + n))
        start += n
    return parts


def mpc_cfg_to_dict(cfg) -> Dict:
    """Extract the JSON-serializable subset of the env cfg that workers need
    to construct CrocoddylQuadrupedMPC instances identical to the serial path."""
    return {
        "robot_name": cfg.robot_name,
        "robot_urdf_path": str(getattr(cfg, "robot_urdf_path", "") or ""),
        "foot_frame_names": dict(cfg.foot_frame_names),
        "hip_offsets": {k: list(map(float, v)) for k, v in cfg.hip_offsets.items()},
        "gait_type": cfg.gait_type,
        "mpc_dt": float(cfg.mpc_dt),
        "mpc_horizon_steps": int(cfg.mpc_horizon_steps),
        "default_step_duration": float(cfg.default_step_duration),
        "default_support_duration": float(cfg.default_support_duration),
        "default_step_height": float(cfg.default_step_height),
        "friction_coefficient": float(cfg.friction_coefficient),
        "mpc_max_iterations": int(cfg.mpc_max_iterations),
        "mpc_verbose": bool(getattr(cfg, "mpc_verbose", False)),
        "mpc_force_standing_contacts": bool(
            getattr(cfg, "mpc_force_standing_contacts", False)
        ),
        "mpc_use_demo_stabilization_weights": bool(
            getattr(cfg, "mpc_use_demo_stabilization_weights", False)
        ),
        "mpc_initial_full_support_duration": float(
            getattr(cfg, "mpc_initial_full_support_duration", 0.0)
        ),
        "mpc_use_feasible_cold_start_rollout": bool(
            getattr(cfg, "mpc_use_feasible_cold_start_rollout", False)
        ),
        "mpc_enable_warm_start": bool(
            getattr(cfg, "mpc_enable_warm_start", True)
        ),
        "mpc_reference_is_root_position": bool(
            getattr(cfg, "mpc_reference_is_root_position", False)
        ),
        "mpc_return_quasi_static_control": bool(
            getattr(cfg, "mpc_return_quasi_static_control", False)
        ),
    }
