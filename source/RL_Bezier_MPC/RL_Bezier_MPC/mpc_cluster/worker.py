# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MPC cluster worker: owns a fixed slice of envs and their controller instances.

Process entrypoint is :func:`worker_main` (spawned by launcher.py). It must stay
importable without IsaacLab/Isaac Sim — only numpy, crocoddyl, pinocchio and the
IsaacLab-free parts of this package are allowed at solve time.

Static env->worker affinity is a hard requirement: warm-start state
(``_prev_xs/_prev_us``) and ``_gait_clock`` live inside the controller objects
created here.
"""

import os
import sys
import traceback
from typing import Callable, Dict, Iterable, Optional

import numpy as np

from .defs import (
    CMD_RESET,
    CMD_SHUTDOWN,
    CMD_SOLVE,
    CTRL_QPOS,
    CTRL_TORQUE,
    FOOT_ORDER,
    INPUT_TENSORS,
    META_CONVERGED,
    META_COST,
    META_STATUS,
    STANDING_JOINTS,
    STATUS_EXCEPTION,
    STATUS_OK,
)


def build_default_controllers(env_ids: Iterable[int], mpc_cfg: Dict) -> Dict[int, object]:
    """Construct per-env CrocoddylQuadrupedMPC identical to the env serial path.

    Imports happen here (not module level) so the local test backend can inject
    dummy controllers without crocoddyl installed.
    """
    from RL_Bezier_MPC.robots.quadruped_cfg import (  # IsaacLab imports are guarded inside
        get_foot_frame_ids,
        load_pinocchio_model,
    )
    from RL_Bezier_MPC.controllers.crocoddyl_quadruped_mpc import CrocoddylQuadrupedMPC

    rmodel, urdf_path = load_pinocchio_model(
        urdf_path=mpc_cfg.get("robot_urdf_path") or None,
        robot_name=mpc_cfg["robot_name"],
        floating_base=True,
    )
    if (rmodel.nq, rmodel.nv) != (19, 18):
        raise ValueError(
            "MPC protocol requires a floating-base 12-DoF model "
            f"(nq=19, nv=18), got nq={rmodel.nq}, nv={rmodel.nv} "
            f"from {urdf_path!r}."
        )
    get_foot_frame_ids(rmodel, mpc_cfg["foot_frame_names"])  # fail fast if frames wrong

    hip_offsets = {k: np.asarray(v) for k, v in mpc_cfg["hip_offsets"].items()}
    controllers = {}
    for i in env_ids:
        controllers[i] = CrocoddylQuadrupedMPC(
            rmodel=rmodel,
            foot_frame_names=mpc_cfg["foot_frame_names"],
            hip_offsets=hip_offsets,
            gait_type=mpc_cfg["gait_type"],
            dt=mpc_cfg["mpc_dt"],
            horizon_steps=mpc_cfg["mpc_horizon_steps"],
            step_duration=mpc_cfg["default_step_duration"],
            support_duration=mpc_cfg["default_support_duration"],
            step_height=mpc_cfg["default_step_height"],
            mu=mpc_cfg["friction_coefficient"],
            max_iterations=mpc_cfg["mpc_max_iterations"],
            verbose=bool(mpc_cfg.get("mpc_verbose", False)),
        )
    return controllers


def worker_loop(
    tensors,
    consumer,
    env_ids: Iterable[int],
    controllers: Dict[int, object],
    horizon_steps: int,
    on_cycle: Optional[Callable] = None,
):
    """Consume triggers until a SHUTDOWN command is seen.

    Every exception is caught and reported through mpc_out_meta status;
    ``consumer.ack()`` is ALWAYS reached — a silent worker deadlocks the
    env-side barrier.
    """
    env_ids = list(env_ids)
    lo, hi = min(env_ids), max(env_ids) + 1
    shutdown = False

    while not shutdown:
        if not consumer.wait(ms_timeout=-1):
            continue

        for name in INPUT_TENSORS:
            tensors.pull(name, lo, hi)

        for i in env_ids:
            cmd = int(tensors.buf["mpc_cmd"][i, 0])
            if cmd & CMD_SHUTDOWN:
                shutdown = True
            if cmd & CMD_RESET:
                try:
                    controllers[i].reset()
                except Exception:
                    traceback.print_exc()
            if not (cmd & CMD_SOLVE):
                continue

            try:
                foot_flat = tensors.buf["mpc_foot_pos"][i].reshape(4, 3)
                solution = controllers[i].solve(
                    current_state=tensors.buf["mpc_states"][i],
                    com_reference=tensors.buf["mpc_com_ref"][i].reshape(horizon_steps, 3),
                    current_foot_positions={
                        name: foot_flat[k].copy() for k, name in enumerate(FOOT_ORDER)
                    },
                    gait_params={
                        "step_length": float(tensors.buf["mpc_gait"][i, 0]),
                        "step_height": float(tensors.buf["mpc_gait"][i, 1]),
                        "step_frequency": float(tensors.buf["mpc_gait"][i, 2]),
                    },
                    warm_start=True,
                )
                out = tensors.buf["mpc_out_ctrl"][i]
                out[CTRL_TORQUE] = solution.control
                if getattr(solution, "predicted_states", None) is not None \
                        and len(solution.predicted_states) > 1:
                    out[CTRL_QPOS] = solution.predicted_states[1][7:19]
                else:
                    out[CTRL_QPOS] = STANDING_JOINTS
                meta = tensors.buf["mpc_out_meta"][i]
                meta[META_COST] = float(solution.cost)
                meta[META_CONVERGED] = 1.0 if solution.converged else 0.0
                meta[META_STATUS] = STATUS_OK
            except Exception:
                traceback.print_exc()
                out = tensors.buf["mpc_out_ctrl"][i]
                out[CTRL_TORQUE] = 0.0
                out[CTRL_QPOS] = STANDING_JOINTS
                meta = tensors.buf["mpc_out_meta"][i]
                meta[META_COST] = 1e6
                meta[META_CONVERGED] = 0.0
                meta[META_STATUS] = STATUS_EXCEPTION

        tensors.push("mpc_out_ctrl", lo, hi)
        tensors.push("mpc_out_meta", lo, hi)

        if on_cycle is not None:
            on_cycle()

        consumer.ack()


def worker_main(
    namespace: str,
    num_envs: int,
    worker_id: int,
    env_start: int,
    env_end: int,
    mpc_cfg: Dict,
    verbose: bool = False,
):
    """Process entrypoint for one EigenIPC-backed worker."""
    # Each worker is one solver lane; Eigen/crocoddyl must not spawn threads.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    from .backend import EigenIPCConsumer, EigenIPCTensorSet
    from .defs import TRIGGER_BASENAME

    env_ids = range(env_start, env_end)
    horizon_steps = mpc_cfg["mpc_horizon_steps"]

    print(f"[MPCWorker {worker_id}] python={sys.executable} "
          f"envs [{env_start}, {env_end}), building controllers...",
          flush=True)
    controllers = build_default_controllers(env_ids, mpc_cfg)
    tensors = EigenIPCTensorSet(namespace, num_envs, horizon_steps,
                                is_server=False, verbose=verbose)
    consumer = EigenIPCConsumer(namespace, TRIGGER_BASENAME, verbose=verbose)
    print(f"[MPCWorker {worker_id}] ready.", flush=True)

    try:
        worker_loop(tensors, consumer, env_ids, controllers, horizon_steps)
    finally:
        consumer.close()
        tensors.close()
        print(f"[MPCWorker {worker_id}] shut down.", flush=True)
