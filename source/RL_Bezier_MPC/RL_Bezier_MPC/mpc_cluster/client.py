# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Env-side API of the MPC cluster.

``MPCClusterClient`` owns the shared tensors (server role) and the trigger
producer. One call to :meth:`solve_all` runs one synchronous barrier cycle:

    write inputs -> trigger -> workers solve in parallel -> wait acks -> read outputs

Phase 2 (pipelined mode) will split this into trigger()/collect(); the internal
structure already separates the two halves.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .defs import (
    CMD_RESET,
    CMD_SHUTDOWN,
    CMD_SOLVE,
    CTRL_QPOS,
    CTRL_TORQUE,
    GAIT_DIM,
    INPUT_TENSORS,
    META_CONVERGED,
    META_COST,
    META_STATUS,
    OUTPUT_TENSORS,
    STATE_DIM,
    TRIGGER_BASENAME,
    partition_envs,
)


def resolve_python_executable(python_executable: Optional[str] = None) -> str:
    """Resolve the interpreter used to launch the external MPC environment."""
    candidate = python_executable or sys.executable
    candidate = os.path.abspath(os.path.expandvars(os.path.expanduser(candidate)))
    if not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"MPC Python executable does not exist: {candidate}. "
            "Set cluster_python_executable to the rlbmpc_mpc environment Python."
        )
    return candidate


def build_launcher_command(
    python_executable: str,
    namespace: str,
    num_envs: int,
    num_workers: int,
    cfg_path: str,
    verbose: bool = False,
) -> list[str]:
    """Build the launcher command without importing Isaac Sim or Crocoddyl."""
    command = [
        python_executable,
        "-m",
        "RL_Bezier_MPC.mpc_cluster.launcher",
        "--namespace",
        namespace,
        "--num-envs",
        str(num_envs),
        "--workers",
        str(num_workers),
        "--mpc-cfg",
        cfg_path,
    ]
    if verbose:
        command.append("--verbose")
    return command


class MPCClusterClient:
    """Server-side (environment) handle to the solver cluster.

    Args:
        num_envs: Number of environments E.
        mpc_cfg: JSON-serializable controller config (see defs.mpc_cfg_to_dict).
        num_workers: Worker process/thread count W.
        namespace: Shared-memory namespace; default ``rlbmpc_<pid>``.
        backend: "eigenipc" (production, Linux) or "local" (thread-based test).
        autostart: eigenipc backend only — Popen the launcher automatically.
            Set False to start it manually in another terminal (debugging).
        timeout_ms: Barrier timeout for one solve cycle. On timeout a
            RuntimeError is raised: a worker died or is stuck — check cluster logs.
        controller_factory: local backend only — ``f(env_ids) -> {i: controller}``.
    """

    def __init__(
        self,
        num_envs: int,
        mpc_cfg: Dict,
        num_workers: int,
        namespace: Optional[str] = None,
        backend: str = "eigenipc",
        autostart: bool = True,
        timeout_ms: int = 30000,
        verbose: bool = False,
        controller_factory: Optional[Callable] = None,
        python_executable: Optional[str] = None,
    ):
        self.num_envs = num_envs
        self.horizon_steps = int(mpc_cfg["mpc_horizon_steps"])
        self.backend = backend
        self.timeout_ms = timeout_ms
        self.namespace = namespace or f"rlbmpc_{os.getpid()}"
        self._partitions = partition_envs(num_envs, num_workers)
        self.num_workers = len(self._partitions)
        self._pending_reset = np.zeros(num_envs, dtype=bool)
        self._proc = None
        self._cfg_path = None
        self._threads = []
        self._closed = False

        if backend == "eigenipc":
            from .backend import EigenIPCProducer, EigenIPCTensorSet

            launcher_python = (
                resolve_python_executable(python_executable) if autostart else None
            )
            self.tensors = EigenIPCTensorSet(
                self.namespace, num_envs, self.horizon_steps,
                is_server=True, verbose=verbose,
            )
            self.producer = EigenIPCProducer(self.namespace, TRIGGER_BASENAME, verbose)
            if autostart:
                self._cfg_path = os.path.join(
                    tempfile.gettempdir(), f"{self.namespace}_mpc_cfg.json"
                )
                with open(self._cfg_path, "w", encoding="utf-8") as f:
                    json.dump(mpc_cfg, f)
                self._proc = subprocess.Popen(
                    build_launcher_command(
                        launcher_python,
                        self.namespace,
                        num_envs,
                        self.num_workers,
                        self._cfg_path,
                        verbose,
                    ),
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                )
                print(f"[MPCClusterClient] launched cluster pid={self._proc.pid} "
                      f"python={launcher_python} namespace={self.namespace} "
                      f"workers={self.num_workers}", flush=True)

        elif backend == "local":
            from .backend import LocalConsumer, LocalHub, LocalProducer, LocalTensorSet
            from .worker import worker_loop

            if controller_factory is None:
                raise ValueError("local backend requires controller_factory")
            hub = LocalHub(num_envs, self.horizon_steps, self.num_workers)
            self.tensors = LocalTensorSet(hub)
            self.producer = LocalProducer(hub)
            for envs in self._partitions:
                consumer = LocalConsumer(hub)
                controllers = controller_factory(list(envs))
                t = threading.Thread(
                    target=worker_loop,
                    args=(LocalTensorSet(hub), consumer, envs, controllers,
                          self.horizon_steps),
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ------------------------------------------------------------------ API

    def mark_reset(self, env_ids: Sequence[int]):
        """Queue controller.reset() for these envs; consumed by the next solve_all."""
        self._pending_reset[np.asarray(env_ids, dtype=int)] = True

    def solve_all(
        self,
        states: np.ndarray,      # (E, 37)
        com_ref: np.ndarray,     # (E, H, 3) or (E, H*3)
        foot_pos: np.ndarray,    # (E, 4, 3) or (E, 12)
        gait: np.ndarray,        # (E, 3)
        solve_mask: Optional[np.ndarray] = None,  # (E,) bool; default all
    ) -> Dict[str, np.ndarray]:
        """One synchronous solve cycle over all envs. Returns copies:
        torques (E,12), qpos (E,12), cost (E,), converged (E,) bool, status (E,).
        """
        E = self.num_envs
        buf = self.tensors.buf
        buf["mpc_states"][:] = states.reshape(E, STATE_DIM)
        buf["mpc_com_ref"][:] = com_ref.reshape(E, self.horizon_steps * 3)
        buf["mpc_foot_pos"][:] = foot_pos.reshape(E, 12)
        buf["mpc_gait"][:] = gait.reshape(E, GAIT_DIM)

        cmd = np.where(
            solve_mask if solve_mask is not None else np.ones(E, dtype=bool),
            CMD_SOLVE, 0,
        ).astype(np.int32)
        cmd |= np.where(self._pending_reset, CMD_RESET, 0).astype(np.int32)
        buf["mpc_cmd"][:, 0] = cmd
        self._pending_reset[:] = False

        for name in INPUT_TENSORS:
            self.tensors.push(name, 0, E)

        t0 = time.perf_counter()
        self.producer.trigger()
        if not self.producer.wait_ack_from(self.num_workers, self.timeout_ms):
            raise RuntimeError(
                f"[MPCClusterClient] barrier timeout after {self.timeout_ms}ms — "
                f"a worker died or is stuck; check cluster logs "
                f"(namespace={self.namespace})."
            )
        self.last_solve_wall_ms = (time.perf_counter() - t0) * 1e3

        for name in OUTPUT_TENSORS:
            self.tensors.pull(name, 0, E)

        return {
            "torques": buf["mpc_out_ctrl"][:, CTRL_TORQUE].copy(),
            "qpos": buf["mpc_out_ctrl"][:, CTRL_QPOS].copy(),
            "cost": buf["mpc_out_meta"][:, META_COST].copy(),
            "converged": buf["mpc_out_meta"][:, META_CONVERGED] > 0.5,
            "status": buf["mpc_out_meta"][:, META_STATUS].copy(),
        }

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.tensors.buf["mpc_cmd"][:, 0] = CMD_SHUTDOWN
            self.tensors.push("mpc_cmd", 0, self.num_envs)
            self.producer.trigger()
            self.producer.wait_ack_from(self.num_workers, 5000)
        except Exception:
            pass
        for t in self._threads:
            t.join(timeout=2.0)
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
        self.producer.close()
        self.tensors.close()
        if self._cfg_path is not None:
            try:
                os.remove(self._cfg_path)
            except FileNotFoundError:
                pass
        print("[MPCClusterClient] shut down.", flush=True)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
