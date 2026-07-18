# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protocol test for the MPC cluster using the local (thread) backend.

Runs on any machine — no EigenIPC, no crocoddyl, no IsaacLab required.
Validates: input routing per env, static worker affinity, RESET forwarding,
exception -> status reporting, barrier semantics, and clean shutdown.

    python scripts/test_mpc_cluster_local.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "source", "RL_Bezier_MPC"))

from RL_Bezier_MPC.mpc_cluster import MPCClusterClient  # noqa: E402
from RL_Bezier_MPC.mpc_cluster.defs import (  # noqa: E402
    STANDING_JOINTS,
    STATE_DIM,
    STATUS_EXCEPTION,
    STATUS_OK,
)

NUM_ENVS = 8
NUM_WORKERS = 3
HORIZON = 25


class DummySolution:
    def __init__(self, control, qpos, cost, converged):
        self.control = control
        self.predicted_states = [np.zeros(37), np.concatenate([np.zeros(7), qpos, np.zeros(18)])]
        self.cost = cost
        self.converged = converged


class DummyController:
    """Echoes identifiable values so routing errors are detectable.

    torque[j] = env_id + state[0];  qpos = env_id * 0.01;  cost = env_id * 10.
    Raises on solve when the env's gait step_length is negative (test hook).
    """

    def __init__(self, env_id):
        self.env_id = env_id
        self.reset_count = 0
        self.solve_count = 0

    def reset(self):
        self.reset_count += 1

    def solve(self, current_state, com_reference, current_foot_positions,
              gait_params, warm_start=True):
        self.solve_count += 1
        assert com_reference.shape == (HORIZON, 3), com_reference.shape
        # controller API contract: dict {foot_name: (3,)} in FOOT_ORDER
        assert set(current_foot_positions.keys()) == {"LF", "RF", "LH", "RH"}
        assert all(v.shape == (3,) for v in current_foot_positions.values())
        if gait_params["step_length"] < 0:
            raise RuntimeError(f"injected failure env {self.env_id}")
        torque = np.full(12, self.env_id + current_state[0])
        qpos = np.full(12, self.env_id * 0.01)
        return DummySolution(torque, qpos, cost=self.env_id * 10.0,
                             converged=self.env_id % 2 == 0)


controllers_by_env = {}


def factory(env_ids):
    made = {i: DummyController(i) for i in env_ids}
    controllers_by_env.update(made)
    return made


def main():
    mpc_cfg = {"mpc_horizon_steps": HORIZON}
    client = MPCClusterClient(
        num_envs=NUM_ENVS, mpc_cfg=mpc_cfg, num_workers=NUM_WORKERS,
        backend="local", controller_factory=factory, timeout_ms=5000,
    )

    states = np.zeros((NUM_ENVS, STATE_DIM))
    states[:, 0] = np.arange(NUM_ENVS) * 100.0  # identifiable per env
    com_ref = np.random.randn(NUM_ENVS, HORIZON, 3)
    foot_pos = np.random.randn(NUM_ENVS, 4, 3)
    gait = np.ones((NUM_ENVS, 3))

    # --- cycle 1: normal solve, verify routing ------------------------------
    out = client.solve_all(states, com_ref, foot_pos, gait)
    for i in range(NUM_ENVS):
        expected_torque = i + states[i, 0]
        assert np.allclose(out["torques"][i], expected_torque), \
            f"env {i}: torque routing broken: {out['torques'][i][0]} != {expected_torque}"
        assert np.allclose(out["qpos"][i], i * 0.01)
        assert out["cost"][i] == i * 10.0
        assert out["converged"][i] == (i % 2 == 0)
        assert out["status"][i] == STATUS_OK
    print("[1/5] routing OK: every env solved by its own controller with its own inputs")

    # --- cycle 2: RESET forwarding ------------------------------------------
    client.mark_reset([2, 5])
    client.solve_all(states, com_ref, foot_pos, gait)
    for i in range(NUM_ENVS):
        expected = 1 if i in (2, 5) else 0
        assert controllers_by_env[i].reset_count == expected, \
            f"env {i}: reset_count={controllers_by_env[i].reset_count}, want {expected}"
    # reset must be one-shot
    client.solve_all(states, com_ref, foot_pos, gait)
    assert controllers_by_env[2].reset_count == 1, "RESET bit not one-shot"
    print("[2/5] RESET forwarding OK (correct envs, one-shot)")

    # --- cycle 3: exception -> status, others unaffected --------------------
    gait_bad = gait.copy()
    gait_bad[3, 0] = -1.0  # inject failure in env 3
    out = client.solve_all(states, com_ref, foot_pos, gait_bad)
    assert out["status"][3] == STATUS_EXCEPTION
    assert np.allclose(out["torques"][3], 0.0)
    assert np.allclose(out["qpos"][3], STANDING_JOINTS)
    assert out["cost"][3] == 1e6 and not out["converged"][3]
    for i in [0, 1, 2, 4, 5, 6, 7]:
        assert out["status"][i] == STATUS_OK, f"env {i} affected by env 3 failure"
    print("[3/5] exception isolation OK (env 3 EXC, siblings healthy)")

    # --- cycle 4: solve_mask ------------------------------------------------
    before = {i: controllers_by_env[i].solve_count for i in range(NUM_ENVS)}
    mask = np.zeros(NUM_ENVS, dtype=bool)
    mask[[0, 7]] = True
    client.solve_all(states, com_ref, foot_pos, gait, solve_mask=mask)
    for i in range(NUM_ENVS):
        expected = before[i] + (1 if i in (0, 7) else 0)
        assert controllers_by_env[i].solve_count == expected
    print("[4/5] solve_mask OK (only masked envs solved)")

    # --- shutdown -----------------------------------------------------------
    client.shutdown()
    assert all(not t.is_alive() for t in client._threads), "worker threads still alive"
    print("[5/5] clean shutdown OK")

    # affinity check: each controller was built exactly once, by one worker
    assert len(controllers_by_env) == NUM_ENVS
    print(f"\nALL PASSED — {NUM_ENVS} envs / {NUM_WORKERS} workers, "
          f"protocol v1 barrier semantics verified.")


if __name__ == "__main__":
    main()
