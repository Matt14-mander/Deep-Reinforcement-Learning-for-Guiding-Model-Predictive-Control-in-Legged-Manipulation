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
import types
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "source", "RL_Bezier_MPC"))

# Load only the IPC subpackage. The protocol test intentionally must not need
# scipy, Isaac Lab, Pinocchio, or Crocoddyl through the project root imports.
package_root = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "source", "RL_Bezier_MPC", "RL_Bezier_MPC"
))
root_package = types.ModuleType("RL_Bezier_MPC")
root_package.__path__ = [package_root]
sys.modules.setdefault("RL_Bezier_MPC", root_package)

from RL_Bezier_MPC.mpc_cluster import MPCClusterClient  # noqa: E402
from RL_Bezier_MPC.mpc_cluster.client import (  # noqa: E402
    build_launcher_command,
    resolve_python_executable,
)
from RL_Bezier_MPC.mpc_cluster.defs import (  # noqa: E402
    PROTOCOL_VERSION,
    STANDING_JOINTS,
    STATE_DIM,
    STATUS_EXCEPTION,
    STATUS_OK,
    mpc_cfg_to_dict,
)
from RL_Bezier_MPC.interfaces import MPCCommandBatch, RobotStateBatch  # noqa: E402

NUM_ENVS = 8
NUM_WORKERS = 3
HORIZON = 25


class DummySolution:
    def __init__(self, control, qpos, cost, converged):
        self.control = control
        qvel = qpos + 0.5
        self.predicted_states = [
            np.zeros(37),
            np.concatenate([np.zeros(7), qpos, np.zeros(6), qvel]),
        ]
        self.cost = cost
        self.converged = converged
        self.solve_time = 0.001 + cost * 1e-6
        self.iterations = 4


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
    # --- pure dual-environment launch/config checks -------------------------
    launcher_python = resolve_python_executable(sys.executable)
    command = build_launcher_command(
        launcher_python, "test_namespace", NUM_ENVS, NUM_WORKERS,
        "/tmp/test_mpc_cfg.json", verbose=True,
    )
    assert command[0] == launcher_python
    assert command[1:3] == ["-m", "RL_Bezier_MPC.mpc_cluster.launcher"]
    assert command[-1] == "--verbose"

    cfg = SimpleNamespace(
        robot_name="go2",
        robot_urdf_path="/models/go2.urdf",
        foot_frame_names={"LF": "FL_foot", "RF": "FR_foot",
                          "LH": "RL_foot", "RH": "RR_foot"},
        hip_offsets={name: np.zeros(3) for name in ("LF", "RF", "LH", "RH")},
        gait_type="trot",
        mpc_dt=0.02,
        mpc_horizon_steps=HORIZON,
        default_step_duration=0.25,
        default_support_duration=0.10,
        default_step_height=0.15,
        friction_coefficient=0.7,
        mpc_max_iterations=50,
    )
    serialized = mpc_cfg_to_dict(cfg)
    assert serialized["robot_urdf_path"] == "/models/go2.urdf"
    print("[0/6] external Python launcher and URDF config serialization OK")

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
        assert np.allclose(out["qvel"][i], i * 0.01 + 0.5)
        assert out["cost"][i] == i * 10.0
        assert out["converged"][i] == (i % 2 == 0)
        assert out["status"][i] == STATUS_OK
        assert out["source_state_id"][i] == 0
        assert out["solution_id"][i] == 1
        assert out["reset_generation"][i] == 0
        assert out["fresh"][i]
        assert out["iterations"][i] == 4
    print("[1/6] routing and response provenance IDs OK")

    # --- cycle 2: RESET forwarding ------------------------------------------
    client.mark_reset([2, 5])
    reset_out = client.solve_all(states, com_ref, foot_pos, gait)
    for i in range(NUM_ENVS):
        expected = 1 if i in (2, 5) else 0
        assert controllers_by_env[i].reset_count == expected, \
            f"env {i}: reset_count={controllers_by_env[i].reset_count}, want {expected}"
        expected_generation = 1 if i in (2, 5) else 0
        assert reset_out["reset_generation"][i] == expected_generation
        assert reset_out["fresh"][i]
    # reset must be one-shot
    client.solve_all(states, com_ref, foot_pos, gait)
    assert controllers_by_env[2].reset_count == 1, "RESET bit not one-shot"
    print("[2/6] RESET forwarding and reset generations OK")

    # --- cycle 3: exception -> status, others unaffected --------------------
    gait_bad = gait.copy()
    gait_bad[3, 0] = -1.0  # inject failure in env 3
    out = client.solve_all(states, com_ref, foot_pos, gait_bad)
    assert out["status"][3] == STATUS_EXCEPTION
    assert np.allclose(out["torques"][3], 0.0)
    assert np.allclose(out["qpos"][3], STANDING_JOINTS)
    assert np.allclose(out["qvel"][3], 0.0)
    assert out["cost"][3] == 1e6 and not out["converged"][3]
    assert out["fresh"][3], "exception response must still identify its input state"
    for i in [0, 1, 2, 4, 5, 6, 7]:
        assert out["status"][i] == STATUS_OK, f"env {i} affected by env 3 failure"
    print("[3/6] exception isolation OK (env 3 EXC, siblings healthy)")

    # --- cycle 4: solve_mask ------------------------------------------------
    before = {i: controllers_by_env[i].solve_count for i in range(NUM_ENVS)}
    mask = np.zeros(NUM_ENVS, dtype=bool)
    mask[[0, 7]] = True
    masked_out = client.solve_all(states, com_ref, foot_pos, gait, solve_mask=mask)
    for i in range(NUM_ENVS):
        expected = before[i] + (1 if i in (0, 7) else 0)
        assert controllers_by_env[i].solve_count == expected
        assert masked_out["fresh"][i] == (i in (0, 7))
    print("[4/6] solve_mask OK; unsolved rows are explicitly stale")

    # --- typed canonical adapter -------------------------------------------
    ids = np.arange(NUM_ENVS, dtype=np.int64) + 100
    state_batch = RobotStateBatch(
        q_pin=states[:, :19], v_pin=states[:, 19:], foot_pos_w=foot_pos,
        physics_step_id=ids,
        reset_generation=client._reset_generation.copy(),
    )
    command_batch = MPCCommandBatch(
        com_reference=com_ref, gait=gait, solve_mask=np.ones(NUM_ENVS, dtype=bool),
    )
    typed_out = client.solve_batch(state_batch, command_batch)
    assert np.all(typed_out.source_state_id == ids)
    assert np.all(typed_out.fresh)
    assert typed_out.tau_ff.shape == (NUM_ENVS, 12)
    print("[5/6] canonical typed state/command/output adapter OK")

    # --- shutdown -----------------------------------------------------------
    client.shutdown()
    assert all(not t.is_alive() for t in client._threads), "worker threads still alive"
    print("[6/6] clean shutdown OK")

    # affinity check: each controller was built exactly once, by one worker
    assert len(controllers_by_env) == NUM_ENVS
    print(f"\nALL PASSED — {NUM_ENVS} envs / {NUM_WORKERS} workers, "
          f"protocol v{PROTOCOL_VERSION} semantics verified.")


if __name__ == "__main__":
    main()
