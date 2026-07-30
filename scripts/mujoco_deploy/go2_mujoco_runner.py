#!/usr/bin/env python3
"""Run a trained Stage 1/2 policy through Crocoddyl in MuJoCo.

Run this script from the MPC environment. Isaac Sim and Isaac Lab are not
imported; the checkpoint actor, Pinocchio/Crocoddyl and MuJoCo form one process.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "RL_Bezier_MPC"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SOURCE_ROOT))

from deployment_core import (  # noqa: E402
    ISAAC_JOINT_ORDER,
    PINOCCHIO_JOINT_ORDER,
    STANDING_JOINTS_PIN,
    ComReferenceManager,
    StageActionProcessor,
    build_policy_observation,
    canonical_joint_name,
    quaternion_wxyz_to_rotation,
    reorder_indices,
    training_compatible_contacts,
)
from policy_runtime import load_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mjcf", required=True, help="Go2 MuJoCo scene.xml")
    parser.add_argument("--urdf", required=True, help="Matching Go2 URDF for Pinocchio")
    parser.add_argument("--stage", type=int, choices=(1, 2), default=2)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--target", type=float, nargs=3, default=(1.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--device", default="cpu", help="Torch actor device; CPU is sufficient for one robot")
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--gait", choices=("trot", "walk", "pace", "bound"), default="trot")
    parser.add_argument("--kp", type=float, default=25.0)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--torque-limit", type=float, default=23.5)
    parser.add_argument("--root-height", type=float, default=0.4)
    parser.add_argument("--contact-mode", choices=("training", "height"), default="training")
    parser.add_argument("--contact-height", type=float, default=0.035)
    parser.add_argument("--mpc-iterations", type=int, default=50)
    parser.add_argument("--mpc-cost-limit", type=float, default=50000.0)
    parser.add_argument("--verbose-mpc", action="store_true")
    parser.add_argument("--no-realtime", action="store_true", help="Do not pace a visible viewer to wall time")
    return parser.parse_args()


def object_name(mujoco, model, object_type, index: int) -> str:
    return mujoco.mj_id2name(model, object_type, index) or f"unnamed_{index}"


def resolve_joint_layout(mujoco, model) -> tuple[int, np.ndarray, np.ndarray]:
    free_joint_ids = [index for index in range(model.njnt) if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free_joint_ids) != 1:
        raise ValueError(f"Expected exactly one free joint, found {len(free_joint_ids)}")
    free_joint_id = free_joint_ids[0]
    scalar_types = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    scalar_ids = [index for index in range(model.njnt) if model.jnt_type[index] in scalar_types]
    names = [object_name(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in scalar_ids]
    pin_indices = reorder_indices(names, PINOCCHIO_JOINT_ORDER)
    selected_ids = np.asarray(scalar_ids, dtype=np.int32)[pin_indices]
    qpos_addresses = model.jnt_qposadr[selected_ids].astype(np.int32)
    dof_addresses = model.jnt_dofadr[selected_ids].astype(np.int32)
    return free_joint_id, qpos_addresses, dof_addresses


def resolve_foot_sites(mujoco, model) -> dict[str, int]:
    result = {}
    site_names = [object_name(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, index) for index in range(model.nsite)]
    aliases = {"LF": ("fl", "lf"), "RF": ("fr", "rf"), "LH": ("rl", "lh"), "RH": ("rr", "rh")}
    for foot, prefixes in aliases.items():
        matches = [
            index for index, name in enumerate(site_names)
            if any(prefix in canonical_joint_name(name) for prefix in prefixes) and "foot" in name.lower()
        ]
        if matches:
            result[foot] = matches[0]
    return result


def root_velocity_world(mujoco, model, data, root_body_id: int) -> tuple[np.ndarray, np.ndarray]:
    spatial = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, root_body_id, spatial, 0)
    # MuJoCo object velocities are [angular, linear]. flg_local=0 requests world axes.
    return spatial[3:].copy(), spatial[:3].copy()


def initialize_pose(mujoco, model, data, free_joint_id: int, qpos_addresses: np.ndarray, height: float) -> None:
    root_qpos = int(model.jnt_qposadr[free_joint_id])
    data.qpos[root_qpos : root_qpos + 7] = (0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0)
    data.qpos[qpos_addresses] = STANDING_JOINTS_PIN
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def main() -> None:
    args = parse_args()
    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "MuJoCo is missing. Install it in rlbmpc_mpc with: "
            "python -m pip install 'mujoco==3.2.7'"
        ) from exc

    from RL_Bezier_MPC.controllers.crocoddyl_quadruped_mpc import CrocoddylQuadrupedMPC
    from RL_Bezier_MPC.robots.quadruped_cfg import load_pinocchio_model

    obs_dim, action_dim = ((45, 12) if args.stage == 1 else (48, 15))
    policy = load_policy(
        args.checkpoint,
        expected_observation_dim=obs_dim,
        expected_action_dim=action_dim,
        device=args.device,
    )
    model = mujoco.MjModel.from_xml_path(str(Path(args.mjcf).expanduser().resolve()))
    data = mujoco.MjData(model)
    # The deployment law supplies generalized torques directly. Neutralize any
    # position/motor gains embedded in the selected MJCF so they cannot add a
    # second, model-specific controller on top of qfrc_applied.
    if model.nu:
        model.actuator_gainprm[:] = 0.0
        model.actuator_biasprm[:] = 0.0
    free_joint_id, qpos_addresses, dof_addresses = resolve_joint_layout(mujoco, model)
    root_body_id = int(model.jnt_bodyid[free_joint_id])
    foot_sites = resolve_foot_sites(mujoco, model)

    rmodel, urdf_used = load_pinocchio_model(str(Path(args.urdf).expanduser().resolve()), floating_base=True)
    expected_names = {canonical_joint_name(item) for item in PINOCCHIO_JOINT_ORDER}
    expected_pin_names = [name for name in rmodel.names[2:] if canonical_joint_name(name) in expected_names]
    reorder_indices(expected_pin_names, PINOCCHIO_JOINT_ORDER)
    controller = CrocoddylQuadrupedMPC(
        rmodel=rmodel,
        foot_frame_names={"LF": "FL_foot", "RF": "FR_foot", "LH": "RL_foot", "RH": "RR_foot"},
        hip_offsets={
            "LF": np.asarray((0.1934, 0.0465, 0.0)),
            "RF": np.asarray((0.1934, -0.0465, 0.0)),
            "LH": np.asarray((-0.1934, 0.0465, 0.0)),
            "RH": np.asarray((-0.1934, -0.0465, 0.0)),
        },
        gait_type=args.gait,
        dt=args.control_dt,
        horizon_steps=25,
        step_duration=0.25,
        support_duration=0.10,
        step_height=0.15,
        mu=0.7,
        max_iterations=args.mpc_iterations,
        verbose=args.verbose_mpc,
    )

    action_processor = StageActionProcessor(stage=args.stage)
    references = ComReferenceManager(dt=args.control_dt, horizon=3.0, mpc_steps=25, blend_steps=5)
    pin_to_isaac = reorder_indices(PINOCCHIO_JOINT_ORDER, ISAAC_JOINT_ORDER)
    initialize_pose(mujoco, model, data, free_joint_id, qpos_addresses, args.root_height)
    root_qpos = int(model.jnt_qposadr[free_joint_id])
    physics_dt = float(model.opt.timestep)
    substeps = max(1, round(args.control_dt / physics_dt))
    effective_control_dt = substeps * physics_dt
    if not np.isclose(effective_control_dt, args.control_dt, atol=1e-9):
        print(f"WARNING: control dt adjusted from {args.control_dt:.6f}s to {effective_control_dt:.6f}s")
    target = np.asarray(args.target, dtype=np.float32)
    last_good = (STANDING_JOINTS_PIN.copy(), np.zeros(12, dtype=np.float64))
    policy_period = 10 if args.stage == 1 else 1
    control_step = 0
    guards = 0
    filtered_vertical_velocity = 0.0

    viewer = None
    if not args.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)

    print("MuJoCo deployment ready")
    print(f"  checkpoint: {policy.checkpoint_path}")
    print(f"  policy: {obs_dim}D -> {policy.info.hidden_dims} -> {action_dim}D")
    print(f"  MuJoCo: dt={physics_dt:.6f}s, control={effective_control_dt:.6f}s, substeps={substeps}")
    print(f"  Pinocchio URDF: {urdf_used}")
    print("  torque path: qfrc_applied (MJCF actuators bypassed for exact PD + feedforward torque)")

    try:
        while data.time < args.duration and (viewer is None or viewer.is_running()):
            wall_start = time.perf_counter()
            root = data.qpos[root_qpos : root_qpos + 7].copy()
            position = root[:3]
            quaternion_wxyz = root[3:7]
            linear_w, angular_w = root_velocity_world(mujoco, model, data, root_body_id)
            joint_position_pin = data.qpos[qpos_addresses].copy()
            joint_velocity_pin = data.qvel[dof_addresses].copy()

            if args.contact_mode == "training":
                contacts = training_compatible_contacts(position[2], args.root_height)
            else:
                if len(foot_sites) != 4:
                    raise ValueError(
                        "--contact-mode height needs four named foot sites; "
                        f"resolved {foot_sites}. Use --contact-mode training."
                    )
                contacts = np.asarray(
                    [
                        float(data.site_xpos[foot_sites[name], 2] <= args.contact_height)
                        for name in ("LF", "RF", "LH", "RH")
                    ],
                    dtype=np.float32,
                )

            observation = build_policy_observation(
                root_position_w=position,
                root_quaternion_wxyz=quaternion_wxyz,
                root_linear_velocity_w=linear_w,
                root_angular_velocity_w=angular_w,
                joint_position_isaac=joint_position_pin[pin_to_isaac],
                joint_velocity_isaac=joint_velocity_pin[pin_to_isaac],
                foot_contacts=contacts,
                target_position_w=target,
                trajectory_phase=references.normalized_phase,
                applied_gait_modifiers=action_processor.previous_gait if args.stage == 2 else None,
            )
            raw_action = policy.act(observation)
            processed = action_processor.process(raw_action)

            if control_step % policy_period == 0 or references.trajectory is None:
                com_reference = references.update(processed.bezier_parameters, position)
            else:
                com_reference = references.current_reference()
                references.advance()

            rotation_wb = quaternion_wxyz_to_rotation(quaternion_wxyz)
            linear_body = rotation_wb.T @ linear_w
            angular_body = rotation_wb.T @ angular_w
            # Mirror the training environment's MPC-only velocity guards. The
            # policy observation above intentionally keeps the unfiltered world velocity.
            filtered_vertical_velocity = 0.3 * linear_body[2] + 0.7 * filtered_vertical_velocity
            linear_body[2] = np.clip(filtered_vertical_velocity, -0.15, 0.15)
            angular_body = np.clip(angular_body, -1.0, 1.0)
            pin_q = np.concatenate((position, quaternion_wxyz[[1, 2, 3, 0]], joint_position_pin))
            pin_v = np.concatenate((linear_body, angular_body, joint_velocity_pin))
            pin_state = np.concatenate((pin_q, pin_v))
            gait = {
                "step_length": float(processed.gait_modifiers[0]),
                "step_height": float(processed.gait_modifiers[1]),
                "step_frequency": float(processed.gait_modifiers[2]),
            }
            try:
                solution = controller.solve(pin_state, com_reference, gait_params=gait, warm_start=True)
                if not np.isfinite(solution.cost) or solution.cost > args.mpc_cost_limit:
                    raise RuntimeError(f"MPC guard: cost={solution.cost}")
                feedforward = np.asarray(solution.control, dtype=np.float64)
                desired_joint_position = (
                    np.asarray(solution.predicted_states[1][7:19], dtype=np.float64)
                    if len(solution.predicted_states) > 1
                    else STANDING_JOINTS_PIN.copy()
                )
                last_good = (desired_joint_position.copy(), feedforward.copy())
            except Exception as exc:
                guards += 1
                desired_joint_position, feedforward = last_good
                if guards <= 5 or guards % 100 == 0:
                    print(f"[MPC Guard #{guards}] {exc}")

            torque = (
                feedforward
                + args.kp * (desired_joint_position - joint_position_pin)
                - args.kd * joint_velocity_pin
            )
            torque = np.clip(torque, -args.torque_limit, args.torque_limit)
            for _ in range(substeps):
                data.ctrl[:] = 0.0
                data.qfrc_applied[:] = 0.0
                data.qfrc_applied[dof_addresses] = torque
                mujoco.mj_step(model, data)
            control_step += 1

            if viewer is not None:
                viewer.sync()
                if not args.no_realtime:
                    time.sleep(max(0.0, effective_control_dt - (time.perf_counter() - wall_start)))
            if control_step % 100 == 0:
                distance = np.linalg.norm(position[:2] - target[:2])
                print(f"t={data.time:6.2f}s  pos={position.round(3)}  target_distance={distance:.3f}  guards={guards}")
    finally:
        if viewer is not None:
            viewer.close()

    print(f"Finished at t={data.time:.2f}s; MPC guards={guards}; final position={data.qpos[root_qpos:root_qpos + 3]}")


if __name__ == "__main__":
    main()
