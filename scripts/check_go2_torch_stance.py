"""Run a two-second fixed-reference Go2 stance acceptance in Isaac Lab."""
import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def run(args):
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import QuadrupedMPCEnv, QuadrupedMPCEnvCfg

    cfg = QuadrupedMPCEnvCfg()
    cfg.mpc_backend = "torch_tracking"
    cfg.use_mpc_cluster = False
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = QuadrupedMPCEnv(cfg=cfg)
    try:
        env.reset()
        action = torch.zeros((1, cfg.action_space), device=env.device)
        target_z = env.stance_command.reference[0, 0, 2].item()
        four_foot_samples = 0
        for step in range(10):
            _, _, terminated, truncated, _ = env.step(action)
            if (terminated | truncated).any():
                raise RuntimeError(f"Episode ended at step {step + 1}, solver status={env.tracking_status.tolist()}")
            data = env.robot.data
            height = (data.root_pos_w - env.scene.env_origins)[0, 2].item()
            up = 1 - 2 * data.root_quat_w[0, 1:3].square().sum().item()
            forces = env.foot_contact_sensor.data.net_forces_w[0, env._contact_sensor_foot_indices, 2]
            contacts = (forces > cfg.foot_contact_force_threshold).sum().item()
            requested = env._pending_joint_efforts.abs().max().item()
            applied = data.applied_torque.abs().max().item()
            print(json.dumps(dict(step=step + 1, height_m=height, up_z=up, contacts=contacts,
                                  requested_Nm=requested, applied_Nm=applied)), flush=True)
            if not (abs(height - target_z) <= 0.05 and up >= 0.95 and
                    0 < applied < float("inf") and 0 < requested < float("inf")):
                raise RuntimeError("Go2 stance height, tilt or effort outside acceptance bounds")
            four_foot_samples += contacts == 4
        if four_foot_samples < 8:
            raise RuntimeError(f"Four loaded feet in only {four_foot_samples}/10 samples")
        print("PASS: single Go2 PhysX torch_tracking stance")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    try:
        run(args)
    finally:
        launcher.app.close()
