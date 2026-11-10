"""Two-Go2 PhysX regression: reset row 1 while row 0 keeps standing."""
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
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device

    class SampledEnv(QuadrupedMPCEnv):
        def sample(self):
            d = self.robot.data
            return torch.cat((
                d.root_link_pose_w[:, :3] - self.scene.env_origins,
                d.root_link_pose_w[:, 3:], d.root_link_vel_w,
                d.joint_pos, d.joint_vel, self._pending_joint_efforts, d.applied_torque,
            ), dim=1).detach().clone()

        def _apply_action(self):
            super()._apply_action()
            self.samples.append(self.sample())

    env = SampledEnv(cfg=cfg)
    try:
        env.samples = []
        env.reset()
        env_ids = torch.tensor([1], device=env.device)
        action = torch.zeros((2, cfg.action_space), device=env.device)
        origins = env.scene.env_origins
        if (origins[0] - origins[1]).norm().item() < 1.0:
            raise RuntimeError(f"Expected distinct origins, got {origins.tolist()}")
        target = env.stance_command.reference[:, 0, :3]
        print("env_origins", origins.tolist(), flush=True)

        def interval():
            env.samples.clear()
            _, _, terminated, truncated, _ = env.step(action)
            if (terminated | truncated).any():
                raise RuntimeError(f"Unexpected episode end: terminated={terminated.tolist()}, "
                                   f"truncated={truncated.tolist()}, status={env.tracking_status.tolist()}")
            env.samples.append(env.sample())
            samples = torch.stack(env.samples)
            if len(samples) != cfg.decimation + 1 or not samples.isfinite().all():
                raise RuntimeError("Missing or nonfinite physics samples")
            position_error = (samples[:, :, :3] - target).norm(dim=-1)
            up = 1 - 2 * samples[:, :, 4:6].square().sum(-1)
            if position_error.max() > 0.05 or up.min() < 0.95:
                raise RuntimeError(f"Stance lost: position error={position_error.max().item()}, "
                                   f"min up={up.min().item()}")
            forces = env.foot_contact_sensor.data.net_forces_w[:, env._contact_sensor_foot_indices, 2]
            loaded = (forces > cfg.foot_contact_force_threshold).all(dim=1)
            effort_active = samples[-1, :, -12:].abs().amax(-1) > 0
            return samples[:, 0], loaded & effort_active

        def jumps(samples):
            # Worst component change per physics step for each signal and unit.
            delta = samples.diff(dim=0).abs()
            return torch.stack([block.max() for block in delta.split((3, 4, 6, 12, 12, 12, 12), dim=1)])

        loaded = []
        baseline = []
        for i in range(10):
            samples, contact = interval()
            loaded.append(contact)
            if i >= 5:
                baseline.append(jumps(samples))
        if (torch.stack(loaded).sum(0) < 8).any():
            raise RuntimeError("Both robots must establish four-foot stance before reset")
        before = env.sample()
        solver = env.tracking.controller.mpc
        if not solver._valid.all():
            raise RuntimeError("Expected two initialized warm starts")
        kept = [x[0].clone() for x in (solver._warm, solver._dual, solver._valid, env.episode_length_buf)]
        env._reset_idx(env_ids)
        after = env.sample()
        torch.testing.assert_close(after[0], before[0], rtol=0, atol=1e-6)
        for actual, expected in zip((solver._warm, solver._dual, solver._valid, env.episode_length_buf), kept):
            torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
        if solver._valid[1] or env.episode_length_buf[1] != 0 or env._pending_joint_efforts[1].any():
            raise RuntimeError("Selected row did not clear its solver, episode and command")
        torch.testing.assert_close(after[1, :3], target[1].to(after), rtol=0, atol=1e-5)
        # Use settled pre-reset noise with explicit floors and upper bounds.
        floor = before.new_tensor([.001, .002, .05, .005, .2, .3, .3])
        cap = before.new_tensor([.005, .01, .2, .02, .8, 1., 1.])
        limit = (4 * torch.stack(baseline).amax(0)).clamp(min=floor, max=cap)
        worst = torch.zeros_like(limit)
        loaded.clear()
        last = after[0:1]
        for i in range(10):
            samples, contact = interval()
            worst = torch.maximum(worst, jumps(torch.cat((last, samples))))
            if (worst > limit).any():
                raise RuntimeError(f"Peer discontinuity after reset at interval {i+1}: "
                                   f"peak={worst.tolist()}, limit={limit.tolist()}")
            last = samples[-1:]
            loaded.append(contact)
        counts = torch.stack(loaded).sum(0)
        if (counts < 8).any() or not loaded[-1].all() or not solver._valid.all():
            raise RuntimeError(f"Post-reset stance recovery failed: loaded samples={counts.tolist()}")
        print(json.dumps(dict(reset_env=1, kept_env=0, loaded_samples=counts.tolist(),
                              signals=["xyz", "wxyz", "root_velocity", "q", "dq", "requested_Nm", "applied_Nm"],
                              peer_max_step_delta=worst.tolist(), limits=limit.tolist())))
        print("PASS: two Go2 PhysX independent reset")
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
