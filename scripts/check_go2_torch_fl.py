"""Single-Go2 PhysX FL lift and reach using the torch_tracking backend."""
import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


def run(args):
    import torch
    from crocoddyl_batched_mpc.models import go2_front_leg_tracking_plan

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import QuadrupedMPCEnv, QuadrupedMPCEnvCfg

    class FrontLegEnv(QuadrupedMPCEnv):
        def feet(self):
            return (self.robot.data.body_link_pos_w[0, self._foot_body_indices.tolist()]
                    - self.scene.env_origins[0]).double()

        def sample(self):
            # End of the preceding MPC interval, before the next command.
            k = self.tick - 1
            feet = self.feet()
            target = self.plan.feet[min(self.tick, len(self.plan.feet) - 1)]
            phase = self.plan.phases[k]
            if args.height_trace:
                data = self.robot.data
                adapter = self.tracking
                state = adapter.pack_state(
                    data.root_link_pose_w.double(), data.root_link_vel_w.double(),
                    data.joint_pos.double(), data.joint_vel.double(), self.scene.env_origins.double(),
                )
                ref = self.plan.reference[min(self.tick, len(self.plan.reference) - 1):][:1]
                fk = adapter.controller.model.foot_positions
                measured_z = fk(state[:, :19])[0, 0, 2]
                # Hold measured base fixed and substitute planned joints to isolate posture error.
                planned_joints_z = fk(torch.cat((state[:, :7], ref[:, 7:19]), dim=1))[0, 0, 2]
                heights = torch.stack((target[0, 2], fk(ref[:, :19])[0, 0, 2],
                                       planned_joints_z, measured_z, feet[0, 2]))
                if self.tick == self.shift_start:
                    self.stance_heights = heights.clone()
                if self.tick >= self.shift_start:
                    ids = adapter.mpc_to_sim[:3]
                    requested = self._pending_joint_efforts[0, ids]
                    applied = data.applied_torque[0, ids]
                    print(json.dumps(dict(
                        trace="FL", phase=phase, tick=k, time_s=self.tick * self.plan.dt,
                        # Ordered as target, planned FK, measured base + planned joints,
                        # measured FK, measured PhysX link. Each uses its own stance baseline.
                        dz_mm=((heights - self.stance_heights) * 1000).tolist(),
                        absolute_z_m=heights.tolist(),
                        FL_joint_error_rad=(state[0, 7:10] - ref[0, 7:10]).tolist(),
                        FL_requested_Nm=requested.tolist(), FL_applied_Nm=applied.tolist(),
                    )), flush=True)
            if self.tick == self.shift_start:
                self.offset = (feet - self.plan.anchors).clone()
            delta = feet - target
            relative = delta - self.offset
            force = self.foot_contact_sensor.data.net_forces_w[0, self._contact_sensor_foot_indices, :]
            row = torch.cat((delta[0].norm()[None], relative[0].norm()[None],
                             relative[1:].norm(dim=-1).max()[None],
                             feet[0] - self.plan.anchors[0] - self.offset[0],
                             force.norm(dim=-1), force[:, 2]))
            if not row.isfinite().all():
                raise RuntimeError(f"Nonfinite foot measurement at tick {k}")
            self.records.setdefault(phase, []).append(row)

        def _solve_mpc_control_step(self):
            if self.tick:
                self.sample()
            self.stance_command = self.plan.command(self.tick, 1, self.tracking.controller.horizon)
            super()._solve_mpc_control_step()
            if self.guard_triggered.any():
                raise RuntimeError(f"MPC rejected tick={self.tick} phase={self.plan.phases[self.tick]} "
                                   f"status={self.tracking_status.tolist()}")
            self.tick += 1

    cfg = QuadrupedMPCEnvCfg()
    cfg.mpc_backend = "torch_tracking"
    cfg.torch_foot_plane_height = args.foot_plane_height
    cfg.use_mpc_cluster = False
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = FrontLegEnv(cfg=cfg)
    try:
        env.reset()
        env.plan = go2_front_leg_tracking_plan(env.tracking.controller.model)
        env.tick = 0
        env.shift_start = env.plan.phases.index("shift")
        env.offset = torch.zeros_like(env.plan.anchors)
        env.records = {}
        if env._foot_body_indices is None:
            raise RuntimeError("Go2 foot link mapping unavailable")
        action = torch.zeros((1, cfg.action_space), device=env.device)
        n = len(env.plan.reference)
        if n % cfg.rl_policy_period:
            raise ValueError("Expected a whole number of policy intervals")
        for _ in range(n // cfg.rl_policy_period):
            _, _, terminated, truncated, _ = env.step(action)
            if (terminated | truncated).any():
                raise RuntimeError(f"Episode ended at tick={env.tick}")
        if env.tick != n:
            raise RuntimeError(f"Expected {n} MPC ticks, got {env.tick}")
        env.sample()
        passed = True
        for phase, records in env.records.items():
            rows = torch.stack(records)
            # FL force norm and the upward loads on FR, RL and RR.
            threshold = cfg.foot_contact_force_threshold
            tripod = (rows[:, 6] < threshold) & (rows[:, 11:14] > threshold).all(-1)
            report = dict(phase=phase, samples=len(rows), tripod_fraction=tripod.float().mean().item(),
                          raw_FL_link_error_max_m=rows[:, 0].max().item(),
                          FL_displacement_error_max_m=rows[:, 1].max().item(),
                          FL_displacement_error_rms_m=rows[:, 1].square().mean().sqrt().item(),
                          support_drift_max_m=rows[:, 2].max().item(),
                          end_FL_displacement_m=rows[-1, 3:6].tolist())
            print(json.dumps(report), flush=True)
            if phase != "stance":
                passed &= report["FL_displacement_error_max_m"] <= .01 and report["support_drift_max_m"] <= .01
            if phase in ("reach", "return", "hold"):
                passed &= report["tripod_fraction"] >= .90
            if phase == "reach":
                passed &= rows[-1, 3].item() >= .006
            if phase == "hold":
                passed &= rows[-1, 5].item() >= .015 and rows[-1, 1].item() <= .005
        print("stance_link_offsets_m", env.offset.tolist())
        if not passed:
            raise RuntimeError("PhysX tripod support or foot tracking outside acceptance bounds")
        print("PASS: single Go2 PhysX FL tracking")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foot-plane-height", type=float, default=0.022,
                        help="stance foot-link plane above physical ground (m); 0 reproduces baseline")
    parser.add_argument("--height-trace", action="store_true", help="Emit FL height diagnostics each MPC tick after settling")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    try:
        run(args)
    finally:
        launcher.app.close()
