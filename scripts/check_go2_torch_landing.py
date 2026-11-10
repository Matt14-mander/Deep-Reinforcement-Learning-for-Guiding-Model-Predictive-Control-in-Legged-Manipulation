"""PhysX contact-confirmed FL landing, followed by a four-foot weight transfer."""
import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def run(args):
    import torch
    from crocoddyl_batched_mpc.models import go2_front_leg_tracking_plan

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import QuadrupedMPCEnv, QuadrupedMPCEnvCfg

    class LandingEnv(QuadrupedMPCEnv):
        def _solve_mpc_control_step(self):
            d = self.robot.data
            a = self.tracking
            state = a.pack_state(d.root_link_pose_w.double(), d.root_link_vel_w.double(),
                                 d.joint_pos.double(), d.joint_vel.double(), self.tracking_origins)
            loads = self.foot_contact_sensor.data.net_forces_w[0, self._contact_sensor_foot_indices, 2]
            feet = (d.body_link_pos_w[0, self._foot_body_indices.tolist()] - self.tracking_origins[0])
            if not torch.isfinite(loads).all() or not feet.isfinite().all():
                raise RuntimeError("Nonfinite foot measurement")
            loaded = loads > self.cfg.foot_contact_force_threshold
            k = self.tick
            if k % 10 == 0 or k in (self.release, self.apex, self.bottom):
                print(json.dumps(dict(stage="mpc", tick=k, landed=self.landed,
                                      confirmed_ticks=self.confirm, stable_ticks=self.stable,
                                      fl_link_z_above_plane_m=feet[0, 2].item(),
                                      foot_loads_N=loads.tolist())), flush=True)
            if k == self.release:
                self.ground = feet[0, 2].item()
            if k >= self.apex:
                height = feet[0, 2].item() - self.ground
                if k == self.apex:
                    if height < .015 or loaded.tolist() != [False, True, True, True]:
                        raise RuntimeError(f"Lift failed: height={height}, loads={loads.tolist()}")
                if not self.landed:
                    near = abs(height) <= .005
                    self.confirm = self.confirm + 1 if near and loaded.all().item() else 0
                    if self.confirm >= 3:
                        self.landed = True
                        self.anchors[0, 0, :2] = feet[0, :2]
                        self.landing_tick = k
                        print(json.dumps(dict(landing_tick=k, height_m=height, loads_N=loads.tolist())), flush=True)
                    elif k >= self.bottom + 100:
                        raise RuntimeError(f"No confirmed landing: height={height}, loads={loads.tolist()}")
            if self.landed:
                idx = max(self.release - 1 - (k - self.landing_tick), self.shift)
                direction = -1 if idx > self.shift else 0
            elif k < self.apex:
                idx, direction = k, 1
            else:
                idx = max(self.release - 1, self.apex - 1 - (k - self.apex))
                direction = -1 if idx > self.release - 1 else 0
            solver = a.controller.mpc
            ids = torch.arange(solver.horizon + 1, device=self.device) * direction + idx
            lo = self.shift if self.landed else self.release - 1 if direction < 0 else 0
            ids = ids.clamp(lo, len(self.plan.reference) - 1)
            ref = self.plan.reference[ids].clone()
            ref[:, 19:] *= direction
            accel = self.plan.acceleration[idx:idx + 1] * abs(direction)
            support = self.plan.support_masks[0 if self.landed or k < self.release else self.release][None]
            if k == self.release or (self.landed and k == self.landing_tick):
                solver.reset()
            effort, result = solver.compute(state, ref[None], accel, support, self.anchors)
            if not result.usable.all():
                raise RuntimeError(f"Unusable MPC at tick={k} status={result.status.tolist()} loads={loads.tolist()}")
            # Establish a small real contact load while the MPC still uses tripod support.
            # The correction vanishes on the confirmed switch to four-foot MPC.
            if not self.landed and k >= self.bottom and near and loaded[1:].all():
                force_n = min(5.0, (k - self.bottom + 1) * self.plan.dt * 10.0)
                kin = solver.model._kinematics(state[:, :19], state[:, 19:])
                jz = kin[4].reshape(1, 4, 3, 18)[:, 0, 2, 6:9]
                effort = effort.clone()
                effort[:, :3] -= force_n * jz
                effort = effort.clamp(-solver.constraints.torque_limits, solver.constraints.torque_limits)
                if k % 10 == 0:
                    print(json.dumps(dict(stage="preload", tick=k, commanded_down_N=force_n,
                                          measured_FL_N=loads[0].item())), flush=True)
            self._pending_joint_efforts.copy_(a.scatter_action(effort).clamp(
                -self.cfg.max_joint_torque, self.cfg.max_joint_torque))
            self._last_mpc_costs = result.cost
            self._last_mpc_converged = result.success
            self._advance_reference_clocks()
            if self.landed and k >= self.landing_tick + self.release - self.shift:
                self.stable = self.stable + 1 if loaded.all().item() else 0
            self.tick += 1

    cfg = QuadrupedMPCEnvCfg()
    cfg.mpc_backend = "torch_tracking"
    cfg.torch_foot_plane_height = .022
    cfg.use_mpc_cluster = False
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    env = None
    try:
        print("[landing] creating environment", flush=True)
        env = LandingEnv(cfg=cfg)
        print("[landing] resetting environment", flush=True)
        env.reset()
        if env._foot_body_indices is None:
            raise RuntimeError("Missing foot link mapping")
        print("[landing] building reference", flush=True)
        env.plan = go2_front_leg_tracking_plan(env.tracking.controller.model)
        env.shift = env.plan.phases.index("shift")
        env.release = env.plan.phases.index("lift")
        env.apex = env.plan.phases.index("reach")
        env.bottom = 2 * env.apex - env.release
        env.anchors = env.plan.anchors[None].clone()
        env.landed = False
        env.tick = env.confirm = env.stable = 0
        action = torch.zeros((1, cfg.action_space), device=env.device)
        while env.tick < env.bottom + 250:
            print(f"[landing] stepping from tick={env.tick}", flush=True)
            _, _, term, trunc, _ = env.step(action)
            if (term | trunc).any():
                raise RuntimeError(f"Episode ended at tick {env.tick}")
            if env.stable >= 50:
                print("PASS: single Go2 PhysX FL landing and four-foot recovery")
                return
        raise RuntimeError("Four-foot recovery did not remain loaded for 50 consecutive MPC ticks")
    except BaseException:
        print(f"[landing] FAILED at tick={getattr(env, 'tick', 'setup')}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    completed = False
    try:
        run(args)
        completed = True
    finally:
        try:
            launcher.app.close()
        finally:
            if not completed:
                raise SystemExit(1)
