"""Two cycles of a fixed Go2 crawl in one PhysX episode."""
import argparse
import json
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def run(args):
    import torch
    from crocoddyl_batched_mpc.models import go2_slow_walk_plan
    from crocoddyl_batched_mpc.models.go2_walk import go2_leg_ik, go2_center_of_mass, smoothstep

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import QuadrupedMPCEnv, QuadrupedMPCEnvCfg

    legs = ("FL", "FR", "RL", "RR")

    class CrawlEnv(QuadrupedMPCEnv):
        def _pre_physics_step(self, actions):
            super()._pre_physics_step(actions)
            if args.endpoint_result is not None and not self.endpoint_ready:
                controls = (self._denormalize_actions(actions) -
                            self._denormalize_actions(torch.zeros_like(actions))).reshape(4, 3)
                self.goal = controls[3, 0].item()
                if abs(self.goal - args.target_mm * .001) > 1e-6:
                    raise ValueError("Endpoint action clipped or decoded incorrectly")
                self.plan = go2_slow_walk_plan(cycles=2, step_length=self.goal/2, swing_height=.02,
                                              support_shift_fraction=args.support_shift_fraction,
                                              dt=self.cfg.mpc_dt, device=self.device)
                self.ref = self.plan.reference.clone()
                self.acc = self.plan.acceleration.clone()
                self.anchors = self.plan.anchors[:1].clone()
                self.endpoint_ready = True
            if args.probe_result is None or self.probe_ready:
                return
            if actions.shape != (1, 12):
                raise ValueError("Expected one 12-D Stage 1 action")
            controls = (self._denormalize_actions(actions) -
                        self._denormalize_actions(torch.zeros_like(actions))).reshape(4, 3).double()
            p = self.plan
            idx = [i for i, name in enumerate(p.phase_names) if name == "shift_RL"]
            first = idx[0]
            end = p.phase_names.index("swing_RL", first)
            self.window = (first, end)
            model = self.tracking.controller.model
            if controls.any():
                # Zero endpoint slope/acceleration at the join to the unmodified crawl.
                t = smoothstep(torch.linspace(0, 1, end-first, device=self.device, dtype=controls.dtype))
                basis = torch.stack(((1-t)**3, 3*t*(1-t)**2, 3*t*t*(1-t), t**3), -1)
                nominal = p.reference[first:end, :19]
                target = go2_center_of_mass(model, nominal) + basis @ controls
                q = nominal.clone()
                for _ in range(5):
                    q[:, :3] += target - go2_center_of_mass(model, q)
                    q = go2_leg_ik(model, q, p.foot_reference[first:end])
                if (go2_center_of_mass(model, q)-target).abs().max() > 1e-5:
                    raise RuntimeError("CoM reference IK residual exceeds 10 micrometers")
                dq = q - nominal
                delta = torch.zeros_like(p.reference[:, 19:])
                delta[first:end, :3] = dq[:, :3]
                delta[first:end, 6:] = dq[:, 7:19]
                dv = torch.zeros_like(delta)
                dv[1:-1] = (delta[2:]-delta[:-2])/(2*p.dt)
                self.ref[first:end, :19] = q
                self.ref[:, 19:] += dv
                self.acc[1:-1] += (dv[2:]-dv[:-2])/(2*p.dt)
            self.controls = controls.tolist()
            self.probe_ready = True

        def align_foot_reference(self, leg, xy):
            # Retain the measured landing XY through stance. Blend the correction
            # out over this leg's next swing so the planned destination stays fixed.
            p, i = self.plan, self.index
            next_swing = f"swing_{legs[leg]}"
            start = next((j for j in range(i+1, len(p.reference))
                          if p.phase_names[j] == next_swing), len(p.reference))
            end = start
            while end < len(p.reference) and p.phase_names[end] == next_swing:
                end += 1
            weight = xy.new_ones(end-i)
            if end > start:
                t = torch.arange(end-start, device=self.device, dtype=xy.dtype) / (end-start)
                weight[start-i:] = 1 - t**3 * (10 + t*(-15 + 6*t))
            target = p.foot_reference[i:end].clone()
            error = xy - target[0, leg, :2]
            target[:, leg, :2] += weight[:, None] * error
            q = go2_leg_ik(self.tracking.controller.model, p.reference[i:end, :19], target)
            joint = slice(7+3*leg, 10+3*leg)
            vel = slice(25+3*leg, 28+3*leg)
            acc = slice(6+3*leg, 9+3*leg)
            self.ref[i:end, joint] = q[:, joint]
            # The first landing reference is held during confirmation/recovery.
            j = torch.arange(i, end, device=self.device)
            prev = (j-1).clamp_min(i)
            nxt = (j+1).clamp_max(len(p.reference)-1)
            self.ref[j, vel] = (self.ref[nxt, joint] - self.ref[prev, joint]) / (2*p.dt)
            self.acc[j, acc] = (self.ref[nxt, vel] - self.ref[prev, vel]) / (2*p.dt)
            print(json.dumps(dict(tick=self.tick, aligned_leg=legs[leg],
                                  anchor_offset_xy_mm=(1000*error).tolist())), flush=True)

        def _reset_idx(self, env_ids):
            if getattr(self, "walking", False):
                raise RuntimeError(f"Episode reset during crawl at tick {self.tick}")
            return super()._reset_idx(env_ids)

        def _solve_mpc_control_step(self):
            p, i = self.plan, self.index
            phase = p.phase_names[i]
            a, d = self.tracking, self.robot.data
            m = a.controller.mpc
            state = a.pack_state(d.root_link_pose_w.double(), d.root_link_vel_w.double(),
                                 d.joint_pos.double(), d.joint_vel.double(), self.tracking_origins)
            feet = d.body_link_pos_w[0, self._foot_body_indices.tolist()].double() - self.tracking_origins[0]
            loads = self.foot_contact_sensor.data.net_forces_w[0, self._contact_sensor_foot_indices, 2]
            if not (state.isfinite().all() and loads.isfinite().all() and feet.isfinite().all()):
                raise RuntimeError(f"Nonfinite measurement at {phase}")
            loaded = loads > self.cfg.foot_contact_force_threshold
            if args.endpoint_result is not None:
                if self.tick == 0:
                    self.initial = state[0].tolist()
                if i == len(p.reference)-1:
                    offset = state[0, :3] - self.base_start
                    speed = state[0, 19:22].norm().item()
                    spin = state[0, 22:25].norm().item()
                    joint_speed = state[0, 25:].abs().max().item()
                    ok = (loaded.all().item() and abs(offset[0].item()-self.goal) <= .004
                          and abs(offset[1].item()) <= .010 and speed <= .010
                          and spin <= .04 and joint_speed <= .10)
                    self.stop_window = self.stop_window + [(offset[0].item(), speed, spin, joint_speed)] if ok else []
                    self.stop_window = self.stop_window[-50:]
                    self.hold_ticks += 1
                    if len(self.stop_window) == 50:
                        xs = [v[0] for v in self.stop_window]
                        self.stopped = max(xs)-min(xs) <= .001
                    else:
                        self.stopped = False
                    if self.hold_ticks > 250 and not self.stopped:
                        raise RuntimeError(f"Stop timeout: goal={self.goal}, offset={offset.tolist()}, "
                                           f"speed={speed}, spin={spin}, joint_speed={joint_speed}, loads={loads.tolist()}")
            mask = p.contact_masks[i].clone()
            if phase != self.previous:
                if self.previous == "center":
                    self.check_progress(state)
                if phase.startswith("swing_"):
                    self.leg = legs.index(phase[6:])
                    self.lift_origin = feet[self.leg].clone()
                    self.airborne = False
                    self.touch = self.settled = self.wait = 0
                    self.confirmed = False
                    m.reset()
                print(json.dumps(dict(tick=self.tick, phase=phase)), flush=True)
                self.previous = phase
            if phase == "start" and self.tick == 20:
                if not loaded.all():
                    raise RuntimeError("Expected settled four-foot stance")
                self.segment = feet.clone()
                self.base_start = state[0, :3].clone()
                self.last_center = self.base_start.clone()
            if phase.startswith("swing_"):
                self.airborne |= bool((feet[self.leg, 2] - self.lift_origin[2] >= .010)
                                      & (~loaded[self.leg]) & loaded[mask].all())
            freeze = phase == "start" and self.tick < 20
            preload = 0.
            if phase.startswith("land_"):
                leg = self.leg
                self.wait += 1
                if self.wait > 180:
                    raise RuntimeError(f"Landing timeout {phase}, loads={loads.tolist()}")
                if not self.airborne:
                    raise RuntimeError(f"No measured tripod clearance for {phase}")
                if not self.confirmed:
                    near = abs((feet[leg, 2] - self.lift_origin[2]).item()) <= .005
                    self.touch = self.touch + 1 if near and loaded.all() else 0
                    if self.touch >= 3:
                        dx = (feet[leg, 0] - self.lift_origin[0]).item()
                        if not .010 <= dx <= .030:
                            raise RuntimeError(f"Foot placement {phase}: dx={dx:.5f} m")
                        self.anchors[0, leg, :2] = feet[leg, :2]
                        self.align_foot_reference(leg, feet[leg, :2])
                        self.segment[leg] = feet[leg]
                        self.confirmed = True
                        m.reset()
                    else:
                        mask[leg] = False
                        if near and loaded[mask].all():
                            preload = min(5., self.wait * p.dt * 10.)
                if self.confirmed:
                    self.settled = self.settled + 1 if loaded.all() else 0
                freeze = self.settled < 20
                if self.settled == 20 and not self.reported:
                    self.landings += 1
                    self.reported = True
                    print(json.dumps(dict(landing=self.landings, leg=legs[leg], recovery_ticks=20,
                                          dx_m=(feet[leg, 0]-self.lift_origin[0]).item(),
                                          loads_N=loads.tolist())), flush=True)
                # Once recovered, run the nominal four-support phase without re-counting it.
                if self.reported:
                    freeze = False
            else:
                self.reported = False
            if self.segment is not None:
                drift = (feet - self.segment).norm(dim=-1)
                self.max_drift = torch.maximum(self.max_drift, torch.where(mask, drift, 0.))
                if (self.max_drift > .010).any():
                    print(json.dumps(dict(tick=self.tick, phase=phase, loads_N=loads.tolist(),
                                          displacement_xyz_mm=((feet-self.segment)*1000).tolist())), flush=True)
                    raise RuntimeError(f"Stance drift at {phase}: {self.max_drift.tolist()} m")
            ids = torch.arange(m.horizon + 1, device=self.device)
            ids = (ids * int(not freeze) + i).clamp_max(len(p.reference)-1)
            ref = self.ref[ids].clone()
            if freeze:
                ref[:, 19:] = 0.
            effort, result = m.compute(state, ref[None], self.acc[i:i+1] * int(not freeze),
                                       mask[None], self.anchors)
            if not result.usable.all():
                raise RuntimeError(f"MPC rejected tick={self.tick} {phase}, status={result.status.tolist()}")
            if args.probe_result is not None:
                if self.tick == 0:
                    self.initial = state[0].tolist()
                if self.window[0] <= i < self.window[1]:
                    masses = d.default_mass[0].to(state)
                    com = (d.body_com_pos_w[0].double() * masses[:, None]).sum(0) / masses.sum()
                    self.samples.append([i, go2_center_of_mass(m.model, ref[:1])[0, 0].item(),
                                         (com[0]-self.tracking_origins[0, 0]).item(), effort[0].tolist()])
            if preload:
                leg = self.leg
                jac = m.model._kinematics(state[:, :19], state[:, 19:])[4].reshape(1, 4, 3, 18)
                effort = effort.clone()
                effort[:, 3*leg:3*leg+3] -= preload * jac[:, leg, 2, 6+3*leg:9+3*leg]
                effort = effort.clamp(-m.constraints.torque_limits, m.constraints.torque_limits)
            self._pending_joint_efforts.copy_(a.scatter_action(effort).clamp(
                -self.cfg.max_joint_torque, self.cfg.max_joint_torque))
            self._last_mpc_costs = result.cost
            self._last_mpc_converged = result.success
            self._advance_reference_clocks()
            if not freeze:
                self.index = min(i+1, len(p.reference)-1)
            self.tick += 1

        def check_progress(self, state):
            delta = state[0, :3] - self.last_center
            if not .010 <= delta[0].item() <= .030:
                raise RuntimeError(f"Cycle forward distance {delta[0].item():.5f} m")
            self.last_center = state[0, :3].clone()
            print(json.dumps(dict(cycle=self.landings//4, cycle_dx_m=delta[0].item())), flush=True)

    cfg = QuadrupedMPCEnvCfg()
    cfg.scene.num_envs = 1
    if args.probe_result is not None or args.endpoint_result is not None:
        cfg.seed = 37
    cfg.episode_length_s = 90.
    cfg.sim.device = args.device
    cfg.mpc_backend = "torch_tracking"
    cfg.torch_foot_plane_height = .022
    cfg.use_mpc_cluster = False
    env = None
    try:
        env = CrawlEnv(cfg=cfg)
        env.reset()
        if env._foot_body_indices is None:
            raise RuntimeError("Foot mapping unavailable")
        env.plan = go2_slow_walk_plan(cycles=2, step_length=.02, swing_height=.02,
                                      support_shift_fraction=args.support_shift_fraction,
                                      dt=cfg.mpc_dt, device=env.device)
        print(f"support_shift_fraction={args.support_shift_fraction}", flush=True)
        env.anchors = env.plan.anchors[0:1].clone()
        env.ref = env.plan.reference.clone()
        env.acc = env.plan.acceleration.clone()
        env.max_drift = env.anchors.new_zeros(4)
        env.index = env.tick = env.landings = 0
        env.segment = env.previous = None
        env.walking = True
        action = torch.zeros((1,cfg.action_space), device=env.device)
        if args.endpoint_result is not None:
            if cfg.action_space != 12 or args.probe_result is not None:
                raise ValueError("Endpoint trial requires the fixed-gait 12-D action alone")
            env.endpoint_ready = env.stopped = False
            env.hold_ticks = 0
            env.stop_window = []
            action[:, 9] = 2 * args.target_mm * .001 / (env.action_high-env.action_low).reshape(-1)[9]
        if args.probe_result is not None:
            env.probe_ready = False
            env.samples = []
            # Express the desired physical perturbation through the real normalized action bounds.
            action[:, [3, 6]] = 2 * args.probe_mm * .001 / (env.action_high-env.action_low).reshape(-1)[[3, 6]]
        while env.index < len(env.plan.reference)-1 or (args.endpoint_result is not None and not env.stopped):
            _, _, term, trunc, _ = env.step(action)
            if (term | trunc).any():
                raise RuntimeError(f"Episode ended at tick {env.tick}")
        delta = env.robot.data.root_link_pos_w[0].double() - env.tracking_origins[0] - env.base_start
        if env.landings != 8 or not .030 <= delta[0].item() <= .050 or abs(delta[1].item()) > .010:
            raise RuntimeError(f"Final displacement {delta.tolist()} m; recovered landings={env.landings}")
        print(json.dumps(dict(dx_m=delta[0].item(), dy_m=delta[1].item(), landings=env.landings,
                              max_stance_drift_mm=(1000*env.max_drift).tolist())), flush=True)
        print("PASS: single Go2 PhysX two-cycle fixed crawl", flush=True)
        if args.endpoint_result is not None:
            xs, speeds, spins, joints = zip(*env.stop_window)
            summary = dict(initial=env.initial, target_mm=args.target_mm, decoded_m=env.goal,
                           dx_m=delta[0].item(), dy_m=delta[1].item(), landings=env.landings,
                           drift_mm=(1000*env.max_drift).tolist(), stable_ticks=len(xs),
                           stop_span_mm=1000*(max(xs)-min(xs)), max_speed_mps=max(speeds),
                           max_spin_rad_s=max(spins), max_joint_rad_s=max(joints))
            if abs(summary["dx_m"]-env.goal) > .004:
                raise RuntimeError(f"Endpoint missed after final physics step: {summary}")
            args.endpoint_result.write_text(json.dumps(summary, allow_nan=False), encoding="utf8")
        if args.probe_result is not None:
            args.probe_result.write_text(json.dumps(dict(initial=env.initial, samples=env.samples,
                controls=env.controls, landings=env.landings, dx_m=delta[0].item(),
                drift_mm=(1000*env.max_drift).tolist()), allow_nan=False), encoding="utf-8")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-result", type=Path)
    parser.add_argument("--target-mm", type=float, choices=(32., 40., 48.), default=40.)
    parser.add_argument("--probe-result", type=Path)
    parser.add_argument("--probe-mm", type=float, default=0.0)
    parser.add_argument("--support-shift-fraction", type=float, default=1.0)
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
