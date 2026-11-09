"""Single-Go2 fixed crawl shared by Stage 1 and PhysX regression tests."""
import json

import torch
from crocoddyl_batched_mpc.models import go2_slow_walk_plan
from crocoddyl_batched_mpc.models.go2_walk import go2_leg_ik, go2_center_of_mass, smoothstep
from .quadruped_mpc_env import QuadrupedMPCEnv

legs = ("FL", "FR", "RL", "RR")


class CrawlFailure(RuntimeError):
    """Measured safety or gait acceptance failure during a rollout."""


class FixedCrawlEnv(QuadrupedMPCEnv):

    def begin_crawl(self, shift_fraction, *, endpoint=False, shape=False, record_probe=False,
                    cycles=2, step_length=.02, require_goal_for_stop=True):
        if self.num_envs != 1 or self._foot_body_indices is None:
            raise ValueError("Fixed crawl requires a single Go2 with mapped feet")
        self.shift_fraction = shift_fraction
        self.check_endpoint = endpoint
        self.shape_enabled = shape
        self.record_probe = record_probe
        self.plan = go2_slow_walk_plan(cycles=cycles, step_length=step_length, swing_height=.02,
                                      support_shift_fraction=shift_fraction,
                                      dt=self.cfg.mpc_dt, device=self.device)
        self.anchors = self.plan.anchors[:1].clone()
        self.ref = self.plan.reference.clone()
        self.acc = self.plan.acceleration.clone()
        self.max_drift = self.anchors.new_zeros(4)
        self.index = self.tick = self.landings = self.hold_ticks = 0
        self.segment = self.previous = None
        self.endpoint_ready = self.stopped = self.probe_ready = False
        self.require_goal_for_stop = require_goal_for_stop
        self.stop_window = []
        self.samples = []
        self.walking = True

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        if self.check_endpoint and not self.endpoint_ready:
            controls = (self._denormalize_actions(actions) -
                        self._denormalize_actions(torch.zeros_like(actions))).reshape(4, 3)
            self.goal = controls[3, 0].item()
            self.plan = go2_slow_walk_plan(cycles=2, step_length=self.goal/2, swing_height=.02,
                                          support_shift_fraction=self.shift_fraction,
                                          dt=self.cfg.mpc_dt, device=self.device)
            self.ref = self.plan.reference.clone()
            self.acc = self.plan.acceleration.clone()
            self.anchors = self.plan.anchors[:1].clone()
            self.endpoint_ready = True
        if not self.shape_enabled or self.probe_ready:
            return
        if actions.shape != (1, 12):
            raise ValueError("Expected one 12-D Stage 1 action")
        controls = (self._denormalize_actions(actions) -
                    self._denormalize_actions(torch.zeros_like(actions))).reshape(4, 3).double()
        if self.check_endpoint:
            controls[3] = 0  # Endpoint already sets the stride length.
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
            raise CrawlFailure(f"Episode reset during crawl at tick {self.tick}")
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
        if self.check_endpoint:
            if self.tick == 0:
                self.initial = state[0].tolist()
            if i == len(p.reference)-1:
                offset = state[0, :3] - self.base_start
                speed = state[0, 19:22].norm().item()
                spin = state[0, 22:25].norm().item()
                joint_speed = state[0, 25:].abs().max().item()
                in_target = abs(offset[0].item()-self.goal) <= .004 and abs(offset[1].item()) <= .010
                ok = (loaded.all().item() and (in_target or not self.require_goal_for_stop) and speed <= .010
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
                    raise CrawlFailure(f"Stop timeout: goal={self.goal}, offset={offset.tolist()}, "
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
                raise CrawlFailure("Expected settled four-foot stance")
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
                raise CrawlFailure(f"Landing timeout {phase}, loads={loads.tolist()}")
            if not self.airborne:
                raise CrawlFailure(f"No measured tripod clearance for {phase}")
            if not self.confirmed:
                near = abs((feet[leg, 2] - self.lift_origin[2]).item()) <= .005
                self.touch = self.touch + 1 if near and loaded.all() else 0
                if self.touch >= 3:
                    dx = (feet[leg, 0] - self.lift_origin[0]).item()
                    if not .010 <= dx <= .030:
                        raise CrawlFailure(f"Foot placement {phase}: dx={dx:.5f} m")
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
                raise CrawlFailure(f"Stance drift at {phase}: {self.max_drift.tolist()} m")
        ids = torch.arange(m.horizon + 1, device=self.device)
        ids = (ids * int(not freeze) + i).clamp_max(len(p.reference)-1)
        ref = self.ref[ids].clone()
        if freeze:
            ref[:, 19:] = 0.
        effort, result = m.compute(state, ref[None], self.acc[i:i+1] * int(not freeze),
                                   mask[None], self.anchors)
        if not result.usable.all():
            raise CrawlFailure(f"MPC rejected tick={self.tick} {phase}, status={result.status.tolist()}")
        if self.record_probe:
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
            raise CrawlFailure(f"Cycle forward distance {delta[0].item():.5f} m")
        self.last_center = state[0, :3].clone()
        print(json.dumps(dict(cycle=self.landings//4, cycle_dx_m=delta[0].item())), flush=True)
