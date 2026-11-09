"""Single Go2 three-cycle episodes with measured touchdown and stance drift."""
import json

import torch

from .fixed_crawl_env import legs
from .replanning_crawl import ReplanningCrawlEnv


class Stage1BCycleEnv(ReplanningCrawlEnv):
    def _solve_mpc_control_step(self):
        phase = self.plan.phase_names[self.index]
        boundary = self.previous == "center" and phase == "shift_RL"
        if self.target_mm != 60 and self.request_tick is None and phase == "swing_FL":
            self.request_tick = self.tick
            self.request_endpoint(self.target_mm / 1000)
            print(json.dumps(dict(stage="request", tick=self.tick, phase=phase,
                                  old_goal_mm=60, new_goal_mm=self.target_mm)), flush=True)
        if self.stride_policy is not None:
            should_apply = phase == "shift_RL" and self.previous in ("start", "center")
        else:
            should_apply = boundary and self.pending_goal is not None
        old = self.anchors.clone()
        count = len(self.replans)
        confirmed = getattr(self, "confirmed", False)
        super()._solve_mpc_control_step()
        if (len(self.replans) != count) != should_apply:
            raise RuntimeError("Replan did not occur on the next cycle boundary")
        jumps = (self.anchors-old).norm(dim=-1)[0]
        if phase.startswith("land_") and not confirmed and self.confirmed:
            jumps[self.leg] = 0  # Only the foot just confirmed may acquire a new anchor.
        if jumps.max() > 1e-9:
            raise RuntimeError(f"Support anchor changed at {phase}: {jumps.tolist()}")
        if should_apply:
            self.splice_jump_mm = max(self.splice_jump_mm, jumps.max().item()*1000)
        if self.segment is None:
            return
        # Accumulate the peak 3-D displacement in each uninterrupted stance
        # interval, ending an interval at lift-off; never clear at a replan.
        mask = torch.ones(4, dtype=torch.bool, device=self.device)
        if phase.startswith("swing_") or (phase.startswith("land_") and not self.confirmed):
            mask[self.leg] = False
        ended = self.stance_mask & ~mask
        self.finished_peaks += torch.where(ended, self.current_peak, 0.)
        feet = self.robot.data.body_link_pos_w[0, self._foot_body_indices.tolist()].double()
        displacement = (feet-self.tracking_origins[0]-self.segment).norm(dim=-1)
        self.current_peak = torch.where(mask, torch.maximum(self.current_peak, displacement), 0.)
        self.stance_mask = mask

    def check_progress(self, state):
        self.distances.append((state[0, 0]-self.last_center[0]).item())
        super().check_progress(state)

    def run_episode(self, target_mm, seed, policy, *, require_goal_for_stop=True):
        self.walking = False
        self.reset(seed=seed)
        self.target_mm = target_mm
        self.begin_forward_crawl(.060, 3, stride_policy=policy, require_goal_for_stop=require_goal_for_stop)
        self.request_tick = None
        self.distances = []
        self.splice_jump_mm = 0.
        self.finished_peaks = self.anchors.new_zeros(4)
        self.current_peak = self.finished_peaks.clone()
        self.stance_mask = torch.zeros(4, dtype=torch.bool, device=self.device)
        action = torch.zeros((1, self.cfg.action_space), device=self.device)
        while not self.stopped:
            _, _, term, trunc, _ = self.step(action)
            if (term | trunc).any():
                raise RuntimeError("Episode ended before final stop")
        delta = self.robot.data.root_link_pos_w[0].double() - self.tracking_origins[0] - self.base_start
        cumulative = 1000*(self.finished_peaks+self.current_peak)
        dx = 1000*delta[0].item()
        xs, speeds, spins, joints = zip(*self.stop_window)
        summary = dict(target_mm=target_mm, stride_commands_mm=[1000*r['stride_m'] for r in self.replans],
                       cycle_dx_mm=[1000*d for d in self.distances], dx_mm=dx,
                       dy_mm=1000*delta[1].item(), final_error_mm=abs(dx-target_mm),
                       landings=self.landings, foot_order=legs, anchor_jump_mm=self.splice_jump_mm,
                       max_stance_mm=(1000*self.max_drift).tolist(), cumulative_stance_mm=cumulative.tolist(),
                       stable_ticks=len(xs), stop_span_mm=1000*(max(xs)-min(xs)),
                       max_speed_mps=max(speeds), max_spin_rad_s=max(spins), max_joint_rad_s=max(joints))
        summary.update(stopped=self.stopped, arrived=abs(dx-target_mm) <= 4 and abs(delta[1].item()) <= .010,
                       recovery_ticks_per_landing=20, cycle_decisions=list(self.replans))
        if (self.landings != 12 or len(xs) != 50 or cumulative.max() > 40
                or len(self.distances) != 3):
            raise RuntimeError(f"Physical crawl gate failed: {summary}")
        return summary
