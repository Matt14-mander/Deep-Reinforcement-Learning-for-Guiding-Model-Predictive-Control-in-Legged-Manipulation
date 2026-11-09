"""Queue a forward endpoint and splice the remaining crawl at a four-foot cycle boundary."""
import json
import math

import torch

from crocoddyl_batched_mpc.models import go2_slow_walk_plan
from .fixed_crawl_env import FixedCrawlEnv


class ReplanningCrawlEnv(FixedCrawlEnv):
    def begin_forward_crawl(self, goal, cycles, *, stride_policy=None):
        """Optionally call policy([remaining meters, cycles left]) once per cycle.

        Input shape (1, 2), float32 on the simulation device. Output (1, 1)
        is the next forward stride in meters. The initial stance settles before
        observing progress. A queued goal becomes visible at the next boundary.
        """
        self.stride_policy = stride_policy
        self.begin_crawl(.75, endpoint=True, cycles=cycles, step_length=goal/cycles)
        self.endpoint_ready = True
        self.goal = goal
        self.total_cycles = cycles
        self.pending_goal = None
        self.replans = []

    def request_endpoint(self, goal):
        if not math.isfinite(goal) or goal <= 0:
            raise ValueError("Expected a finite positive forward endpoint in meters")
        self.pending_goal = goal

    def _solve_mpc_control_step(self):
        at_entry = (self.previous in ("start", "center")
                    and self.plan.phase_names[self.index] == "shift_RL")
        if at_entry and (self.stride_policy is not None or self.pending_goal is not None):
            self._replan_remaining()
        super()._solve_mpc_control_step()

    def _replan_remaining(self):
        remaining = self.total_cycles - self.landings // 4
        pose = self.robot.data.root_link_pos_w[0].double() - self.tracking_origins[0]
        dx = (pose[0] - self.base_start[0]).item()
        goal = self.goal if self.pending_goal is None else self.pending_goal
        if self.stride_policy is None:
            stride = (goal - dx) / remaining
        else:
            obs = torch.tensor([[goal - dx, remaining]], device=self.device, dtype=torch.float32)
            with torch.no_grad():
                action = self.stride_policy(obs)
            if not isinstance(action, torch.Tensor) or action.shape != (1, 1) or not action.is_floating_point():
                raise ValueError("Stride policy must return a (1, 1) tensor in meters")
            if not ((action >= .010) & (action <= .030)).item():
                raise ValueError(f"Stride {action.item()} outside the 10-30 mm crawl envelope")
            stride = action.item()
        first = self.landings == 0
        # Use a post-center cycle entry: the first cycle starts from a different body pose.
        plan = go2_slow_walk_plan(cycles=remaining+int(not first), step_length=stride, swing_height=.02,
                                  support_shift_fraction=self.shift_fraction,
                                  dt=self.cfg.mpc_dt, device=self.device)
        entry = plan.phase_names.index("shift_RL", 0 if first else plan.phase_names.index("center"))
        offset = self.plan.foot_reference[self.index].mean(0) - plan.foot_reference[entry].mean(0)
        plan.reference[:, :3] += offset
        plan.foot_reference += offset
        plan.anchors += offset
        old_stride = self.plan.step_length
        self.plan, self.index = plan, entry
        self.ref, self.acc = plan.reference.clone(), plan.acceleration.clone()
        # Preserve the actual MPC contact anchors and reapply each measured touchdown
        # correction until its next swing. No clock, foot segment or solver reset here.
        if not first:
            for leg in range(4):
                self.align_foot_reference(leg, self.anchors[0, leg, :2])
        self.goal = goal
        self.pending_goal = None
        event = dict(tick=self.tick, completed_cycles=self.landings//4, dx_m=dx,
                     remaining_m=goal-dx, cycles_left=remaining, goal_m=self.goal, old_stride_m=old_stride, stride_m=stride)
        self.replans.append(event)
        print(json.dumps(dict(stage="replan", **event)), flush=True)
