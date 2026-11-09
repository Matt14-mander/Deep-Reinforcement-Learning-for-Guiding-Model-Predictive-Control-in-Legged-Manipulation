"""Three continuous Go2 crawl cycles. Change the endpoint during the first FL swing."""
import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


def run(args):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import QuadrupedMPCEnvCfg
    from RL_Bezier_MPC.envs.replanning_crawl import ReplanningCrawlEnv
    from RL_Bezier_MPC.envs.fixed_crawl_env import legs

    class AuditEnv(ReplanningCrawlEnv):
        def _solve_mpc_control_step(self):
            phase = self.plan.phase_names[self.index]
            boundary = self.previous == "center" and phase == "shift_RL"
            if self.request_tick is None and phase == "swing_FL":
                self.request_tick = self.tick
                self.request_endpoint(self.target_mm / 1000)
                print(json.dumps(dict(stage="request", tick=self.tick, phase=phase,
                                      old_goal_mm=60, new_goal_mm=self.target_mm)), flush=True)
            if args.cycle_policy:
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

    cfg = QuadrupedMPCEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = 37
    cfg.sim.device = args.device
    cfg.mpc_backend = "torch_tracking"
    cfg.use_mpc_cluster = False
    cfg.torch_foot_plane_height = .022
    cfg.episode_length_s = 60.
    env = AuditEnv(cfg=cfg)
    def trial(target_mm):
        actions = []

        def policy(obs):
            # Leave a small residual to the final cycle. This distinguishes a
            # consumed policy action from the old equal-division replanner.
            mean = obs[:, :1] / obs[:, 1:]
            stride = torch.where(obs[:, 1:] > 1, .020 + .95*(mean-.020), mean)
            actions.append((obs.clone(), stride.item()))
            return stride

        env.walking = False
        env.reset(seed=cfg.seed)
        env.target_mm = target_mm
        env.begin_forward_crawl(.060, 3, stride_policy=policy if args.cycle_policy else None)
        env.request_tick = None
        env.distances = []
        env.splice_jump_mm = 0.
        env.finished_peaks = env.anchors.new_zeros(4)
        env.current_peak = env.finished_peaks.clone()
        env.stance_mask = torch.zeros(4, dtype=torch.bool, device=env.device)
        action = torch.zeros((1, cfg.action_space), device=env.device)
        while not env.stopped:
            _, _, term, trunc, _ = env.step(action)
            if (term | trunc).any():
                raise RuntimeError("Episode ended before final stop")
        delta = env.robot.data.root_link_pos_w[0].double() - env.tracking_origins[0] - env.base_start
        cumulative = 1000*(env.finished_peaks+env.current_peak)
        dx = 1000*delta[0].item()
        xs, speeds, spins, joints = zip(*env.stop_window)
        summary = dict(target_mm=target_mm, stride_commands_mm=[1000*a for _, a in actions],
                       cycle_dx_mm=[1000*d for d in env.distances], dx_mm=dx,
                       dy_mm=1000*delta[1].item(), final_error_mm=abs(dx-target_mm),
                       landings=env.landings, foot_order=legs, anchor_jump_mm=env.splice_jump_mm,
                       max_stance_mm=(1000*env.max_drift).tolist(), cumulative_stance_mm=cumulative.tolist(),
                       stable_ticks=len(xs), stop_span_mm=1000*(max(xs)-min(xs)),
                       max_speed_mps=max(speeds), max_spin_rad_s=max(spins), max_joint_rad_s=max(joints))
        print(json.dumps(summary, allow_nan=False), flush=True)
        if args.cycle_policy:
            if len(actions) != 3 or len(env.replans) != 3 or len(env.distances) != 3:
                raise RuntimeError("Expected exactly one stride decision for each of three cycles")
            for i, ((obs, stride), event, measured) in enumerate(zip(actions, env.replans, env.distances)):
                goal = .060 if i == 0 else target_mm / 1000
                if not (event["completed_cycles"] == i and obs[0, 1].item() == 3-i
                        and abs(obs[0, 0].item() - (goal-event["dx_m"])) < 1e-6
                        and abs(event["stride_m"]-stride) < 1e-8 and abs(measured-stride) <= .002):
                    raise RuntimeError(f"Invalid observation, applied stride or measured response: {event}")
            direction = 1 if target_mm > 60 else -1
            if not (env.replans[0]["tick"] < env.request_tick < env.replans[1]["tick"]
                    and all(direction*(a-actions[0][1]) > .002 for _, a in actions[1:])
                    and all(direction*(d-env.distances[0]) > .002 for d in env.distances[1:])):
                raise RuntimeError("Expected signed stride adjustment on the next cycle")
        else:
            if len(env.replans) != 1 or len(env.distances) != 3:
                raise RuntimeError("Expected one replan and three completed cycles")
            event = env.replans[0]
            if not (event["completed_cycles"] == 1 and env.request_tick < event["tick"]
                    and event["stride_m"] > event["old_stride_m"]+.002
                    and all(abs(d-event["stride_m"]) <= .002 for d in env.distances[1:])):
                raise RuntimeError("Endpoint replan did not increase cycle distance")
        if not (env.landings == 12 and len(xs) == 50 and abs(dx-target_mm) <= 4
                and abs(delta[1].item()) <= .010 and cumulative.max() <= 40):
            raise RuntimeError("Stage 1B stride response, recovery, drift or endpoint failed")
        return torch.tensor(env.initial, dtype=torch.float64)

    try:
        initial = trial(68)
        if args.cycle_policy:
            if not torch.allclose(initial, trial(52), rtol=0, atol=1e-6):
                raise RuntimeError("Initial states of lengthen and shorten trials differ")
            print("PASS: Stage 1B cycle policy lengthens and shortens stride", flush=True)
        else:
            print("PASS: Stage 1B three-cycle crawl with live endpoint replan", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-policy", action="store_true", help="Run 60->68 and 60->52 mm policy trials")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    success = False
    try:
        run(args)
        success = True
    except Exception:
        traceback.print_exc()
        raise
    finally:
        try:
            launcher.app.close()
        finally:
            if not success:
                raise SystemExit(1)
