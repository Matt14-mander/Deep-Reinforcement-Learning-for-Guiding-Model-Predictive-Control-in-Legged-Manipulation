"""PhysX on-policy PPO for one gait choice and one stride at each four-foot boundary."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher
from check_go2_slow_trot import (SETTLE, HOLD, evaluate_mixed, gait_env_type, make_plan)

GOALS = (20, 42, 64)


def make_policy(speed_task=False, period_actions=False):
    import torch
    from torch import nn
    from torch.distributions import Bernoulli, Normal

    inputs_for_legacy = 3 if speed_task else 2

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            inputs = (3 if speed_task else 2) + int(period_actions)
            self.actor = nn.Linear(inputs, 2)
            self.value = nn.Sequential(nn.Linear(inputs,16), nn.Tanh(), nn.Linear(16,1))
            if period_actions:
                self.period_actor = nn.Linear(inputs, 3)
                nn.init.zeros_(self.period_actor.weight)
                with torch.no_grad():
                    self.period_actor.bias.copy_(torch.tensor([0.,1.,0.]))
            self.log_std = nn.Parameter(torch.tensor(-.65))
            # Distance-based initialization. PPO updates both outputs.
            with torch.no_grad():
                weights = [[2.,0.,7.5],[.15,0.,0.]] if speed_task else [[2.,0.],[.15,0.]]
                if period_actions:
                    weights = [row+[0.] for row in weights]
                self.actor.weight.copy_(torch.tensor(weights))
                self.actor.bias.copy_(torch.tensor([0.,0.]))

        def load_compatible_state(self, state):
            state = dict(state)
            if period_actions and "period_actor.weight" not in state:
                # Extend a legacy policy with a neutral previous-period feature.
                for key in ("actor.weight", "value.0.weight"):
                    weight = state[key]
                    if weight.shape[1] != inputs_for_legacy:
                        raise ValueError("Checkpoint observation layout does not match this task")
                    state[key] = torch.cat((weight,weight.new_zeros(weight.shape[0],1)),dim=1)
                state["period_actor.weight"] = self.period_actor.weight.detach().clone()
                state["period_actor.bias"] = self.period_actor.bias.detach().clone()
            self.load_state_dict(state)

        def period_distribution(self, obs):
            return torch.distributions.Categorical(logits=self.period_actor(obs))

        def action_log_prob(self, obs, gait, stride, period=None):
            gd, zd = self.distributions(obs)
            value = gd.log_prob(gait) + zd.log_prob(stride)
            if period is not None:
                value = value + self.period_distribution(obs).log_prob(period)
            return value

        def distributions(self, obs):
            logits, mean = self.actor(obs).unbind(-1)
            return Bernoulli(logits=logits), Normal(mean,self.log_std.clamp(-3,0).exp())

    return Policy()


def run(args):
    import torch
    from torch import nn
    from torch.distributions import Bernoulli, Normal
    from PIL import Image, ImageDraw

    schedule = []
    Env = gait_env_type(schedule, True)
    from RL_Bezier_MPC.envs import QuadrupedMPCEnvCfg
    cfg = QuadrupedMPCEnvCfg()
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 1, args.seed, args.device
    cfg.mpc_backend, cfg.use_mpc_cluster, cfg.torch_foot_plane_height = "torch_tracking", False, .022
    cfg.episode_length_s = 80.
    cfg.rl_policy_period, cfg.decimation = 1, cfg.mpc_decimation
    cfg.viewer.eye, cfg.viewer.lookat, cfg.viewer.resolution = (.65,-1.,.51),(.05,0,.20),(800,480)
    # Render explicitly only while recording the final deterministic rollout.
    cfg.sim.render_interval = 1000000

    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    exploration = torch.Generator().manual_seed(args.seed+1)
    policy = make_policy()
    opt = torch.optim.Adam(policy.parameters(), lr=.003)
    first_update = 1
    if args.checkpoint or args.resume:
        checkpoint = torch.load(args.checkpoint or args.resume, map_location="cpu", weights_only=True)
        policy.load_state_dict(checkpoint["policy"])
        if args.resume:
            opt.load_state_dict(checkpoint["optimizer"])
            first_update = checkpoint["iteration"] + 1
            if first_update > args.updates:
                raise ValueError("--updates must exceed the resumed iteration")
    env = Env(cfg, render_mode="rgb_array" if args.video else None)
    dt = cfg.mpc_dt
    mpc = env.tracking.controller.mpc
    zero = torch.zeros((1,cfg.action_space),device=env.device)
    nominal = make_plan(mpc.model,dt,[("crawl",.020)],cycle_decisions=True)
    initial_states = {}
    reset_states = {}

    def episode(goal, mode, video=False):
        schedule.clear()
        env.active = False
        env.robot.set_joint_effort_target(torch.zeros_like(env.robot.data.joint_pos))
        env.reset(seed=args.seed)
        data = env.robot.data
        reset_state = torch.cat((data.root_link_pose_w[0],data.root_link_vel_w[0],
                                 data.joint_pos[0],data.joint_vel[0])).detach().cpu()
        if goal in reset_states and not torch.allclose(reset_state,reset_states[goal],atol=1e-5,rtol=0):
            raise RuntimeError("Reset root/joint state mismatch")
        reset_states.setdefault(goal,reset_state)
        mpc.reset()
        env.ref,env.acc,env.phases = nominal
        env.mask = torch.ones(1,4,dtype=torch.bool,device=env.device)
        env.anchors = mpc.model.contact_positions[None].clone()
        env.origin = env.anchors[0].clone()
        env.rows,env.drift,env.tick,env.active = [],[0.]*4,0,True
        boundary, decisions, samples, writer = SETTLE, [], [], None
        finishing = False
        if video:
            writer = subprocess.Popen(["ffmpeg","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
                "-s","800x480","-r",str(1/dt),"-i","pipe:0","-c:v","libx264","-pix_fmt","yuv420p",
                "-movflags","+faststart",str(args.output/"policy_far.mp4")],stdin=subprocess.PIPE)
        try:
            while env.tick < len(env.ref):
                current_x = env.robot.data.root_link_pos_w[0,0].item()
                if env.tick == boundary and not finishing:
                    if not decisions:
                        start_x = current_x
                        d = env.robot.data
                        settled = torch.cat((d.root_link_pose_w[0], d.root_link_vel_w[0],
                                             d.joint_pos[0], d.joint_vel[0])).detach().cpu()
                        diff = (settled-initial_states.setdefault(goal,settled)).abs()
                        settling_delta = dict(xyz_m=diff[:3].max().item(),quat=diff[3:7].max().item(),
                            velocity=diff[7:13].max().item(),q_rad=diff[13:25].max().item(),
                            dq_rad_s=diff[25:].max().item())
                        if mode=="policy" and any(v>lim for v,lim in zip(settling_delta.values(),(.001,.001,.02,.003,.03))):
                            raise RuntimeError(f"Settled initial states differ: {settling_delta}")
                    remaining = goal-1000*(current_x-start_x)
                    loads = env.foot_contact_sensor.data.net_forces_w[0,env._contact_sensor_foot_indices,2]
                    if not (loads>5).all():
                        raise RuntimeError(f"Cycle boundary without four loaded feet: {loads.tolist()}")
                    if decisions and (remaining<=4 or len(schedule)>=7):
                        finishing = True
                    else:
                        prev = bool(schedule and schedule[-1][0]=="trot")
                        obs = torch.tensor([(remaining-30)/15,float(prev)],dtype=torch.float32)
                        with torch.no_grad():
                            gait_dist,stride_dist = policy.distributions(obs)
                            if mode=="sample":
                                gait = torch.bernoulli(gait_dist.probs, generator=exploration)
                                z = stride_dist.mean + stride_dist.stddev * torch.randn((),generator=exploration)
                                samples.append((obs,gait,z,gait_dist.log_prob(gait)+stride_dist.log_prob(z)))
                            elif mode=="policy":
                                gait,z = gait_dist.logits>=0,stride_dist.mean
                            else:
                                gait,z = remaining>=30,torch.tensor(0.)
                        name = "trot" if bool(gait) else "crawl"
                        # Common physical decoder: shorten near the target, bounded residual +/- 3 mm.
                        stride = max(10.,min(24.,min(remaining,21.)+3*z.tanh().item()))
                        schedule.append((name,stride/1000))
                        decisions.append(dict(tick=env.tick,remaining_mm=remaining,gait=name,stride_mm=stride))
                        env.ref,env.acc,env.phases = make_plan(mpc.model,dt,schedule,cycle_decisions=True)
                        boundary = len(env.ref)-HOLD
                _,_,term,trunc,_ = env.step(zero)
                if (term|trunc).any():
                    raise RuntimeError("Episode ended before terminal hold")
                if writer:
                    env.sim.render()
                    rgb = env.render()
                    for _ in range(8):
                        if rgb.size and rgb[40:430:8,::8,0].std()>8:
                            break
                        env.sim.set_camera_view(cfg.viewer.eye,cfg.viewer.lookat)
                        env.sim.render()
                        rgb = env.render()
                    else:
                        raise RuntimeError("No usable camera image")
                    frame = Image.fromarray(rgb).convert("RGB")
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0,435,800,480),fill="#182532")
                    row = env.rows[-1]
                    dx = 1000*(current_x-start_x) if decisions else 0.
                    draw.text((12,444),f"Go2  |  Stage 2 PPO   {row['gait']}   {env.tick*dt:.2f} s    "
                              f"x {dx:.1f} / {goal} mm    cycle decisions {len(decisions)}")
                    draw.text((12,462),"Measured foot forces (N)   "+"   ".join(f"{v:.0f}" for v in row["loads"]))
                    writer.stdin.write(frame.tobytes())
        finally:
            if writer:
                writer.stdin.close()
                if writer.wait():
                    raise RuntimeError("Video encoding failed")
        m = evaluate_mixed(env.rows, env.drift, dt, schedule, target_mm=goal)
        stable = 0
        arrival = None
        for r in env.rows[boundary:]:
            ok = (abs(1000*(r["x"]-env.rows[SETTLE]["x"])-goal)<=4 and
                  abs(r["y"]-env.rows[SETTLE]["y"])<=.010 and min(r["loads"])>5 and
                  r["speed"]<.01 and r["spin"]<.04 and r["joint"]<.1)
            stable = stable+1 if ok else 0
            if stable==50:
                arrival = (r["tick"]-SETTLE)*dt
                break
        m.update(mode=mode,settled_initial_delta=settling_delta, decisions=decisions, arrival_s=arrival,
                 completion_s=(len(env.rows)-SETTLE)*dt)
        m["checks"]["timed_arrival"] = arrival is not None
        print(json.dumps(dict(stage="episode",mode=mode,goal=goal,arrival_s=arrival,
            error_mm=m["final_x_error_mm"],gaits=[v[0] for v in schedule],checks=m["checks"])),flush=True)
        return m,samples

    try:
        for update in range(first_update, (0 if args.checkpoint else args.updates)+1):
            exploration.manual_seed(args.seed+update)
            batch = []
            for goal in GOALS:
                m, samples = episode(goal,"sample")
                terminal = (2. if all(m["checks"].values()) else -2.)-abs(m["final_x_error_mm"])/4
                # Return-to-go charges simulated time from this action onward.
                for decision, row in zip(m["decisions"],samples):
                    time_left = m["completion_s"]-(decision["tick"]-SETTLE)*dt
                    batch.append((*row,terminal-.10*time_left))
            obs,gaits,z,old,returns = zip(*batch)
            obs,gaits,z,old = map(torch.stack,(obs,gaits,z,old))
            returns = torch.tensor(returns)
            advantage = (returns-policy.value(obs).squeeze(-1)).detach()
            advantage = (advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-6)
            before = nn.utils.parameters_to_vector(policy.actor.parameters()).detach().clone()
            for _ in range(4):
                gd,sd = policy.distributions(obs)
                ratio = (gd.log_prob(gaits)+sd.log_prob(z)-old).exp()
                loss = -torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean()
                loss += .5*(policy.value(obs).squeeze(-1)-returns).square().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(),1.)
                opt.step()
            change = (nn.utils.parameters_to_vector(policy.actor.parameters())-before).norm().item()
            if not torch.isfinite(loss) or not change>0:
                raise RuntimeError("PPO update did not produce a finite actor update")
            latest = args.output/f"stage2_{update:04d}.pt"
            torch.save(dict(iteration=update,policy=policy.state_dict(),optimizer=opt.state_dict()),latest)
            print(json.dumps(dict(stage="ppo_update",iteration=update,decisions=len(batch),actor_change=change)),flush=True)
        if not args.checkpoint:
            policy.load_state_dict(torch.load(latest,map_location="cpu",weights_only=True)["policy"])
        policy.eval()
        import omni.usd
        comparisons = []
        for goal in GOALS:
            # Fresh paired scenes avoid comparing different PhysX contact histories.
            env.close()
            omni.usd.get_context().new_stage()
            env = Env(cfg,render_mode=None)
            mpc = env.tracking.controller.mpc
            initial_states.clear()
            reset_states.clear()
            baseline,_ = episode(goal,"rule")
            env.close()
            omni.usd.get_context().new_stage()
            video = args.video and goal==GOALS[-1]
            env = Env(cfg, render_mode="rgb_array" if video else None)
            mpc = env.tracking.controller.mpc
            learned,_ = episode(goal,"policy",video=video)
            time_delta = None
            if baseline["arrival_s"] is not None and learned["arrival_s"] is not None:
                time_delta = learned["arrival_s"] - baseline["arrival_s"]
            delta = dict(arrival_time_s=time_delta,
                         absolute_error_mm=abs(learned["final_x_error_mm"])-abs(baseline["final_x_error_mm"]))
            comparisons.append(dict(target_mm=goal,rule=baseline,policy=learned,policy_minus_rule=delta))
            print(json.dumps(dict(stage="comparison",target_mm=goal,policy_minus_rule=delta,
                                  switches=len(learned["switches"]),stopped=learned["checks"]["stopped"])),
                  flush=True)
        passed = all(all(e[k]["checks"].values()) for e in comparisons for k in ("rule","policy"))
        (args.output/"comparison.json").write_text(json.dumps(dict(goals=comparisons,physical_checks=passed),
                                                              indent=2,allow_nan=False))
        if not passed:
            raise RuntimeError("Stage 2 evaluation failed; see comparison.json")
        print("PASS: Stage 2 cycle policy and rule evaluation",flush=True)
    finally:
        env.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--updates",type=int,default=2)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume",type=Path,help="Resume optimizer and actor to --updates total")
    source.add_argument("--checkpoint",type=Path,help="Evaluate a saved policy without training")
    parser.add_argument("--video",action="store_true")
    parser.add_argument("--seed",type=int,default=37)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.updates<1:
        parser.error("--updates must be positive")
    if args.video:
        args.enable_cameras=True
    launcher=AppLauncher(args)
    try:
        run(args)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()
