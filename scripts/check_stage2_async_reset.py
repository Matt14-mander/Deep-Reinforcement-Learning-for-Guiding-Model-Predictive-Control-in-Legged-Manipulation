"""Two asynchronous Stage 2 policy clocks sharing one batched MPC and PhysX scene."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

from isaaclab.app import AppLauncher
from check_go2_slow_trot import SETTLE, HOLD, evaluate_mixed, make_plan, swing_groups, PERIOD_SCALES


def async_types(choose, verbose=False, period_actions=False):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import QuadrupedMPCEnv

    from RL_Bezier_MPC.envs.go2_cycle import CycleClock

    class Clock(CycleClock):
        def __init__(self,env,goal,delay=0):
            super().__init__(env,goal,delay,choose=choose,period_actions=period_actions,verbose=verbose)

    class AsyncEnv(QuadrupedMPCEnv):
        live = False

        def _reset_idx(self,ids):
            if self.live and (ids.tolist()!=self.allowed_reset_ids):
                raise RuntimeError(f"Unexpected reset {ids.tolist()} at world tick {self.world}")
            super()._reset_idx(ids)

        def _solve_mpc_control_step(self):
            d,a = self.robot.data,self.tracking
            mpc = a.controller.mpc
            state = a.pack_state(d.root_link_pose_w.double(),d.root_link_vel_w.double(),
                                 d.joint_pos.double(),d.joint_vel.double(),self.tracking_origins)
            feet = d.body_link_pos_w[:,self._foot_body_indices.tolist()].double()-self.tracking_origins[:,None]
            loads = self.foot_contact_sensor.data.net_forces_w[:,self._contact_sensor_foot_indices,2]
            if not all(t.isfinite().all() for t in (state,feet,loads)):
                raise RuntimeError("Nonfinite measurement")
            mask = torch.ones_like(self.mask)
            refs, accs, phases = [], [], []
            for r,c in enumerate(self.clocks):
                c.prepare(state[r,0].item(),loads[r])
                ref,acc,phase = c.plan
                i = max(0,min(c.tick,len(ref)-1)) if not c.parked else 0
                cycle,pair,swing = phase[i]
                gait = c.schedule[cycle][0] if 0<=cycle<len(c.schedule) else "stance"
                if swing:
                    mask[r,list(swing_groups(gait)[pair])] = False
                refs.append(ref[(self.nodes+i).clamp_max(len(ref)-1)])
                accs.append(acc[i])
                phases.append((cycle,pair,swing,gait))
                if c.tick==SETTLE:
                    self.origins[r] = feet[r]
            changed = mask ^ self.mask
            if changed.any():
                gained = mask & changed
                self.anchors[:,:,:2][gained] = feet[:,:,:2][gained]
                self.origins[gained] = feet[gained]
                mpc.reset(changed.any(dim=1))
            self.mask = mask
            tau,result = mpc.compute(state,torch.stack(refs),torch.stack(accs),mask,self.anchors)
            if not result.usable.all():
                raise RuntimeError(f"MPC rejected tick {self.world}: {result.status.tolist()}, phases={phases}")
            self._pending_joint_efforts.copy_(a.scatter_action(tau).clamp(-self.cfg.max_joint_torque,self.cfg.max_joint_torque))
            self._last_mpc_costs,self._last_mpc_converged = result.cost,result.success
            for r,c in enumerate(self.clocks):
                cycle,pair,swing,gait = phases[r]
                if c.tick>=SETTLE and not c.parked:
                    self.drift[r] = torch.maximum(self.drift[r],(feet[r]-self.origins[r]).norm(dim=-1)*mask[r])
                if c.tick>=0 and not c.parked:
                    q = d.root_link_quat_w[r].tolist()
                    w,x,y,z = q
                    row = dict(tick=c.tick,world_tick=self.world,cycle=cycle,pair=pair,swing=swing,gait=gait,
                        decision_count=len(c.schedule),x=state[r,0].item(),y=state[r,1].item(),quat=q,
                        roll=math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))),
                        pitch=math.degrees(math.asin(max(-1.,min(1.,2*(w*y-z*x))))),
                        yaw=math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))),
                        speed=state[r,19:22].norm().item(),spin=state[r,22:25].norm().item(),
                        joint=state[r,25:].abs().max().item(),loads=loads[r].tolist(),anchors=self.anchors[r].tolist(),
                        requested=self._pending_joint_efforts[r].tolist(),applied=d.applied_torque[r].tolist())
                    c.rows.append(row)
                    if c.finishing:
                        steady = (min(row["loads"])>5 and row["speed"]<.01 and row["spin"]<.04 and
                                  row["joint"]<.1 and abs(1000*(row["x"]-c.start_x)-c.goal)<=4 and
                                  abs(row["y"]-c.rows[SETTLE]["y"])<=.010)
                        c.stable = c.stable+1 if steady else 0
                c.tick += 1
            self.world += 1
            self._advance_reference_clocks()

        def peer_snapshot(self):
            mpc = self.tracking.controller.mpc
            c = self.clocks[1]
            return dict(local_tick=c.tick,count=len(c.schedule),boundary=c.boundary,
                        phase=c.plan[2][c.tick],target=c.goal,
                        tensors=[v[1].clone() for v in (self.anchors,self.origins,self.mask,self.drift,
                            self._pending_joint_efforts,self.robot.data.applied_torque,self.robot.data.root_link_pose_w,
                            self.robot.data.root_link_vel_w,self.robot.data.joint_pos,self.robot.data.joint_vel,
                            mpc._warm,mpc._dual,mpc._valid,self.episode_length_buf)])

    return Clock, AsyncEnv


def run(args):
    import torch
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"source/RL_Bezier_MPC"))
    from RL_Bezier_MPC.envs import QuadrupedMPCEnv, QuadrupedMPCEnvCfg

    # Frozen deterministic actor from stage2_0002.pt, so the regression also runs
    # on a fresh checkout. Use --checkpoint to test another trained actor.
    actor = torch.nn.Linear(2, 2)
    if args.checkpoint:
        state = torch.load(args.checkpoint,map_location="cpu",weights_only=True)["policy"]
        actor.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("actor.")})
    else:
        actor.load_state_dict(dict(
            weight=torch.tensor([[2.023329734802246,-.018651263788342476],
                                 [.1350843757390976,.020735003054142]],dtype=torch.float32),
            bias=torch.tensor([-.023278076201677322,.02399066649377346])))
    actor.eval()

    def choose(clock,obs,remaining):
        logit,z = actor(obs).tolist()
        return ("trot" if logit>=0 else "crawl"),z
    Clock, AsyncEnv = async_types(choose, verbose=True)

    cfg = QuadrupedMPCEnvCfg()
    cfg.scene.num_envs,cfg.scene.env_spacing,cfg.seed,cfg.sim.device = 2,1.3,37,args.device
    cfg.mpc_backend,cfg.use_mpc_cluster,cfg.torch_foot_plane_height = "torch_tracking",False,.022
    cfg.episode_length_s = 40.
    cfg.rl_policy_period,cfg.decimation = 1,cfg.mpc_decimation
    cfg.viewer.eye,cfg.viewer.lookat,cfg.viewer.resolution = (.23,-3.1,.92),(.04,0,.20),(1120,560)
    cfg.sim.render_interval = cfg.decimation
    args.output.mkdir(parents=True,exist_ok=False)
    env = AsyncEnv(cfg,render_mode="rgb_array" if args.video else None)
    writer = None
    try:
        env.reset(seed=cfg.seed)
        dt,mpc = cfg.mpc_dt,env.tracking.controller.mpc
        env.nominal = make_plan(mpc.model,dt,[("crawl",.020)],cycle_decisions=True)
        near,far = Clock(env,20),Clock(env,64,delay=270)
        env.clocks = [near,far]
        env.mask = torch.ones((2,4),device=env.device,dtype=torch.bool)
        env.anchors = mpc.model.contact_positions.expand(2,-1,-1).clone()
        env.origins,env.drift = env.anchors.clone(),env.anchors.new_zeros(2,4)
        env.nodes = torch.arange(mpc.horizon+1,device=env.device)
        env.world,env.live,env.allowed_reset_ids = 0,True,[]
        action = torch.zeros((2,cfg.action_space),device=env.device)
        reset = None
        if args.video:
            writer = subprocess.Popen(["ffmpeg","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s",
                "1120x560","-r",str(1/dt),"-i","pipe:0","-c:v","libx264","-pix_fmt","yuv420p",
                "-movflags","+faststart",str(args.output/"two_go2.mp4")],stdin=subprocess.PIPE)
        while far.stable<50:
            if env.world>1450:
                raise RuntimeError("Timed out before far robot stopped")
            _,_,term,trunc,_ = env.step(action)
            if (term|trunc).any():
                raise RuntimeError("Unexpected episode termination")
            if near.stable>=50 and reset is None:
                before = env.peer_snapshot()
                if far.rows[-1]["gait"]!="trot" or not far.rows[-1]["swing"]:
                    raise RuntimeError("Reset must test an active far trot swing")
                ids = torch.tensor([0],device=env.device)
                near_report = evaluate_mixed(near.rows,env.drift[0].tolist(),dt,near.schedule,target_mm=20)
                env.allowed_reset_ids = [0]
                env._reset_idx(ids)
                env.allowed_reset_ids = []
                env.anchors[0] = mpc.model.contact_positions
                env.origins[0] = env.anchors[0]
                env.mask[0] = True
                env.drift[0].zero_()
                idle = Clock(env,20)
                idle.parked = True
                env.clocks[0] = idle
                after = env.peer_snapshot()
                for old,new in zip(before.pop("tensors"),after.pop("tensors")):
                    torch.testing.assert_close(new,old,atol=0,rtol=0)
                if before!=after or mpc._valid[0] or env.episode_length_buf[0]!=0 or env._pending_joint_efforts[0].any():
                    raise RuntimeError("Reset isolation failed")
                reset = dict(world_tick=env.world,far_clock=before,instantaneous_peer_delta=0)
                print(json.dumps(dict(stage="near_reset",**reset)),flush=True)
            if writer:
                rgb = env.render()
                for _ in range(8):
                    if rgb.size and rgb[40:450:8,::8,0].std()>8:
                        break
                    env.sim.set_camera_view(cfg.viewer.eye,cfg.viewer.lookat)
                    env.sim.render()
                    rgb=env.render()
                else:
                    raise RuntimeError("Camera lost scene")
                frame=Image.fromarray(rgb).convert("RGB")
                draw=ImageDraw.Draw(frame)
                draw.rectangle((0,503,1120,560),fill="#162431")
                for r,c in enumerate(env.clocks):
                    row = c.rows[-1] if c.rows else None
                    label = "RESET / standing" if c.parked else row["gait"] if row else "waiting"
                    dx = 1000*(row["x"]-c.start_x) if row and c.start_x is not None else 0.
                    side = "right" if env.scene.env_origins[r,0]>0 else "left"
                    draw.text((12,510+21*r),f"GO2 {r} ({side})    {dx:.1f} / {c.goal} mm    {label}    "
                              f"decisions {len(c.schedule)}    world {env.world*dt:.2f} s")
                writer.stdin.write(frame.tobytes())
        if reset is None:
            raise RuntimeError("Near robot never reset")
        far_report=evaluate_mixed(far.rows,env.drift[1].tolist(),dt,far.schedule,target_mm=64)
        tick=reset["world_tick"]
        window=[r for r in far.rows if tick-3<=r["world_tick"]<tick+3]
        def jump(key):
            return max(abs(x-y) for a,b in zip(window,window[1:]) for x,y in zip(a[key],b[key]))
        reset.update(requested_step_Nm=jump("requested"),applied_step_Nm=jump("applied"),
                     anchor_jump_mm=1000*max(math.dist(x,y) for r in window for x,y in zip(r["anchors"],window[0]["anchors"])))
        if not (len(window)==6 and len({r["decision_count"] for r in window})==1 and
                all(b["tick"]==a["tick"]+1 for a,b in zip(window,window[1:])) and
                reset["anchor_jump_mm"]<.001 and max(jump("requested"),jump("applied"))<3 and
                any(d["world_tick"]>tick and d["gait"]=="crawl" for d in far.decisions) and
                far_report["switches"] and all(near_report["checks"].values()) and all(far_report["checks"].values())):
            raise RuntimeError(f"Acceptance failed: {reset}, {near_report}, {far_report}")
        m=dict(near=near_report,far=far_report,reset=reset,near_decisions=near.decisions,far_decisions=far.decisions,
               env_origins=env.scene.env_origins.tolist(),far_start_delay_s=270*dt)
        (args.output/"measurements.json").write_text(json.dumps(m,indent=2,allow_nan=False))
    finally:
        try:
            if writer:
                writer.stdin.close()
                if writer.wait():
                    raise RuntimeError("Video encoding failed")
        finally:
            env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--video",action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.video:
        args.enable_cameras=True
    launcher=AppLauncher(args)
    try:
        run(args)
        print("PASS: Stage 2 dual Go2 asynchronous decisions and isolated reset",flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        launcher.app.close()
