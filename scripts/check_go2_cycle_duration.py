"""Single Go2, fixed stride and gait, three cycle durations and a full stop."""
import argparse
from itertools import groupby
import json
import math
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher
from check_go2_slow_trot import (
    SETTLE, cycle_timing, evaluate_mixed, gait_env_type, make_plan, swing_groups,
)

SCALES = {"crawl": (1.2, 1., .8), "trot": (1.2, 1., .9)}
STRIDE = .020
CYCLES = 3


def boundary_torque_step(rows):
    starts = [i for i in range(1,len(rows))
              if rows[i]["cycle"] != rows[i-1]["cycle"] and rows[i]["cycle"] >= 0]
    return max(abs(a-b) for i in starts
               for prev,cur in zip(rows[max(0,i-2):i+2],rows[max(0,i-2)+1:i+3])
               for a,b in zip(prev["requested"],cur["requested"]))


def run(args):
    import torch
    import omni.usd

    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for gait in args.gaits:
        schedule = [(gait, STRIDE)] * CYCLES
        Env = gait_env_type(schedule, True)
        from RL_Bezier_MPC.envs import QuadrupedMPCEnvCfg

        for scale in (args.scales or SCALES[gait]):
            print(json.dumps(dict(stage="start", gait=gait, time_scale=scale)), flush=True)
            cfg = QuadrupedMPCEnvCfg()
            cfg.scene.num_envs, cfg.seed, cfg.sim.device = 1, 37, args.device
            cfg.mpc_backend, cfg.use_mpc_cluster = "torch_tracking", False
            cfg.torch_foot_plane_height = .022
            cfg.rl_policy_period, cfg.decimation = 1, cfg.mpc_decimation
            cfg.episode_length_s = 60.
            cfg.sim.render_interval = 1000000
            omni.usd.get_context().new_stage()
            env = Env(cfg)
            trace_path = args.output / f"{gait}_{scale:.1f}_trace.json"
            try:
                env.reset(seed=37)
                mpc = env.tracking.controller.mpc
                dt = cfg.mpc_dt
                env.ref, env.acc, env.phases = make_plan(
                    mpc.model, dt, schedule, cycle_decisions=True, time_scale=scale)
                reference_feet = mpc.model.foot_positions(env.ref[:, :19]).tolist()
                env.mask = torch.ones(1, 4, device=env.device, dtype=torch.bool)
                env.anchors = mpc.model.contact_positions[None].clone()
                env.origin = env.anchors[0].clone()
                env.rows, env.drift, env.tick = [], [0.] * 4, 0
                env.active = env.period_diagnostics = True
                zero = torch.zeros((1, cfg.action_space), device=env.device)
                while env.tick < len(env.ref):
                    _, _, term, trunc, _ = env.step(zero)
                    if (term | trunc).any():
                        raise RuntimeError("Unexpected episode end")
                m = evaluate_mixed(env.rows, env.drift, dt, schedule,
                                   target_mm=1000*STRIDE*CYCLES, time_scale=scale)
                errors = []
                for (_, pair, swing), group in groupby(
                        env.rows, key=lambda r: (r["cycle"], r["pair"], r["swing"])):
                    if not swing:
                        continue
                    samples = list(group)
                    first = samples[0]["tick"]
                    legs = swing_groups(gait)[pair]
                    for row in samples:
                        for leg in legs:
                            actual = [a-b for a,b in zip(row["feet"][leg], env.rows[first]["feet"][leg])]
                            planned = [a-b for a,b in zip(reference_feet[row["tick"]][leg], reference_feet[first][leg])]
                            errors.append(1000*math.dist(actual, planned))
                cycle_ticks = [sum(p[0] == i for p in env.phases) for i in range(CYCLES)]
                moving = env.rows[SETTLE:SETTLE+sum(cycle_ticks)]
                last = env.rows[SETTLE+sum(cycle_ticks)]
                speed = 1000*(last["x"]-env.rows[SETTLE]["x"])/(sum(cycle_ticks)*dt)
                raw_peak = max(abs(v) for r in moving for v in r["raw_requested"])
                applied_peak = max(abs(v) for r in moving for v in r["applied"])
                torque_step = max(abs(a-b) for prev,cur in zip(moving,moving[1:])
                                  for a,b in zip(prev["requested"],cur["requested"]))
                m.update(gait=gait, time_scale=scale, timing_ticks=cycle_timing(scale),
                         stride_mm=1000*STRIDE, cycle_s=[t*dt for t in cycle_ticks],
                         measured_mm_s=speed,
                         swing_displacement_error_max_mm=max(errors),
                         swing_displacement_error_rms_mm=math.sqrt(sum(e*e for e in errors)/len(errors)),
                         raw_torque_peak_Nm=raw_peak, applied_torque_peak_Nm=applied_peak,
                         requested_step_peak_Nm=torque_step,
                         boundary_requested_step_Nm=boundary_torque_step(env.rows))
                m["checks"].update(
                    speed_tracking=all(abs(c["measured_mm_s"]/c["command_mm_s"]-1)<.25 for c in m["cycles"]),
                    swing_tracking=max(errors)<=10.,
                    torque_unsaturated=raw_peak<cfg.max_joint_torque,
                    cycle_boundary_torque=m["boundary_requested_step_Nm"]<3.,
                )
                reports.append(m)
                print(json.dumps(dict(stage="trial", **m), allow_nan=False), flush=True)
                (args.output/"measurements.json").write_text(json.dumps(reports, indent=2, allow_nan=False))
            finally:
                try:
                    if getattr(env, "rows", None):
                        trace_path.write_text(json.dumps(env.rows, allow_nan=False))
                finally:
                    env.close()
    ordered = all(
        all(a["measured_mm_s"] < b["measured_mm_s"] for a,b in zip(group, group[1:]))
        for gait in args.gaits
        for group in [[m for m in reports if m["gait"] == gait]]
    )
    passed = ordered and all(all(m["checks"].values()) for m in reports)
    (args.output/"comparison.json").write_text(json.dumps(
        dict(passed=passed, speed_ordered=ordered, trials=reports), indent=2, allow_nan=False))
    if not passed:
        raise RuntimeError("Variable period acceptance failed; see comparison.json")
    print("PASS: single Go2 variable cycle duration", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaits", nargs="+", choices=("crawl", "trot"), default=["crawl", "trot"])
    parser.add_argument("--scales", nargs="+", type=float,
                        help="Descending positive time scales; default crawl 1.2/1/0.8, trot 1.2/1/0.9")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.scales and (any(not math.isfinite(s) or not .5 <= s <= 2. for s in args.scales)
                        or any(a <= b for a,b in zip(args.scales,args.scales[1:]))):
        parser.error("Scales must decrease strictly and lie between 0.5 and 2")
    if len(set(args.gaits)) != len(args.gaits):
        parser.error("Select each gait once")
    launcher = AppLauncher(args)
    passed = False
    try:
        run(args)
        passed = True
    except Exception:
        traceback.print_exc()
    finally:
        launcher.app.close()
    if not passed:
        raise SystemExit(1)
