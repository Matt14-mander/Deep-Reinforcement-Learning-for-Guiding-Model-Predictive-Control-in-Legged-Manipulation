"""Read milestone metrics without importing Torch or Isaac Sim."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("run",type=Path)
parser.add_argument("--require-complete",action="store_true")
args=parser.parse_args()
manifest=json.loads((args.run/"run.json").read_text())
reports=json.loads((args.run/"milestones.json").read_text())
if args.require_complete:
    assert manifest["training_complete"],"Training/evaluation is incomplete"
    assert [r["iteration"] for r in reports]==manifest["milestones"],"Missing evaluations"
print("iter controller target_mm error_mm mean_mm_s episode_s arrival_s arrived stopped drift_mm gaits")
for report in reports:
    for episode in report["episodes"]:
        gaits=",".join(d["gait"] for d in episode["decisions"])
        arrival="--" if episode["arrival_time_s"] is None else f'{episode["arrival_time_s"]:.2f}'
        print(f'{report["iteration"]:4d} {episode["controller"]:10s} {episode["target_mm"]:9d} '
              f'{episode["error_mm"]:8.3f} {episode["mean_forward_speed_mps"]*1000:9.3f} '
              f'{episode["elapsed_s"]:9.2f} {arrival:9s} {str(episode["success"]):7s} {str(episode["stopped"]):7s} '
              f'{episode["max_stance_drift_mm"]:8.3f} {gaits}')
    print(json.dumps(dict(iteration=report["iteration"],policy_all_arrived=report["policy_all_arrived"],
        improvement_over_untrained=report["improvement_demonstrated"])))
print(json.dumps(dict(training_complete=manifest["training_complete"],evaluations=manifest["evaluations"])))

if reports:
    first={e["target_mm"]:e for e in reports[0]["episodes"] if e["controller"]=="post"}
    last={e["target_mm"]:e for e in reports[-1]["episodes"] if e["controller"]=="post"}
    for goal in sorted(first.keys() & last.keys()):
        before,after=first[goal],last[goal]
        arrival_gain=None
        if before["arrival_time_s"] is not None and after["arrival_time_s"] is not None:
            arrival_gain=before["arrival_time_s"]-after["arrival_time_s"]
        print(json.dumps(dict(stage="change_from_start",from_iteration=reports[0]["iteration"],
            to_iteration=reports[-1]["iteration"],target_mm=goal,
            error_improvement_mm=before["error_mm"]-after["error_mm"],
            speed_change_mps=after["mean_forward_speed_mps"]-before["mean_forward_speed_mps"],
            arrival_time_improvement_s=arrival_gain,arrived_before=before["success"],arrived_after=after["success"],
            max_drift_change_mm=after["max_stance_drift_mm"]-before["max_stance_drift_mm"])))
