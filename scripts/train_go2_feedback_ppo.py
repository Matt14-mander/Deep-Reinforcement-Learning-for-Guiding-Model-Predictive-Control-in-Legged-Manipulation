"""Resume to an absolute iteration with isolated deterministic milestone evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from go2_feedback_training import load_checkpoint,save_checkpoint,save_json,milestones


def main(args):
    state=load_checkpoint(args.checkpoint)
    stops=milestones(state["iteration"],args.total_iterations,args.eval_iterations)
    if args.rollout_steps<2:
        raise ValueError("At least two rollout steps required")
    if "training_task" in state and state["training_task"]["rollout_steps"]!=args.rollout_steps:
        raise ValueError("Resume rollout length differs from checkpoint")
    args.output.mkdir(parents=True,exist_ok=False)
    current=args.output/f"feedback_{state['iteration']:04d}.pt"
    save_checkpoint(current,state)
    script_dir=Path(__file__).resolve().parent
    completed=state["iteration"]
    reports=[]
    manifest=dict(start_checkpoint=str(args.checkpoint.resolve()),start_iteration=completed,
        total_iterations=args.total_iterations,milestones=stops,rollout_steps=args.rollout_steps,
        evaluations=[],training_complete=False)
    save_json(args.output/"run.json",manifest)
    def launch(command):
        print(json.dumps(dict(stage="launch",command=[str(x) for x in command])),flush=True)
        subprocess.run([str(x) for x in command],check=True)
    for stop in stops:
        if stop>completed:
            launch([sys.executable,"-u",script_dir/"train_go2_feedback_worker.py","--checkpoint",current,
                "--output",args.output,"--until",stop,"--rollout-steps",args.rollout_steps,
                "--headless","--device",args.device])
            current=args.output/f"feedback_{stop:04d}.pt"
            if load_checkpoint(current)["iteration"]!=stop:
                raise RuntimeError("Worker did not save requested iteration")
            completed=stop
        destination=args.output/f"eval_{stop:04d}"
        launch([sys.executable,"-u",script_dir/"evaluate_go2_feedback_ppo.py","--checkpoint",current,
            "--output",destination,"--headless","--device",args.device])
        report=json.loads((destination/"comparison.json").read_text())
        if not report["evaluation_completed"]:
            raise RuntimeError("Incomplete milestone evaluation")
        reports.append(dict(iteration=stop,summary=report["summary"],comparisons=report["comparisons"],
            episodes=report["episodes"],policy_all_arrived=report["policy_all_arrived"],
            improvement_demonstrated=report["improvement_demonstrated"]))
        manifest["evaluations"].append(stop)
        manifest["last_checkpoint"]=str(current.resolve())
        save_json(args.output/"run.json",manifest)
        save_json(args.output/"milestones.json",reports)
        print(json.dumps(dict(stage="milestone",iteration=stop,summary=report["summary"],
            policy_all_arrived=report["policy_all_arrived"],improvement_demonstrated=report["improvement_demonstrated"])),flush=True)
    manifest["training_complete"]=True
    save_json(args.output/"run.json",manifest)
    print(json.dumps(dict(stage="complete",completed_iteration=completed,milestones=stops,
        all_milestone_policy_arrived=all(r["policy_all_arrived"] for r in reports))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--total-iterations",type=int,default=12)
    parser.add_argument("--eval-iterations",type=int,nargs="+",default=[1,4,8,12])
    parser.add_argument("--rollout-steps",type=int,default=96)
    parser.add_argument("--device",default="cuda:0")
    main(parser.parse_args())
