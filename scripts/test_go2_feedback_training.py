"""Cumulative numbering, continuation guards and milestone orchestration."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from go2_feedback_training import milestones,save_checkpoint,load_checkpoint,validate_task
from train_go2_feedback_ppo import main


class TrainingTests(unittest.TestCase):
    def test_milestones_are_absolute(self):
        self.assertEqual(milestones(1,12,[1,4,8,12]),[1,4,8,12])
        self.assertEqual(milestones(5,12,[1,4,8,12]),[5,8,12])
        self.assertEqual(milestones(12,12,[1,4,8,12]),[12])
        with self.assertRaises(ValueError):
            milestones(8,4,[4])

    def test_task_guard(self):
        validate_task({},dict(rollout_steps=96))
        validate_task(dict(training_task=dict(rollout_steps=96)),dict(rollout_steps=96))
        with self.assertRaises(ValueError):
            validate_task(dict(training_task=dict(rollout_steps=96)),dict(rollout_steps=8))

    def test_parent_schedule_and_checkpoint_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            start=root/"start.pt"
            save_checkpoint(start,dict(iteration=1,observation_size=57,action_size=4,
                optimizer={"test":1},policy={"weight":torch.tensor(1.)},initial_policy={"weight":torch.tensor(0.)},
                rng_cpu=torch.get_rng_state(),rng_cuda=[]))
            calls=[]
            def child(command,check):
                self.assertTrue(check)
                checkpoint=Path(command[command.index("--checkpoint")+1])
                state=load_checkpoint(checkpoint)
                output=Path(command[command.index("--output")+1])
                if "--until" in command:
                    stop=int(command[command.index("--until")+1])
                    calls.append(("train",state["iteration"],stop))
                    for i in range(state["iteration"]+1,stop+1):
                        state["iteration"]=i
                        state["policy"]["weight"]=torch.tensor(float(i))
                        save_checkpoint(output/f"feedback_{i:04d}.pt",state)
                else:
                    calls.append(("eval",state["iteration"]))
                    self.assertEqual(state["policy"]["weight"].item(),state["iteration"])
                    output.mkdir()
                    (output/"comparison.json").write_text(json.dumps(dict(evaluation_completed=True,
                        summary={},comparisons=[],episodes=[],policy_all_arrived=True,improvement_demonstrated=False)))
            args=argparse.Namespace(checkpoint=start,output=root/"run",total_iterations=12,
                eval_iterations=[1,4,8,12],rollout_steps=96,device="cpu")
            with patch("train_go2_feedback_ppo.subprocess.run",side_effect=child),contextlib.redirect_stdout(io.StringIO()):
                main(args)
            self.assertEqual(calls,[("eval",1),("train",1,4),("eval",4),("train",4,8),("eval",8),("train",8,12),("eval",12)])
            self.assertEqual(load_checkpoint(args.output/"feedback_0012.pt")["iteration"],12)
            report=json.loads((args.output/"run.json").read_text())
            self.assertTrue(report["training_complete"])
            self.assertEqual(report["evaluations"],[1,4,8,12])


if __name__=="__main__":
    unittest.main()
