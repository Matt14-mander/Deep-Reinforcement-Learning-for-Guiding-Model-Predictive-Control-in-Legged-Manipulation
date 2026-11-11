"""Regression checks for Stage 2 resume state and milestone reporting."""
import ast
import contextlib
import io
import json
from types import SimpleNamespace
import copy
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import stage2_training_state
from stage2_training_state import (
    save_checkpoint, restore_checkpoint, sampling_state, restore_sampling,
    compare_evaluation, learning_summary, period_probes,
)


class TrainingStateTests(unittest.TestCase):
    def test_resume_restores_adam_iteration_and_sampling(self):
        task = dict(seed=37, speed_task=True, period_actions=True)
        model = torch.nn.Linear(4, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        def update():
            optimizer.zero_grad()
            model(torch.ones(4)).square().sum().backward()
            optimizer.step()
        update()
        noise = [torch.Generator().manual_seed(37+i) for i in range(8)]
        goals = [random.Random(37+i) for i in range(8)]
        bags = [[20, 42] for _ in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"checkpoint.pt"
            save_checkpoint(path,model,optimizer,4,task,noise,goals,bags)
            expected = [torch.randn(5,generator=g) for g in noise]
            expected_goals = [g.random() for g in goals]
            update()
            expected_weights = copy.deepcopy(model.state_dict())
            bags[0].pop()
            iteration, mode = restore_checkpoint(path,model,optimizer,task,noise,goals,bags)
            self.assertEqual((iteration,mode),(4,"restored"))
            self.assertEqual(bags[0],[20,42])
            for generator,values in zip(noise,expected):
                self.assertTrue(torch.equal(torch.randn(5,generator=generator),values))
            self.assertEqual([g.random() for g in goals],expected_goals)
            update()
            for key,value in model.state_dict().items():
                self.assertTrue(torch.equal(value,expected_weights[key]))
            changed = dict(task,seed=38)
            with self.assertRaises(ValueError):
                restore_checkpoint(path,model,optimizer,changed,noise,goals,bags)

    def test_evaluation_sampling_isolation_and_legacy_fallback(self):
        task = dict(seed=37,speed_task=True,period_actions=True)
        model = torch.nn.Linear(4,3)
        optimizer = torch.optim.Adam(model.parameters())
        noise = [torch.Generator().manual_seed(i) for i in range(8)]
        goals = [random.Random(i) for i in range(8)]
        bags = [[] for _ in range(8)]
        state = sampling_state(noise,goals,bags)
        expected = torch.rand(5)
        torch.rand(50)
        bags[0].append(64)
        restore_sampling(state,noise,goals,bags)
        self.assertTrue(torch.equal(torch.rand(5),expected))
        self.assertEqual(bags[0],[])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"legacy.pt"
            torch.save(dict(policy=model.state_dict(),optimizer=optimizer.state_dict(),
                            iteration=1,speed_task=True,period_actions=True),path)
            self.assertEqual(restore_checkpoint(path,model,optimizer,task,noise,goals,bags),
                             (1,"legacy_reseeded"))

    def test_reports_pair_rows_across_different_iterations(self):
        def episodes(iteration):
            result=[]
            for row in range(8):
                speed=3. if row%2==0 else 6.
                result.append(dict(identity=[iteration,row,0],target_mm=64,
                    desired_mm_s=speed,success=True,speed_ok=True,walking_mm_s=speed,
                    arrival_s=22. if speed==3. else 12.,final_x_error_mm=-1.,
                    switches=[],cycles=[dict(gait="crawl",time_scale=1.)],
                    decisions=[dict(period_slot=1)],max_stance_mm=[4.]*4))
            return result
        reports=[compare_evaluation(episodes(i),list(reversed(episodes(0))),i,True,[])
                 for i in (1,4,8,12)]
        self.assertTrue(all(m["passed"] for m in reports))
        summary=learning_summary(reports)
        self.assertEqual([m["iteration"] for m in summary["milestones"]],[1,4,8,12])
        self.assertFalse(summary["final_period_adaptation_observed"])
        self.assertTrue(summary["all_evaluations_passed"])
        reports[-1]["passed"]=False
        self.assertFalse(learning_summary(reports)["all_evaluations_passed"])
        bad=episodes(0)
        bad[0]["target_mm"]=32
        with self.assertRaises(RuntimeError):
            compare_evaluation(episodes(12),bad,12,True,[])

    def test_runner_resumes_to_12_and_replays_milestones(self):
        # Exercise the actual runner orchestration, replacing only PhysX collection
        # and the numerical PPO update, which are covered by the on-device smoke run.
        source=Path(__file__).with_name("train_stage2_async8.py").read_text()
        run=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="run")
        run.body=[n for n in run.body
                  if not (isinstance(n,ast.Import) and any(a.name=="omni.usd" for a in n.names))
                  and not (isinstance(n,ast.ImportFrom) and n.module in ("PIL","RL_Bezier_MPC.envs"))]
        replacement=ast.parse(
            "def rollout(mode,update,output=None):\n"
            "    return fake_rollout(mode,update)\n"
            "def update_policy(samples,reports):\n"
            "    return 0.01,0.02\n")
        replacements={n.name:n for n in replacement.body}
        run.body=[replacements.get(n.name,n) if isinstance(n,ast.FunctionDef) else n for n in run.body]
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.actor=torch.nn.Linear(4,2)
                self.period_actor=torch.nn.Linear(4,3)
            def load_compatible_state(self,state):
                self.load_state_dict(state)
            def period_distribution(self,obs):
                return torch.distributions.Categorical(logits=self.period_actor(obs))
        calls=[]
        def fake_rollout(mode,iteration):
            calls.append((mode,iteration))
            reports=[]
            for row in range(8):
                speed=3. if row%2==0 else 6.
                reports.append(dict(identity=[iteration,row,0],target_mm=64,desired_mm_s=speed,
                    success=True,speed_ok=True,walking_mm_s=speed,arrival_s=22. if speed==3. else 12.,
                    final_x_error_mm=0.,switches=[],cycles=[dict(gait="crawl",time_scale=1.)],
                    decisions=[dict(period_slot=1)],max_stance_mm=[1.]*4))
            return reports,[],1.
        ns=dict(vars(stage2_training_state), random=random, shutil=shutil, fake_rollout=fake_rollout,
                make_policy=lambda **kwargs:Policy(), async_types=lambda *a,**kw:(None,None),
                N=8,SPEED_GOAL_MM=64,SPEEDS=(3.,6.),GOALS=(20,42,64),
                PERIOD_SCALES={"crawl":(1.2,1.,.8),"trot":(1.2,1.,.9)})
        exec(compile(ast.fix_missing_locations(ast.Module(body=[run],type_ignores=[])),
                     "runner-orchestration","exec"),ns)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            model=Policy()
            checkpoint=root/"legacy.pt"
            torch.save(dict(policy=model.state_dict(),
                            optimizer=torch.optim.Adam(model.parameters(),lr=.003).state_dict(),
                            iteration=1,speed_task=True,period_actions=True),checkpoint)
            args=SimpleNamespace(seed=37,speed_task=True,period_actions=True,
                checkpoint=None,resume=checkpoint,output=root/"training",
                until=12,updates=None,eval_only=False,evaluate_run=None,
                eval_iterations=[1,4,8,12],period_probe_only=False)
            with contextlib.redirect_stdout(io.StringIO()):
                ns["run"](args)
            self.assertEqual([i for mode,i in calls if mode=="sample"],list(range(2,13)))
            self.assertEqual([i for mode,i in calls if mode=="policy"],[1,4,8,12])
            self.assertEqual([i for mode,i in calls if mode=="rule"],[0])
            summary=json.loads((args.output/"summary.json").read_text())
            self.assertTrue(summary["completed"])
            self.assertEqual(summary["last_evaluated_iteration"],12)
            self.assertEqual(len(list(args.output.glob("async8_*.pt"))),12)
            calls.clear()
            args.resume=None
            args.evaluate_run=args.output
            args.output=root/"replay"
            with contextlib.redirect_stdout(io.StringIO()):
                ns["run"](args)
            self.assertFalse(any(mode=="sample" for mode,i in calls))
            self.assertEqual([i for mode,i in calls if mode=="policy"],[1,4,8,12])
            calls.clear()
            args.resume=root/"training"/"async8_0008.pt"
            args.evaluate_run=None
            args.output=root/"resumed8"
            with contextlib.redirect_stdout(io.StringIO()):
                ns["run"](args)
            self.assertEqual([i for mode,i in calls if mode=="sample"],[9,10,11,12])
            self.assertEqual([i for mode,i in calls if mode=="policy"],[1,4,8,12])
            self.assertTrue((args.output/"async8_0001.pt").is_file())
            self.assertTrue((args.output/"async8_0004.pt").is_file())

    def test_period_probes_change_only_requested_speed(self):
        class Policy:
            def period_distribution(self,obs):
                return torch.distributions.Categorical(
                    logits=torch.stack((-obs[2]*3,obs[2]*0,obs[2]*3+1)))
        probes=period_probes(Policy())
        self.assertEqual(len(probes),18)
        self.assertTrue(all(p["speed_changes_slot"] for p in probes))


if __name__=="__main__":
    unittest.main()
