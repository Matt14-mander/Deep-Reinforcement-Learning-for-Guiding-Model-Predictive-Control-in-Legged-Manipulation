"""Reward decomposition and same-episode effective action attribution."""
import importlib.util
from pathlib import Path
import unittest
import torch
from go2_learning_signal import reward_parts,summarize
p=Path(__file__).resolve().parents[1]/"source/RL_Bezier_MPC/RL_Bezier_MPC/envs/go2_feedback.py"
spec=importlib.util.spec_from_file_location("feedback",p)
feedback=importlib.util.module_from_spec(spec)
spec.loader.exec_module(feedback)


class SignalTests(unittest.TestCase):
    def test_reward_parts(self):
        torch.manual_seed(3)
        args=(torch.randn(8),torch.randn(8)*.01,torch.full((8,),.006),torch.randn(8,12),
              torch.randn(8,4),torch.tensor([True,False]*4),torch.tensor([False,True]*4),.25)
        self.assertTrue(torch.allclose(reward_parts(*args).sum(-1),feedback.dense_reward(*args),atol=1e-5))

    def test_boundary_not_persistent_request(self):
        batch=dict(advantage=torch.tensor([[1.],[-1.],[2.]]),actions=torch.tensor([[[1.,0.,0.,0.]],[[1.,0.,0.,0.]],[[1.,0.,0.,0.]]]),
            identity=torch.tensor([[[0,0,0]],[[0,0,1]],[[0,1,0]]]),
            mode_counts=torch.tensor([[[0,0,13,0]],[[0,0,12,0]],[[13,0,0,0]]]),
            parts=torch.zeros(3,1,6),categorical_change_cost=torch.tensor([[0.],[-.02],[0.]]))
        events=[dict(step=0,env=0,identity=[0,0,0],gait_probability=[.2,.8])]
        result=summarize(batch,events)
        self.assertEqual(result["effective_cycle_decisions"],1)
        self.assertEqual(events[0]["advantage"],1.)
        self.assertAlmostEqual(result["unapplied_categorical_change_penalty_total"],-.02)
        wrong=[dict(step=2,env=0,identity=[0,0,0],gait_probability=[.2,.8])]
        with self.assertRaises(AssertionError):
            summarize(batch,wrong)


if __name__=="__main__":
    unittest.main()
