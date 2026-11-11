"""Validated iteration checkpoints and milestone planning (no Isaac dependency)."""
import json
from pathlib import Path
import torch
from go2_feedback_ppo import FeedbackPolicy


def milestones(start,total,requested):
    if start<1 or total<start:
        raise ValueError("Total iteration must be at least the checkpoint iteration")
    return sorted({start,total}|{i for i in requested if start<=i<=total})


def save_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def save_checkpoint(path,state):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+".tmp")
    torch.save(state,temporary)
    temporary.replace(path)


def load_checkpoint(path,device="cpu"):
    state=torch.load(path,map_location=device,weights_only=True)
    if type(state.get("iteration")) is not int or state["iteration"]<1:
        raise ValueError("Invalid completed iteration")
    if state.get("observation_size")!=57 or state.get("action_size")!=4:
        raise ValueError("Incompatible policy interface")
    if not state.get("optimizer") or "rng_cpu" not in state or "rng_cuda" not in state:
        raise ValueError("True resume requires optimizer and RNG state")
    if "initial_policy" not in state:
        if state["iteration"]!=1:
            raise ValueError("Legacy reconstruction supports only iteration 1")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(37)
            pre=FeedbackPolicy()
            rollout=torch.load(Path(path).parent/"rollout.pt",map_location="cpu",weights_only=True)
            with torch.no_grad():
                obs=rollout["observations"].flatten(0,1)
                log=pre.log_prob(obs,rollout["latent"].flatten(0,1))
                if not torch.allclose(log,rollout["log_prob"].flatten(),atol=2e-5,rtol=0):
                    raise ValueError("Initial policy reconstruction mismatch")
                if not torch.allclose(pre.value(obs),rollout["values"].flatten(),atol=2e-5,rtol=0):
                    raise ValueError("Initial critic reconstruction mismatch")
            state["initial_policy"]={k:v.detach().clone() for k,v in pre.state_dict().items()}
    return state


def restore_rng(state):
    torch.set_rng_state(state["rng_cpu"].cpu())
    if torch.cuda.is_available() and state["rng_cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["rng_cuda"]])


def validate_task(state,task):
    if "training_task" in state and state["training_task"]!=task:
        raise ValueError("Resume training configuration differs from checkpoint")
