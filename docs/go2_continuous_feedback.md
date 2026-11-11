# Go2 continuous feedback environment

Gym ID: `Isaac-Go2-AugMPC-Direct-v0`. The environment reuses the existing
robot, sensors, state packing, Torch MPC and effort actuator integration.
The old cycle-level PPO scripts remain separate experiments; their checkpoints
are not compatible with this new observation/action interface.

## Timing and actions

Default physics is 200 Hz, MPC 50 Hz, and policy 4 Hz. A policy step advances
50 physics ticks. The global MPC phase persists across policy steps and
individual resets, giving alternating 13 and 12 MPC solves per policy step.

Four normalized actions are clipped to [-1,1]: gait (positive=trot, otherwise
crawl), stride residual, period slot (round(action+1)), and forward CoM residual.
The first three are requests consumed at the next safe cycle boundary. The
fourth drives a critically damped filter at 50 Hz, bounded to a 1 mm target,
so policy actions also affect control within the active cycle. Existing swing
schedules are retained. This is a feedback interface acceptance, not proof of
successful PPO learning or large-batch throughput.

## Observation and transition contract

The 57-D observation contains remaining distance and desired speed, projected
gravity, body angular velocity, joint displacement/velocity, last action,
body linear velocity, planned/measured contacts and phase summary, and MPC
health. See `go2_feedback.py` for field order and scaling. Measured velocity is
the default policy input. MPC velocity is explicitly a one-step prediction
initialized from measured state, not an independent state estimator. Both are
logged with their own timestamps and a prediction-valid flag. Switching to
`mpc_prediction` is a configuration option, not an estimator validation.

Each transition has [environment, episode, decision] identity. Automatic reset
returns the new episode observation; `info['transition']['bootstrap_observation']`
is the final pre-reset observation for value estimation. Natural termination
(success or failure) suppresses bootstrap. Time truncation keeps bootstrap but
cuts GAE recursion. `generalized_advantage` implements both masks. Do not use
the automatically reset observation as the terminal next value.

Explicit `reset_envs(ids)` returns `(observations, event)`. The collector must
mark the identified last transition as truncated, replace its next value with
that from the event's pre-reset observation, and cut recursion. A decision ID
of -1 means no action has occurred in that episode, so there is no transition
to modify. Reset clears only those environments' action requests, filtered
residuals, warm starts, contact schedules and episode counters.

## Server acceptance

```bash
conda activate limx
cd ~/workspace/Deep-Reinforcement-Learning-for-Guiding-Model-Predictive-Control-in-Legged-Manipulation
git switch go2-torch-integration
git pull --ff-only origin go2-torch-integration
export PYTHONPATH="$HOME/workspace/Crocoddyl-batched-mpc/src${PYTHONPATH:+:$PYTHONPATH}"
RUN="logs/go2_augmpc/check_$(date +%Y%m%d_%H%M%S)"
python -u scripts/check_go2_augmpc.py --headless --device cuda:0 --seconds 60 --output "$RUN"
python -m json.tool "$RUN/acceptance.json"
```

The acceptance executes at least 60 seconds of world simulation with eight
environments, command changes, natural termination, time truncation and two
individual resets during swing. It checks row-isolated MPC/cache responses,
physical peer continuity on reset, action/reward identity, finite outputs,
exact frequencies, and terminal-value semantics. Stance drift is reported as
a diagnostic; this test does not introduce a new locomotion drift threshold.
Pure helper tests: `python scripts/test_go2_feedback.py`.

## Local acceptance result (2026-11-11)

Eight Go2s completed 60 s in PhysX: 1,920 policy transitions, 3,000 MPC ticks,
12 natural terminations, 11 time truncations, and two explicit resets during
swing. Four target changes were exercised. All isolation, finite-value,
frequency and bootstrap assertions passed. Reported maximum stance drift was
5.2115 mm (diagnostic). Three feedback helper tests and five existing Stage 2
training-state tests passed. Extracted planner outputs matched the previous
planner for crawl/trot across all three period scales.

## Eight-environment PPO smoke acceptance

```bash
conda activate limx
(
  set -e
  cd ~/workspace/Deep-Reinforcement-Learning-for-Guiding-Model-Predictive-Control-in-Legged-Manipulation
  git switch go2-torch-integration
  git pull --ff-only origin go2-torch-integration
  export PYTHONPATH="$HOME/workspace/Crocoddyl-batched-mpc/src${PYTHONPATH:+:$PYTHONPATH}"
  RUN="logs/go2_augmpc/ppo_$(date +%Y%m%d_%H%M%S)"
  python -u scripts/check_go2_feedback_ppo.py --headless --device cuda:0 --steps 96 --output "$RUN"
  python -m json.tool "$RUN/acceptance.json"
)
```

This collects 96 steps per environment (768 transitions, 24 simulated seconds),
then performs ONE PPO update with four full-batch optimization epochs. The actor
uses categorical gait/period choices and tanh-transformed Gaussian stride/CoM
residuals. Likelihood includes the tanh Jacobian; latent samples are retained
rather than reconstructing them from clipped actions. The sampled categorical
choices map exactly to the environment's gait threshold and period slots.

The first three action components are pending requests, consumed only at safe
cycle boundaries, while the fourth affects ongoing control. PPO records requests
at their submission time; it does not claim all components execute immediately.

The acceptance checks rollout identities, executed actions, finite signals, and
termination masks. GAE uses pre-reset final observations for time-limit bootstrap
and cuts recursion for all episode boundaries. It is independently checked per
row. Environment 0 starts at an already-reached goal and environment 6 at a far
goal to guarantee coverage of natural termination and timeout; these are test
fixtures, not a training target distribution.

Outputs are `rollout.pt`, `feedback_0001.pt`, and `acceptance.json`. The checkpoint
stores model, Adam state, CPU/CUDA RNG state, iteration and interface dimensions.
Exact deterministic outputs after reload, an identical next Adam update, and
identical stochastic samples after RNG restoration are checked. Four real PhysX
steps then exercise the reloaded mean policy. The optimizer comparison is a
throwaway probe; the saved checkpoint remains iteration 1. This checkpoint does
not serialize PhysX state or in-flight episodes, so restarting collection must
reset the environment and begin a fresh rollout.

This acceptance is not a full-episode policy-vs-rule evaluation or evidence of
learned arrival behavior. Next, add matched-seed complete-episode evaluations
before starting longer training. Unit tests: `python scripts/test_go2_feedback_ppo.py`.

Local PPO acceptance (2026-11-11): 768 transitions / 24 s, 3 natural
terminations, 6 time truncations, zero failures. One update (4 epochs) changed
actor parameters by L2 0.048533 and deterministic actions by maximum 0.029177.
Loss was finite (2.030428). Model/optimizer/RNG reload checks and four post-reload
PhysX steps passed. Two probability/optimizer tests and three feedback helper
tests passed. These changes establish a working training pipeline, not measured
improvement in goal-reaching performance.

## Deterministic full-episode comparison

Run the evaluator on the actual iteration-1 checkpoint from the smoke test:

```bash
conda activate limx
(
  set -e
  cd ~/workspace/Deep-Reinforcement-Learning-for-Guiding-Model-Predictive-Control-in-Legged-Manipulation
  git switch go2-torch-integration
  git pull --ff-only origin go2-torch-integration
  export PYTHONPATH="$HOME/workspace/Crocoddyl-batched-mpc/src${PYTHONPATH:+:$PYTHONPATH}"
  CHECKPOINT=$(python -c 'from pathlib import Path; p=list(Path("logs/go2_augmpc").rglob("feedback_0001.pt")); assert p, "No feedback_0001.pt found; run the PPO smoke acceptance first"; print(max(p,key=lambda x:x.stat().st_mtime).resolve())')
  RUN="logs/go2_augmpc/evaluation_$(date +%Y%m%d_%H%M%S)"
  echo "Checkpoint: $CHECKPOINT"
  python -u scripts/evaluate_go2_feedback_ppo.py --headless --device cuda:0 \
    --checkpoint "$CHECKPOINT" --output "$RUN"
  python -m json.tool "$RUN/comparison.json"
)
```

Defaults: 32/40/48 mm targets, desired speed 0.006 m/s, 60 s timeout. Nine
independent robots run three goals for each of pre-update, post-update, and rule
controllers in the same simulation. Their local initial states are checked equal.
No action sampling occurs: categories use argmax and continuous actions use tanh
of Gaussian means. The rule uses trot when desired speed exceeds 0.0045 m/s and
remaining distance exceeds 30 mm, otherwise crawl; stride/period/residual inputs
are zero. It is a fixed reference controller, not a separately optimized baseline.

New smoke checkpoints include initial weights. Legacy iteration-1 checkpoints
reconstruct the seed-37 model and MUST match both log probabilities and critic
values in the sibling `rollout.pt`; missing/mismatching evidence aborts evaluation.
Keep the original checkpoint and rollout together when transferring them.

Each arm records exactly the first complete episode, using a physical snapshot
before automatic reset. Later auto-reset episodes of completed rows are excluded.
Stopping is independent of arrival: over the last 1 s, position span must be below
1 mm, body speed below 0.01 m/s, angular speed below 0.04 rad/s, joint speeds below
0.1 rad/s, and every foot load above 5 N. Arrival additionally requires endpoint
error at most 4 mm and no failure. Signed error, time to arrival, total-episode
mean forward speed, sampled body velocity, final speed, and maximum stance drift
are reported. Drift is the per-stance-anchor maximum retained at MPC frequency;
it is not a cumulative path-length measurement. The fixed target set gives a
small acceptance-set arrival rate, not a statistical generalization estimate.

`comparison.json` distinguishes evaluation completion, policy_all_arrived, and
improvement_demonstrated (every goal improves endpoint error by more than 0.1 mm).
A completed evaluation is not a performance PASS. `traces.json` stores sampled
physical signals. The script never changes the input checkpoint or trains a model.

Local full-episode result (2026-11-11, iteration 1): all nine episodes arrived
within 4 mm and stopped, with zero failures. Mean absolute endpoint errors were
1.3002 mm (pre), 1.3022 mm (post), and 1.1833 mm (rule). Maximum stance drift was
5.2726 / 5.2781 / 5.2696 mm respectively. Mean whole-episode forward speeds were
0.002239 / 0.002236 / 0.003139 m/s. Pre/post mean policies both selected crawl;
the rule used trot for the first cycle and crawl thereafter. Error changes were
-0.0414, -0.0336, +0.0689 mm improvements at 32, 40, 48 mm. Thus the pipeline and
complete-episode acceptance pass, but one PPO update has not demonstrated
performance improvement. This closes the feedback-environment/training-pipeline
milestone, not the learned-controller-performance milestone.

## Resume continuous PPO through iteration 12

`train_go2_feedback_ppo.py` accepts an ABSOLUTE `--total-iterations`. Starting
from iteration 1 performs 11 new updates. Each update defaults to 96 policy steps
in eight environments (768 transitions, four optimization epochs). It evaluates
1/4/8/12 with the unchanged 32/40/48 mm complete-episode evaluator. The initial
checkpoint is evaluated before any further training. The original untrained
policy remains the `pre` baseline throughout; `post` is the current milestone.

Training uses independent reset-time uniform sampling of 32/40/48 mm goals,
6 mm/s speed, and 60 s episode timeout. It deliberately replaces the smoke test's
zero/far termination fixtures and the old 64-100 mm defaults. This curriculum is
stored in checkpoints and checked on subsequent resumes. No claim is made that
12 updates suffice for learning or that a higher arrival rate is guaranteed.

Within a training segment, physical episodes persist across update boundaries.
Evaluation runs in a separate process after the worker exits and cannot change
its policy/optimizer/RNG. A new segment starts fresh physical episodes because
PhysX state is not checkpointed. Rollout-end values bootstrap normally and no
GAE recursion crosses into the fresh rollout. CPU/CUDA action RNG, Adam moments,
and the independent goal sampler RNG are restored. Individual in-segment resets
remain asynchronous. Evaluation never supplies samples to PPO.

Launch on the server (the output directory must not already exist):

```bash
conda activate limx
(
  set -e
  cd ~/workspace/Deep-Reinforcement-Learning-for-Guiding-Model-Predictive-Control-in-Legged-Manipulation
  git switch go2-torch-integration
  git pull --ff-only origin go2-torch-integration
  export PYTHONPATH="$HOME/workspace/Crocoddyl-batched-mpc/src${PYTHONPATH:+:$PYTHONPATH}"
  START=$(python -c 'from pathlib import Path; p=list(Path("logs/go2_augmpc").rglob("feedback_0001.pt")); assert p, "Missing iteration-1 checkpoint"; print(max(p,key=lambda x:x.stat().st_mtime).resolve())')
  RUN="$PWD/logs/go2_augmpc/pilot12_$(date +%Y%m%d_%H%M%S)"
  nohup bash -c '
    python -u scripts/train_go2_feedback_ppo.py --checkpoint "$1" --output "$2" \
      --total-iterations 12 --eval-iterations 1 4 8 12 --rollout-steps 96 --device cuda:0
    rc=$?
    echo "TRAIN_EXIT_CODE=$rc"
    exit "$rc"
  ' _ "$START" "$RUN" > "$RUN.log" 2>&1 < /dev/null &
  echo "PID=$!"
  echo "RUN=$RUN"
  echo "LOG=$RUN.log"
)
```

Inspect the newest run without a placeholder path:

```bash
cd ~/workspace/Deep-Reinforcement-Learning-for-Guiding-Model-Predictive-Control-in-Legged-Manipulation
RUN=$(python -c 'from pathlib import Path; p=list(Path("logs/go2_augmpc").glob("pilot12_*/run.json")); assert p; print(max(p,key=lambda x:x.stat().st_mtime).parent.resolve())')
grep -E '"stage": "(ppo_update|milestone|complete)"|TRAIN_EXIT_CODE=|Traceback|RuntimeError:' "$RUN.log" | tail -15
python scripts/report_go2_feedback_training.py "$RUN" --require-complete
```

The final report checks all planned milestone reports exist, then lists gait
sequences, endpoint errors, mean speeds, arrival times, stopping and drift for
all controllers and goals. Completion is separate from performance success.
Check `run.json`, `milestones.json`, `eval_0001/comparison.json`, etc. Training
emits effective transitions/s and counts of successful/failed/timed-out episodes.
Every completed update is atomically saved as `feedback_NNNN.pt`, not just the
milestones. An interrupted update can be restarted from the last complete file.

For an interrupted run, choose its last checkpoint and a NEW output directory;
keep the same rollout length. Resume evaluates the starting checkpoint and the
remaining scheduled milestones; older reports remain in the previous directory.
For example, with `RUN` still pointing to the interrupted run:

```bash
export PYTHONPATH="$HOME/workspace/Crocoddyl-batched-mpc/src${PYTHONPATH:+:$PYTHONPATH}"
START=$(python -c 'from pathlib import Path; import sys; p=list(Path(sys.argv[1]).glob("feedback_*.pt")); assert p; print(max(p,key=lambda x:int(x.stem.split("_")[-1])).resolve())' "$RUN")
NEW="${RUN}_resume_$(date +%Y%m%d_%H%M%S)"
nohup bash -c '
  python -u scripts/train_go2_feedback_ppo.py --checkpoint "$1" --output "$2" \
    --total-iterations 12 --eval-iterations 1 4 8 12 --rollout-steps 96 --device cuda:0
  rc=$?; echo "TRAIN_EXIT_CODE=$rc"; exit "$rc"
' _ "$START" "$NEW" > "$NEW.log" 2>&1 < /dev/null &
echo "RUN=$NEW"
```

Regression tests: `python scripts/test_go2_feedback_training.py` and
`python scripts/test_go2_feedback_ppo.py`. A changed curriculum or rollout length
is rejected on resume rather than silently changing the experiment.

Local continuation validation (2026-11-11): real PhysX resumed the original
iteration-1 checkpoint, performed updates 2 and 3, evaluated complete episodes
at 3, restarted the worker from 3, updated to 4, and evaluated complete episodes
again. The reduced test rollout length was 8 steps per update, not the production
96-step setting. All nine episodes at each evaluation arrived and stopped with
zero failures. Real Adam step counters were 4/8/12/16 at iterations 1/2/3/4, and
goal RNG reload matched exactly. The completion report command passed. Three
schedule/configuration regression tests and two PPO probability/optimizer tests
passed. Full 12-round training is to be run on the server; this local smoke run
is not evidence of learning improvement or a completed 12-round experiment.
