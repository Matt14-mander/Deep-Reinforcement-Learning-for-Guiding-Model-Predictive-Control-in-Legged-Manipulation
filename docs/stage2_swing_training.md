# Stage 2 gait/swing modulation

Stage 2 keeps the 12 CoM Bezier actions and adds three gait modifiers:

1. `step_length`: scales horizontal lift-off to landing displacement.
2. `step_height`: scales the Bezier swing-trajectory apex.
3. `step_frequency`: inversely scales swing and support durations.

The resulting 15D action is bounded, exponentially smoothed and rate-limited
before it reaches Crocoddyl. The applied three gait modifiers are appended to
the policy observation, so Stage 2 uses 48 observations instead of 45.

Stage 1 remains selected by default. Stage 2 is enabled only with `--stage2`.

## Local dependency-light tests

```bash
python scripts/test_stage2_modulation.py
python scripts/test_mpc_cluster_local.py
```

## AutoDL smoke test

Use the environment variables from `docs/autodl_dual_env_run.md`, then run:

```bash
cd "$PROJECT_ROOT"
OMNI_KIT_ACCEPT_EULA=YES "$ISAAC_ENV/bin/python" scripts/train_quadruped_mpc.py \
  --headless \
  --stage2 \
  --num_envs 4 \
  --max_iterations 2 \
  --use_mpc_cluster \
  --cluster_workers 2 \
  --mpc_python "$MPC_ENV/bin/python" \
  --robot_urdf "$GO2_URDF"
```

Expected startup values:

- `Training stage: Stage 2`
- `Observation dim: 48`
- `Action dim: 15`
- `RL policy rate: 50 Hz`

The first Stage 2 smoke run should start from a new policy. A Stage 1 actor has
45 input features and 12 outputs, while Stage 2 has 48 inputs and 15 outputs;
loading the old checkpoint directly is therefore not shape-compatible. Weight
migration or teacher/distillation initialization is a separate training step.

## Acceptance checks

- All MPC workers reach `ready`.
- PPO completes two iterations without shape or observation-group errors.
- MPC barrier does not time out.
- Guard failures do not occur continuously.
- Changing each gait modifier independently changes the corresponding planned
  foothold displacement, swing height, or contact timing.

Strict 5 Hz policy / 50 Hz MPC execution is not implemented by merely ignoring
nine actions. It requires one RL environment step to execute ten MPC updates;
until that refactor, Stage 2 consumes actions at 50 Hz and uses the safety filter.
