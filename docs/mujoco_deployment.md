# MuJoCo deployment of trained Stage 1/2 policies

The RSL-RL checkpoint does **not** need to be converted before Python-based
MuJoCo evaluation. The deployment runner loads `model_*.pt` directly. ONNX and
TorchScript export are optional packaging and cross-runtime checks.

The policy is a high-level controller, not a joint-torque policy. Keep the full
chain when evaluating it:

```text
MuJoCo state -> 45D/48D observation -> normalized actor -> 12D/15D action
  -> Bezier CoM reference + gait modulation -> Crocoddyl MPC
  -> desired joint position + feedforward torque -> PD + feedforward -> MuJoCo
```

## 1. Required files

- A Stage 1 checkpoint (`45 -> 12`) or Stage 2 checkpoint (`48 -> 15`).
- A Go2 MuJoCo `scene.xml`, for example `unitree_go2/scene.xml` from MuJoCo
  Menagerie.
- The same Go2 URDF used by Pinocchio/Crocoddyl during training.

MJCF drives MuJoCo physics; URDF builds the MPC model. They are not substitutes
for each other. Joint names, signs, base orientation and link inertias must be
checked before interpreting transfer results.

## 2. Prepare the MPC environment

Run deployment in `rlbmpc_mpc`, not in the Isaac Sim environment:

```bash
export RLBMPC_ROOT=/root/autodl-tmp/rlbmpc_workspace
export MPC_ENV="$RLBMPC_ROOT/envs/rlbmpc_mpc"
export PROJECT="$RLBMPC_ROOT/RL_Bezier_MPC"

conda activate "$MPC_ENV"
python -m pip install "mujoco==3.2.7"

# The native .pt runner also needs Torch. Install a CPU wheel if this clean MPC
# environment does not already contain torch; a GPU Torch build is unnecessary.
python -c "import torch" || \
  python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.7.0"

python -c "import torch, mujoco, pinocchio, crocoddyl; print('deployment imports OK')"
```

The environment must already import compatible Pinocchio and Crocoddyl builds.
Do not repair the old mixed NumPy environment in place if it still segfaults;
use the clean, verified MPC environment.

## 3. Validate the checkpoint contract

Stage 2:

```bash
cd "$PROJECT"
python scripts/mujoco_deploy/validate_isaac_mujoco_io.py \
  --stage 2 \
  --checkpoint /path/to/model_XXXX.pt
```

The command must report `obs=48`, `action=15`, finite output, and the intended
normalizer keys. A `45 -> 12` model is Stage 1 and must use `--stage 1`.

For numerical parity, save an Isaac-side sample first:

```bash
conda activate "$RLBMPC_ROOT/envs/rlbmpc45"
cd "$PROJECT"
python scripts/play_quadruped_mpc.py \
  --stage2 --headless --num_envs 1 --num_episodes 1 --max_steps 1 \
  --checkpoint /path/to/model_XXXX.pt \
  --policy_io_sample /tmp/isaac_policy_sample.npz
```

Then compare it in the MPC environment:

```bash
python scripts/mujoco_deploy/validate_isaac_mujoco_io.py \
  --stage 2 \
  --checkpoint /path/to/model_XXXX.pt \
  --sample /path/to/isaac_policy_sample.npz
```

## 4. Run MuJoCo

First run headless for 5 seconds:

```bash
python scripts/mujoco_deploy/go2_mujoco_runner.py \
  --stage 2 \
  --checkpoint /path/to/model_XXXX.pt \
  --mjcf /path/to/mujoco_menagerie/unitree_go2/scene.xml \
  --urdf /path/to/go2_description.urdf \
  --duration 5 \
  --headless
```

Then remove `--headless` to open the MuJoCo viewer on a machine/session with a
working display server. A normal AutoDL SSH terminal generally needs headless
mode. The default target is `[1, 0, 0]`; change it with `--target X Y Z`.

The runner deliberately bypasses MJCF actuator gains and applies the computed
PD-plus-feedforward torque through `qfrc_applied`. This makes the control law
independent of whether the chosen MJCF defines motor, position or general
actuators.

The default `--contact-mode training` reproduces the current Isaac task's
temporary root-height contact observation. `--contact-mode height` uses named
foot-site heights, but changes the observation distribution and should only be
used after training is changed to use real contact sensors too.

## 5. Optional export

The exported graph includes observation normalization and the actor. It does
not include action denormalization, gait filtering, Bezier generation or MPC.

```bash
python scripts/mujoco_deploy/export_policy.py \
  --stage 2 \
  --checkpoint /path/to/model_XXXX.pt \
  --output /path/to/export/stage2_policy \
  --format both
```

This creates:

- `stage2_policy.ts`
- `stage2_policy.onnx`
- `stage2_policy.json`

If `onnxruntime` is installed, verify the exported output:

```bash
python scripts/mujoco_deploy/validate_isaac_mujoco_io.py \
  --stage 2 \
  --checkpoint /path/to/model_XXXX.pt \
  --onnx /path/to/export/stage2_policy.onnx
```

## 6. Acceptance order

1. Checkpoint dimensions and normalizer are correct.
2. Initial MuJoCo pose and joint order match Isaac/Pinocchio.
3. A 5-second headless run has finite state and limited MPC guard events.
4. The robot can stand before judging walking quality.
5. Compare trajectories, joint states, torques and policy I/O between simulators.
6. Only then tune contact/friction, PD gains or apply domain randomization.

MuJoCo success is a simulator-to-simulator transfer test, not proof of safe
real-robot deployment. Real hardware additionally needs state estimation,
latency handling, command watchdogs, torque/rate limits and an emergency stop.
