# Isaac Lab Training and Evaluation

Run all commands from the isolated Starling package:

```bash
cd "${PX4_GYM_ROOT}/rlPx4Controller/starling2max_px4_isaac_port"
```

If Isaac Lab uses a dedicated virtual environment, launch through Isaac Sim
with `PYTHONEXE` set to that environment. Otherwise replace the two-line prefix
below with `"${ISAACSIM_PYTHON}"`.

```bash
export ISAACLAB_PYTHON="/path/to/IsaacLab/environment/bin/python"
```

## Registered Tasks

| Script | Task ID | Observation | Policy rate |
|---|---|---:|---:|
| `train_starling2max_velocity.py` | `AerialIsaac-Starling2Max-Velocity-v0` | 18 | 100 Hz |
| `train_starling2max_position.py` | `AerialIsaac-Starling2Max-Position-v0` | 18 | 100 Hz |
| `train_starling2max_attitude_delta.py` | `AerialIsaac-Starling2Max-AttitudeDelta-v0` | 98 | 100 Hz |
| `train_starling2max_rates.py` | `AerialIsaac-Starling2Max-Rates-v0` | 98 | 100 Hz |
| `train_starling2max_navigation.py` | `AerialIsaac-Starling2Max-Navigation-v0` | 81 | 10 Hz |
| `train_starling2max_depth_attitude_delta.py` | `AerialIsaac-Starling2Max-DepthAttitudeDelta-v0` | 81 | 50 Hz |

The attitude, rate, and depth tasks use the Torch-native RL-PX4 cascade. The
position and velocity waypoint tasks retain the Lee controller path for their
existing policy contracts.

## Smoke Tests

The depth smoke tests validate camera geometry, normalization, frozen DCE
weights, latent size, controller stepping, and observation shape:

```bash
PYTHONEXE="${ISAACLAB_PYTHON}" "${ISAACSIM_PYTHON}" \
  scripts/smoke_starling2max_navigation.py \
  --headless --device cuda:0 --num-envs 4 --dump-images /tmp/starling_nav

PYTHONEXE="${ISAACLAB_PYTHON}" "${ISAACSIM_PYTHON}" \
  scripts/smoke_starling2max_depth_attitude_delta.py \
  --headless --device cuda:0 --num-envs 4
```

Run the controller unit tests with the environment that has Torch and pytest:

```bash
python -m pytest tests
```

## Training

State-only tasks normally use more parallel environments:

```bash
PYTHONEXE="${ISAACLAB_PYTHON}" "${ISAACSIM_PYTHON}" \
  scripts/train_starling2max_velocity.py --headless --device cuda:0 --num-envs 4096

PYTHONEXE="${ISAACLAB_PYTHON}" "${ISAACSIM_PYTHON}" \
  scripts/train_starling2max_position.py --headless --device cuda:0 --num-envs 4096

PYTHONEXE="${ISAACLAB_PYTHON}" "${ISAACSIM_PYTHON}" \
  scripts/train_starling2max_attitude_delta.py --headless --device cuda:0 --num-envs 4096

PYTHONEXE="${ISAACLAB_PYTHON}" "${ISAACSIM_PYTHON}" \
  scripts/train_starling2max_rates.py --headless --device cuda:0 --num-envs 4096
```

Depth rendering and encoding are heavier; start with 256 environments:

```bash
PYTHONEXE="${ISAACLAB_PYTHON}" "${ISAACSIM_PYTHON}" \
  scripts/train_starling2max_navigation.py --headless --device cuda:0 --num-envs 256

PYTHONEXE="${ISAACLAB_PYTHON}" "${ISAACSIM_PYTHON}" \
  scripts/train_starling2max_depth_attitude_delta.py \
  --headless --device cuda:0 --num-envs 256
```

Use `--max-epochs` and fewer environments for short debug runs. Checkpoints and
summaries are written below `runs/` using the RL-Games experiment name.

## Evaluation

Every training script also evaluates with `--play`, one environment, and an
absolute checkpoint path. For example:

```bash
PYTHONEXE="${ISAACLAB_PYTHON}" "${ISAACSIM_PYTHON}" \
  scripts/train_starling2max_rates.py \
  --play --device cuda:0 --num-envs 1 \
  --checkpoint /absolute/path/to/starling2max_rates.pth
```

For either depth task, fix the initial obstacle count and select a camera view:

```bash
PYTHONEXE="${ISAACLAB_PYTHON}" "${ISAACSIM_PYTHON}" \
  scripts/train_starling2max_navigation.py \
  --play --device cuda:0 --num-envs 1 \
  --curriculum-level 45 --view follow \
  --checkpoint /absolute/path/to/starling2max_navigation.pth
```

Replace the script and checkpoint with the matching task. Never mix task YAML,
checkpoint, observation size, or deployment entry point.

## RL-Games Configs

All six configs live in `config/` and follow the script names:

```text
rl_games_starling2max_velocity.yaml
rl_games_starling2max_position.yaml
rl_games_starling2max_attitude_delta.yaml
rl_games_starling2max_rates.yaml
rl_games_starling2max_navigation.yaml
rl_games_starling2max_depth_attitude_delta.yaml
```

Each training script accepts `--config` for an alternate YAML without replacing
the checked-in default.
