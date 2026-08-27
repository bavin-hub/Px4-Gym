# Starling 2 Max PX4 Isaac Lab Port

This package contains the six Starling 2 Max Isaac Lab/RL-Games tasks used by
PX4-Gym. The attitude, direct-rate, and navigation environments use a batched
Torch port of the RL-PX4/PX4 cascade so vectorized training stays on the GPU.

Install and run from this directory:

```bash
"${ISAACSIM_PYTHON}" -m pip install --editable .

PYTHONEXE="${ISAACLAB_PYTHON}" "${ISAACSIM_PYTHON}" \
  scripts/train_starling2max_attitude_delta.py \
  --headless --device cuda:0 --num-envs 4096
```

## Tasks

- `AerialIsaac-Starling2Max-Velocity-v0`
- `AerialIsaac-Starling2Max-Position-v0`
- `AerialIsaac-Starling2Max-AttitudeDelta-v0`
- `AerialIsaac-Starling2Max-Rates-v0`
- `AerialIsaac-Starling2Max-Navigation-v0`
- `AerialIsaac-Starling2Max-DepthAttitudeDelta-v0`

## Important Files

- `aerial_isaac_lab/core/px4_attitude_rate_control.py`: Torch attitude/rate loop.
- `aerial_isaac_lab/core/px4_velocity_control.py`: Torch velocity outer loop.
- `aerial_isaac_lab/core/motor_px4.py`: allocation and motor dynamics.
- `config/allocation_starling2max.yaml`: vehicle geometry and actuator constants.
- `aerial_isaac_lab/perception/`: DCE implementation and frozen weights.
- `scripts/`: task-specific training/evaluation and depth smoke tests.

Use the repository-level guides for complete commands:

- [`docs/isaac_lab_training_eval.md`](../../docs/isaac_lab_training_eval.md)
- [`docs/starling2max_rlpx4.md`](../../docs/starling2max_rlpx4.md)
- [`docs/px4_sitl_policy_test.md`](../../docs/px4_sitl_policy_test.md)

The root C++/pybind package is retained for controller experiments, but this
training package does not import it at runtime.
