# Px4-Gym

[![PX4](https://img.shields.io/badge/PX4-v1.14.3-silver.svg)](https://github.com/PX4/PX4-Autopilot/releases/tag/v1.14.3)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-main-silver.svg)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-ubuntu--22.04-orange.svg)](https://releases.ubuntu.com/22.04/)
[![GPU](https://img.shields.io/badge/GPU-RTX%205090-green.svg)](https://www.nvidia.com/)

PX4-Gym contains reinforcement-learning training, evaluation, and SITL policy
testing pipelines for PX4-compatible quadrotors.  The Isaac Lab side trains and
evaluates policies at multiple control levels, and the ROS 2/PX4 side runs the
trained policies against PX4 SITL in Gazebo.

The repository is organized around a sim-to-SITL workflow: train a policy in
Isaac Lab, evaluate it in the same task wrapper, then test the exported policy
through PX4 offboard interfaces in Gazebo.  The current Isaac Lab tasks target
the X500 quadrotor and keep the deployment contract close to PX4 by preserving
the policy observation/action semantics used by the ROS 2 policy-test nodes.
The low-level controller, motor model, allocation, frame conversions, and task
logic are kept explicit so that policies trained in Isaac Lab can be tested
against PX4 with minimal glue code.

The available task set covers position, velocity, and attitude-delta policy
interfaces.  Each task has its own RL-Games config, smoke test, training script,
and evaluation path.  The rewards for all tasks were tuned heavily so the
trained policies remain stable when moved from Isaac Lab evaluation into PX4
SITL.  A few tasks may still require further reward tuning depending on the
vehicle model, command envelope, and deployment setup.

Known working versions and tested platform:

- PX4: `v1.14.3`
- Isaac Sim: `5.1.0`
- Isaac Lab: `main`
- PX4 ROS message/bridge repositories: PX4 `1.14` release branches
- Python: `3.11`
- Ubuntu: `22.04`
- GPU: NVIDIA RTX `5090`

## Documentation

Start with the setup guide, then use the workflow-specific guide you need:

- [Workspace setup](docs/workspace_setup.md): install Isaac Sim, Isaac Lab,
  PX4, Micro XRCE-DDS Agent, and the ROS 2 policy-test workspace.
- [Isaac Lab training and evaluation](docs/isaac_lab_training_eval.md): run
  the X500 position, velocity, and attitude-delta training/evaluation scripts.
- [PX4 SITL policy testing](docs/px4_sitl_policy_test.md): run a trained
  policy through PX4 `v1.14.3`, Micro XRCE-DDS, ROS 2, and Gazebo.
- [Aerial Isaac Lab notes](docs/aerial_isaac.md): implementation details for
  the Aerial Gym task/controller port and policy observation contract.

## Results

X500 attitude policy tested with PX4 SITL:

[![X500 attitude policy tested with PX4 SITL](results/x500_attitude.gif)](results/x500_attitude.gif)

Regenerate the GIF from the source MP4 with:

```bash
scripts/mp4_to_gif.sh results/x500_attitude.mp4 results/x500_attitude.gif
```

## TODO

- Add training pipelines for more control levels:
  - thrust and body rates
  - thrust and torques
- Add an EKF path to simulate real-time sensor noise.
- Add fixed-wing support.

## Acknowledgement

The low-level geometric controller used by these policies is ported from
[ntnu-arl/aerial_gym_simulator](https://github.com/ntnu-arl/aerial_gym_simulator).
Thank you to the Aerial Gym Simulator authors for releasing the controller and
simulation work that this Isaac Lab port builds on.
