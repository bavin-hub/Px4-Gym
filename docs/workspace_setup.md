# Workspace Setup

This guide installs the complete Starling 2 Max training and SITL evaluation
stack used by the `s2m_rlpx4` branch.

## Clone Px4-Gym

```bash
git clone --branch s2m_rlpx4 https://github.com/bavin-hub/Px4-Gym.git
cd Px4-Gym
export PX4_GYM_ROOT="$(pwd)"
```

Keep `PX4_GYM_ROOT` set in every terminal used below. Commands use repository
relative paths and do not assume a particular home-directory layout.

## Isaac Sim and Isaac Lab

Install Isaac Sim 5.1.0 and export its launchers:

```bash
export ISAACSIM_PATH="${HOME}/isaacsim"
export ISAACSIM_PYTHON="${ISAACSIM_PATH}/python.sh"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PYTHON}"
```

Verify the installation:

```bash
"${ISAACSIM_PATH}/isaac-sim.sh" --help
"${ISAACSIM_PYTHON}" -c "print('Isaac Sim Python is ready')"
```

Install Isaac Lab with RL-Games support:

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
ln -s "${ISAACSIM_PATH}" _isaac_sim
./isaaclab.sh --install rl_games
```

Install the Starling task package into the Python environment used to launch
Isaac Lab:

```bash
cd "${PX4_GYM_ROOT}/rlPx4Controller/starling2max_px4_isaac_port"
"${ISAACSIM_PYTHON}" -m pip install --editable .
```

The root C++ bindings are optional. If they are needed for separate controller
experiments, install Eigen and the package in the desired Python environment:

```bash
sudo apt install libeigen3-dev
python3 -m pip install --editable "${PX4_GYM_ROOT}/rlPx4Controller"
```

## Pegasus Simulator

Pegasus v5.1.0 is required for PX4 SITL evaluation inside Isaac Sim 5.1.0. Use
the official [Pegasus installation guide](https://pegasussimulator.github.io/PegasusSimulator/source/setup/installation.html),
including its `isaac_run` shell function, then install the extension as a
library in Isaac Sim's Python:

```bash
git clone --branch v5.1.0 https://github.com/PegasusSimulator/PegasusSimulator.git
cd PegasusSimulator/extensions
"${ISAACSIM_PYTHON}" -m pip install --editable pegasus.simulator
```

Open a new terminal after defining `isaac_run`, then verify:

```bash
isaac_run --help
"${ISAACSIM_PYTHON}" -c "import pegasus.simulator; print('Pegasus is ready')"
```

The standalone evaluator requires a Starling 2 Max USD with articulation prims
named `body`, `rotor0`, `rotor1`, `rotor2`, and `rotor3`. Export the included
Starling URDF through Isaac Sim's URDF importer, verify those prim names, and
save the result outside Git or as:

```text
sitl/isaac/starling2max.usd
```

The evaluator accepts the file through `--vehicle-usd`; it no longer depends on
Pegasus' machine-local `ROBOTS["Modal"]` entry.

## PX4 and SITL Assets

Clone and build the tested PX4 version:

```bash
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
git checkout v1.14.3
git submodule update --init --recursive
export PX4_AUTOPILOT_ROOT="$(pwd)"
make px4_sitl_default none
```

Install the checked-in Starling Gazebo assets:

```bash
cp -a "${PX4_GYM_ROOT}/sitl/gz/starling2max" \
  "${PX4_AUTOPILOT_ROOT}/Tools/simulation/gz/models/"
cp -a "${PX4_GYM_ROOT}/sitl/gz/starling2max_depth" \
  "${PX4_AUTOPILOT_ROOT}/Tools/simulation/gz/models/"
cp "${PX4_GYM_ROOT}/sitl/gz/default_trees.sdf" \
  "${PX4_AUTOPILOT_ROOT}/Tools/simulation/gz/worlds/"
cp "${PX4_GYM_ROOT}/rlPx4Controller/starling2max_px4_isaac_port/assets/robots/starling2max/4007_gz_starling2max" \
  "${PX4_AUTOPILOT_ROOT}/ROMFS/px4fmu_common/init.d-posix/airframes/"
cp "${PX4_GYM_ROOT}/sitl/gz/4008_gz_starling2max_depth" \
  "${PX4_AUTOPILOT_ROOT}/ROMFS/px4fmu_common/init.d-posix/airframes/"
cp "${PX4_GYM_ROOT}/sitl/isaac/10050_pegasus_starling2max" \
  "${PX4_AUTOPILOT_ROOT}/ROMFS/px4fmu_common/init.d-posix/airframes/"
```

Add `4007_gz_starling2max`, `4008_gz_starling2max_depth`, and
`10050_pegasus_starling2max` to the
`px4_add_romfs_files(...)` list in
`ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt`, then rebuild:

```bash
cd "${PX4_AUTOPILOT_ROOT}"
make px4_sitl_default
```

Install the repository DDS map and rebuild whenever it changes:

```bash
cp "${PX4_GYM_ROOT}/dds_topics.yaml" \
  "${PX4_AUTOPILOT_ROOT}/src/modules/uxrce_dds_client/dds_topics.yaml"
cd "${PX4_AUTOPILOT_ROOT}"
make px4_sitl_default
```

## Micro XRCE-DDS Agent

```bash
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
```

The agent is started during evaluation with:

```bash
MicroXRCEAgent udp4 -p 8888
```

## ROS 2 Workspace

Use ROS 2 Humble on Ubuntu 22.04. Add the matching PX4 message repositories to
the checked-in workspace:

```bash
cd "${PX4_GYM_ROOT}/ros_ws/src"
git clone --branch release/1.14 https://github.com/PX4/px4_msgs.git
git clone --branch release/v1.14 https://github.com/PX4/px4_ros_com.git
```

Install Python runtime dependencies in the Python used by ROS 2:

```bash
python3 -m pip install --user numpy pillow PyYAML rl-games
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cu128
```

Build and source the workspace:

```bash
cd "${PX4_GYM_ROOT}/ros_ws"
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro humble -y
colcon build --symlink-install
source install/setup.bash
```

For Gazebo depth evaluation, also install the ROS-Gazebo image bridge matching
ROS Humble and Gazebo Garden. Pegasus publishes `/starling/raw_depth` directly
through Isaac Sim's ROS 2 bridge.

Continue with [training](isaac_lab_training_eval.md) or
[SITL evaluation](px4_sitl_policy_test.md).
