"""Train/evaluate the Starling 2 Max depth attitude-delta navigation task.

The Deep Collision Encoder is frozen and lives inside the environment, so the
81-D observation already carries the 64-D depth latent when rl_games sees it.
Only the MLP+GRU actor-critic is trained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--max-epochs", type=int, default=None)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--play", action="store_true")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument(
    "--curriculum-level",
    type=int,
    default=None,
    help="Override the starting obstacle count (default: curriculum min_level, 45).",
)
parser.add_argument(
    "--view",
    choices=("follow", "world"),
    default="follow",
    help="follow: chase camera on the drone (default). world: static overview of the box.",
)
parser.add_argument(
    "--view-env",
    type=int,
    default=0,
    help="Which environment the viewport camera tracks.",
)
parser.add_argument(
    "--config",
    type=Path,
    default=ROOT / "config/rl_games_starling2max_depth_attitude_delta.yaml",
)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher

simulation_app = AppLauncher({"headless": args.headless, "device": args.device}).app

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from aerial_isaac_lab.tasks import (
    STARLING2MAX_DEPTH_ATTITUDE_DELTA_TASK_ID,
    Starling2MaxDepthAttitudeDeltaEnvCfg,
)

class CurriculumAlgoObserver(IsaacAlgoObserver):
    """IsaacAlgoObserver that reports the curriculum level on each new best reward.

    rl_games prints ``saving next best rewards: [...]`` from inside
    ``a2c_common`` whenever the mean reward improves, and saves the checkpoint at
    the same moment.  There is no hook on that line, so this observer watches
    ``algo.last_mean_rewards`` (which rl_games updates immediately after the save)
    and prints the obstacle count alongside it.
    """

    def __init__(self, env):
        super().__init__()
        self._env = env
        self._best = None

    def after_print_stats(self, frame, epoch_num, total_time):
        super().after_print_stats(frame, epoch_num, total_time)
        best = getattr(self.algo, "last_mean_rewards", None)
        if best is None:
            return
        if self._best is None or best > self._best:
            self._best = best
            task = self._env.unwrapped
            print(
                f"  -> new best reward {best:.3f} | curriculum level "
                f"{task.curriculum_level} (active obstacles) | "
                f"progress fraction {task._curriculum_progress_fraction:.3f}"
            )


with args.config.open() as stream:
    agent_cfg = yaml.safe_load(stream)
env_cfg = Starling2MaxDepthAttitudeDeltaEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = args.device
if args.curriculum_level is not None:
    env_cfg.curriculum.min_level = args.curriculum_level
env_cfg.viewer.env_index = args.view_env
if args.view == "world":
    env_cfg.follow_viewer = False
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.eye = (-8.0, -14.0, 8.0)
    env_cfg.viewer.lookat = (4.0, 0.0, 0.0)
env = gym.make(STARLING2MAX_DEPTH_ATTITUDE_DELTA_TASK_ID, cfg=env_cfg)
env = RlGamesVecEnvWrapper(env, rl_device=args.device, clip_obs=float("inf"), clip_actions=1.0)
vecenv.register(
    "AerialIsaacRlg",
    lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
)
env_configurations.register(
    "rlgpu",
    {"vecenv_type": "AerialIsaacRlg", "env_creator": lambda **kwargs: env},
)
agent_cfg["params"]["config"]["num_actors"] = args.num_envs
agent_cfg["params"]["config"]["device"] = args.device
agent_cfg["params"]["config"]["device_name"] = args.device
if args.max_epochs is not None:
    agent_cfg["params"]["config"]["max_epochs"] = args.max_epochs
if args.checkpoint:
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint

runner = Runner(CurriculumAlgoObserver(env))
runner.load(agent_cfg)
runner.reset()
runner.run({"train": not args.play, "play": args.play, "checkpoint": args.checkpoint})
env.close()
simulation_app.close()
