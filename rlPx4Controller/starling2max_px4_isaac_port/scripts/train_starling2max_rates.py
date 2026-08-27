"""Train/evaluate the Starling 2 Max direct body-rate task with RL-Games."""

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
    "--config",
    type=Path,
    default=ROOT / "config/rl_games_starling2max_rates.yaml",
)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher

simulation_app = AppLauncher({"headless": args.headless, "device": args.device}).app

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from aerial_isaac_lab.simpleflight_rl_games import register_simpleflight_rl_games
from aerial_isaac_lab.tasks import STARLING2MAX_RATES_TASK_ID, Starling2MaxRatesEnvCfg

register_simpleflight_rl_games()

with args.config.open() as stream:
    agent_cfg = yaml.safe_load(stream)
env_cfg = Starling2MaxRatesEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = args.device
env = gym.make(STARLING2MAX_RATES_TASK_ID, cfg=env_cfg)
# clip_actions is inf because the task applies SimpleFlight's tanh squash in
# _pre_physics_step; a finite bound here would clamp the sample before tanh.
env = RlGamesVecEnvWrapper(env, rl_device=args.device, clip_obs=float("inf"), clip_actions=float("inf"))
vecenv.register(
    "AerialIsaacRlg",
    lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
)
env_configurations.register(
    "rlgpu",
    {"vecenv_type": "AerialIsaacRlg", "env_creator": lambda **kwargs: env},
)
agent_cfg["params"]["config"]["num_actors"] = args.num_envs
rlg_cfg = agent_cfg["params"]["config"]
rlg_batch = args.num_envs * rlg_cfg["horizon_length"]
num_minibatches = rlg_cfg.pop("simpleflight_num_minibatches", 16)
if rlg_batch % num_minibatches != 0:
    raise ValueError(
        f"rollout batch {rlg_batch} must be divisible by {num_minibatches} minibatches"
    )
rlg_cfg["minibatch_size"] = rlg_batch // num_minibatches
rlg_cfg["device"] = args.device
rlg_cfg["device_name"] = args.device
if args.max_epochs is not None:
    rlg_cfg["max_epochs"] = args.max_epochs
if args.checkpoint:
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint

runner = Runner(IsaacAlgoObserver())
runner.load(agent_cfg)
runner.reset()
runner.run({"train": not args.play, "play": args.play, "checkpoint": args.checkpoint})
env.close()
simulation_app.close()
