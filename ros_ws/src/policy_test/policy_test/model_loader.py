from pathlib import Path

import torch
import yaml
from rl_games.algos_torch import model_builder


def load_policy_model(
    *,
    policy_config: str,
    checkpoint: str,
    device: str,
    actions_num: int,
    input_shape: int,
):
    """Load an RL-Games policy from user-provided config and checkpoint paths."""
    if not policy_config:
        raise ValueError("Missing required ROS parameter: policy_config")
    if not checkpoint:
        raise ValueError("Missing required ROS parameter: checkpoint")

    policy_config_path = Path(policy_config).expanduser()
    checkpoint_path = Path(checkpoint).expanduser()

    if not policy_config_path.is_file():
        raise FileNotFoundError(f"Policy config not found: {policy_config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")

    with policy_config_path.open() as f:
        params = yaml.safe_load(f)["params"]

    network = model_builder.ModelBuilder().load(params)
    config = params["config"]

    model = network.build({
        "actions_num": actions_num,
        "input_shape": (input_shape,),
        "num_seqs": 1,
        "value_size": 1,
        "normalize_value": config.get("normalize_value", False),
        "normalize_input": config.get("normalize_input", False),
    })

    checkpoint_data = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    weights = {
        k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
        for k, v in checkpoint_data["model"].items()
    }

    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    return model
