"""RL-Games adapters for the SimpleFlight PPO network and critic loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import A2CBuilder
from rl_games.common import common_losses


class SimpleFlightA2CBuilder(A2CBuilder):
    """Build separate actor/critic MLPs with SimpleFlight-style LayerNorm."""

    class Network(A2CBuilder.Network):
        def _build_mlp(
            self,
            input_size,
            units,
            activation,
            dense_func,
            norm_only_first_layer=False,
            norm_func_name=None,
            d2rl=False,
        ):
            if d2rl:
                return super()._build_mlp(
                    input_size,
                    units,
                    activation,
                    dense_func,
                    norm_only_first_layer,
                    norm_func_name,
                    d2rl,
                )

            layers = []
            if norm_func_name == "layer_norm":
                layers.append(torch.nn.LayerNorm(input_size))
            in_size = input_size
            for unit in units:
                layers.append(dense_func(in_size, unit))
                layers.append(self.activations_factory.create(activation))
                if norm_func_name == "layer_norm":
                    layers.append(torch.nn.LayerNorm(unit))
                elif norm_func_name == "batch_norm":
                    layers.append(torch.nn.BatchNorm1d(unit))
                in_size = unit
            return torch.nn.Sequential(*layers)

    def build(self, name, **kwargs):
        return self.Network(self.params, **kwargs)


def _simpleflight_critic_loss(
    model,
    value_preds_batch,
    values,
    curr_e_clip,
    return_batch,
    clip_value,
):
    """SimpleFlight's clipped Huber value loss with delta=10."""
    del model
    original = F.huber_loss(values, return_batch, reduction="none", delta=10.0)
    if not clip_value:
        return original
    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
        -curr_e_clip, curr_e_clip
    )
    clipped = F.huber_loss(value_pred_clipped, return_batch, reduction="none", delta=10.0)
    return torch.maximum(original, clipped)


def register_simpleflight_rl_games() -> None:
    """Register task-local network/loss behavior before constructing Runner."""
    model_builder.register_network("simpleflight_actor_critic", SimpleFlightA2CBuilder)
    common_losses.critic_loss = _simpleflight_critic_loss

