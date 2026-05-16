from __future__ import annotations

from typing import Any


def is_nnsight_mlp_input_hook(hook_name: str) -> bool:
    return "hook_mlp_in" in hook_name or "ln2.hook_normalized" in hook_name


def save_nnsight_hook_outputs(model: Any, hook_name: str, layer_num: int):
    if "resid_post" in hook_name:
        return model.layers_output[layer_num].save()
    if "resid_pre" in hook_name:
        if layer_num == 0:
            return model.embeddings_output.save()
        return model.layers_output[layer_num - 1].save()
    if is_nnsight_mlp_input_hook(hook_name):
        return model.mlps_input[layer_num].save()
    raise ValueError(f"Unsupported hook name for nnsight: {hook_name}")