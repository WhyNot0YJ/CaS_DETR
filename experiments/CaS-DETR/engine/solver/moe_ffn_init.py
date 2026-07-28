from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


def _checkpoint_model_state(checkpoint: Dict) -> Dict[str, torch.Tensor]:
    if 'ema' in checkpoint:
        return checkpoint['ema']['module']
    if 'model' in checkpoint:
        return checkpoint['model']
    raise KeyError("MoE FFN initialization checkpoint must contain 'ema' or 'model'.")


@torch.no_grad()
def load_structured_moe_ffn_init(
    model: nn.Module,
    source: str,
    root: Path,
) -> Dict[str, int]:
    """Compress trained wide MoE FFNs into the current model by neuron saliency."""
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = (root / source_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f'MoE FFN initialization checkpoint not found: {source_path}')

    source_state = _checkpoint_model_state(torch.load(source_path, map_location='cpu'))
    target_state = model.state_dict()
    w1_keys = sorted(
        key for key in target_state
        if key.endswith('.decoder_moe_layer.expert_w1')
    )
    if not w1_keys:
        raise RuntimeError('Target model has no decoder MoE FFN layers.')

    loaded_tensors = 0
    selected_neurons = 0
    for w1_key in w1_keys:
        prefix = w1_key[:-len('expert_w1')]
        b1_key = prefix + 'expert_b1'
        w2_key = prefix + 'expert_w2'
        b2_key = prefix + 'expert_b2'
        keys = (w1_key, b1_key, w2_key, b2_key)
        missing = [key for key in keys if key not in source_state]
        if missing:
            raise KeyError(f'Missing source MoE FFN tensors: {missing}')

        source_w1 = source_state[w1_key]
        source_b1 = source_state[b1_key]
        source_w2 = source_state[w2_key]
        source_b2 = source_state[b2_key]
        target_w1 = target_state[w1_key]
        target_b1 = target_state[b1_key]
        target_w2 = target_state[w2_key]
        target_b2 = target_state[b2_key]

        experts, target_hidden, d_model = target_w1.shape
        expected_source = (experts, source_w1.shape[1], d_model)
        if tuple(source_w1.shape) != expected_source or source_w1.shape[1] < target_hidden:
            raise ValueError(
                f'Invalid source/target expert_w1 shapes for {w1_key}: '
                f'{tuple(source_w1.shape)} -> {tuple(target_w1.shape)}'
            )
        source_hidden = source_w1.shape[1]
        expected_shapes = {
            b1_key: (experts, source_hidden),
            w2_key: (experts, d_model, source_hidden),
            b2_key: tuple(target_b2.shape),
        }
        actual_shapes = {
            b1_key: tuple(source_b1.shape),
            w2_key: tuple(source_w2.shape),
            b2_key: tuple(source_b2.shape),
        }
        if actual_shapes != expected_shapes:
            raise ValueError(
                f'Incompatible source MoE FFN tensors for {prefix}: '
                f'expected {expected_shapes}, got {actual_shapes}'
            )

        if source_hidden == target_hidden:
            indices = torch.arange(source_hidden).expand(experts, -1)
        else:
            scores = source_w1.float().norm(dim=2) * source_w2.float().norm(dim=1)
            indices = scores.argsort(dim=1, descending=True, stable=True)[:, :target_hidden]
        w1_index = indices.unsqueeze(-1).expand(-1, -1, d_model)
        w2_index = indices.unsqueeze(1).expand(-1, d_model, -1)

        target_w1.copy_(source_w1.gather(1, w1_index))
        target_b1.copy_(source_b1.gather(1, indices))
        target_w2.copy_(source_w2.gather(2, w2_index))
        target_b2.copy_(source_b2)
        loaded_tensors += len(keys)
        selected_neurons += indices.numel()

    return {
        'layers': len(w1_keys),
        'tensors': loaded_tensors,
        'selected_neurons': selected_neurons,
    }
