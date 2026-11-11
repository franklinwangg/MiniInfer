"""Simple greedy autoregressive generation with an explicit KV cache."""

from __future__ import annotations

import torch
from torch import Tensor

from .model import MiniQwen2ForCausalLM


@torch.inference_mode()
def greedy_generate(
    model: MiniQwen2ForCausalLM, input_ids: Tensor, max_new_tokens: int, eos_token_id: int | None = None
) -> Tensor:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Generation requires one nonempty, unpadded prompt.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative.")
    if input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus requested generation exceeds max_position_embeddings.")
    if max_new_tokens == 0:
        return input_ids
    output = model(input_ids, use_cache=True)
    tokens = [input_ids]
    for step in range(max_new_tokens):
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(next_token)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
        if step + 1 < max_new_tokens:
            output = model(next_token, past_key_values=output.past_key_values, use_cache=True)
    return torch.cat(tokens, dim=1)
