"""Qwen2 inference using PyTorch operations, independent of the HF forward pass."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, Qwen2Config


KVPair = tuple[Tensor, Tensor]
KVCache = tuple[KVPair, ...]


@dataclass
class ModelOutput:
    logits: Tensor
    past_key_values: KVCache | None


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normalized.to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        freqs = torch.outer(positions.float(), self.inv_freq.float())
        angles = torch.cat((freqs, freqs), dim=-1)
        return angles.cos().to(dtype)[None, None, :, :], angles.sin().to(dtype)[None, None, :, :]


def rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, past: KVPair | None, use_cache: bool
    ) -> tuple[Tensor, KVPair | None]:
        batch, length, hidden = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        past_length = 0 if past is None else past[0].shape[2]
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        present = (k, v) if use_cache else None
        expanded_k = k.repeat_interleave(self.groups, dim=1)
        expanded_v = v.repeat_interleave(self.groups, dim=1)
        scores = torch.matmul(q, expanded_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        query_positions = torch.arange(past_length, past_length + length, device=x.device)
        key_positions = torch.arange(past_length + length, device=x.device)
        causal_mask = key_positions[None, :] > query_positions[:, None]
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(probabilities, expanded_v).transpose(1, 2).contiguous()
        return self.o_proj(output.reshape(batch, length, hidden)), present


class MLP(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, past: KVPair | None, use_cache: bool
    ) -> tuple[Tensor, KVPair | None]:
        attention, present = self.self_attn(self.input_layernorm(x), cos, sin, past, use_cache)
        x = x + attention
        return x + self.mlp(self.post_attention_layernorm(x)), present


class Decoder(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_attention_heads, config.rope_theta)


class MiniQwen2ForCausalLM(nn.Module):
    """Unpadded, equal-length batches; cache contains rotated keys and raw values."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        if config.model_type != "qwen2" or config.hidden_act != "silu":
            raise ValueError("Only Qwen2 models with SiLU activation are supported.")
        if config.rope_scaling is not None or config.use_sliding_window:
            raise ValueError("RoPE scaling and sliding-window attention are not supported.")
        if (
            config.num_attention_heads <= 0
            or config.num_key_value_heads <= 0
            or config.hidden_size % config.num_attention_heads
            or config.num_attention_heads % config.num_key_value_heads
            or (config.hidden_size // config.num_attention_heads) % 2
        ):
            raise ValueError("Invalid attention head dimensions.")
        self.config = config
        self.model = Decoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self, input_ids: Tensor, past_key_values: KVCache | None = None, use_cache: bool = False
    ) -> ModelOutput:
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [nonempty batch, nonempty sequence].")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must have dtype torch.long.")
        if past_key_values is not None and not use_cache:
            raise ValueError("Passing past_key_values requires use_cache=True.")
        past_length = 0
        if past_key_values is not None:
            if len(past_key_values) != len(self.model.layers):
                raise ValueError("Cache must contain one key/value pair per layer.")
            past_length = past_key_values[0][0].shape[2]
            expected = (input_ids.shape[0], self.config.num_key_value_heads, past_length,
                        self.config.hidden_size // self.config.num_attention_heads)
            for key, value in past_key_values:
                if key.shape != expected or value.shape != expected:
                    raise ValueError("Cache tensors have inconsistent shapes.")
                if key.device != input_ids.device or value.device != input_ids.device:
                    raise ValueError("Cache tensors and input_ids must be on the same device.")
        length = input_ids.shape[1]
        if past_length + length > self.config.max_position_embeddings:
            raise ValueError("Sequence exceeds max_position_embeddings.")
        positions = torch.arange(past_length, past_length + length, device=input_ids.device)
        x = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(positions, x.dtype)
        cache: list[KVPair] = []
        for index, layer in enumerate(self.model.layers):
            past = None if past_key_values is None else past_key_values[index]
            x, present = layer(x, cos, sin, past, use_cache)
            if present is not None:
                cache.append(present)
        return ModelOutput(self.lm_head(self.model.norm(x)), tuple(cache) if use_cache else None)


def from_reference(reference: nn.Module) -> MiniQwen2ForCausalLM:
    """Reuse the reference weights without running its transformer forward pass."""
    with torch.device("meta"):
        model = MiniQwen2ForCausalLM(reference.config)
    model.load_state_dict(reference.state_dict(), strict=True, assign=True)
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    # Non-persistent RoPE frequencies are not in the state dict.
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    model.model.rotary_emb = RotaryEmbedding(head_dim, model.config.rope_theta).to(
        model.model.embed_tokens.weight.device
    )
    return model.eval()


def load_pretrained(model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "cpu") -> MiniQwen2ForCausalLM:
    reference = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model = from_reference(reference)
    del reference
    model = model.to(device)
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    return model.eval()
