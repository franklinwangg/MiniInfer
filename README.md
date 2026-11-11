# MiniInfer: Qwen2.5 baseline

This is a minimal, readable PyTorch inference engine for the **base**
`Qwen/Qwen2.5-0.5B` model. It implements the transformer forward pass and
greedy decoding itself; Transformers is used for model configuration, loading
weights and the tokenizer, and as a numerical reference in tests. It does not
call Hugging Face's `generate` or reuse its transformer forward pass.
The existing [vLLM setup](vllm/README.md) is separate.

## Architecture and data flow

The published Qwen2.5-0.5B configuration has 24 decoder layers, hidden size 896,
14 query heads, 2 key/value heads (64 dimensions per head), MLP size 4864,
vocabulary size 151936, RoPE base 1000000, and tied token embeddings and output
weights. Sliding-window attention and RoPE scaling are disabled.

1. Token IDs `[batch, sequence]` become embeddings `[batch, sequence, 896]`.
2. Each pre-normalized decoder layer projects Q to `[batch, 14, sequence, 64]`
   and K and V to `[batch, 2, sequence, 64]`. RoPE rotates Q and K using
   absolute token positions. The cache stores the rotated K and V before
   repeating KV heads to match the query heads.
3. Scaled QK scores receive a causal mask, then float32 softmax and a weighted
   sum of V; the output projection and a residual connection restore the hidden
   state. A second RMSNorm, gated SiLU MLP and residual complete the layer.
4. Final RMSNorm and the tied vocabulary projection produce logits
   `[batch, sequence, vocabulary]`.
5. Generation prefills the prompt once, chooses the largest final-position
   logit, and feeds each generated token back with its per-layer KV cache.
   The causal mask and RoPE positions account for the cached prefix.

## Setup and run

Install Python dependencies in a dedicated environment, then run:

```powershell
python -m pip install -r requirements.txt
python -m miniinfer "The capital of France is" --max-new-tokens 16
```

Use `--device cuda` with a CUDA-enabled PyTorch installation and sufficient GPU
memory. Weights and tokenizer are downloaded from Hugging Face on first use.
The loader uses float32 for a straightforward numerical comparison, so the
checkpoint requires roughly 2 GB for weights, plus memory for activations and
cache. CPU inference is slow. `--model` can select another compatible Qwen2
checkpoint; incompatible RoPE scaling and sliding-window configurations are
rejected. The base checkpoint is a text-completion model, not an instruction
chat template.

## Verify

```powershell
python -m unittest discover -s tests -v
$env:MINIINFER_TEST_PRETRAINED = "1"
python -m unittest discover -s tests -v
```

The first command compares a small, randomly initialized Qwen2 reference with
the independent implementation. The second additionally downloads and checks
the actual pretrained model, comparing full-sequence logits, cached
single-token and multi-token logits, and greedy token IDs against eager
Hugging Face inference. Tests use float32 with `atol=rtol=2e-5` for the tiny
model and `2e-4` for the pretrained model. No reference `generate` call is used.

## Scope

The forward pass supports unpadded equal-length batches; generation currently
accepts one unpadded prompt. Padding masks, sampling, beam search, training,
scaled RoPE, sliding windows and cache eviction are not implemented. Cache
appends use `torch.cat`, so they prioritize clarity over memory efficiency.
Requests beyond `max_position_embeddings` are rejected.

## Benchmark

Measure single-request greedy inference and save a JSON report:

```powershell
python -m miniinfer.benchmark --device auto
```

By default this tests all combinations of 32- and 256-token prompts and 32- and
128-token outputs, with one warmup and three measured runs per case. Prompts are
repeated tokenized text truncated to an exact length. EOS is ignored to make the
output length repeatable. Results are saved under `benchmark-results` with a
timestamp; pass `--output path.json` for a specific new file. Existing results
are not overwritten.

The report records individual runs, medians, device and library versions, input
lengths, and a hash of the prompt token IDs. Prefill time includes producing the
first token; decode throughput counts only the remaining output tokens.
Overall throughput includes both phases. Loading, tokenization, and final token
concatenation are outside the timed interval. CUDA is synchronized at phase
boundaries; peak CUDA allocated memory includes the model weights. These are
single-request measurements, not multi-request serving throughput.

Use `--device cpu` or `--device cuda` for explicit selection, and optionally
`--runs 3 --warmup 1 --prompt-lengths 32 256 --output-lengths 32 128` to
change the workload. CUDA requires a CUDA-enabled PyTorch build. Keep the
device, model, input lengths, and environment fixed when comparing results.
