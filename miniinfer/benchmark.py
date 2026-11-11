"""Reproducible, single-request greedy inference benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from torch import Tensor
from transformers import AutoTokenizer

from .model import MiniQwen2ForCausalLM, load_pretrained


PROMPT_TEXT = (
    "Artificial intelligence can help people understand complex systems. "
    "A useful explanation starts with a clear example and builds on it. "
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure_run(
    model: MiniQwen2ForCausalLM, input_ids: Tensor, output_tokens: int
) -> tuple[dict[str, float | int | None], Tensor]:
    """Time prompt prefill including first token, then remaining cached decode."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Benchmark requires one nonempty prompt.")
    if output_tokens < 2:
        raise ValueError("Benchmark requires at least two output tokens.")
    if input_ids.shape[1] + output_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus output exceeds the model's maximum context.")
    device = input_ids.device
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = model(input_ids, use_cache=True)
    token = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    synchronize(device)
    prefill_end = time.perf_counter()

    tokens = [token]
    for _ in range(output_tokens - 1):
        result = model(token, past_key_values=result.past_key_values, use_cache=True)
        token = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tokens.append(token)
    synchronize(device)
    end = time.perf_counter()

    prefill = prefill_end - start
    decode = end - prefill_end
    total = end - start
    metrics: dict[str, float | int | None] = {
        "prefill_seconds": prefill,
        "decode_seconds": decode,
        "total_seconds": total,
        "decode_tokens": output_tokens - 1,
        "decode_tokens_per_second": (output_tokens - 1) / decode,
        "overall_tokens_per_second": output_tokens / total,
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    return metrics, torch.cat(tokens, dim=1)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--prompt-lengths", type=positive_int, nargs="+", default=[32, 256])
    parser.add_argument("--output-lengths", type=positive_int, nargs="+", default=[32, 128])
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--warmup", type=positive_int, default=1)
    parser.add_argument("--output", type=Path, help="JSON output path; existing files are not overwritten")
    args = parser.parse_args()
    if min(args.output_lengths) < 2:
        parser.error("--output-lengths must be at least 2")
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name if args.device == "auto" else args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must be cpu, cuda, or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable in this Python environment")

    print(f"Loading {args.model} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_pretrained(args.model, str(device))
    if max(args.prompt_lengths) + max(args.output_lengths) > model.config.max_position_embeddings:
        parser.error("prompt plus output exceeds the model's maximum context")
    seed_tokens = tokenizer(PROMPT_TEXT, add_special_tokens=False)["input_ids"]
    if not seed_tokens:
        raise ValueError("The benchmark prompt tokenized to an empty sequence.")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "dtype": str(model.model.embed_tokens.weight.dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
        "cpu_threads": torch.get_num_threads(),
        "warmup_runs_per_case": args.warmup,
        "measured_runs_per_case": args.runs,
        "workload": (
            "Single request, greedy decoding, fixed output length; EOS ignored. "
            "Prompt is repeated tokenized text truncated to the requested length."
        ),
        "timing": (
            "Prefill includes prompt forward pass and first token selection. Decode includes "
            "remaining cached forward passes and token selections. Model loading, tokenization, "
            "and final token concatenation are excluded. CUDA is synchronized at phase boundaries."
        ),
        "cases": [],
    }
    for prompt_length in args.prompt_lengths:
        ids = (seed_tokens * ((prompt_length + len(seed_tokens) - 1) // len(seed_tokens)))[:prompt_length]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        prompt_hash = hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()
        for output_length in args.output_lengths:
            print(f"Prompt {prompt_length} tokens, output {output_length} tokens...", flush=True)
            for _ in range(args.warmup):
                measure_run(model, input_ids, output_length)
            runs = []
            for _ in range(args.runs):
                metrics, generated = measure_run(model, input_ids, output_length)
                if generated.shape[1] != output_length:
                    raise RuntimeError("The number of generated tokens did not match the workload.")
                runs.append(metrics)
            median_prefill = statistics.median(run["prefill_seconds"] for run in runs)
            median_decode = statistics.median(run["decode_seconds"] for run in runs)
            median_total = statistics.median(run["total_seconds"] for run in runs)
            report["cases"].append({
                "prompt_tokens": prompt_length,
                "prompt_token_ids_sha256": prompt_hash,
                "output_tokens": output_length,
                "runs": runs,
                "median": {
                    "prefill_seconds": median_prefill,
                    "decode_seconds": median_decode,
                    "total_seconds": median_total,
                    "decode_tokens_per_second": (output_length - 1) / median_decode,
                    "overall_tokens_per_second": output_length / median_total,
                },
            })
            print(
                f"  prefill {median_prefill:.3f}s; decode "
                f"{(output_length - 1) / median_decode:.2f} tokens/s; "
                f"overall {output_length / median_total:.2f} tokens/s",
                flush=True,
            )
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = Path("benchmark-results") / f"benchmark-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
