"""Run: python -m miniinfer "Your prompt"."""

import argparse

import torch
from transformers import AutoTokenizer

from .generation import greedy_generate
from .model import load_pretrained


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explicit Qwen2 greedy inference.")
    parser.add_argument("prompt")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_pretrained(args.model, args.device)
    input_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(args.device)
    with torch.inference_mode():
        result = greedy_generate(model, input_ids, args.max_new_tokens, tokenizer.eos_token_id)
    print(tokenizer.decode(result[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
