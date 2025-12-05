from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .tokenizer import Tokenizer
from .model import TinyTransformerLM, ModelConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--vocab_path", type=str, default="data/vocab.json")
    p.add_argument("--prompt", type=str, default="once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> TinyTransformerLM:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt["config"]
    cfg = ModelConfig(**cfg_dict)
    model = TinyTransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def sample(
    model: TinyTransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    device: Optional[torch.device] = None,
) -> str:
    device = device or next(model.parameters()).device
    idx = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor(idx, dtype=torch.long, device=device)[None, :]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond, targets=None)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    out_ids = idx[0].tolist()
    return tokenizer.decode(out_ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tok = Tokenizer.load(args.vocab_path)
    model = load_model(Path(args.checkpoint), device)

    text = sample(
        model,
        tok,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print("=== Generated text ===")
    print(text)


if __name__ == "__main__":
    main()

