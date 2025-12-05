from __future__ import annotations
import argparse
from pathlib import Path
import math

import torch
from torch.utils.data import DataLoader

from .tokenizer import Tokenizer
from .dataset import TextDataset
from .model import TinyTransformerLM, ModelConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--vocab_path", type=str, default="data/vocab.json")
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
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
def compute_ppl(
    model: TinyTransformerLM,
    dataset: TextDataset,
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size)
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        B, T = y.shape
        total_loss += loss.item() * B * T
        total_tokens += B * T
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    text = Path(args.data_path).read_text(encoding="utf-8")
    tok = Tokenizer.load(args.vocab_path)
    dataset = TextDataset(text, tok, block_size=args.block_size)
    model = load_model(Path(args.checkpoint), device)

    ppl = compute_ppl(model, dataset, batch_size=args.batch_size, device=device)
    print(f"Perplexity on {args.data_path}: {ppl:.4f}")


if __name__ == "__main__":
    main()

