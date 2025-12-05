from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from .tokenizer import Tokenizer, TokenizerConfig
from .dataset import TextDataset
from .model import TinyTransformerLM, ModelConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--vocab_path", type=str, default="data/vocab.json")
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="cs336-language-modeling")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    wandb = None
    if args.use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(project=args.wandb_project, config=vars(args))
        except ImportError:
            print("[WARN] wandb not installed, proceeding without logging.")
            wandb = None

    data_path = Path(args.data_path)
    text = data_path.read_text(encoding="utf-8")

    tok = Tokenizer(TokenizerConfig())
    tok.build_vocab(text)
    vocab_path = Path(args.vocab_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(vocab_path))

    dataset = TextDataset(text, tok, block_size=args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    cfg = ModelConfig(
        vocab_size=len(tok.stoi),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        block_size=args.block_size,
    )
    model = TinyTransformerLM(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    print(f"Using device: {device}")
    print(f"Vocab size: {cfg.vocab_size}, #params: {sum(p.numel() for p in model.parameters())}")

    global_step = 0
    model.train()
    for epoch in range(1, args.n_epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.n_epochs}")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix({"loss": loss.item()})

            if wandb is not None:
                wandb.log({"train/loss": loss.item(), "epoch": epoch, "step": global_step})

        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")

        if wandb is not None:
            wandb.log({"epoch": epoch, "checkpoint": str(ckpt_path)})

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

