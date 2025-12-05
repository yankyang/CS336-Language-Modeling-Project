from __future__ import annotations
from typing import Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


class TextDataset(Dataset):
    def __init__(self, text: str, tokenizer: Tokenizer, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.ids = tokenizer.encode(text, add_special_tokens=False)

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.block_size - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.ids[idx : idx + self.block_size]
        y = self.ids[idx + 1 : idx + 1 + self.block_size]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

