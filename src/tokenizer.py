from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
from collections import Counter
import json
import re

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


@dataclass
class TokenizerConfig:
    min_freq: int = 1
    lowercase: bool = True


class Tokenizer:
    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    def _basic_tokenize(self, text: str) -> List[str]:
        if self.config.lowercase:
            text = text.lower()
        return [t for t in re.split(r"\W+", text) if t]

    def build_vocab(self, corpus: str) -> None:
        tokens = self._basic_tokenize(corpus)
        counter = Counter(tokens)
        self.stoi = {}
        for tok in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]:
            self.stoi[tok] = len(self.stoi)
        for tok, freq in counter.items():
            if freq >= self.config.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.stoi)
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if not self.stoi:
            raise ValueError("Vocabulary is empty. Call build_vocab() first.")
        tokens = self._basic_tokenize(text)
        ids = [self.stoi.get(tok, self.stoi[UNK_TOKEN]) for tok in tokens]
        if add_special_tokens:
            ids = [self.stoi[BOS_TOKEN]] + ids + [self.stoi[EOS_TOKEN]]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if not self.itos:
            self.itos = {i: s for s, i in self.stoi.items()}
        tokens: List[str] = []
        for i in ids:
            tok = self.itos.get(int(i), UNK_TOKEN)
            if skip_special_tokens and tok in {PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN}:
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def save(self, path: str) -> None:
        obj = {
            "config": self.config.__dict__,
            "stoi": self.stoi,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        tok = cls(TokenizerConfig(**obj["config"]))
        tok.stoi = {k: int(v) for k, v in obj["stoi"].items()} if isinstance(
            next(iter(obj["stoi"].values())), str
        ) else obj["stoi"]
        tok.itos = {i: s for s, i in tok.stoi.items()}
        return tok

