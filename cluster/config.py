"""Toy model config. Small enough to run on a laptop, real enough to exercise
the whole pipeline (attention + KV cache + lm_head)."""
from dataclasses import dataclass


@dataclass
class ToyConfig:
    n_layers: int = 8
    hidden: int = 256
    n_heads: int = 4
    ffn: int = 768
    vocab: int = 512
    rope_theta: float = 10000.0
    eps: float = 1e-6
    seed: int = 1234

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads
