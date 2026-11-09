"""Toy LLaMA-style transformer in MLX, with an explicit KV cache that can be
cropped to `start_pos` (the rollback trick: a rejected speculative run is
overwritten by re-running from an earlier position, no snapshot needed)."""
import mlx.core as mx
import numpy as np

from .config import ToyConfig


def rms_norm(x, w, eps):
    return (x / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)) * w


def rope(x, positions, cfg):
    """x: [1, L, NH, HD] -> rotated. Interleaved-halves (GPT-NeoX style)."""
    half = cfg.head_dim // 2
    inv = cfg.rope_theta ** (-mx.arange(half, dtype=mx.float32) * (2.0 / cfg.head_dim))
    ang = positions.astype(mx.float32)[:, None] * inv[None, :]      # [L, half]
    cos = mx.cos(ang)[None, :, None, :]                            # [1, L, 1, half]
    sin = mx.sin(ang)[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def attention(x, w, slot, positions, start_pos, cfg):
    b, L, H = x.shape
    nh, hd = cfg.n_heads, cfg.head_dim
    q = (x @ w["wq"]).reshape(b, L, nh, hd)
    k = (x @ w["wk"]).reshape(b, L, nh, hd)
    v = (x @ w["wv"]).reshape(b, L, nh, hd)
    q = rope(q, positions, cfg)
    k = rope(k, positions, cfg)

    if slot["k"] is None:
        slot["k"], slot["v"] = k, v
    else:
        # crop stale speculative KV, then append
        if slot["k"].shape[1] > start_pos:
            slot["k"] = slot["k"][:, :start_pos, :, :]
            slot["v"] = slot["v"][:, :start_pos, :, :]
        slot["k"] = mx.concatenate([slot["k"], k], axis=1)
        slot["v"] = mx.concatenate([slot["v"], v], axis=1)

    K, V = slot["k"], slot["v"]
    T = K.shape[1]
    scale = mx.array(cfg.head_dim ** -0.5, mx.float32)
    scores = mx.matmul(q.transpose(0, 2, 1, 3), K.transpose(0, 2, 3, 1)) * scale
    qpos = positions[:, None]                    # [L, 1]
    kpos = mx.arange(T)[None, :]                 # [1, T]
    mask = (kpos <= qpos)[None, None, :, :]
    scores = mx.where(mask, scores, mx.array(-1e9, mx.float32))
    probs = mx.softmax(scores, axis=-1)
    o = mx.matmul(probs, V.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3).reshape(b, L, H)
    return o @ w["wo"]


def mlp(x, w):
    a = x @ w["w1"]
    a = a * mx.sigmoid(a)                       # SiLU
    return a @ w["w2"]


class ToyModel:
    """Weights are generated from a numpy RNG, so every process builds the
    exact same model and the token-level comparison is meaningful."""

    def __init__(self, cfg: ToyConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        H, F, V = cfg.hidden, cfg.ffn, cfg.vocab

        def mat(shape, scale):
            return mx.array(rng.normal(0, scale, size=shape).astype(np.float32))

        self.tok_emb = mat((V, H), 0.02)
        self.layers = [
            dict(
                wq=mat((H, H), H ** -0.5), wk=mat((H, H), H ** -0.5),
                wv=mat((H, H), H ** -0.5), wo=mat((H, H), H ** -0.5),
                w1=mat((H, F), H ** -0.5), w2=mat((F, H), F ** -0.5),
                n1=mx.ones((H,)), n2=mx.ones((H,)),
            )
            for _ in range(cfg.n_layers)
        ]
        self.final_norm = mx.ones((H,))
        self.lm_head = mat((H, V), 0.02)

    # ---- pieces the ring needs ------------------------------------------
    def embed(self, tokens):
        ids = mx.array(list(tokens), mx.int32)
        return mx.take(self.tok_emb, ids, axis=0)[None]          # [1, L, H]

    def forward_range(self, h, start_pos, lo, hi, kv):
        cfg = self.cfg
        positions = mx.arange(start_pos, start_pos + h.shape[1])
        for i in range(lo, hi):
            n = rms_norm(h, self.layers[i]["n1"], cfg.eps)
            h = h + attention(n, self.layers[i], kv[i], positions, start_pos, cfg)
            n = rms_norm(h, self.layers[i]["n2"], cfg.eps)
            h = h + mlp(n, self.layers[i])
        return h

    def logits_all(self, h):
        cfg = self.cfg
        hn = rms_norm(h, self.final_norm, cfg.eps)
        return hn @ self.lm_head, hn                        # [1, L, V]

    def crop_kv(self, kv, lo, hi, length):
        for i in range(lo, hi):
            if kv[i]["k"] is not None and kv[i]["k"].shape[1] > length:
                kv[i]["k"] = kv[i]["k"][:, :length, :, :]
                kv[i]["v"] = kv[i]["v"][:, :length, :, :]


def new_kv(n_layers):
    return [dict(k=None, v=None) for _ in range(n_layers)]
