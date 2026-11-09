"""Fixed-shape decode: preallocated per-layer KV slabs + `mx.compile`.

Why this exists: in the live loop every step has a different KV length, so MLX
re-plans and re-dispatches the whole graph every token; a layer that costs
0.10ms in a fixed-shape loop costs 0.38ms in the real loop. The fix is to make
the shape constant:

  * KV lives in a slab per layer, allocated once at max context. Writing
    position `pos` is `slice_update`, never `concatenate`: no realloc, no shape
    change, so no re-plan.
  * Attention reads the whole slab and masks positions > pos. We waste FLOPs on
    masked-out keys, which is affordable here because batch=1 decode is
    launch-bound, not FLOP-bound.
  * The layer stack is `mx.compile`d per (lo, hi), collapsing ~200 kernels into
    one dispatch.

The slab is per layer, not one big (L, ctx, nh, hd) tensor, because slice_update
returns a NEW array: with one big tensor each write copies everything (8 layers
× ctx × heads × 64 × 4B), which measured out to 1.5× SLOWER than not doing this
at all. Per-layer slabs copy 1/8 as much.
"""
import mlx.core as mx

from .config import ToyConfig
from .model import ToyModel, rms_norm, rope as rope_multi


def rope_one(x, pos, cfg):
    """x: [1, NH, HD]; pos: scalar int array. Interleaved-halves (GPT-NeoX)."""
    half = cfg.head_dim // 2
    inv = cfg.rope_theta ** (-mx.arange(half, dtype=mx.float32) * (2.0 / cfg.head_dim))
    ang = pos.astype(mx.float32) * inv
    cos, sin = mx.cos(ang)[None, None, :], mx.sin(ang)[None, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class FastModel:
    def __init__(self, cfg: ToyConfig, ctx_len=640, compile=True):
        self.cfg, self.ctx, self.use_compile = cfg, ctx_len, compile
        self.w = ToyModel(cfg)                              # same weights, same seed
        nl, nh, hd = cfg.n_layers, cfg.n_heads, cfg.head_dim
        self.ks = [mx.zeros((ctx_len, nh, hd)) for _ in range(nl)]
        self.vs = [mx.zeros((ctx_len, nh, hd)) for _ in range(nl)]
        self.ar = mx.arange(ctx_len)
        self.scale = mx.array(cfg.head_dim ** -0.5, mx.float32)
        self._cache = {}

    def _step_impl(self, x, pos, ks, vs, lo, hi):
        cfg, W = self.cfg, self.w
        nh, hd = cfg.n_heads, cfg.head_dim
        ko, vo = [], []
        for i in range(cfg.n_layers):
            L = W.layers[i]
            if not (lo <= i < hi):
                ko.append(ks[i])
                vo.append(vs[i])
                continue

            n = rms_norm(x, L["n1"], cfg.eps)
            q = rope_one((n @ L["wq"]).reshape(1, nh, hd), pos, cfg)
            kk = rope_one((n @ L["wk"]).reshape(1, nh, hd), pos, cfg)[0]
            vv = (n @ L["wv"]).reshape(1, nh, hd)[0]
            ki = mx.slice_update(ks[i], kk, mx.array([pos, 0, 0]),
                                 mx.array([0, 1, 2]))
            vi = mx.slice_update(vs[i], vv, mx.array([pos, 0, 0]),
                                 mx.array([0, 1, 2]))
            ko.append(ki)
            vo.append(vi)

            scores = (q[0][:, None, :] @ ki.transpose(1, 2, 0))[:, 0] * self.scale
            scores = mx.where(self.ar <= pos, scores, mx.array(-1e9, mx.float32))
            probs = mx.softmax(scores, axis=-1)                       # [nh, ctx]
            o = (probs[:, None, :] @ vi.transpose(1, 0, 2))[:, 0]     # [nh, hd]
            x = x + (o.reshape(1, nh * hd) @ L["wo"])

            n = rms_norm(x, L["n2"], cfg.eps)
            a = n @ L["w1"]
            x = x + (a * mx.sigmoid(a)) @ L["w2"]
        return x, ko, vo

    def step(self, x, pos, lo=0, hi=None):
        hi = self.cfg.n_layers if hi is None else hi
        if self.use_compile:
            key = (lo, hi)
            if key not in self._cache:
                self._cache[key] = mx.compile(
                    lambda a, p, ks, vs: self._step_impl(a, p, ks, vs, lo, hi))
            fn = self._cache[key]
        else:
            fn = lambda a, p, ks, vs: self._step_impl(a, p, ks, vs, lo, hi)
        out, self.ks, self.vs = fn(x, pos, self.ks, self.vs)
        return out

    # -- many positions at once (prefill) --------------------------------
    def _batch_impl(self, x, start, ks, vs, lo, hi, L):
        cfg, W = self.cfg, self.w
        nh, hd = cfg.n_heads, cfg.head_dim
        qpos = start + mx.arange(L)
        ko, vo = [], []
        for i in range(cfg.n_layers):
            Lw = W.layers[i]
            if not (lo <= i < hi):
                ko.append(ks[i])
                vo.append(vs[i])
                continue
            n = rms_norm(x, Lw["n1"], cfg.eps)                     # [1, L, H]
            q = rope_multi((n @ Lw["wq"]).reshape(1, L, nh, hd), qpos, cfg)
            kk = rope_multi((n @ Lw["wk"]).reshape(1, L, nh, hd), qpos, cfg)
            vv = (n @ Lw["wv"]).reshape(1, L, nh, hd)
            ki = mx.slice_update(ks[i], kk[0], mx.array([start, 0, 0]),
                                 mx.array([0, 1, 2]))
            vi = mx.slice_update(vs[i], vv[0], mx.array([start, 0, 0]),
                                 mx.array([0, 1, 2]))
            ko.append(ki)
            vo.append(vi)

            qh = q.transpose(0, 2, 1, 3)[0]                        # [nh, L, hd]
            scores = (qh @ ki.transpose(1, 2, 0)) * self.scale     # [nh, L, ctx]
            ok = self.ar[None, :] <= qpos[:, None]                 # [L, ctx]
            scores = mx.where(ok[None], scores, mx.array(-1e9, mx.float32))
            probs = mx.softmax(scores, axis=-1)
            o = probs @ vi.transpose(1, 0, 2)                      # [nh, L, hd]
            o = o.transpose(1, 0, 2).reshape(1, L, nh * hd)
            x = x + o @ Lw["wo"]

            n = rms_norm(x, Lw["n2"], cfg.eps)
            a = n @ Lw["w1"]
            x = x + (a * mx.sigmoid(a)) @ Lw["w2"]
        return x, ko, vo

    def forward_batch(self, x, start, lo=0, hi=None):
        """x: [1, L, H]. One compiled dispatch per L (shape must stay static)."""
        hi = self.cfg.n_layers if hi is None else hi
        L = x.shape[1]
        key = ("batch", lo, hi, L)
        if self.use_compile:
            if key not in self._cache:
                self._cache[key] = mx.compile(
                    lambda a, s, ks, vs: self._batch_impl(a, s, ks, vs, lo, hi, L))
            fn = self._cache[key]
        else:
            fn = lambda a, s, ks, vs: self._batch_impl(a, s, ks, vs, lo, hi, L)
        out, self.ks, self.vs = fn(x, start, self.ks, self.vs)
        return out

    # -- helpers ----------------------------------------------------------
    def embed_one(self, token):
        return mx.take(self.w.tok_emb, mx.array([token], mx.int32), axis=0)

    def embed(self, tokens):
        return mx.take(self.w.tok_emb, mx.array(list(tokens), mx.int32), axis=0)[None]

    def logits(self, x):
        hn = rms_norm(x, self.w.final_norm, self.cfg.eps)
        return hn @ self.w.lm_head, hn

    def prefill(self, tokens, lo=0, hi=None, ctx_cap=None):
        """Token at a time on purpose: the shape stays constant, so no recompile."""
        cap = ctx_cap or self.ctx
        for p, t in enumerate(tokens):
            if p >= cap:
                raise RuntimeError(f"prefill longer than slab ({cap})")
            self.step(self.embed_one(t), mx.array(p, mx.int32), lo, hi)
