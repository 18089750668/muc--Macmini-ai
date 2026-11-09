"""真模型接入（Qwen2.5-0.5B）：和 ToyModel 完全同一套接口。

為什麼要接：玩具模型的權重是隨機的，所以「投機草稿能猜中多少」這件事
根本無從量起——隨機權重下任何草稿都沒有預測力，g 永遠貼近 1。而 k（每輪
接受多少 token）正是整條路線真正的瓶頸。要量 k，就必須有真權重。

三個和玩具模型不同的地方，都是真模型才有的坑：

1. **GQA**：14 個 query 頭共用 2 個 KV 頭。KV 只存 2 份（省 7 倍記憶體），
   算 attention 前再展開成 14 份。分片時每段各自展開，不影響正確性。
2. **attention 有 bias**：Qwen2 的 q/k/v 投影帶 bias，玩具模型沒有。
3. **tie_word_embeddings**：lm_head 與 token embedding 是同一塊權重，
   所以 head 那台不必多存一份 272MB。

精度：一律 float16（bf16 的 32K 表格 numpy 存不了，而且協議走 <f2 也是 fp16）。
整條鏈都用 fp16，所以「單機」和「拆成 N 段」比對時是完全同一條數值路徑。
"""
import json
import os
import struct

import mlx.core as mx
import numpy as np

from .config import ToyConfig
from .model import new_kv, rms_norm, rope


class RealConfig:
    """把 HF 的 config.json 攤成 ToyConfig 那幾個會被用到的欄位。"""

    def __init__(self, path):
        c = json.load(open(os.path.join(path, "config.json")))
        self.n_layers = c["num_hidden_layers"]
        self.hidden = c["hidden_size"]
        self.ffn = c["intermediate_size"]
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.head_dim = self.hidden // self.n_heads
        self.vocab = c["vocab_size"]
        self.eps = c["rms_norm_eps"]
        self.rope_theta = float(c["rope_theta"])
        self.tie = c["tie_word_embeddings"]
        self.max_pos = c["max_position_embeddings"]
        self.arch = c["architectures"][0]


_CACHE = {}

_DTYPE = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}


class Safetensors:
    """按 tensor 讀，不整檔載入。

    這一版很要緊：先前是 `mx.load()` 整檔 988MB 再丟掉不要的層，四個 stage
    同時起就是四份 988MB 的峰值 —— 在 16GB 的機器上直接換頁，於是「加機器
    變快」被自己的載入方式拖死。真正的分片從「只讀自己那幾層」開始。
    """

    def __init__(self, path):
        self.path = os.path.join(path, "model.safetensors")
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.data = 8 + n

    def names(self):
        return [k for k in self.header if k != "__metadata__"]

    def get(self, name, dtype=mx.float16):
        info = self.header[name]
        b, e = info["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.data + b)
            raw = f.read(e - b)
        a = np.frombuffer(raw, dtype=_DTYPE[info["dtype"]]).reshape(info["shape"])
        x = mx.array(a)
        if info["dtype"] == "BF16":
            x = x.view(mx.bfloat16)
        return x.astype(dtype)


def _load(path):
    if path not in _CACHE:
        _CACHE[path] = Safetensors(path)
    return _CACHE[path]


class RealModel:
    def __init__(self, path, lo=0, hi=None, need_embed=True, need_head=True,
                 dtype=mx.float16):
        self.cfg = RealConfig(path)
        cfg = self.cfg
        hi = cfg.n_layers if hi is None else hi
        self.lo, self.hi = lo, hi
        w = _load(path)
        f16 = lambda p: w.get(p, dtype)

        self.embed_w = f16("model.embed_tokens.weight") if need_embed else None
        self.norm_w = f16("model.norm.weight") if need_head else None
        self.lm_head = None
        if need_head:
            key = "lm_head.weight"
            # tie 的時候檔案裡沒有這張，輸出矩陣就是 embedding 本身。最後一台
            # 因此仍然必須放 vocab×hidden（151936×896 ≈ 272MB fp16）——
            # 這塊切不開，是每條鏈的固定成本。
            if key in w.names():
                self.lm_head = f16(key)
            elif self.embed_w is not None:
                self.lm_head = self.embed_w
            else:
                self.lm_head = f16("model.embed_tokens.weight")

        self.layers = [None] * cfg.n_layers
        for i in range(lo, hi):
            p = f"model.layers.{i}."
            self.layers[i] = dict(
                q=f16(p + "self_attn.q_proj.weight"),
                bq=f16(p + "self_attn.q_proj.bias"),
                k=f16(p + "self_attn.k_proj.weight"),
                bk=f16(p + "self_attn.k_proj.bias"),
                v=f16(p + "self_attn.v_proj.weight"),
                bv=f16(p + "self_attn.v_proj.bias"),
                o=f16(p + "self_attn.o_proj.weight"),
                n1=f16(p + "input_layernorm.weight"),
                n2=f16(p + "post_attention_layernorm.weight"),
                g=f16(p + "mlp.gate_proj.weight"),
                u=f16(p + "mlp.up_proj.weight"),
                d=f16(p + "mlp.down_proj.weight"),
            )

    # ---- 和 ToyModel 一樣的四個接縫 -----------------------------------
    def _rope(self, x, positions):
        """[B,L,NH,HD]，positions 可為 [L] 或 [B,L]（左補齊時每列的內容位置不同）。

        原本 model.py 的 rope 只吃一維 positions；左補齊之後每一列的「真實位置」
        不一樣（第 r 列的 0 落在物理位置 valid_from[r]），所以必須接受 [B,L]。
        """
        cfg = self.cfg
        if positions.ndim == 1:
            positions = positions[None]
        half = cfg.head_dim // 2
        inv = cfg.rope_theta ** (-mx.arange(half, dtype=mx.float32)
                                * (2.0 / cfg.head_dim))
        ang = positions.astype(mx.float32)[..., None] * inv[None, None, :]   # [B,L,half]
        cos = mx.cos(ang)[:, :, None, :]
        sin = mx.sin(ang)[:, :, None, :]
        x1, x2 = x[..., :half], x[..., half:]
        return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

    def embed(self, tokens):
        ids = mx.array(list(tokens), mx.int32)
        return mx.take(self.embed_w, ids, axis=0)[None]           # [1, L, H]

    def embed_rows(self, rows):
        """多條序列一起 embed：rows 是等長的 B 條 token 序列 → [B, L, H]。

        batching 的入口。注意這跟 `embed` 不是同一個東西：`embed([[a],[b]])`
        會被壓成 [1, 2, H]（兩條擠在同一列），batch 必須走這條路。
        """
        ids = mx.array([list(r) for r in rows], mx.int32)
        return mx.take(self.embed_w, ids, axis=0)                 # [B, L, H]

    def _attn(self, x, w, slot, positions, start_pos, valid_from=None):
        cfg = self.cfg
        b, L, H = x.shape
        nh, nkv, hd = cfg.n_heads, cfg.n_kv, cfg.head_dim

        q = (x @ w["q"].T + w["bq"]).reshape(b, L, nh, hd)
        k = (x @ w["k"].T + w["bk"]).reshape(b, L, nkv, hd)
        v = (x @ w["v"].T + w["bv"]).reshape(b, L, nkv, hd)
        q = self._rope(q, positions)
        k = self._rope(k, positions)

        if nkv != nh:                       # GQA：2 個 KV 頭展開成 14 個
            rep = nh // nkv
            k = mx.broadcast_to(k[:, :, :, None, :], (b, L, nkv, rep, hd))
            v = mx.broadcast_to(v[:, :, :, None, :], (b, L, nkv, rep, hd))
            k = k.reshape(b, L, nh, hd)
            v = v.reshape(b, L, nh, hd)

        if slot["k"] is None:
            slot["k"], slot["v"] = k, v
        else:
            if slot["k"].shape[1] > start_pos:
                slot["k"] = slot["k"][:, :start_pos, :, :]
                slot["v"] = slot["v"][:, :start_pos, :, :]
            slot["k"] = mx.concatenate([slot["k"], k], axis=1)
            slot["v"] = mx.concatenate([slot["v"], v], axis=1)

        K, V = slot["k"], slot["v"]
        T = K.shape[1]
        scale = mx.array(hd ** -0.5, mx.float32)
        scores = mx.matmul(q.transpose(0, 2, 1, 3),
                           K.transpose(0, 2, 3, 1)).astype(mx.float32) * scale
        # 因果遮罩用「物理位置」（KV 的索引），不是 rope 用的內容位置。
        qpos = mx.arange(start_pos, start_pos + L)[None, :, None]   # [1,L,1]
        kpos = mx.arange(T)[None, None, :]                          # [1,1,T]
        keep = kpos <= qpos
        if valid_from is not None:
            # 左補齊：每列前面那幾個 pad 佔了 KV 槽，但不能被看見。
            keep = keep & (kpos >= valid_from[:, None, None])       # [B,1,T]
        keep = keep[:, None]                    # [B,1,L,T]，對齊 scores 的頭維
        scores = mx.where(keep, scores, mx.array(-1e9, mx.float32))
        probs = mx.softmax(scores, axis=-1).astype(q.dtype)
        o = mx.matmul(probs, V.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
        return o.reshape(b, L, H) @ w["o"].T

    def _mlp(self, x, w):
        a = x @ w["g"].T
        a = a * mx.sigmoid(a)                    # SiLU
        return (a * (x @ w["u"].T)) @ w["d"].T

    def forward_range(self, h, start_pos, lo, hi, kv, rope_pos=None,
                      valid_from=None):
        """rope_pos: 每列自己的內容位置（左補齊時不同）；valid_from: 真實 token 起點。"""
        cfg = self.cfg
        L = h.shape[1]
        if rope_pos is None:
            rope_pos = mx.arange(start_pos, start_pos + L)
        for i in range(lo, hi):
            w = self.layers[i]
            n = rms_norm(h, w["n1"], cfg.eps)
            h = h + self._attn(n, w, kv[i], rope_pos, start_pos, valid_from)
            n = rms_norm(h, w["n2"], cfg.eps)
            h = h + self._mlp(n, w)
        return h

    def logits_all(self, h):
        hn = rms_norm(h, self.norm_w, self.cfg.eps)
        return hn @ self.lm_head.T, hn            # [1, L, V]

    def crop_kv(self, kv, lo, hi, length):
        for i in range(lo, hi):
            if kv[i]["k"] is not None and kv[i]["k"].shape[1] > length:
                kv[i]["k"] = kv[i]["k"][:, :length, :, :]
                kv[i]["v"] = kv[i]["v"][:, :length, :, :]

    def new_kv(self):
        return new_kv(self.cfg.n_layers)


def make(cfg_or_none=None, path=None, lo=0, hi=None, embed=True, head=True):
    return RealModel(path, lo=lo, hi=hi, need_embed=embed, need_head=head)
