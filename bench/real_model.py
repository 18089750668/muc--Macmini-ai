"""真模型（Qwen2.5-0.5B）第一次跑起來：先證明它會講話，再證明切開不變。

兩個檢查，缺一不可：
  1. 完整模型 greedy 生成一句，看是不是人話（權重 / attention / RoPE 有沒有接錯）。
  2. 同一份權重在第 12 層切開、手動傳遞 hidden，產生完全一樣的 token
     （分片正確性的前提；後面 2 台、4 台都靠這個）。

    PYTHONPATH=. .venv-arm/bin/python bench/real_model.py
"""
import time

import mlx.core as mx
from tokenizers import Tokenizer

from cluster.model import new_kv
from cluster.realmodel import RealModel

PATH = "models/qwen2.5-0.5b"


def make(lo, hi, embed, head):
    return RealModel(PATH, lo=lo, hi=hi, need_embed=embed, need_head=head)


def prefill_and_first(models, ids):
    """models: [(model, lo, hi)]，第一段負責 embed、最後一段負責 logits。"""
    n_layers = models[-1][0].cfg.n_layers
    kv = new_kv(n_layers)
    h = models[0][0].embed(ids)
    for m, lo, hi in models:
        h = m.forward_range(h, 0, lo, hi, kv)
    last = models[-1][0]
    lg, _ = last.logits_all(h)
    return kv, int(mx.argmax(lg[0, -1]))


def step(models, kv, tok, pos):
    h = models[0][0].embed([tok])
    for m, lo, hi in models:
        h = m.forward_range(h, pos, lo, hi, kv)
    last = models[-1][0]
    lg, _ = last.logits_all(h)
    return int(mx.argmax(lg[0, -1]))


def generate(models, tok, prompt, n_new, label):
    ids = tok.encode(prompt).ids
    kv, nxt = prefill_and_first(models, ids)
    out = [nxt]
    t0 = time.perf_counter()
    for _ in range(n_new - 1):
        pos = len(ids) + len(out) - 1
        out.append(step(models, kv, out[-1], pos))
    dt = time.perf_counter() - t0
    text = tok.decode(out)
    print(f"{label}: {dt/(n_new-1)*1e3:.1f}ms/token  {(n_new-1)/dt:.1f} tok/s")
    print(f"  {prompt!r} -> {text!r}")
    return out


def main():
    tok = Tokenizer.from_file(f"{PATH}/tokenizer.json")
    whole = make(0, 24, True, True)
    print(f"n_layers={whole.cfg.n_layers} hidden={whole.cfg.hidden} "
          f"heads={whole.cfg.n_heads}/{whole.cfg.n_kv} vocab={whole.cfg.vocab}")

    a = generate([(whole, 0, 24)], tok, "The capital of France is", 24, "完整 24 層")

    left = make(0, 12, True, False)
    right = make(12, 24, False, True)
    b = generate([(left, 0, 12), (right, 12, 24)], tok,
                 "The capital of France is", 24, "切成 0:12 | 12:24")

    print("token 逐位相同:" , a == b)


if __name__ == "__main__":
    main()
