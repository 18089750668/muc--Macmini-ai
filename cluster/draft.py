"""草稿來源（drafter）：投機解碼唯一真正的瓶頸是「草稿對不對」。

為什麼要這支檔案：整條路線的速度公式是

    每 token 成本 ≈ (tau_chain + hop) / g         g = 每輪被接受的 token 數

`g = 1` 等於沒有投機（每次送 block 個候選、只收 1 個，純虧）。所以要嘛把 draft
選對，要嘛這條路線的快只能來自 batching（吞吐），單人延遲不會動。

兩個現成的草稿來源，成本不一樣：

- `LookupDraft`（prompt-lookup / n-gram）：**零成本、零額外記憶體**，在已經看過的
  上下文裡找相同的 n-gram，把後面的 token 當草稿。對「有重複結構的文字」（列表、
  表格、程式碼、引用原文）有效；對自由散文幾乎無效（Shard 量到 0.012 接受率）。
- `RepeatDraft`：拿上一個 token 重複 N 次。最爛的草稿，用來當下界。

模型型草稿（0.5B 配 1.5B 那種）需要「同一個 tokenizer 的更小權重」，本機沒有這個
檔案，所以只留接口說明、不寫沒驗過的實作：**沒量過的草稿等於沒有草稿。**
"""
import collections
import time

import mlx.core as mx


class RepeatDraft:
    """最笨的草稿：上一個 token 重複 block 次。當下界用。"""

    name = "repeat"

    def propose(self, ctx, block):
        return [ctx[-1]] * block


class LookupDraft:
    """prompt-lookup：在 ctx 裡找「最近的相同 n-gram」，把它後面的 token 當草稿。

    細節三個：
    - **從近往遠找**（`range(len(ctx)-1-n, -1, -1)`），因為最近的出現最可能重演。
    - **n 從大到小退**：3-gram 找不到就退 2-gram、1-gram，找不到就回空清單（這一輪
      就退化成 greedy，只虧一個空草稿的判斷，不虧 block）。
    - **草稿長度可以短於 block**：長度短的驗證也成立（chunk 大小是可變的），
      所以不需要硬湊。
    """

    def __init__(self, n=3):
        self.n = n
        self.name = f"lookup{n}"

    def propose(self, ctx, block):
        if len(ctx) < 2:
            return []
        hi = min(self.n, len(ctx) - 1)
        for n in range(hi, 0, -1):
            key = list(ctx[-n:])
            for i in range(len(ctx) - 1 - n, -1, -1):
                if list(ctx[i:i + n]) == key:
                    cand = list(ctx[i + n:i + n + block])
                    if cand:
                        return cand
        return []


class MixedDraft:
    """先問 lookup，找不到就退回 repeat。

    這個組合的用意：lookup 在「有重複」的文字上很準、在自由散文上完全空手；
    repeat 在散文上偶爾能中一個（例如「的」「,」這種高頻接續）。兩者都不會讓
    正確性出問題——驗證那一步只接受對得上的前綴。
    """

    def __init__(self, n=3):
        self.lookup = LookupDraft(n)
        self.repeat = RepeatDraft()
        self.name = f"mixed{ n }"

    def propose(self, ctx, block):
        d = self.lookup.propose(ctx, block)
        if d:
            return d
        return self.repeat.propose(ctx, block)


class SelfDraft:
    """自我草稿（early exit）：**同一個模型只跑到第 d 層就輸出 logits**。

    為什麼值得試：模型型草稿要「同 tokenizer 的更小權重」，本機沒有那個檔案；
    但同一個模型的前 d 層 + 同一顆 lm_head 就是一個現成的、tokenizer 必然相容的
    小草稿。代價是每步只讀 d 層權重（加一次 lm_head，那塊 272MB 省不掉）。

    兩個不變量：
    - 草稿有自己的 KV（不能跟主鏈共用，主鏈的 KV 在別的機器/別的形狀裡）。
    - 草稿 KV 永遠「自我對齊」：每次 propose 開頭檢查長度，短了就補算、
      長了就裁掉。所以呼叫端不需要記得幫它裁。
    """

    def __init__(self, cfg, model_dir, layers=4):
        from .realmodel import RealModel
        from .model import new_kv
        self.cfg = cfg
        self.layers = layers
        self.model = RealModel(model_dir, lo=0, hi=layers, need_embed=True,
                               need_head=True)
        self.kv = new_kv(cfg.n_layers)
        self.n = 0
        self.name = f"self{layers}"
        self.draft_ms = 0.0

    def reset(self):
        from .model import new_kv
        self.kv = new_kv(self.cfg.n_layers)
        self.n = 0

    def _fwd(self, toks, pos):
        x = self.model.embed(toks)
        h = self.model.forward_range(x, pos, 0, self.layers, self.kv)
        lg, _ = self.model.logits_all(h)
        mx.eval(lg)
        return lg

    def propose(self, ctx, block):
        t0 = time.perf_counter()
        target = len(ctx) - 1                     # 草稿 KV 應該有的長度
        if self.n > target:                       # 被拒的那幾個 token 要丟掉
            self.model.crop_kv(self.kv, 0, self.layers, target)
            self.n = target
        if self.n < target:                       # 落後的補算（一輪通常只落後 1~2 個）
            gap = list(ctx[self.n:target])
            self._fwd(gap, self.n)
            self.n = target
        drafts, cur, p = [], ctx[-1], target
        for _ in range(block):
            lg = self._fwd([cur], p)              # 寫入位置 p 的 KV
            self.n = p + 1
            cur = int(mx.argmax(lg[0], axis=-1).tolist()[-1])
            drafts.append(cur)
            p += 1
        self.draft_ms += (time.perf_counter() - t0) * 1e3
        self.name = f"self{self.layers}"
        return drafts


def make(kind, n=3):
    kind = (kind or "lookup").lower()
    if kind.startswith("lookup"):
        return LookupDraft(n)
    if kind == "repeat":
        return RepeatDraft()
    if kind == "mixed":
        return MixedDraft(n)
    raise SystemExit(f"unknown drafter: {kind}")


class HitStats:
    """草稿命中率的旁觀統計（和 metrics 的 g 是兩件事：g 算接受幾個 token）。"""

    def __init__(self):
        self.rounds = 0
        self.proposed = 0
        self.empty = 0
        self.lengths = collections.Counter()

    def note(self, drafts):
        self.rounds += 1
        self.proposed += len(drafts)
        self.lengths[len(drafts)] += 1
        if not drafts:
            self.empty += 1

    def summary(self):
        return dict(rounds=self.rounds, avg_len=self.proposed / max(self.rounds, 1),
                    empty_rate=self.empty / max(self.rounds, 1),
                    lengths=dict(self.lengths))
