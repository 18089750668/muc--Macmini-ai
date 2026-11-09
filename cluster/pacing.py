"""pacing：投機解碼唯一需要「線上調」的參數是 block，而 block 該由 k 決定。

為什麼不能固定 block：實測同一台機器同一個模型，散文的 k 飽和在 0.34（block 8 反而
比 block 4 慢），照抄的 k 到 4.20（block 8 給 3.9×）。所以「固定 block=8」在散文上
是純虧，在照抄上還不夠。**block 要跟 k 匹配。**

這個控制器只做三件事：

1. **估 k**：每輪知道「送了幾個草稿、收了幾個」，用 EWMA 追（文字類型會漂，
   不能累積平均）。
2. **定 block**：`block ≈ 2k + 1`，夾在 `[1, max_block]`。k→0 時直接回 0（不投機），
   因為「送 block 個只收 1 個」比 greedy 更慢（實測 -6%）。
3. **記帳**：tau（每輪計算）、hop（一輪的往返），讓 `g` 與「實際每秒」可以事後對帳。

它不是 PI 控制器，也不預測未來。基礎模型上「k 的變化是文字類型造成的階躍」，
不是連續漂移，用 EWMA + 閾值就夠了；多餘的迴路只會讓數字更難解釋。
"""


class Pacing:
    def __init__(self, max_block=8, alpha=0.3, spec_floor=0.08, warmup_block=4,
                 probe_every=5, probe_block=2):
        self.max_block = max_block
        self.alpha = alpha
        self.spec_floor = spec_floor   # k 低於這個就關掉投機（block=0）
        self.k = None                  # EWMA of accepted drafts per round
        self.tau_ms = None
        self.hop_ms = None
        self.rounds = 0
        self.accepted = 0
        self.drafted = 0
        self.greedy_rounds = 0
        self.probe_every = probe_every   # 關掉投機之後，每隔幾輪探一次
        self.probe_block = probe_block
        self.probes = 0
        self._since_probe = 0
        self._warmup = warmup_block

    # -- 決策 -------------------------------------------------------------
    def block(self):
        """這一輪該送幾個草稿。0 = 這一輪走 greedy（不投機）。"""
        if self.k is None:
            return self._warmup
        if self.k < self.spec_floor:
            # **一定得偶爾探一次**：block=0 就不會送草稿，也就不會再有「接受/被拒」
            # 的證據，k 會永遠卡在 0（實測就是這樣卡住的）。探一次只虧一點點。
            self._since_probe += 1
            if self._since_probe >= self.probe_every:
                self._since_probe = 0
                self.probes += 1
                return self.probe_block
            return 0
        # 每多收一個 token，就值得多送兩個候選：驗證多算的位置很便宜（權重只讀一次），
        # 但被拒的位置也要算，所以斜率取 2 而不是更大。
        return max(1, min(self.max_block, int(round(2 * self.k)) + 1))

    # -- 觀測 -------------------------------------------------------------
    def observe(self, accepted, drafted, tau_ms=0.0, hop_ms=0.0):
        """accepted 含 bonus（= 這輪吐出的 token 數），drafted 是送出去的草稿數。"""
        self.rounds += 1
        k = max(0, accepted - 1)
        if drafted == 0:
            self.greedy_rounds += 1
        else:
            self._since_probe = 0
        self.accepted += accepted
        self.drafted += drafted
        if self.k is None:
            self.k = float(k)
        else:
            self.k = self.alpha * k + (1 - self.alpha) * self.k
        if tau_ms:
            self.tau_ms = tau_ms if self.tau_ms is None else (
                self.alpha * tau_ms + (1 - self.alpha) * self.tau_ms)
        if hop_ms:
            self.hop_ms = hop_ms if self.hop_ms is None else (
                self.alpha * hop_ms + (1 - self.alpha) * self.hop_ms)

    # -- 報帳 -------------------------------------------------------------
    def report(self):
        return dict(k=(self.k or 0.0), block=self.block(), rounds=self.rounds,
                    acceptance=self.accepted / max(self.rounds, 1),
                    drafted_per_round=self.drafted / max(self.rounds, 1),
                    tau_ms=self.tau_ms or 0.0, hop_ms=self.hop_ms or 0.0,
                    greedy_rounds=self.greedy_rounds, probes=self.probes)

    def line(self):
        r = self.report()
        return (f"k={r['k']:.2f} block={r['block']} 每輪吐 {r['acceptance']:.2f} token/"
                f"送 {r['drafted_per_round']:.2f} 草稿 tau={r['tau_ms']:.1f}ms "
                f"hop={r['hop_ms']:.1f}ms")
