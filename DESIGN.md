# cluster — 多機流水線推理原型（路線 C）

目標不是跑 671B，而是把**四層**跑通並可量測。速度只有接上真集群才有意義。

## 四層

| 層 | 頻率 | 本原型狀態 |
|---|---|---|
| Placement：層分配、ring 排序、協調器定位 | 一次性 | `planner.py`（DP 最小化最大段） |
| **Pacing：在飛幀數、塊大小、背壓** | **每 token** | `coordinator.py`（唯一需要自己發明的） |
| Transport：直回、校驗、心跳 | 每層 | `protocol.py` + `stage.py` |
| Verification：故障檢測 ≠ 誠實驗證 | 每次返回 | `protocol.py` 的 crc + shape/dtype 檢查 |

## 三個關鍵設計決定

1. **KV 按 `start_pos` 裁切**。投機被拒時用更早的 `start_pos` 重跑，舊 KV 直接被覆蓋，不做快照。回滾因此是無操作。
2. **在飛深度硬 clamp 在 `block + 1`**。Shard 用 10 台實測換來的：`max_inflight = block_size + 1`，第 10 台以後永遠閒著。
3. **尾節點直回協調器**，不原路返回。

## 模組

```
cluster/
  config.py        ToyConfig（層數/hidden/heads/vocab）
  model.py         ToyModel：rmsnorm / rope / attn(KV) / mlp / embed / logits
  protocol.py      48-byte struct header + raw payload，一帧一次 sendall / 一次 slab read
  stage.py         StageServer：一个层区间 + KV，收帧、forward、回 RESULT
  coordinator.py   Head：prefill、greedy 解码、投機 chunk 驗證、accept、裁切、forward_rows（批次）
  serve.py         最小批次服務器：收多條請求 → 左補齊湊批 → 一起解碼 → 分別串流
  draft.py         草稿來源（repeat / prompt-lookup / early-exit）、命中統計
  pacing.py        block 控制器：k 的 EWMA → block ≈ 2k+1，k 太低就關掉投機（會探測）
  planner.py       DP 分段：時間 + 記憶體雙約束，objective = max | sum
  footprint.py     逐節點記憶體核算（權重 + head + KV × ctx），不跑模型就能算爆不爆
  probe.py         在活迴圈量 tau 與 stage_fixed，寫 profile.json 餵回 planner
  metrics.py       tau / L / g / frames_per_token / useful_inflight
  main.py          CLI：single | run | stage | plan
bench/
  m0_baseline.py   逐 op 启动开销 + 每层 tau + 单机 decode tok/s
  transport_only.py 真协议 + 空模型的纯传输往返
  transport_micro.py 裸 TCP RTT + CPU 端逐项拆解
  fault_test.py    跑到一半 SIGKILL 掉 stage，验 head 会不会沉默
```

## 四個後來補上的語意（都在實測之後才寫對）

1. **目標函數不是「最小化最慢段」而是「最小化總和」**。batch=1 是接力不是並行：16 台各 1 層 = 每 token 走 16 段，時間相加。`max` 只在投機解碼把 sum 問題改造成 max 問題之後才成立。
2. **每段有固定成本 `stage_fixed`（本機 0.274ms）**，段數越多虧得越多；規劃時必須 `fixed + n*tau` 而不是 `n*tau`。
3. **記憶體是硬約束，且不對稱**：權重與 KV 都掛在持有該層那台，node 0 另外背 embed + lm_head。
4. **容錯的第一要求是「不要沉默」**，不是「自動恢復」：掉線 → 立刻 `RuntimeError`；卡住 → 15s 後 `treat the cluster as degraded, not as slow`。

## 第五個補上的語意：形狀必須是常數（`FastModel`）

活回圈的教訓：同一段 8 層，固定形狀暖回圈 0.10ms/層、KV 每步在長的活回圈 0.38ms/層。差 4 倍，差別不在計算而在**每步都要重新規劃與派發**（還在長 KV + 每次不同形狀）。

所以解碼路徑改成 `cluster/fastmodel.py`：

- KV 每層一塊預分配 slab `(ctx_len, nh, hd)`；寫入用 `slice_update`，不 `concatenate`。
- attention 讀滿 slab，`ar <= pos` 遮罩未來位置（浪費 FLOPs，換掉形狀變化）。
- 層棧按 `(lo, hi)` 只 `mx.compile` 一次。
- prefill 用另一個按 `L` 快取的批次 graph（`forward_batch`），否則逐 token 走鏈 = 鏈延遲 × prompt 長度。
- **副產品**：投機回滾不再需要 crop。遮罩讓未被覆寫的舊 KV 自動失效，`crop()` 在 fast 模式是 no-op。

量到的邊界（重要）：
- **只用固定 slab、不 compile 是負收益**（8 層 4.911ms vs 3.334ms）。加速來自 compile，不是來自預分配。
- **單一大 slab 是負收益**（0.64×）：`slice_update` 返回新陣列，大 tensor 一次寫入複製全部。必須每層獨立。
- fixed 幾乎沒降（0.274 → 0.260ms/段），tau 降 1.8×（0.383 → 0.211）。仍有與層數無關的固定開銷沒被吃掉。
- **`slice_update` 是純函數，每步都複製整塊 slab**：640 → 1.84ms/8層，8192 → 5.04ms。成本是 O(ctx) 不是 O(L)，所以（a）短上下文 + 長生成才划算；（b）投機的 5-token 批次反而更慢（4 台 66.67 vs 103.12 tok/s）。要上 32K 得改成分塊/paged slab，而 `mx.compile` 不支援原地寫，這超出 Python 層能解的範圍。

## Pacing 控制律（待 M3 標定）

- 量三個數：`tau`（最慢段單幀時間）、`L`（環一圈 RTT）、`g`（每塊接受 token 數）
- 目標：`useful_inflight = tavg / frames_per_token`，維持接近 `block + 1` 但**不填滿**
- 填滿會腰斬：Shard 實測 block 8 填到 99.6% 時 30.5 → 14.1 tok/s，診斷指標 `topup_disagree` 從 0 漲到 1496
- 控制律：`depth ≤ block + 1`；refill 只在 `depth < block + 1` **且** disagreement 為 0 時觸發，一旦上升就退回 floor=1

## 失效與恢復語意（已實作並驗證）

| 事件 | 偵測 | 主機判定 | 動作 |
|---|---|---|---|
| 節點被殺 / 崩潰 | socket EOF，**立刻** | `StageLost(kind=closed)` 可恢復 | 整鏈重建 → 重放前綴 → 續跑 |
| 節點卡住（沒死但無回應） | 15s 無 RESULT | `StageLost(kind=timeout)`，訊息 `degraded, not slow` | 同上 |
| 恢復次數超過上限 | — | 放棄 | 拋出原錯誤（不假裝成功） |

三條規則：

1. **重放是唯一不會錯的恢復**。head 持有權威 token 前綴，重建後全鏈重跑該前綴，每個 stage 的 KV 必然與 head 一致。保留存活節點省下的幾百毫秒，換的是 KV 悄悄錯位——會輸出看起來正常的錯 token。
2. **恢復有成本，就是前綴長度**。841 tokens 重放 0.31s（≈0.37ms/token）。所以「容錯能力」實際受上下文長度限制。
3. **planner 說「不要分散」時要照做**。`objective=sum` 下 4 台的最優解是 1 台（248.8 vs 91.5 tok/s）。`run --nodes 4 --objective sum` 會直接落回單機本地跑，這是正確行為。

`--on-loss rebuild|shrink|abort`：rebuild 原樣重建；shrink 剔除節點後按剩餘數量重規劃（4→3 台實測通過）；abort 直接失敗。

## 本輪刻意簡化（原型限制，不是設計）

- 鏈式分片已支援任意節點數（1/2/3/4/8 實測，逐 token 對照單機 hash 一致），但 coordinator 與 head 合併在一進程，全部是本機 loopback 的進程間，不是跨機器。完整設計裡 coordinator 不持有任何層。
- 容錯只做到**偵測**（斷線立刻報錯、卡住 15s 超時），沒有重連、沒有心跳、沒有掉一台後按剩餘機器重新分片。
- 加密（ChaCha20-Poly1305）未實作，只有 crc32 + 長度/shape 校驗。本機 localhost 測試。
- 草稿：已換成真的 prompt-lookup（`cluster/draft.py`），實測 k = 0.09~4.20（隨文字
  類型變），散文 1.3×、程式 2.1×、照抄 3.9×，全部逐 token 與 greedy 相同。
  **early-exit（同模型前 d 層）實測 k = 0**：24/24 層能完全重現主模型的預測，
  砍到 18/24 就開始不同，所以問題不在接線而在「這顆模型砍一半深度就崩」。
  想要更高的 k，只能等一個**真的同 tokenizer 小模型**或訓練過的 draft head。
- 後端只有 MLX。numpy 後端與 Linux VM 節點是 TODO。

## 多機連接（實體層語意）

- **度數是硬限制**：基礎版 M4 mini 有 3 個 TB4 口，每條線吃兩端各一個 → 最大度數 3。環用 2、全互聯要 n-1（16 台 120 條線，不可行）。所以拓撲不是選出來的，是口數決定的。
- **stage 不一定要本機 spawn**：`--peers host:port,...` 表示「這些 stage 已經在別台跑著」，head 只負責連第一個；`--advertise host:port` 是最後一段回報 head 用的位址。少了這兩個參數，多機就只能靠同機 spawn 假裝。
- **head 永遠是 node 0**：它持有 embed 與完整 token 序列，容錯時重放靠它。因此 head 所在的機器是單點——但它是「便宜的那台」，不是最貴的那台。
- **每台一個 L2**：TB4 點對點直連各成網段，靜態 /30 或 169.254.x；不要 DHCP，不要串交換機（串了就拿不到 TB4 的低延遲）。

## 心跳與「卡住 vs 死掉」

固定 15 秒超時是個謊：一條往返 4ms 的鏈，等 15 秒才判異常等於寬容 3750 倍；
而遠程鏈路抖一下又遠不到 15 秒，於是「慢」和「死」被混在一起，恢復策略也就
只能猜。所以門檻必須跟著實測基線走：`deadline = clamp(20 × 每輪延遲, 0.5s, 15s)`。

判斷分三種，只有三種處理方式：

| 現象 | 訊號 | 判定 | 處理 |
|---|---|---|---|
| 進程沒了 / 機器關機 | socket EOF、RST，立刻到 | `closed` | 立即重建 + 重放 |
| 進程在但不算（SIGSTOP、卡死、被系統凍結） | 到門檻仍無 RESULT，心跳也不回 | `hung` | 立即重建 + 重放 |
| 機器活著只是慢 | 到門檻仍無 RESULT，但心跳回得很快 | `timeout` | 再等，別亂重建 |

心跳（`HEARTBEAT`）必須**整條鏈走一圈**才算數：頭在 `in_sock` 上等，只有尾段
回報那條會回到這裡。所以量到的是全鏈往返，順便也就是一次免費的 hop 量測。
在開工前先 `ping()` 一次，比跑到一半才發現鏈路早就死了便宜太多——遠程場景
筆記本放半小時再回來，這件事只有問了才知道。

## 真模型接入（realmodel.py）

玩具模型的權重是隨機的，所以「投機草稿能猜中多少」量不出來——隨機權重下任何
草稿都沒有預測力。而 k 正是整條路線真正的瓶頸，要量它必須先有真權重。現在接的
是 Qwen2.5-0.5B（fp16，24 層，hidden 896，14 頭 / 2 個 KV 頭）。

真模型和玩具模型只差四個地方，但每個都會安靜地錯：

1. **GQA**：14 個 query 頭共用 2 個 KV 頭。KV 只存 2 份（省 7 倍記憶體），算
   attention 前才展開成 14 份。展開順序要跟 HF 一致（同一組內連續複製），
   否則形狀合法、結果全歪。
2. **q/k/v 投影帶 bias**：Qwen2 有，玩具沒有。漏掉 bias 不會報錯，只會慢慢變笨。
3. **tie_word_embeddings**：輸出矩陣就是 embedding 本身。所以最後一台仍然必須
   放 vocab×hidden（151936×896 ≈ 272MB fp16）——這塊切不開，是每條鏈的固定成本。
4. **權重載入方式**：一開始用 `mx.load()` 整檔 988MB 再丟掉不要的層，四個 stage
   同時起就是四份 988MB 峰值，16GB 機器直接換頁，於是「加機器更快」被自己的
   載入方式拖死（4 台比 1 台慢 24%）。改成按 tensor 讀 safetensors（讀 header 拿
   `data_offsets`，只讀自己那幾層）之後，同樣 4 台只慢 8%。**分片要從「只讀自己
   那幾層」開始，不是從「算完丟掉」開始。**

## 第六個補上的語意：batch 軸（`forward_rows`）

`step(tokens, start_pos)` 是「一條序列、逐 token」。批次解碼要的是另一支：`forward_rows(rows, start_pos)`，
`rows` 是 B 條**等長**序列，回傳 B 個「下一 token」。四條不變量：

1. **KV 的 batch 軸 = 所有列共用同一個位置索引**。所以第一版硬性要求等長 + 同步：
   一起 prefill、一起前進一格。等長讓「共用一個 `start_pos`」免費成立；不等長就得
   左補齊再加 per-row mask，那是下一步，不是現在。
2. **B=1 時保留舊語意**（每個位置一個 argmax），因為**投機 chunk 驗證要靠它**：送 4 個
   候選 token，回 4 個 argmax，才驗得出前綴對不對。B>1 只取每列最後一個位置。
   `forward_rows` 在 B=1 時自己把結果截成最後一個，否則 prefill 會回 L 個 token、
   對不上 `len(rows)`。
3. **`_exchange` 是共用的**：batch 與單序列走完全一樣的「送出 → 等 → 逾時判斷 →
   心跳探活」。掉線語意不能因為走的是 batch 就變成另一套。
4. **`--fast` 路徑沒有 batch 版**（`FastModel` 的 slab 是固定 ctx 的單列），所以
   `stage._forward` 收到 3 維就直接 `SystemExit`。安靜地降級成單列是最糟的處理。

**k 是整個速度公式裡唯一的乘數。** 速度 ≈ (tau_chain + hop) / g，`g = 1 + k`。
`k = 0` 時每輪送 block 個候選、只收 1 個，**比不投機更慢**（實測 -6%）——因為驗證
要算 block+1 個位置。所以「加大 block」不是萬靈藥：散文的 k 在 0.34 就飽和，
block 8 反而比 block 4 慢；照抄的 k 到 4.20，block 8 給 3.9×。**block 要跟 k 匹配**，
而 k 由文字類型與草稿品質決定，不由機器數決定。

**batch 買的是吞吐，不是延遲。** 一個人一次問一句話，B 永遠是 1，每 token 的
100ms 一點都不會變；batching 只在「同時有多條請求」時才有收益。這條界線要寫在
文件裡，不然會被誤讀成「加了 batching 就變快」。

**不等長：左補齊，不是右補齊。** 右補齊的話最後一格是 pad，argmax 出的是「pad 之後」的
token，直接錯；左補齊讓每條真實 token 都結束在最後一格。代價是 pad 進了 KV，要兩個
東西同時到位才不會歪：`valid_from`（per-row 遮罩，擋掉前面的 pad）與**內容位置**
（絕對位置模型要看到 0..len-1，不是 pad 之後的編號）。兩個都漏掉任何一個，輸出都會
「看起來正常但是錯的」。

## 第九個被迫承認的設計：控制器會把自己鎖死

pacing 依 k 決定 block，k 由「送出去的草稿有幾個被接受」量出來。這兩個東西是一個
迴路，而迴路有正回饋的死角：**k 低 → block=0 → 不再送草稿 → 沒有新的證據 → k 永遠是 0**。
第一次上線就是這樣卡住（serve 日誌連續 `k=0.00 block=0`，速度掉回 greedy）。

修法不是「調大 alpha」這種參數遊戲，而是**偶爾探一次**（每 5 輪送 2 個草稿）。
一條「量不到就等於不存在」的迴路，必須自帶探索，否則它會把一個「其實有 0.3」的
世界誤判成「永遠是 0」。這跟 bandit 裡的 exploration 是同一個問題，不是調參問題。

另一條界線要寫清楚：**pacing 決定的是 block，不是「要不要投機」。**
要不要投機是 k 的函數；k 又依賴文字類型。所以「加了投機就變快」不成立，
成立的是「在有重複結構的文字上，投機值 1.3×~3.9×」。

## 第七個補上的語意：frame 的尾巴（可變 meta）

48 字節定長 header 只裝得下五個純量（`req / chunk_id / start_pos / n_in / tau_ms`）。
batch 需要 `valid_from` 這種「每列都不一樣」的東西，而它原本的命運是**被靜默丟掉**：
head 端算的是左補齊 + 遮罩，stage 端算的是沒有遮罩、位置從 0 開始，兩邊都不報錯。
單機完全正確，2 台 batch 從第一個 decode token 就歪。

修法：用 header 的 `flags` 欄位存「JSON 尾巴的長度」，尾巴接在 payload 後面。
沒有額外鍵時尾巴是空的，熱路徑一個字節都不多。**代價最小的做法是把可推導的東西
留在家裡算**：`valid_from` 是 B 個整數，位置編號由 `start_pos + i - valid_from`
在每個 stage 自己推，不必上線傳。

教訓寫進這裡而不是只寫在 commit：**一個「只裝得下五個欄位」的 header，會在功能
擴張時安靜地吃掉新欄位**。它不會撞壞，只會讓兩端算不同的東西。

## 遠程連接（出門在外，不靠公網 IP）

家裡機器在 NAT / CGNAT 後面，**沒有可開的埠**。所以連線方向必須反過來：家裡那台
主動往外撥、把連線一直掛著。這就是全部設計：

- **一個中轉機**（雲 VM 或任何有公網 IP 的盒子）跑 `remote gateway`。它只做兩件事：
  驗 token、把兩條連線接成一條。
- **家裡**跑 `remote publish --name m1 --to 127.0.0.1:<stage端口>`：連上 gateway 掛號，
  等人來配對。
- **筆記本**跑 `remote consume --listen 9001 --name m1`：在本地開一個埠，任何連進來的
  東西都轉去 m1。

於是**模型協議完全不用改**：遠端 stage 在 head 眼裡就是 `127.0.0.1:9001`。反向那條
（tail 回報 head）用同樣一對原語：筆記本 `publish --name headin --to 127.0.0.1:<head埠>`，
家裡最後一段 `--reply 127.0.0.1:9002`，而 9002 是 `consume --name headin`。

三個踩過的細節，都是「靜默錯誤」的來源：

1. **publish 要先連上自己的後端，再去 gateway 掛號**。順序反了會出現「掛了號、後端卻
   已死」，配對出去的那條一寫就是 broken pipe，訊息完全指不到真因。
2. **gateway 的 OK 必須在配對成功之後才回**。先回 OK 等於對兩邊說「可以了」，實際上
   對面根本還沒有人，寫進去的東西直接掉。
3. **stage 加 `--persist`**：ABORT 只結束這一輪，不結束進程；上游斷了也回到 accept
   等下一次。新上游連進來時**清空 KV**——因為 head 重連後會重放整個前綴，殘留的
   舊 KV 只會把 token 悄悄算錯。

認證與「不用再認證」：token 存在 `~/.cluster/token`（0600），publish/consume 迴圈
失敗就退避重連。開機自啟靠 launchd（`RunAtLoad` + `KeepAlive`）：
`python -m cluster.remote plist --relay <host:port> --name m1 --to-port <stage端口>` 印出
plist，裝進 `~/Library/LaunchAgents/` 就好。之後重開機不需要任何手動認證。

**安全**：gateway 只搬 bytes，不加密。token 是認證不是加密，被偷就是全開。對外之前
必須套 WireGuard / Tailscale，或在 gateway 前面加 TLS。

## 容量是這條路線唯一無條件成立的東西（`cluster/capacity.py`）

設計上的三條硬規則，寫進 planner 之前就必須成立：

1. **可用記憶體要用 24GB（32GB 機型），不是 32GB。** 系統與 Metal 保留吃掉約 8GB。
   用標稱值算，表會樂觀三成。
2. **KV 跟權重同一本帳，掛在持有那幾層的那台。** 所以「層多的台」付兩次；不對稱是真的。
3. **頭尾兩台額外各揹一份 vocab×hidden**：頭揹 embedding，尾揹輸出矩陣。tie embeddings
   的模型也一樣——那顆權重必須實體存在於尾那台的記憶體，所以整條鏈付兩份。

另外兩個實測出來的規劃餘裕（`bench/verify_capacity.py`）：

- **載入 embed/lm_head 的瞬間會多一份轉換暫存**（0.5B 上是 +280MB，等於一份 embedding）。
  「每台剛好塞滿」的配置會在啟動那一刻失敗，不是跑起來才失敗。
- 中間段 RSS = 自己的層 + ~40MB 固定開銷；固定開銷不隨台數消失。

結論：96GB（4×32GB 可用）能跑 120B dense，5 台能跑 200B dense 的 32K 上下文。這是
容量帳；速度仍是接力帳（tok/s 不隨 n 長），只有 MoE 的低 active 參數能改後者。

## 為什麼不做「按專家切分」（EP）

結論先寫：**MoE 該用 pipeline，不該用 expert parallel**，至少單人使用時如此。

三條理由，數字在 `cluster/moe.py` 與 RESULTS.md：

1. **EP 的稅按層數收，pipeline 的稅按機器數收。** EP 每層兩次 all-to-all（dispatch
   + combine），48 層就是 96 次往返；pipeline 4~16 台只有 3~15 次。單人場景 96 ≫ 15。
2. **attention 切不開。** 專家切分只切 MoE 的 FFN；attention/embedding 是稠密部分，
   每台都要有一份。batch=1 時它只能落在一個節點上，125B 的稠密部分 12.5GB 每 token
   整份讀一遍 = 104ms，比整條 pipeline 鏈還慢。671B 更誇張：稠密部分 67GB，一台
   24GB 根本放不下。
3. **「切完就變 10B」只在算力上成立。** 權重 62.5GB 不會消失，容量該多台還是多台。

EP 什麼時候才對：多用戶（batch 大）、星型 1 跳 fabric、稠密部分放得下單台——也就是
vLLM/SGLang 的部署形態，靠 `deepep_low_latency` 服務 decode。家用單人不是那個形態。

## 跑法

```bash
# 單機基線（全部層在一個進程）
.venv-arm/bin/python -m cluster.main single --max-tokens 32

# 兩節點：tail 在另一個進程/端口，head 驅動
.venv-arm/bin/python -m cluster.main run --layers 0:4 --tail 127.0.0.1:5201 --max-tokens 32 --mode greedy
.venv-arm/bin/python -m cluster.main run --layers 0:4 --tail 127.0.0.1:5201 --max-tokens 32 --mode spec --block 4

# M0 量測
.venv-arm/bin/python bench/m0_baseline.py
```

## 第十個被迫承認的設計：鏈不是環，回程要自己造

**物理事實先擺前面**：家裡的接法是「每台只接上下家」，4 台就是 3 條線，16 台是 15
條線。端點只佔 1 個 TB4 口，中間每台佔 2 個。不是全互聯（4 台就要 6 條線、每台
3 個口全滿），也不是環（環要多拉一條「尾巴直連頭」的線）。

**這條事實把程式裡一個默認假設打穿了**：舊版最後一台用 `--reply host:head_port`
直接把結果送回 head，等於假設「尾巴能一步回到頭」。只有相鄰兩台有線的鏈上，
這個假設是假的 —— TB4 每條線是獨立的點對點 L2，沒有跨跳路由。

### 試過但被否掉的：在同一條 socket 上雙向收發

第一版讓中間那台在送完下游之後，就著同一條 `conn` 讀 RESULT、再往上游寫。
實測**不穩**：同一個進程裡第二次建鏈，整條鏈靜默吐出錯 token（`[171,171,...]`），
而且 head 端送出的 hidden 本身就先錯了。這條路後來被證明不是「雙向收發」的問題
（見下面第 3 條），但既然一個機制能讓錯誤變成「不報錯的錯答案」，就不留它。

### 採用的：每台一個獨立回程口（`--ret-port`）

- 每台（除了尾巴）多開一個監聽埠，起一個 `_relay` 執行緒：收下游送上來的 frame，
  原樣轉給自己的 `reply_to`（上游那一台的回程口；最前面那台就是 head）。
- 第 i 台的 `--reply` 指「上游那一台的回程口」，尾巴的 `--reply` 指它的上家。
- head 恢復「等第二條連線」，只是那條連線現在來自**最近的一台**而不是尾巴。

代價：回程每 frame 多一次「收 + 轉發」，也就是每多一台多一跳。實測 4 台
114.9 vs 直連 123.7 tok/s（同一顆模型、同一組切分、逐 token 完全一致）。

### 順手被這件事逼出來的三個真 bug

1. **只有尾巴可以連 `--reply`**。中間那些台如果也連了，head 的 `accept()` 會先
   搶到一條「永遠沒人讀」的空連線：心跳全部超時，但沒有任何錯誤訊息。症狀是
   「2 台正常、4 台必掛」——因為 2 台時中間根本沒有別的台。
2. **head 的監聽埠不能是固定號**（`--head-port 0` 交給核心配置）。固定號在反覆
   重建時會被上一次沒退乾淨的尾巴搶走 accept，head 於是跟一個死掉的尾巴握手。
3. **玩具模型一個進程只能建一次權重**。同一個 seed、同一個 cfg，第二個 `ToyModel`
   開始同一個 prompt 的 hidden 就從 1.2418 漂到 0.9852 —— 不報錯、形狀全對。
   這件事一度被誤判成「鏈式回程的 bug」，實際上是權重重建不穩定；現在按 cfg
   快取，一個進程一份權重（真實部署本來也只載入一次）。

### 冷啟動不是掉線

`--chain` 之後，第一跳的 stage 要 accept、再往下一跳連、再往上游連回程口，
三段 TCP 握手疊起來，第一個心跳很容易超過 250ms。所以 `ping()` 的預算是
「沒量到基線就用 2s，最多試 3 次」，`_probe_alive()` 才是「只問一次」的那把尺
（它的工作是分辨「慢」還是「卡死」，不是再給一次機會）。

## 驗收不變量

- `single` 與 `run --mode greedy` 的輸出 token 序列**逐 token 相同**
- `max_inflight ≤ block + 1`
- 每輪 `start_pos` 與實際 KV 長度一致（裁切正確）
