# 實測結果（M0 / M1 / M2 / 傳輸升級）

機器：MacBook Air M1（16GB）、macOS 27.2、MLX（arm64 venv, Python 3.9）
模型：ToyModel 8 層 / hidden 256 / 4 heads / vocab 512（隨機權重，同一 seed 各進程一致）

> env 的坑：`/usr/local/bin/python3` 是 **x86_64（Rosetta）**，MLX 沒有 x86 wheel，必須用 `/usr/bin/python3`（arm64 native）建 venv。

---

## M0：是不是 launch-bound？—— 是，而且非常嚴重

| 指標 | 數值 |
|---|---|
| 單 op（含強制 sync） | median **184.5 µs**，min 162.7 µs |
| 1 層 forward（L=1） | 260.5 µs |
| 8 層 forward（L=1） | 803 µs → **100 µs/層**（攤薄後） |
| 8 層 decode 端到端 | 4.32 ms/token（231 tok/s） |

**解讀**：8 層 forward 只要 0.8ms，但一個完整 decode step（embed + 8 層 + lm_head + argmax + sync）要 4.3ms。差出來的 3.5ms 是排隊與調度。以每 token 約 200 個 kernel 估算：純調度約 **20 µs/op**，強制 sync 約 **160 µs/op**。

**冷啟動**：同一個 16 token 測試，第一次 27.7 tok/s，第二次 275.4 tok/s——**10 倍差距**，與 Shard 說的「階段間 JIT 串行」同一現象。任何數字不標明冷/熱都是假的。

---

## M1：兩節點 ring 的正確性與代價

分段：head `[0:4]`（含 embed）＋ tail `[4:8]`（含 lm_head），tail 直接回傳 argmax。

| 模式 | token hash | tok/s（熱） | 相對單機 |
|---|---|---|---|
| single（8 層同進程） | `ac06b785c2ea2a82` | 275.4 | 1.00× |
| ring 0:4 / 4:8，greedy | `ac06b785c2ea2a82` | 169.1 → **189.1**（傳輸升級後） | **0.61× → 0.69×** |

逐 token 完全一致，分片沒有改變數學。速度如預期低於單機：batch=1 的 pipeline 是接力，總時間是各段相加再加跳數。

---

## M2：投機 chunk 驗證 + KV 裁切機制

草稿是**佔位實作**（重複上一個 token），接受率必然很低，只用來驗證機制：

| 指標 | 數值 |
|---|---|
| 輸出 hash | `ac06b785c2ea2a82`（**與 greedy 完全相同**） |
| g（每塊接受 token 數） | 1.07（幾乎全靠 bonus token） |
| max_inflight | 5 = `block + 1` ✓ |

**這是機制層最重要的驗證**：用一個刻意很差的草稿，輸出仍與 greedy 逐 token 相同，說明 chunk 驗證索引正確、accept 取最長前綴加 bonus 正確、**KV 按 `start_pos` 裁切的回滾正確**。傳輸升級後重跑，hash 依然一致。

---

## 傳輸升級（v2）：3.1ms → 0.07ms

### 先診斷，不要猜

| 元件 | 成本 |
|---|---|
| 裸 TCP 往返 8KB（echo 進程） | **0.06 ms** |
| JSON encode / decode | 3.06 / 6.61 µs |
| struct pack / unpack | 0.31 / 0.22 µs |
| crc32 over 8KB | 0.33 µs |
| np.tobytes / frombuffer+copy | 0.22 / 0.83 µs |
| mx ← np / np ← mx | 1.36 / 2.06 µs |

**CPU 端全部加起來約 14 µs，線路 60 µs——沒有任何一項能解釋 3.1ms。**

真因是 syscall 次數與隨之而來的多次阻塞：v1 每方向 **3 次 sendall**（4 字節長度、JSON header、payload）＋ **3 次 recv**，每次 recv 都可能重新睡進內核再被喚醒。JSON 從來不是問題，`zlib.crc32` 也不是。

### v2 的做法

- **48 字節定長 struct header**（magic / type / flags / req / chunk_id / start_pos / n_in / ndim / dtype / 3 × dims / nbytes / crc / tau），熱路徑欄位全部進 header，不再有 JSON
- **一幀一次 `sendall`**：header 與 payload 拼成一個 buffer 送出
- **一次 slab read**：接收端預配置 128 KiB `bytearray`，`recv_into` 一次讀進來就地解析；小幀不會再有「讀 4 字節 → 再讀 header → 再讀 payload」的三次等待
- `TCP_NODELAY`、`np.frombuffer` 零拷貝視圖（交由 `mx.array` 那一次必然的拷貝吸收），RESULT 的 token 改用 int32 payload 而不是 JSON 陣列
- 可選的 **polling window**（`--spin-us`）：預期資料要來時不睡進內核，改為 20 µs 輪詢。代價是燒一個核，這是從 syscall 走向 kernel bypass 時一定會調的那個旋鈕

### 結果

| 指標 | v1 | v2（spin=0） | v2（spin=4ms） |
|---|---|---|---|
| ring tok/s | 169.1 | 189.1 | **206.1** |
| round_latency | 3.63 ms | 3.17 ms | 3.05 ms |
| hop（latency − tau_tail） | ≈3.1 ms | 1.14 ms | 1.02 ms |
| **純傳輸往返（無模型）** | — | **0.071 ms**（min 0.034） | 0.061 ms |

`bench/transport_only.py` 用真協議、空模型量出 **0.071 ms median / 0.034 ms min**——**線路這一段達標了（0.1ms 級）**。同時它也證明 polling window 對純傳輸幾乎沒用（0.071 → 0.061），喚醒不是瓶頸。

### 但 hop 還有 1.0ms，而且不在線路上

用不同分段掃一遍每段 tau，看成本隨層數怎麼長（單步、L=1）：

| 某段的層數 | 該段 tau |
|---|---|
| 1 層 | 0.80 ms |
| 4 層 | 2.14 ms |
| 7 層 | 3.06 ms |

擬合 ≈ **0.38 ms/層 + 0.42 ms/段固定**。而 M0 在固定形狀的暖迴圈裡量到的是 **0.10 ms/層**。

**所以活迴圈裡每層貴了約 4 倍，原因是 KV 每步都在長、形狀每次不同，MLX 每步都要重新配置與重新派發。** 剩下那 1.0ms 的 hop 也同理：head 端的 `h[0]` 切片 + D2H 轉換、tail 端的 `logits_all` + `argmax().tolist()`，全是幾十個微操作的固定開銷。

**這正是 Shard 為什麼一定要做 CUDA graph capture、也是它為什麼被 CUDA 綁死**：固定形狀能把幾十個 kernel 壓成一次 graph launch。MLX 這邊對應的手段是 `mx.compile` + 固定長度的 KV slab（一次配置到最大上下文，位置用索引寫入而不是 `concatenate`），不是換更快的網卡。

**結論：transport 這把尺子修好了（0.07ms），但它量出來的真相是——現在的瓶頸是每層的動態形狀派發，不是通訊。**

---

## 任務分配：測了什麼、會出什麼錯

### 一、分配正確性（逐 token 對照單機）

| 節點 / 分段 | prompt | 生成 | 模式 | 輸出 hash | 與單機 |
|---|---|---|---|---|---|
| 1（8 層同進程） | 512 | 32 | greedy | `25b084baf8c43f24` | 基準 |
| 2（4 / 4） | 512 | 32 | greedy | `25b084baf8c43f24` | 一致 |
| **3（2 / 3 / 3，不能整除）** | 512 | 32 | greedy | `25b084baf8c43f24` | 一致 |
| 4（2 / 2 / 2 / 2） | 512 | 32 | greedy | `25b084baf8c43f24` | 一致 |
| 3（2 / 3 / 3） | 512 | 64 | 投機 block4 | `158e23a8919692a3` | 與單機 64 token 一致 |
| **8（1 層 × 8，極端不均）** | 128 | 16 | greedy | `3e6516f959cff6a4` | 一致 |
| 2（4 / 4） | 16 | 16 | 投機 block4 | `ac06b785c2ea2a82` | 一致 |

涵蓋：節點數 1→2→3→4→8、層數不能整除、greedy 與投機、128 與 512 上下文、32 與 64 生成步。
**結論：任意切法下輸出都逐 token 等於單機。** 這是「分配準確」的定義，這一項是真的過了。

### 二、會出錯的四類地方（實測踩到的）

**1）進程編排錯——端口。** 我把 head 監聽端口設成 stage 端口 +1，結果 head 和第一個 stage 撞在同一端口；head 連上的是上一個殘留進程，收到層區間不符的張量。表現是收到 `(512,256)` 卻在 `attention` 裡炸 `too many values to unpack`。**這一類最耗時間**，因為錯誤訊息指向模型，真因在編排。修法：head 用獨立端口區段、先 bind head 再 spawn 鏈、stage 之間重試 10 秒。

**2）協議／形狀錯——目前是 fail-fast 的。** 形狀不符直接 `ValueError`，不會靜默算錯。但要注意反面：**如果層區間接錯一格，形狀完全合法，不會 crash，只會安靜地輸出錯的 token**。所以 hash 基準測試不是可選項。

**3）層數不能整除——數學上無害。** 8 層分 3 台 = 2/3/3、分 8 台 = 1×8，輸出都正確。它不影響正確性，只影響時間：**整體由最慢那一段決定**。

**4）記憶體爆掉——最陰的一類，權重放得下不代表跑得長。** 用新加的 `cluster/footprint.py` 算（節點可用 24GB / 32GB 機器）：

| 模型 | 節點 | 峰值 | 最長上下文 | 判定 |
|---|---|---|---|---|
| 200B 4bit（權重 87GB） | 2 | 50.0G | — | 不通過 |
| 200B | 4 | 25.9G | 9K | **不通過（32K）** |
| 200B | 8 | 13.9G | 279K | OK |
| 70B 4bit（權重 35GB） | 2 | 23.7G | 34K | 勉強 OK |
| 70B | 4 | 12.5G | 173K | OK |

**關鍵不對稱**：權重和 KV 都掛在「持有那幾層」的那台機器上，層多的同時背最重的權重和最多的 KV；第 0 台還要多背 embed + lm_head（200B 模型這部分是 2.2GB）。所以「16 台 = 512GB 共用池」是錯的直覺——**它是一個必須逐節點核算的靜態分片**。

### 三、最重要的發現：目標函數選錯了

原本的 planner 只最小化「最慢的那一段」——這假設**各段是並行的**。但 batch=1 的普通流水線是**接力**，總時間是各段相加：

| 目標 | 8 台 1 層 | 1 台 8 層 | 真實含義 |
|---|---|---|---|
| `max`（最小化最慢段） | 2.0ms | 8.0ms | 8 台看起來贏 4 倍——**假象** |
| `sum`（最小化總時間） | 16.0ms | 9.0ms | 1 台才對，8 台純虧 |

當 hop≈tau 時，**每多加一台就多賠一整個 hop**。這正是投機解碼存在的理由：它把「sum 問題」變成「max 問題」。所以 planner 現在有 `--objective max|sum`，選錯就是選錯方案，而不是選錯參數。

**異質性能**已經能算：`node_speed=[1.0, 0.55, 1.0]` 分 8 層 → `n0[0:3] → n1[3:5] → n2[5:8]`，慢機器只拿 2 層。但這用的是**假設的速度**，還沒有「上線實測再重規劃」的回路。

### 四、還沒測的（不要當成已驗證）

- **真模型**：還是隨機權重玩具模型，沒有真 tokenizer、沒有真 attention 實作、沒有 GQA。
- **真多機**：全部在本機 loopback 的進程間，不是跨機器、不是 TB 鏈路。
- **長上下文**：只到 512。32K 以上的索引正確性沒驗（本機 16GB 也跑不動真模型）。
- **多請求並發**：現在是單請求嚴格串行，batch=1。
- **慢節點 / 重規劃**：斷線能被偵測（見下一節），但沒有重連、沒有心跳、沒有掉一台後重新分片。


## 分配層工程化：記憶體約束 + 實測速度 + 容錯（三件都已跑）

### 一、記憶體約束併進 planner

planner 現在同時回答「快不快」和「放不放得下」。`mem=dict(layer_w, layer_kv_token, ctx, head, capacity=[每台可用 bytes])`，DP 內每段檢查 `stage_bytes ≤ capacity`；不對稱的地方是 node 0 額外背 embed + lm_head。

CLI 實測（32K 上下文、每台 24GB 可用）：

| 模型 | 台數 | 結果 |
|---|---|---|
| 200B（80 層） | 4 | **不可行** |
| 200B | 16 | `n0[0:5] → … → n15[75:80]`，cost 5.20ms，peak 7.9G |
| 200B | 8 | peak 13.9G、最長 279K |
| 70B | 2 / 4 | 23.7G/34K、12.5G/173K |
| 8B | 2 | 3.9G/384K |

「一台放不下」被量化了：24GB/台的約束下，200B 給 4 台無論怎麼切都必爆（每台得背 20 層）。

### 二、速度反饋：tau 不用手填

`cluster/probe.py` 在**活迴圈**（KV 逐 token 增長）下量 1/2/4/8 層：

```
1 層 0.657 / 2 層 1.064 / 4 層 1.896 / 8 層 3.334 ms
→ tau = 0.383 ms/層, fixed = 0.274 ms/段   (已寫 profile.json)
```

關鍵在「**每多一段就固定多付 0.27ms**」——接力模式下段越多、fixed 加得越多，這是純虧損。用 profile + 實測 hop 0.071ms 重算：

| 目標函數 | 台數 | 結果 | 每 token 成本 |
|---|---|---|---|
| sum（接力，正確） | 8 | `n0[0:8]` 一台全吃 | **3.41ms** |
| max（假設並行，錯） | 8 | 8×1 層 | 0.73ms |
| max | 4 | 2/2/2/2 | 1.11ms |

3.41ms 與實測單機 8 層 decode 的 round_latency 3.39ms 吻合到 1% 內——**cost model 現在能預測實測**，這是後面所有判斷的前提。異質機器也驗過：`node_speed=[1.0, 0.55, 1.0]` → 慢機器只拿 2 層。

### 三、容錯：掉線不再變成「看起來在思考」

`RingHead` 兩條 socket 都設 `timeout_s=15`，三種錯誤各自映射成能據以行動的訊息；`protocol._Reader._wait` 尊重 socket timeout（select 帶超時），避免 recv 無限阻塞。

故障注入實測（`bench/fault_test.py`：2 節點跑起來後 `SIGKILL` 掉 stage）：

```
殺掉 stage pid=93467（模擬節點崩潰/掉線）
head 在 0.0s 內結束, exit=1
RuntimeError: stage closed the connection (crashed or was killed); in-flight chunk is lost
```

**0.0 秒**結束——不是等到 15s 超時，是 socket 立刻 EOF。這正是要的行為：壞消息立刻說，不要拖成沉默。逾時那條路是留給「機器沒死但卡住」的（例如 swap 換到地獄），訊息寫 `treat the cluster as degraded, not as slow`。

回歸：容錯改動後重跑 single / 2 節點 / 3 節點，hash 仍全部 `25b084baf8c43f24`（單機基準）。

### 四、仍然沒解決的

- 沒有**重連**、沒有**重新分片**：掉一台就是整個 run 失敗，不是降級繼續。
- 沒有**心跳**：`HEARTBEAT` 幀型別定義了但沒用，所以「慢」與「死」只能靠單次請求的 timeout 區分。
- `min_nodes` 訊息仍誤導（只搜到 `--nodes` 為止，200B @4 台會說「一台一層也放不下」，其實 8 台就放得下）。


- **誠實驗證**：仍然無解（見 DESIGN.md 第四層）。



## 4 台的分配與掉線恢復（實測）

### 一、planner 現在自己檢查，並回答「4 台到底能不能、怎麼分」

新增三件事：`validate()`（層區間是否連續、有無空洞、每台是否放得下）、`min_nodes()` 搜到 `n_layers`（不再只搜到你問的那個數字）、`max_ctx()`（這台數下最大上下文）。

```
plan --nodes 4 --profile profile.json --hop-ms 0.071
 n0[0:2] -> n1[2:4] -> n2[4:6] -> n3[6:8]
 [自檢] 通過：層區間連續、無空洞、每台放得下

plan --nodes 4 --spare 1        # 4 台 = 3 工作 + 1 熱備
 n0[0:2] -> n1[2:5] -> n2[5:8]  cost 1.11 → 1.49ms

plan --nodes 4 --model 200B --ctx 32768
 不可行：4 台（4 台工作）× 24GB 放不下
 [自檢] 最少需要 5 台：n0[0:16] → … → n4[64:80]
 [自檢] 4 台能跑的最大上下文: 17K（每台 24GB 可用）
```

注意「最少需要 5 台」推翻了先前口頭說的 8 台：那是被測過的配置，不是下限。自我檢測的價值就在這。

### 二、4 台實測：越加越慢，而且和 planner 預測一致

同樣 512 prompt / 32 tokens、greedy、本機 loopback：

| 配置 | 分段 | tok/s | round_latency | hop |
|---|---|---|---|---|
| 1 台（`--objective sum` 自動選出） | `n0[0:8]` | **248.80** | 0.00ms | 0 |
| 2 台 | 4/4 | 123.62 | 3.43ms | 3.45ms |
| 3 台（= 4 台留 1 熱備） | 2/3/3 | 116.23 | 5.77ms | 3.85ms |
| 4 台 | 2/2/2/2 | **91.51** | 6.18ms | 4.75ms |

hash 全部 `25b084baf8c43f24`（與單機一致）。

**這張表就是「4 台如何分配」的答案**：在沒有投機解碼的接力模式下，4 台的最優分配是「只用 1 台」。多加一台不是多一份算力，是多一次 hop；本機 hop ≈ 4.75ms，而每層只要 0.38ms——切走一層換來 4.75ms 的來回，怎麼算都是虧。planner 用 `sum` 目標會直接選 `n0[0:8]`，這是對的，不是保守。

### 三、掉線 → 主機判斷 → 重建 → 重放 → 續跑（已驗證到 hash）

`bench/fault_recover.py`：同一條 4 台鏈跑兩次，第二次在生成途中 `SIGKILL` 掉中間那台。

```
[host] 掉线: stage closed the connection (crashed or was killed); in-flight chunk is lost
[host] 判定: 可恢复 1/3  policy=rebuild  已知前缀=842 tokens  重建次数=2
[host] 重放 841 tokens 重建 KV 完成: 0.31s，继续生成
recoveries=1 builds=2
hash=836243a18cf95b30   ← 與乾淨跑逐位相同
```

`policy=shrink` 也跑過：殺掉中間節點後主機把它剔除，按剩下 3 台重規劃成 `n0[0:2] → n1[2:5] → n2[5:8]`，重放 842 tokens 用 0.06s，最終 hash 同樣一致。

三個關鍵設計：

1. **掉線是「立刻」通知，不是等超時**：socket EOF 立刻 `StageLost(kind="closed")`；卡住才走 15s `kind="timeout"`，訊息寫 `degraded, not slow`。
2. **恢復的唯一正確做法是重放前綴**：head 是唯一握有完整 token 序列的節點，所以重建全鏈 + 重放 = 不可能錯。保留存活節點去省那零點幾秒，換來的是 KV 悄悄錯位——那會輸出看似正常的錯 token，比崩潰危險得多。
3. **重放成本就是容錯的代價**：841 tokens 花 0.31s（約 0.37ms/token）。真模型上這是整段重新 prefill，ctx 越長越貴——所以「能不能容錯」實際是「你的上下文有多長」的問題。

### 四、這一輪仍然沒做到的

- `--spare` 目前只是「不參與計算」，**掉線後並不會真的讓熱備接管**：`shrink` 是把 4 台縮成 3 台繼續跑，熱備節點從頭到尾沒被調用。真正的接管要等「動態加入 ring」。
- 恢復是**整鏈重建**，不是增量修復；重放是 O(前綴長度)。
- 沒有心跳：判斷「慢」與「死」仍靠單次請求的 timeout。
- 仍是本機 loopback 的進程，不是跨機器；跨機器時 4.75ms 的 hop 會變成十幾毫秒，上表會更難看。



## M3：固定形狀 + `mx.compile` —— 每層 0.383 → 0.211ms

### 為什麼要做

傳輸修好之後（往返 0.071ms），hop 仍有 1.0ms 不在線路上。用不同分段掃出來的結論是：**同一個 8 層段，固定形狀暖回圈 0.10ms/層、活回圈 0.38ms/層——活回圈貴 4 倍**。因為 KV 每步在長、形狀每次不同，MLX 每步都要重新規劃與派發。這正是 Shard 非做 CUDA graph capture 不可（也因此被 CUDA 綁死）的原因，MLX 的對應手段是 `mx.compile` + 固定長度 KV slab。

### 做法（`cluster/fastmodel.py`）

- KV 存在**每層獨立**的 slab `(ctx_len, nh, hd)`，一次分配到最大上下文。
- 寫入位置用 `mx.slice_update` 而不是 `concatenate`：不 realloc、不改形狀 → 不重新規劃。
- attention 讀整個 slab，用 `ar <= pos` 遮罩掉未來位置。浪費 FLOPs，但 batch=1 解碼是 launch-bound 不是 FLOP-bound。
- 層棧按 `(lo, hi)` 只 compile 一次，約 200 個 kernel 合成一次派發。
- 遮罩同時免費送了一個好處：**投機回滾不需要 crop**。重跑較早的 `start_pos` 時，舊 KV 直接被覆蓋，沒被覆蓋到的位置又被遮罩排除，天然正確。

### 被證僞的做法（別再試）

單一大 slab `(n_layers, ctx, nh, hd)` + `slice_update`：8 層 **5.182ms**，比不做還慢（0.64×）。原因是 `slice_update` 返回新陣列，單一大 tensor 每次寫入都複製全部（8 層 × ctx × heads × 64 × 4B）。改成每層獨立 slab（只複製 1/8）才有效。第一版還量出 `fixed = -0.027ms/段`——負的固定成本是明顯的錯數警訊。

### 隔離實驗：加速到底來自誰

| 每段層數 | 原 ToyModel（活回圈） | 只固定 slab、不 compile | 固定 slab + compile |
|---|---|---|---|
| 1 層 | 0.657ms | 0.855ms | **0.471ms**（1.39×） |
| 2 層 | 1.064ms | 1.400ms | **0.694ms**（1.53×） |
| 4 層 | 1.896ms | 2.416ms | **1.174ms**（1.61×） |
| 8 層 | 3.334ms | 4.911ms | **1.947ms**（1.71×） |

**固定 slab 單獨用是負收益（全部更慢），加速完全來自 `mx.compile`。** 這件事必須分開量，否則很容易誤以為「預分配 KV」本身有用。

擬合：**tau = 0.211 ms/層（原 0.383）、fixed = 0.260 ms/段（原 0.274）**。每層便宜 1.8×，每段的固定成本幾乎沒動——這說明還有一塊與層數無關的 Python/派發開銷沒被吃掉。

### 正確性

`bench/m3_compile.py` 先比 hash 再量速度：baseline `25b084baf8c43f24`、compiled `25b084baf8c43f24`，一致。

### 接到真實鏈路之後（`--fast`）

原本只有單進程 bench 變快，多節點沒動；現已接進 `stage.py` / `coordinator.py`（`RingHead` 與 `StageServer` 都可用 `FastModel`，CLI 加 `--fast`）。prefill 走**批次固定形狀**（`forward_batch`，按 L 快取 compile），解碼走單步 `step`——若 prefill 也逐 token 走鏈，等於把鏈延遲乘以 prompt 長度，512 token 會吃掉 0.7s（實測踩到）。

統一條件：512 prompt / 128 tokens / greedy / 本機 loopback / 全鏈 hash 皆 `126227e02636ad3a`。

| 節點 | 原路徑 tok/s | `--fast` tok/s | 提升 | fast tau_head/tail | fast hop |
|---|---|---|---|---|---|
| 1（單機） | 222.97 | 230.20（同一後端，屬量測抖動） | — | 3.88 / 0.00 | 0 |
| 2 | 172.83 | **197.18** | +14% | 1.80 / 1.76 | 1.10ms |
| 3 | 150.35 | **166.10** | +10% | 1.13 / 1.47 | 2.95ms |
| 4 | 124.83 | **148.29** | +19% | 1.11 / 1.10 | 3.93ms |

### 讀這張表要注意的三件事

1. **prefill 成本變高了**（4 台：54ms → 81ms）。批次 prefill 每個進程第一次要 compile 一次，而且 512 長度的 graph 比逐 token 大。所以短生成（32 tokens）時 fast 反而看起來更慢——那不是變慢，是**固定成本被少數 token 攤不開**。判斷加速與否要用長生成（128+）。
2. **hop 也一起降了**（4 台 4.80 → 3.93ms）。不是線路變快，是每段計算時間縮短後，量測到的 `latency - tau_tail` 跟著縮小。
3. **仍然越加越慢**：4 台 148.29 對 1 台 230.20，接力模式下加機器照樣虧——目標函數選錯這件事，`mx.compile` 救不了。它只讓「每一段更便宜」，不改變「總時間是相加」。

### 但代價是「每步複製整塊 slab」，成本隨上下文長度成長

`slice_update` 是純函數操作，每次寫入都返回新陣列——也就是**每個 token 都要複製一次整塊 KV**。這筆帳在短上下文看不出來，長上下文會把 compile 賺到的全部吐回去：

| slab 長度 | 8 層解碼 | KV 記憶體 |
|---|---|---|
| 640 | 1.84 ms/token | 10.5 MB |
| 2048 | 2.70 ms/token | 33.6 MB |
| 4096 | 3.43 ms/token | 67.1 MB |
| 8192 | 5.04 ms/token | 134.2 MB |

大約每多 2K 上下文，8 層就多付 0.4~0.8ms。線性外推到 32K，8 層會回到 ~15ms/token（≈1.9ms/層），**比不做 compile 還差**。所以這條路只在「短上下文 + 長生成」成立；要上 32K 必須改成**分塊 slab（只複寫 pos 所在那一塊）**或 paged KV——而 MLX 的 `mx.compile` 不支援原地寫，這就得靠 kernel 層的手段，不是 Python 層能解的。

### 投機模式下 fast 反而更慢（誠實記錄）

| 4 台 / 512 prompt / 64 tokens / spec block4 | tok/s | tau_head | tau_tail | hop |
|---|---|---|---|---|
| 原路徑 | **103.12** | 1.73ms | 3.05ms | 5.52ms |
| `--fast` | 66.67 | 3.37ms | 3.37ms | 9.05ms |

hash 兩邊都是 `158e23a8919692a3`（**回滾正確性沒問題**，`crop` 變 no-op 之後靠遮罩照樣對）。慢的原因是：投機每輪送 5 個 token，slab 複製成本是 O(ctx) 而不是 O(L)——同樣一次複製，裝 1 個 token 划算，裝 5 個也不會更便宜，而且 5 個 query 還要對滿 slab 做 attention。

結論：**`--fast` 適合貪婪解碼（1 token/輪），不適合投機批次（5 token/輪）**。這不是 bug，是兩種負載的成本結構不同。

### 這一輪踩到的形狀坑（都已修）

- `q[0] @ kl.transpose(...)`：MLX 會把 2-D 左運算元提升成 batch 維，`(4,64) @ (4,64,ctx)` 得到 `(4,4,ctx)` 而不是 `(4,ctx)`。正確寫法是 `(q[0][:, None, :] @ kl.transpose(1, 2, 0))[:, 0]`。
- 同一段程式在 `ctx == hd` 時「看起來正確」（64×64 剛好對上），只有 `ctx ≠ hd` 才暴露——所以形狀測試必須用不對稱維度。
- `mx.argmax(lg[0])` 對 `[L, V]` 取是錯的（回扁平索引）；但反過來，單 token 時 logits 是 `[1, V]`，用 `lg[0, -1]` 會取到單一純量再 argmax → 永遠 0。兩者要看形狀決定，統一做法是在 stage 內把 logits 補成 `[1, L, V]`。

## 經濟帳：容量、速度、錢（把「加機器虧」放回正確前提）

先修正一個比較對象。前面那些「4 台比 1 台慢」的表，模型都是玩具 8 層——**一台機器裝得下**，所以「1 台」那一行存在，加機器就只是多付跳數。真實情境是模型裝不下：200B / 4-bit 要 87.4GB，24GB 可用的機器一台根本不存在，比較對象不是「1 台」而是「更貴的大機，或根本買不到」。

`cluster/econ.py` 用 roofline 算（每層時間 = 層權重 bytes / 可用頻寬），假設都印在輸出的第一段：

| n | 每段層 | 最重那台 | 裝得下 | 接力 tok/s | 投機(k=3) tok/s | 通訊/token |
|---|---|---|---|---|---|---|
| 5 | 16 | 21.1G | OK | 1.37 | 20.95 | 1.5ms |
| 8 | 10 | 13.9G | OK | 1.36 | 33.45 | 2.4ms |
| 16 | 5 | 7.9G | OK | 1.36 | 66.48 | 4.8ms |
| 對照：Studio 512GB（819GB/s，¥112,899） | 80 | 87.4G | OK | **9.37** | 28.10 | — |

三件事同時成立，而且必須分開講：

1. **容量是線性的**：5 台就塞得下 200B，加機器確實換到容量。這是叢集存在的唯一理由。
2. **接力模式的速度與 n 無關**（1.36~1.37 恆定）。一次只有一段在算，聚合頻寬永遠是一台的 120GB/s；加機器只是把權重切開輪流讀，不是把工作切開並行。所以「16 台 = 1.92TB/s」只在**各段同時在算**時成立。
3. **只有投機流水線讓速度隨 n 長**（20.95 → 66.48）。8 台 ¥7.2 萬 ≈ 33 tok/s，比 Studio 512GB ¥11.29 萬的 28 tok/s 更快也更便宜——**這才是「多台更省錢」的正確版本**。

但 k=3 是假設，不是量測：玩具模型上重複上一個 token 的爛草稿只有 g≈1.15，Shard 的 n-gram 草稿接受率是 0.012。**k 才是這條路線的真正瓶頸**，不是機器數量、不是頻寬。所以 33/66 tok/s 是「有了真草稿模型之後的條件承諾」。

## 怎麼把這麼多台接起來（實體層，不是比喻）

`cluster/topology.py` 回答兩個不能含糊的問題：

- **度數**：基礎版 M4 mini 有 3 個 TB4 口，每條線吃兩端各一個 → 最大可用度數 3。**環用 2（留 1 給外設/管理）**；全互聯需要度數 n-1：8 台 28 條線還塞得下但沒口可用，16 台 **120 條線、度數 15，物理不可行**。
- **誰連誰**：ring 不是「過一台交換機的邏輯環」，是 n 條點對點 TB4 線，每條各自一個 L2（靜態 /30 或 169.254.x，不要靠 DHCP；網段跟著線走，機器換口就要重配）。

啟動命令由 `topology.py` 直接產出，例如 16 台 / 80 層：

```
[m0 / head] python -m cluster.main run --peers m1:7601 --advertise 192.168.1.10:7610 ...
[m1] python -m cluster.main stage --layers 5:10  --port 7601 --next m2:7602
[m2] python -m cluster.main stage --layers 10:15 --port 7602 --next m3:7603
...
[m15] python -m cluster.main stage --layers 75:80 --port 7615 --reply 192.168.1.10:7610
```

為此 `main.py` 加了 `--peers`（接上已經在別台跑起來的 stage，不在本機 spawn）與 `--advertise`（最後一段要回報的 head 位址）。**已驗證**：外部進程單獨起 stage、head 用 `--peers` 接上，hash `158e23a8919692a3` 與本機 spawn 完全相同——「跨機器」現在只剩「換 IP」這一步，不再是未實作的缺口。

延遲預算（每多一台就多一次往返，接力模式下直接相加）：

| 連線 | 每跳 | 來源 |
|---|---|---|
| 本機 loopback（v2 協議） | **0.071ms** | 實測 |
| TB4 上的 TCP | 0.1~0.3ms | 估（40Gb/s 鏈路 + 內核協議棧） |
| TB5 + RDMA（僅 M4 Pro/M5、macOS 26.2） | <0.02ms | 文件 |

8 台 = 8 跳 ≈ 1~2.4ms/token 純通訊。相對 200B 每段 100~150ms 的計算，**這一項小到可以忽略**——和玩具模型剛好相反（玩具每段只有 0.4ms，跳數就變成主項）。也就是說：模型越大，多機的相對代價越小。

**試著用雲主機當第二台真機器，失敗了（誠實記錄）**：本機只有一台 Mac，另一個環境是雲 VM。VM 的 22 埠從 Mac 連得上，但自建的 7801/7803 埠連過去時——TCP 握手「成功」，VM 端卻從未出現 accept 記錄，payload 也從沒到達。也就是說 Mac 這一側有個本地代理會先吃下連線、然後把資料黑洞掉，不是真的可路由。結論：**跨機器連線在程式面已就緒（`--peers` + hash 一致），但沒有第二台可路由的實體機器就無法真的驗**。`bench/cross_machine_rtt.py` 已寫好並標註 unverified，等有 TB4 對端再跑。

## 容量階梯：N 台 32GB 到底能跑多大（已用真權重檔驗過公式）

這是整個專案唯一無條件成立的理由：**容量線性**。不是「多台更快」，是「一台根本裝不下」。

前提先寫死：**32GB 機器實際只能給模型 24GB**（扣掉系統與 Metal 保留），512GB Studio 算 480GB。
用標稱 32GB 去算的表會樂觀約 30%，不能用。

`cluster/capacity.py`，4-bit 權重、fp16 KV，每台的帳 = 自己那幾層的權重 + 自己那幾層的 KV，
頭那台額外背 embedding、尾那台額外背輸出矩陣：

| 機型 | 8B | 32B | 70B | 120B | 200B |
|---|---|---|---|---|---|
| 1× 32GB | 3.8G ✓ | 17.5G ✓ | 需 2 台 | 需 3 台 | 需 4 台 |
| 2× 32GB | 2.2G ✓ | 9.1G ✓ | 19.0G ✓ | 需 3 台 | 需 4 台 |
| **4× 32GB** | 1.4G ✓ | 4.9G ✓ | **10.1G ✓** | **16.9G ✓** | 23.6G ✓（4K ctx）／32K 需 5 台 |
| 8× 32GB | 1.0G ✓ | 2.9G ✓ | 5.7G ✓ | 9.2G ✓ | 12.7G ✓ |

（數字是「最重那台」的 GB，上下文 4K；32K 時 70B 在 4 台上是 12.5G/台，200B 要 5 台。）

也就是說：**4 台 32GB = 96GB 可用，跑得動 120B dense；5 台跑得動 200B dense（32K 上下文）**。
這正是「不要一台超大內存」的正確版本——但它是容量版本，速度是另一本帳（pipeline 接力，
tok/s 與台數無關；只有 MoE 的低 active 參數能讓每 token 的計算量真的變小）。

### 三個用真檔驗出來的結論（`bench/verify_capacity.py`）

拿 Qwen2.5-0.5B 的真 safetensors，逐 tensor 對帳、逐 stage 量子進程 peak RSS：

1. **層權重公式與檔案一致到 0.020%**（7 個層段全過，誤差 < 0.1MB）。所以上面那張表
   的「裝不裝得下」是算術，不是樂觀估計。
2. **記憶體只隨「自己那幾層」長**：中間段 `6:12` 必須放 178.9MB，RSS 218.6MB，
   多出來的就是 python+mlx 的 39.7MB 固定開銷。証明「只讀自己那幾層」不只是設計口號。
3. **兩個會讓容量算錯的坑**：
   - **tie embeddings 的模型，尾那台仍然要放一整份 embedding 當 lm_head**（0.5B 是
     272MB，200B 級是 1GB+）。整條鏈總共付兩份，不是一份。
   - **載入 embed/lm_head 時會多一份轉換暫存**：實測 +280MB（等於一份 embedding）。
     這是峰值不是常駐，但「每台剛好塞滿」的配置會在啟動那一刻 OOM。
     所以每台要留「一個層段 + 一份 embedding」的餘裕。

用法：

    .venv-arm/bin/python -m cluster.capacity --ctx 32768
    PYTHONPATH=. .venv-arm/bin/python bench/verify_capacity.py

## 按專家切分（EP）：GitHub 上有什麼，以及為什麼家用場景會輸

### 現況（查到的東西，不是猜的）

| 類別 | 代表 | 做法 | 能不能搬到家用 |
|---|---|---|---|
| 真正的 expert parallel | **DeepEP**（DeepSeek）、vLLM/SGLang 的 EP 部署、DeepSpeed-MoE | 每層 dispatch+combine 兩次 all-to-all；DeepEP 明寫要 Hopper+、NVLink 域內 + RDMA 域間 | ✗ 需要 RDMA/NVLink 級 fabric |
| 家用多機 | **exo**、**mesh-llm**、PRIMA.CPP | **一律按層切**（exo 按記憶體加權分層；mesh-llm 用 contiguous layer stages，MoE 家族也是 package-backed stages），沒有一個做專家切 | ✓ 但都不是 EP |
| 專家放慢速記憶體（同一台機器內） | **KTransformers**（清華 MADSYS）、MoE-Infinity 那類 expert caching | 專家放 CPU RAM / NVMe，NUMA-aware，671B 用 1 GPU + 大記憶體 | ✓ 但這是「機器內」，不是「機器間」 |

一句話：**專家切分沒人拿來做家用多機，不是因為沒人想到，而是因為下面兩筆稅。**

順便記一筆教訓：某篇二手介紹說 mesh-llm「MoE 用 expert sharding、零跨節點流量」，
去看 README 本人，它講的是 layer stages。二手摘要不可信，要看原始碼／README。

### 兩筆稅（`cluster/moe.py` 算出來的）

**EP 的稅按「層數」收**：每個 MoE 層要 dispatch + combine 各一次，48 層 = 96 次往返。
**pipeline 的稅按「機器數」收**：n 台就是 n−1 次。單人使用時 96 ≫ 3~15，EP 必輸。

125B / 每 token 激活 10B / 48 層 / 128 專家 top8 / 4-bit（62.5GB 權重）：

| 方案 | 每台權重 | 每 token | roofline tok/s |
|---|---|---|---|
| pipeline 4 台 | 15.6GB ✓ | 41.7ms 計算 + 0.4ms 通訊 | **23.7** |
| EP 4 台 | 25.0GB ✗（放不下 24G） | 112ms 計算 + 14.4ms 通訊 | 7.9 |
| pipeline 6 台 | 10.4GB ✓ | 41.7ms + 0.8ms | **24.0** |
| EP 6 台 | 20.8GB ✓ | 110ms + 14.4ms | 9.1（batch=32 也只有 9.1） |

**EP 的第二個死穴：attention 不會被專家切分。** batch=1 時它只能待在一個節點上，
那 12.5GB 稠密權重每 token 都要整份讀一遍 = 104ms，比整條 pipeline 鏈（41.7ms）
還慢。資料中心是靠 MLA + NVLink 域內的 tensor parallel 把稠密部分也切開，家用
機器沒有那個 fabric。

DeepSeek-V3（671B / 37B active、61 層、256 專家）在 16 台上：

| 方案 | 每台 | 結論 |
|---|---|---|
| pipeline | 21.0GB ✓ | 154ms/token |
| EP | **83.9GB ✗** | 光是 attention+embedding 就 67GB，切不開，根本放不下 |

### 所以「125B 切完就變 10B」這句話，對一半

- **算力上對**：每 token 只算 active 那 10B，這是 MoE 的全部意義。所以 125B MoE
  在 4~6 台機器上，roofline 是 **24 tok/s**，跟一個 10B dense 模型差不多。
- **容量上不對**：權重還是 62.5GB，一台 24GB 放不下，所以仍然要靠多台——
  但靠的是**按層切**，不是按專家切。
- **單人場景**：pipeline 完勝 EP（24 vs 9）。
- **多用戶場景**：反過來。batch 大了以後 attention 變資料並行、專家權重被 batch
  攤掉，EP 的通訊也按 batch 攤（14.4ms/32 = 0.45ms），這時 EP 才划算——這正是
  vLLM/SGLang 的部署形態（`deepep_low_latency` 專門服務 decode）。

### 還有個必須說清楚的落差

上面 24 tok/s 是**硬體上限**（每 token 要讀的權重 bytes ÷ 帶寬）。我們原型在 0.5B
上實測 9.4 tok/s，而同一個 roofline 是 ~120 tok/s —— **只跑到 8%**，因為 batch=1
解碼是 launch-bound（M0 量到每個 op 184µs）。所以「換對方案能到 24 tok/s」和
「現在能跑到 24 tok/s」是兩件事，中間隔著 kernel/編譯/批次這一段。

用法：

    .venv-arm/bin/python -m cluster.moe --total 125 --active 10 --nodes 6
    .venv-arm/bin/python -m cluster.moe --total 671 --active 37 --layers 61 --experts 256 --nodes 16

## 心跳與「卡住 vs 死掉」（已實測）

`bench/heartbeat_test.py`：同一條 2 段鏈，分別 SIGSTOP（活著但不算）與 SIGKILL。

| 情況 | 舊：固定 15s | 現在 | 判定 |
|---|---|---|---|
| 心跳往返（整條鏈 2 段） | — | **0.15ms 中位**（5 次：0.19/0.15/0.15/0.09/0.16） | — |
| SIGSTOP 卡住 | 15000ms | **1511ms** | `hung`（開著的 socket + 不回應心跳） |
| SIGKILL 死掉 | 即時（EOF） | **3ms** | `closed` |

門檻照實測基線走：每輪 3.37ms → `clamp(20×3.37, 500, 15000) = 500ms`，再加一次
1s 心跳詢問（問「是慢還是死」）。訊息也換成能直接讀的：「waited 505ms, baseline
3.4ms, deadline 500ms」＋「machine not computing」，而不是一句看不出原因的
unresponsive。

## 真模型（Qwen2.5-0.5B）：第一次用真權重跑通環（已實測）

為什麼要接：隨機權重的玩具模型沒有任何預測力，**k（投機每輪接受多少 token）根本
量不出來**，而 k 才是整條路線真正的瓶頸。

| 項目 | 結果 |
|---|---|
| 單機 greedy（24 層、8 token prompt） | 101.6ms/token = **9.8 tok/s**，輸出「 Paris. It is the largest city in Europe…」 |
| 同權重切成 0:12 \| 12:24（同進程傳 hidden） | **token 逐位相同**，同一句 |
| 環上 1 / 2 / 4 台（32 token，greedy） | **9.47 / 9.35 / 8.74 tok/s**，hash 全 `f38128e60975fec8` |
| 環上 1 / 2 / 4 台（16 token，greedy） | 9.38 / 9.02 / 8.29 tok/s，hash 全 `4773507657e1e78a` |
| 投機（重複上一個 token 當草稿，block 4） | **g = 1.00**（佔位草稿；真草稿的 k 見後面「投機解碼」一節） |

三件事同時確立：

1. **玩具模型上「4 台比 1 台慢 24~40%」這個結論，在真模型上不成立。** 真模型每
   token 要 100ms，而 hop 是 0.1~1ms：通訊佔比從 25% 掉到 1~8%。4 台 8.74 對
   1 台 9.47 只是 -8%，其中一半還是「每個 stage 冷啟動自己的權重」。
   **「模型越大，多機的相對代價越小」這句原本是推算，現在是小模型上的實測。**
2. **載入方式本身就是分片的一部分。** 第一版用 `mx.load()` 整檔 988MB 再丟，
   四個 stage 同時起 → 四份 988MB 峰值 → 16GB 機器換頁 → 4 台比 1 台慢 24%。
   改成按 tensor 讀 safetensors 之後同樣 4 台只慢 8%。差別不在算法，在「有沒有
   真的只放自己那幾層」。
3. **k 仍然是 1.00，也就是瓶頸還在草稿。** 重複上一個 token 這種爛草稿在真權重上
   一個都不會被接受（玩具模型上是 g≈1.15 的假象）。要讓速度隨機器數量長，必須
   有真的草稿模型（同一家族的小模型，或 n-gram/prompt-lookup 這種零成本草稿），
   這件事現在終於可以在真權重上量了。

用法：

    .venv-arm/bin/python -m cluster.main run --nodes 2 --model-dir models/qwen2.5-0.5b \
      --stage-port 7601 --head-port 7611 --prompt-len 8 --max-tokens 32
    PYTHONPATH=. .venv-arm/bin/python bench/real_model.py    # 單機 vs 切開的逐位比對

## 批次解碼（batching）：每步幾乎不隨 B 漲，所以 tok/s 隨 B 長（已實測）

`bench/batch_decode.py`：四條 prompt 補 eos 到等長（8 tokens），greedy 12 token，
單機 24 層與 2 台 12|12 各一組，每組重起乾淨鏈（不同的端口，避免 TIME_WAIT 誤判）。

| B | 1 台 ms/step | 1 台 tok/s | 2 台 ms/step | 2 台 tok/s |
|---|---|---|---|---|
| 1 | 102.0 | 9.81 | 120.5 | 8.30 |
| 2 | 112.6 | 17.77 | 123.3 | 16.22 |
| 4 | 110.7 | 36.14 | 119.0 | 33.62 |
| 8 | 113.7 | 70.33 | 112.3 | **71.23** |

四件事：

1. **假設成立，而且是乾淨地成立。** 每步從 B=1 的 102.0ms 到 B=8 的 113.7ms，只漲
   11%，而 tok/s 從 9.81 長到 70.33（**7.2×**）。這正是 M0 診斷的推論：batch=1
   解碼九成以上是 kernel 調度，攤給 8 條序列之後，那筆固定成本被攤掉，真正漲的
   只有「權重讀一次 × 8 列的 attention」那一小塊。
2. **B=1 → B=2 有 10ms 的階躍，之後完全平。** 階躍是 attention 從「單列」變成
   「8 列矩陣」的那一次形狀切換（並不是每加一列再加成本）。這也是最好的一條證據：
   成本結構是「一次固定 + 每列極小邊際」。
3. **只有 batch > 1 才有這個收益 —— 對單人聊天完全沒用。** 這是必須講清楚的限制：
   一個人打字、一次只問一句話，B 永遠是 1，102ms/token 一點都不會變。batching
   買的是**吞吐**（同時服務多條請求），不是**延遲**。單機 0.5B 一個人用 9.8 tok/s，
   八個人同時用每個人還是 8.8 tok/s（70/8）——這才是它的正確用途。
4. **2 台在 B=8 時反超單機（112.3 vs 113.7ms）。** 不是因為它算得快，而是每步
   固定成本裡那一次 hop（ring 只有 1 跳）本來就被 batch 攤薄了；B=8 時 2 台多付
   的 1 次 hop 相對 112ms 已經可以忽略。**「機器越多越慢」在 batch>1 下同樣不成立。**

正確性（同樣是這支 bench 的一部分，不是推論）：

- 單機：4 條一起跑 vs 每條單獨跑，**逐 token 全一致**（batch 沒有污染任何一列）。
  （第一版 4 條一起跑時 110.9ms/step、36.07 tok/s，兩次跑數字一致。）
- 2 台：batch 的輸出與單機逐 token 相同 `[True, True, True, True]`；再逐列單獨
  跑（先 `reset()` 清掉 head 與所有 stage 的 KV）也全一致。

**prefill 不適用「攤固定成本」這套說法**（同一支模型、L=15，單機 3 次取最後一次）：

| B | prefill |
|---|---|
| 1 | 112ms |
| 2 | 109ms |
| 4 | 117ms |
| 8 | 186ms |

decode 是「讀 0.5B 權重一次 + 每列一點算」，所以每步成本幾乎不隨 B 漲；
prefill 是「15 × B 個位置都要算」，B=8 就開始被算力壓住了。**兩者的成本結構不同**，
不能拿 decode 的結論去推 prefill。

三個實作坑（都踩到了）：

1. **KV 的 batch 軸是「所有列共用同一個位置索引」**，所以第一版硬性要求等長 + 同步。
   不等長要左補齊 + per-row mask，尚未做（見「已知限制」）。
2. **B=1 的舊語意必須保留**：`stage` 對 B=1 仍回「每個位置一個 argmax」（投機 chunk
   驗證靠它），B>1 才只取每列最後一列的位置。`forward_rows` 在 B=1 時自己截成
   最後一個，否則 prefill 會回 L 個 token、對不上 `len(rows)`。
3. **`_check_ports` 原本不帶 `SO_REUSEADDR`**，於是「剛關掉一條鏈、馬上重建」會被
   自己的預檢擋下來（Errno 48，但 `lsof` 上根本沒有 LISTEN，只剩 TIME_WAIT）。
   改成帶 `SO_REUSEADDR` 之後語意才對：TIME_WAIT 可 reuse，**真的有人 LISTEN 仍會擋**（已回歸驗過：故意占住 7701，head 立即退出並印出說明）。

## 批次服務（`cluster/serve.py`）：把 batching 接到真的請求上（已實測）

bench 只證明「同時 8 條一起跑不會變慢」。要用得上，缺的是排程：一個人打字時 B 永遠是 1，
只有**同時有多條請求**才湊得出 B。所以多了一個最小服務器：收請求 → 湊批 → 一起解碼 →
分別串流回去。`bench/serve_load.py` 開 4 條長度不同的 prompt（12 / 47 / 88 / 96 tokens），
先逐條單獨跑記下文字，再同時跑，比對是否逐字相同。

| 台數 | 併發 | 總 tok/s | 單條延遲中位 | 與單獨跑一致 |
|---|---|---|---|---|
| 1 台 | 1 | 9.45 | 1.69s | 一致 |
| 1 台 | 2 | 17.92 | 1.79s | 一致 |
| 1 台 | 4 | **34.78** | 1.84s | 一致 |
| 2 台 | 1 | 9.50 | 1.68s | 一致 |
| 2 台 | 2 | 17.98 | 1.78s | 一致 |
| 2 台 | 4 | **33.60** | 1.90s | 一致 |

四件事：

1. **吞吐真的隨併發長，延遲幾乎不動。** 4 條併發 → 34.78 tok/s（3.7×），而單條延遲
   只從 1.69s 變 1.84s（**+9%**）。這才是「多機/多列」該有的曲線：加的是吞吐，不是等待。
2. **2 台跟 1 台在併發下幾乎沒有差別（33.60 vs 34.78，-3%）**——每步多付的那一跳，
   在 100ms 級別的模型計算面前仍然是零頭。
3. **同一句話在「單獨跑」與「跟別人一起跑」逐字相同。** 這是最容易錯的地方：左補齊的
   pad 進了 KV、每列的位置編號還不一樣，錯了不會 crash，只會輸出一段看似合理的話。
4. **窗口 15ms 的代價可忽略**：單條 1.69s 對照「進來就跑」的 1.71s，差在雜訊裡。

### 這一段踩到的三個坑（都不是崩潰，是「看起來正常」）

**一、KV 的 batch 軸跟上一批的人數綁死。** 第一批 B=1 跑完，KV 是 `[1, 8, 14, 64]`；
第二批 B=2 進來，prefill 從 start_pos=0 開始，裁切後留下 `(1,0,14,64)` 再跟
`(2,9,14,64)` 拼接 → `concatenate` 形狀不符。**每個新批次都必須先 `reset()` 清 KV**
（`crop(0)` 不夠，它只裁長度、不動 batch 維）。

**二、frame header 只有五個純量欄位，多出來的 meta 被靜默丟掉。** 這是本輪最陰的一個：
`valid_from` 放進 `meta` 之後，`send_frame` 只把 `req / chunk_id / start_pos / n_in / tau_ms`
打包進 48 字節 header，其餘鍵**直接消失**。於是 head 端算的是「左補齊 + per-row 遮罩」，
stage 端算的是「沒有遮罩、位置從 0 開始」：**單機完全正確，2 台 batch 第一個 decode token
就開始歪，而兩邊都沒有任何錯誤訊息**。修法是給 header 的 `flags` 欄位存「JSON 尾巴
的長度」，把它接在 payload 後面；沒有額外鍵時尾巴是空的，熱路徑一個字節都不多。
（`valid_from` 是 B 個整數，很小；位置編號由 `start_pos + i - valid_from` 在
stage 端自己算，不必上線傳。）

**三、絕對位置模型不能只補齊、不改位置。** Qwen 用絕對位置編號：第 r 列被左補了 10 格，
它的第一個真實 token 的位置必須是 0 而不是 10，否則整個 attention 幾何都偏掉。所以
`realmodel._attn` 現在吃兩個東西：`rope_pos`（每列的內容位置，可以不同）與
`valid_from`（每列真實 token 的物理起點，用來遮掉前面的 pad）。遮罩仍然用物理位置，
因為它比的是「這個 KV 槽在不在因果範圍內」。

## 投機解碼：把 k 從 1.00 變成有數字的東西（已實測）

前面所有的「多台更快」都掛在一個未量過的假設上：`g`（每輪被接受的 token 數）。
`g = 1` 就是「送 block 個候選、只收 1 個」，**純虧**——因為驗證那一步要把 block+1 個
位置全算一遍。舊的佔位草稿（重複上一個 token）量出來正好是 `g = 1.00`，等於沒投機。

這一節換三個真的草稿來源，量 k 隨文字類型怎麼變。

**草稿不是模型，是「在已看過的文字裡找重複」**（`LookupDraft`，prompt-lookup）：
從近往遠找最近的相同 3-gram，把後面的 token 當草稿；找不到就退回 2-gram、1-gram，
再找不到就回空清單（這一輪退化成 greedy，只虧一次判斷、不虧 block）。
**零成本、零額外記憶體、不需要額外權重**。

### 一、k 隨文字類型變化極大（1 台，48 token，block=4）

| 案例 | greedy | repeat（舊） | lookup3 | 逐 token 一致 |
|---|---|---|---|---|
| 散文（"The capital of France is"） | 9.57 tok/s | 9.02, k=0.00 | **12.53, k=0.34** | 一致 |
| 清單（行星編號） | 9.82 | 9.02, k=0.00 | 10.27, k=0.09 | 一致 |
| 程式（`def add/sub/mul`） | 9.85 | 8.99, k=0.00 | **18.39, k=1.04** | 一致 |
| 照抄（repeat 一句話） | 9.87 | 9.10, k=0.00 | **28.67, k=2.64** | 一致 |

兩個讀數：

1. **k 不是機器的性質，是文字與工作負載的性質。** 同一台機器同一個模型，k 從 0.09
   到 2.64，速度從 10.3 到 28.7 tok/s。任何「N 台能跑 X tok/s」的宣稱，必須先說
   它的 k 是多少——這是整份文件裡最容易被混過去的一個數字。
2. **repeat 這一列證明了「有投機但 k=0」比不投機更慢**（9.02 vs 9.57，-6%）：
   驗證 block+1 個位置的成本，一分都收不回來。

### 二、block 加大會繼續賺，但只在 k 高的文字上（1 台）

| block | 散文 k / tok/s | 清單 | 程式 | 照抄 |
|---|---|---|---|---|
| 2 | 0.31 / 12.28 | 0.07 / 10.07 | 0.78 / 15.86 | 1.58 / 22.11 |
| 4 | 0.34 / 12.47 | 0.09 / 10.22 | 1.04 / 17.87 | 2.64 / 28.53 |
| 6 | 0.34 / 12.01 | 0.09 / 10.11 | 1.45 / 20.65 | **3.64 / 36.37** |
| 8 | 0.34 / 12.25 | 0.09 / 10.09 | **1.45 / 20.80** | **4.20 / 38.49** |

- 散文的 k 在 0.34 就**飽和**了：再大的 block 只是把驗證成本從 3 個位置加到 9 個，
  所以 12.25 反而比 12.47 差。**block 要跟 k 匹配，不是越大越好。**
- 照抄從 22.11（block 2）一路漲到 38.49（block 8）——這是 3.9× 的單機加速，
  而且**全部逐 token 與 greedy 相同**（驗證只接受對得上的前綴）。

### 三、多台上，k 有沒有把 hop 攤薄？——部分攤薄，不是全部

| 台數 | greedy | spec（照抄, block 8） | spec（散文, block 4） |
|---|---|---|---|
| 1 台 | 9.66 | 38.49 | 12.47 |
| 2 台 | 9.59 | 28.31（block 4）/- | 12.27 |
| 4 台 | 9.24 | **35.38** | **11.63** |

四台 greedy 9.24、四台投機（照抄 block 8）**35.38 tok/s**：**3.8×**，而且與單機的
38.49 只差 8%。但要誠實：4 台對 1 台的折損（-8%）跟 greedy 的折損（-3%~-4%）
是同一量級，**不是「k 越大 hop 越便宜」那種乾淨的攤薄**——因為驗證一個 block 的
位置本身就要多算，多台時那部分也跟著乘以台數。真正被攤薄的只有「一輪一次 hop」。

### 四、試過但失敗的：把同一個模型的前 d 層當草稿（early exit）

想法很省：同 tokenizer 必然相容、不用下載新權重。實測**完全不行**：

| 草稿深度 | 第一個 token 是否等於 24 層的預測 | k | 速度 |
|---|---|---|---|
| 24/24 | 相同（`12095`） | — | — |
| 18/24 | **不同**（`30743`） | 0.00 | — |
| 12/24 | **不同**（`98964`） | 0.00 | 2.61 tok/s |
| 4/24 | 不同 | 0.00 | 3.79 tok/s |

模型自己的驗證很乾淨：24/24 層跑出來的 draft 與主模型逐 token 相同（`12095`），
所以**不是我的草稿接錯了，是這顆 0.5B 的預測在砍掉一半深度之後就崩了**。
early-exit 草稿能用，前提是那顆草稿被**訓練**過（trained draft head / 蒸餾），
不是拿來就能用。這條路留給「有真草稿模型」的時候，不是現在。

### 五、所以 k 這件事的結論

- **免費的草稿只值 1.3×（散文）到 3.9×（照抄），中間取決於你的工作負載。**
- 想把散文的 k 從 0.34 抬到 2 以上，只有一條路：**一個真的、同 tokenizer 的小模型**
  （0.5B 配 0.4B/0.6B 那種比例），或者一個訓練過的 draft head。這一項沒做，
  因為本機沒有那個權重檔；**沒量過的草稿不算草稿**。
- `RESULTS` 裡所有「8 台 33 tok/s / 16 台 66 tok/s」（`cluster/econ.py`）仍然是
  **條件承諾**，條件就是 k≈3。現在我們知道 k≈3 對「有重複結構的文字」成立，
  對自由散文不成立。

## pacing 控制器 + serve 層容錯（已實測）

### 一、為什麼 block 不能固定，以及控制器會怎麼卡死

前一節的數字有一半是「固定 block」量出來的，但真實的文字會變：
同一個模型，散文的 k=0.19~0.94、清單 0.09、程式 1.04~1.45、照抄 4.20。
**固定 block=8 在 k=0.19 的散文上是純虧（驗證 9 個位置只換 1.2 個 token）。**

`cluster/pacing.py` 用 `block ≈ 2k + 1`（夾在 `[1, max_block]`）線上決定，k 用 EWMA 追。
用同樣 4 條散文 prompt 量到的**匹配關係**（1 台，32 token）：

| prompt | greedy | block2 | block4 | block8 | k |
|---|---|---|---|---|---|
| 0（France） | 9.37 | **11.12** | 10.56 | 10.91 | 0.19 |
| 1（宇宙膨脹） | 9.75 | 14.19 | 16.91 | **16.92** | 0.94 |
| 2（機器學習） | 9.91 | 9.63 | 9.55 | 9.43 | 0.00 |
| 3（童話） | 9.56 | 9.73 | 9.77 | **9.67** | 0.03 |

- k=0.19 那一列，**block2 最好**（11.12 > 10.56 > 10.91 的排法說明「加大只是多算」）。
- k=0.94 那一列，block4/8 都到 16.9，block2 只有 14.19——**該大**。
- k=0.00 那一列，任何 spec 都比 greedy 差：控制器該做的是關掉投機。

**控制器自己踩到的坑（很典型）**：第一次上線時它把 block 調到 0，然後 k 永遠停在 0——
因為 block=0 就不送草稿，也就不再有「接受/被拒」的證據，收縮到 0 之後再也爬不回來。
修法是**每隔 5 輪探一次**（送 2 個草稿），寧可虧一次也不要把通道關死。
日誌裡看得到它自己在爬：`k=0.00 block=0 → k=0.81 block=3 → k=0.30 block=2`。

### 二、serve 的兩種模式（延遲 vs 吞吐）

`cluster/serve.py --mode batch|spec|auto`：
- `batch`：多條一起跑，買吞吐（併發 4 → 34.36 tok/s）。
- `spec`：單條投機，買延遲（k 由 pacing 決定，草稿是 prompt-lookup）。
- `auto`：只有一個人時走 spec，一有人一起就退回 batch。

實測（2 台，24 token，散文 prompt）：`mode=spec` 9.53 tok/s、`mode=batch` 9.57 tok/s。
**這一組數字是誠實的：這 4 條散文 prompt 的 k 只有 0.0~0.9，投機在這裡幾乎沒有東西可賺。**
投機能賺的是「有重複結構的文字」（照抄 3.9×），不是「一般散文」——這條界線不該被藏起來。

### 三、serve 層的掉線：重放前綴續跑，不是丟掉請求（已驗證到逐字）

鏈路層的容錯（`fault_recover.py`）驗的是「重建 + 重放」，但服務層原本只做到「重建」，
在飛的請求直接回報錯。現在補完：每一列的完整上下文永遠是 `ids + ids_out`，
所以掉線後只要**用現在的前綴重跑一次 prefill**，就能接著生成——不必從頭。

`bench/serve_fault.py`：同一個 prompt 跑兩次，第二次在生成 2.5s 時 SIGKILL 中間那台。

| 模式 | 基準（不掉線） | 掉線那次 | 串流段數 | 與基準逐字相同 |
|---|---|---|---|---|
| spec | 9.78s / 96 token | **10.18s / 96 token**（無錯誤） | 96（沒有重送） | **True** |
| batch | 10.12s / 96 token | **10.62s / 96 token**（無錯誤） | 96 | **True** |

兩個細節值得記：
- **代價只有 0.4~0.5s**（≈ 重建鏈 + 重放約 30 個 token 的前綴），不是「重跑整段」。
- 客戶端看到的串流**沒有重複的段**：`ids_out` 只在「確實推進」時增加，重放不會重播舊 token。

### 四、這一段抓到的一個 bug（`rebuild` 的 NoneType）

`Chain.stop()` 會把 `self.srv` 關掉並設成 `None`（這是對的，不關就 Errno 48）；
但 `Chain.rebuild()` 直接呼叫 `start()` → `connect()` 裡 `self.srv.settimeout()` → `NoneType`。
所以「重建鏈」這件事在 serve 層第一次跑就死在這裡，訊息是
`rebuild failed: 'NoneType' object has no attribute 'settimeout'`。
修法：`rebuild()` 先自己 `_open_srv()` 再 `start()`。

## 遠程連接：出門在外也能接家裡的機器（已實測）

家裡機器在 NAT 後面沒有可開的埠，所以用**反向上線 + 中轉機**：家裡 `publish` 主動
撥出去掛著，筆記本 `consume` 成本地埠。模型協議一行都不用改——遠端 stage 在 head
眼裡就是 `127.0.0.1:9001`。

實測（同一台機器上跑完整鏈：head → consume → gateway → publish → stage，反向也走
同一條）：

| 項目 | 結果 |
|---|---|
| 經中轉機的 hash | `158e23a8919692a3`，與直連 spawn **完全相同** |
| 吞吐 | 147.80 / 159.01 tok/s（兩次；直連 173.83，**約 -10%**：多了兩段用戶態轉發 + 兩跳） |
| 跑完一輪後 stage | **不死**（`--persist`），回到 accept |
| kill 中轉機再重啟、不重啟任何 stage | 第二次照樣 `158e23a8919692a3` / 159.01 tok/s ✅ |

也就是「筆記本闔蓋 → 換 Wi-Fi → 回家 → 隔天再開」這條路徑，實測不需要手動重啟
任何 stage、不需要重新認證：publish/consume 迴圈自己重連，`~/.cluster/token`（0600）
就是憑證，launchd 負責開機拉起。

過程中修掉四個真問題（都會表現成「連線莫名斷」而不是明確報錯）：

1. gateway 對 publish 先回 OK 才等 consumer → 配對到的是「掛了號但沒後端」的死連線，
   head 一寫就 `BrokenPipe`。改成**配對成功才回 OK**。
2. publish 先向 relay 掛號、才去連本地後端 → 同樣的死連線。改成**先連後端再掛號**。
3. stage 上游一斷就自殺（原本的設計是給 head spawn 用），遠端場景變成「筆記本一離線
   家裡就死一片」。加 `--persist`：ABORT 只結束一輪，上游斷了回 accept 等下一次。
4. 新上游進來時**必須清 KV**，否則上一輪的殘留位置會讓重放的 token 悄悄算錯。

順手修掉 `topology.py` 兩個舊 bug：head 埠取 `base+10`，16 台時正好撞上第 10 台的
stage 埠；以及遠端命令把所有機器都指向同一個 `--to` 埠。

## 已知限制（不是 bug，是刻意留的）

- **遠程那條中轉鏈路沒有加密**（只有 token 認證）。真要對公網開放，必須先套 WireGuard / Tailscale 或 TLS。
- **端口撞到殘留 stage 會靜默輸出錯 token**，這一輪又踩到一次：殘留進程佔著 7003，4 台跑的 hash 變成 `fbb3884b4293cdd0`（正確是 `126227e02636ad3a`），而且速度看起來更快（因為有一段根本沒在算）。已加防護：`spawn()` 前先 `_check_ports()`，撞到就直接 `SystemExit` 並說明後果，不讓它「合法但錯」。已驗證防護會擋下（佔用 7101 時 head 立即退出）。
- 鏈式分片已能跑 2 / 3 / 4 / 8 節點（實測），但 coordinator 與 head 仍合併在同一進程；`--peers` 已讓 stage 可在別的機器上跑，**跨機器這件事程式面通了，但還沒在真的 TB4 上跑過**（本機只有一台 Mac）。
- 傳輸未加密，只有 crc32 + shape/dtype 校驗。故障檢測 ≠ 誠實驗證。
- 草稿是佔位實作，接受率無意義（隨機權重上任何草稿都沒有預測力）。
- 只有 MLX 後端；Linux VM 節點是 TODO。
- 沒有 pacing 控制器（M3），目前 depth 由迴圈隱式決定。
- `--fast` 路徑沒有 batch 版。**batching 買吞吐不買延遲**：單人 B=1 的 102ms/token 不會變。
  只測到 B=8；B=16/32 的拐點（KV 讀取與 attention 開始壓過權重讀取）還沒量。
- **每個批次都從零 prefill**：`serve.py` 沒有前綴快取，也沒有「同一段對話續聊」；
  先 EOS 的列留在批次裡繼續算（輸出丟掉），因為抽掉中間一列就得重排 KV。
- **服務器沒有背壓、沒有優先級、沒有超時**：只有 `max_batch` 與 15ms 窗口兩個參數。

---

## 下一步

1. ~~**把記憶體約束併進 planner**~~ ✅ 已完成（上節一）
2. ~~**速度反饋回路**~~ ✅ 已完成（`cluster/probe.py` → `profile.json` → `plan --profile`）
3. ~~**故障注入**~~ ✅ 已完成：掉線偵測（上節「分配層工程化」三）、主機判斷 + 重建 + 重放恢復（4 台節四）
4. ~~**`mx.compile` + 固定 KV slab**~~ ✅ 已完成一半：0.383 → 0.211 ms/層（1.8×，上節 M3），端到端 +10~19%。**還沒到 0.10 那一檔**，剩下的 fixed 0.26ms/段與 Python 側開銷待拆。
5. **M3 pacing**：`depth = block + 1`、refill floor、disagreement 退回機制，跑 block 5/6/8/10 × floor 1/3/5 的 sweep。
5b. ~~**batching（B>1 批次解碼）**~~ ✅ 每步 102 → 113.7ms（B=1 → 8），tok/s 9.81 → 70.33。
    ~~請求佇列 + 動態湊批~~ ✅ `cluster/serve.py`（併發 4 → 34.78 tok/s，單條延遲 +9%）。
    ~~不等長左補齊 + per-row mask~~ ✅（`valid_from` + 內容位置，2 台逐字驗過）。
    **仍缺 (c) prefill 按長度分桶、前綴快取、背壓/優先級**；這些是「同一個對話續聊」
    與「長 prompt 不重算」的前提。
6. **換真模型 + 真草稿**（Qwen3 0.6B / 0.4B）：接受率才有意義。
7. **熱備真正接管**：現在 `--spare` 只是閒置，掉線走 `shrink` 縮容。要讓備機頂上，需要「動態加入 ring + 只重放新節點」。
8. **心跳**：用背景 HEARTBEAT 區分「慢」與「死」，把判定從單次 timeout 升級成持續監控。

建議順序 **1 → 2 → 3 → 4 → 5 → 6**：先把「分配」變成一件有約束、有反饋、能容錯的事，再去談速度。

M3 之後最該先補的兩件事（依現在的實測）：
- **固定成本 0.26ms/段** 已成為接力模式的第二大項（4 台 × 0.26 ≈ 1ms/token，佔 round 的 ~20%）。要拆它得先分清「Python 端陣列組裝」與「第一次 dispatch」。
- **prefill 的批次 graph 是新的固定成本**（每進程首次要 compile 一次，4 台多付 ~27ms）。長上下文時要改成按 power-of-2 分桶，而不是每個 L 各編一次。

## 重跑指令

```bash
cd /Users/xieyuteng/Doubao/chats/2026-11-09/new-chat-1

# 單機基線（冷啟動第一次會慢 10 倍，看第二次）
.venv-arm/bin/python -m cluster.main single --max-tokens 16 --repeat 3

# 兩節點 ring（tail 由 --tail 的端口自動起子進程）
.venv-arm/bin/python -m cluster.main run --layers 0:4 \
  --tail 127.0.0.1:5201 --head-port 5200 --max-tokens 16 --repeat 3

# 投機模式（輸出必須與 greedy 同 hash）
.venv-arm/bin/python -m cluster.main run --layers 0:4 \
  --tail 127.0.0.1:5202 --head-port 5203 --mode spec --block 4 --max-tokens 16

# 量測
PYTHONPATH=. .venv-arm/bin/python bench/m0_baseline.py
PYTHONPATH=. .venv-arm/bin/python bench/transport_only.py     # 純傳輸往返
.venv-arm/bin/python bench/transport_micro.py                  # 各元件拆解

# 任意節點數的鏈式分片（自動規劃 + 起 N-1 個 stage 子進程）
# 注意：head-port 必須和 stage-port 區段錯開，撞了會連到錯誤的對端
.venv-arm/bin/python -m cluster.main run --nodes 3 \
  --stage-port 5300 --head-port 5400 --prompt-len 512 --max-tokens 32

# 長上下文 + 投機：輸出必須與單機 greedy 同 hash
.venv-arm/bin/python -m cluster.main run --nodes 3 \
  --stage-port 5320 --head-port 5420 --prompt-len 512 --max-tokens 64 \
  --mode spec --block 4

# 量測
PYTHONPATH=. .venv-arm/bin/python bench/m0_baseline.py
PYTHONPATH=. .venv-arm/bin/python bench/transport_only.py     # 純傳輸往返
.venv-arm/bin/python bench/transport_micro.py                  # 各元件拆解

# 記憶體核算：會不會爆，最長能跑多長上下文
.venv-arm/bin/python -m cluster.footprint --summary --ctx 32768
.venv-arm/bin/python -m cluster.footprint --nodes 4 --ctx 32768

# 實測本機 tau / fixed，寫 profile.json（活迴圈量，不是固定形狀）
.venv-arm/bin/python -m cluster.probe --write

# 分段規劃：接力的時候要用 sum，不是 max
.venv-arm/bin/python -m cluster.main plan --nodes 8 --profile profile.json --hop-ms 0.071 --objective sum
.venv-arm/bin/python -m cluster.main plan --nodes 8 --profile profile.json --hop-ms 0.071 --objective max

# 時間 + 記憶體一起算：200B 在 24GB/台 上到底要幾台
.venv-arm/bin/python -m cluster.main plan --nodes 4 --model 200B --ctx 32768
.venv-arm/bin/python -m cluster.main plan --nodes 16 --model 200B --ctx 32768

# 故障注入：跑到一半 SIGKILL 掉 stage，驗 head 是否立刻給出理由（不是無限等）
.venv-arm/bin/python bench/fault_test.py

# 殺掉一台後能不能「恢復並且結果正確」：乾淨跑 vs 被殺跑的 hash 必須一致
.venv-arm/bin/python bench/fault_recover.py                  # 整鏈重建
ON_LOSS=shrink .venv-arm/bin/python bench/fault_recover.py   # 縮成 3 台繼續

# 4 台到底怎麼分：自檢 + 熱備 + 記憶體
.venv-arm/bin/python -m cluster.main plan --nodes 4 --profile profile.json --hop-ms 0.071
.venv-arm/bin/python -m cluster.main plan --nodes 4 --spare 1 --profile profile.json --hop-ms 0.071
.venv-arm/bin/python -m cluster.main plan --nodes 4 --model 200B --ctx 32768

# 4 台實測（注意：接力模式下 4 台比 1 台慢，這是正確結果不是 bug）
.venv-arm/bin/python -m cluster.main run --nodes 4 --stage-port 5720 --head-port 5730 --prompt-len 512 --max-tokens 32
.venv-arm/bin/python -m cluster.main run --nodes 4 --stage-port 5700 --head-port 5710 --prompt-len 512 --max-tokens 32 --objective sum

# 固定形狀 + compile：先驗正確性（hash 必須與原路徑相同），再看速度
PYTHONPATH=. .venv-arm/bin/python bench/m3_compile.py

# 帶 --fast 的端到端（生成要夠長，否則 compile/prefill 的固定成本攤不開）
.venv-arm/bin/python -m cluster.main run --nodes 4 --stage-port 6200 --head-port 6210 \
  --prompt-len 512 --max-tokens 128 --fast
.venv-arm/bin/python -m cluster.main run --nodes 2 --stage-port 6220 --head-port 6230 \
  --prompt-len 512 --max-tokens 128 --fast

# 用 FastModel 重新標定 tau / fixed（會覆寫 profile.json）
PYTHONPATH=. .venv-arm/bin/python -m cluster.probe --fast --write

# 容量/速度/錢：模型裝不下時，每多一台換到什麼
.venv-arm/bin/python -m cluster.econ --model 200B --nodes 5 8 16 --k 3 --hop 0.3
.venv-arm/bin/python -m cluster.econ --model 70B --nodes 2 4 8 --k 3 --hop 0.3

# 多機連接：布線、度數、層切分、每台機器要跑的指令
.venv-arm/bin/python -m cluster.topology --nodes 8
.venv-arm/bin/python -m cluster.topology --nodes 16 --hosts m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15

# 遠程：中轉機 + 家裡 publish + 筆記本 consume（出門在外用這組）
.venv-arm/bin/python -m cluster.remote newtoken            # 一次，寫 ~/.cluster/token
.venv-arm/bin/python -m cluster.remote gateway --port 8800 &   # 中轉機上
# 家裡（每台，裝進 launchd 就開機自動跑）：
.venv-arm/bin/python -m cluster.remote publish --relay relay:8800 --name m1 \
  --to 127.0.0.1:7001 --wait 300
# 筆記本：把遠端 stage 變成本地埠，之後照原來的方式跑 head
.venv-arm/bin/python -m cluster.remote consume --relay relay:8800 --listen 9001 --name m1 &
.venv-arm/bin/python -m cluster.main run --nodes 2 --peers 127.0.0.1:9001 --advertise ... --head-port ...
# 開機自啟的 plist（貼進 ~/Library/LaunchAgents/ 即可，之後不必再認證）
.venv-arm/bin/python -m cluster.remote plist --relay relay:8800 --name m1 --to-port 7001

# 真的「head 在一台、stage 在另一台」：先在各台起 stage，再用 --peers 接
# （同一台機器上先驗語意，換成 TB4 的 IP 就是真的跨機器）
.venv-arm/bin/python -m cluster.main stage --layers 4:8 --port 7001 --reply 127.0.0.1:7010 &
.venv-arm/bin/python -m cluster.main run --nodes 2 --peers 127.0.0.1:7001 \
  --advertise 127.0.0.1:7010 --head-port 7010 --prompt-len 512 --max-tokens 64

# 批次解碼：先驗「4 條一起跑 == 每條單獨跑」，再掃 B=1/2/4/8 的每步延遲
# （每個 B 換一組端口，避免 TIME_WAIT 讓預檢誤判）
PYTHONPATH=. .venv-arm/bin/python bench/batch_decode.py --tokens 12 --batch 1 2 4 8

# 批次服務：真的收多條請求，左補齊湊批，逐段串流回去
.venv-arm/bin/python -m cluster.serve --model-dir models/qwen2.5-0.5b --nodes 2 \
  --stage-port 7801 --head-port 7901 --port 8100 --max-batch 8 --window-ms 15
# 壓測：4 條長度不同的 prompt，先逐條單獨跑，再一起跑，逐字比對
PYTHONPATH=. .venv-arm/bin/python bench/serve_load.py --nodes 1 --clients 1 2 4
PYTHONPATH=. .venv-arm/bin/python bench/serve_load.py --nodes 2 --clients 1 2 4

# 投機：量 k（先比「逐 token 與 greedy 相同」，再看速度）
PYTHONPATH=. .venv-arm/bin/python bench/spec_k.py --nodes 1 --tokens 48 --block 8
PYTHONPATH=. .venv-arm/bin/python bench/spec_k.py --nodes 4 --tokens 48 --block 8 \
  --stage-port 7641 --head-port 7681
# pacing：讓 block 自己跟 k 匹配（1 台、32 token）
PYTHONPATH=. .venv-arm/bin/python bench/spec_k.py --nodes 1 --tokens 32 --block 8

# serve 的兩種模式 + serve 層容錯（掉線要續跑，不是報錯）
PYTHONPATH=. .venv-arm/bin/python bench/serve_load.py --nodes 2 --mode spec --clients 1
PYTHONPATH=. .venv-arm/bin/python bench/serve_load.py --nodes 2 --mode batch --clients 1 2 4
PYTHONPATH=. .venv-arm/bin/python bench/serve_fault.py --mode spec --tokens 96 --after 2.5
PYTHONPATH=. .venv-arm/bin/python bench/serve_fault.py --mode batch --tokens 96 --after 2.5

# early-exit 草稿（負面結果，可跳）
PYTHONPATH=. .venv-arm/bin/python bench/spec_k.py --nodes 1 --tokens 24 \
  --drafters lookup,self --draft-layers 12
```
