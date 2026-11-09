# 多台 Mac mini 聯合跑本地大模型（最小原型）

用 N 台普通 Mac（32GB、TB4 互連）跑一個**單機裝不下**的本地大模型。這不是「把模型切開放到別台」的演示，
而是一個可量測的原型：協議、分片、投機解碼、批次服務、掉線恢復，每一項都有實測數字與正確性基準。

核心結論先寫在前面（全部有數據，見 [`RESULTS.md`](RESULTS.md)）：

1. **加機器換的是容量，不是速度。** batch=1 的解碼是「接力」不是「並行」：N 台各算一段，總時間相加。
   實測 4 台 greedy 9.24 tok/s vs 1 台 9.66 tok/s——加機器反而略慢。
2. **要讓速度隨機器數成長，只有投機解碼。** 速度 ≈ (每輪鏈上計算 + 一次 hop) / g，`g = 1 + k`。
   `k` 是草稿被接受的個數。實測：散文 k≈0.0~0.9、程式 1.0~1.45、照抄一段文字 2.6~4.2。
3. **batching 買的是吞吐不是延遲。** 單機 B=1→8 每步只從 102ms 漲到 113.7ms（tok/s 9.81→70.33），
   但單人 B=1 的 102ms/token 一點不會變。4 條併發 → 34.8 tok/s，單條延遲只 +9%。
4. **32GB 機器實際只有約 24GB 給模型**（系統 + Metal 保留）。4 台 = 96GB 跑 120B dense；
   200B / 32K 上下文**最少需要 5 台**（用真 safetensors 逐 tensor 對帳，誤差 +0.020%）。

## 這個原型驗證了什麼

| 能力 | 實測結果 |
|---|---|
| 分片正確性 | 1/2/3/4/8 節點（含層數不能整除、極端 1 層×8）逐 token 與單機相同，多組 hash 基準 |
| 傳輸 | 48 字節定長 header + 單次 `sendall` + `recv_into` 預配置 slab：**0.071ms** 往返（原 3.1ms） |
| 批次解碼 | 4 條一起跑與每條單獨跑逐 token 相同；左補齊 + per-row 遮罩支援不等長 |
| 批次服務 | 請求佇列 + 15ms 湊批窗口 + 逐段串流；併發 4 → 34.8 tok/s |
| 投機解碼 | prompt-lookup 草稿；散文 1.0~1.3×、程式 2.1×、照抄 3.9×，全部與 greedy 逐 token 相同 |
| pacing | `block ≈ 2k+1` 線上自適應（含「關掉投機就量不到 k」的探測修正） |
| 掉線判定 | SIGKILL → `closed` **3ms**；SIGSTOP → `hung` **1511ms**；心跳走整條鏈，RTT 中位 0.15ms |
| 掉線恢復 | 鏈路層與服務層都做到「重建 → 重放前綴 → 續跑」；96 token 請求中途被殺，10.18s 完成、逐字相同、串流無重送 |
| 遠程接入 | 反向 `publish`/`consume` 中轉（NAT/CGNAT 後的家用機）；token + launchd，開機免重新認證 |
| 容量規劃 | planner 併入記憶體約束與實測 profile；預測 3.41ms vs 實測 3.39ms（1% 內） |

## 目錄結構

```
cluster/
  main.py         head/stage 進程編排、端點預檢（殘留端口會導致「合法形狀 + 錯誤 token」）
  stage.py        單段 stage：層區間、批次 argmax、心跳轉發、--persist
  coordinator.py   head：prefill、greedy、投機 chunk 驗證、KV 裁切、forward_rows（批次）
  protocol.py      48 字節 header + 可選 JSON 尾巴（batch meta）、零拷貝收發
  realmodel.py     Qwen2.5 真模型：GQA、q/k/v bias、tie embeddings、只讀自己那幾層的 safetensors
  serve.py         批次服務器：batch / spec / auto 三種模式，掉線重放續跑
  draft.py         草稿來源：repeat / prompt-lookup / early-exit（後者實測 k=0，留作負面記錄）
  pacing.py        block 控制器（k 的 EWMA、低於門檻就關投機、定期探測）
  planner.py       分片規劃（DP，目標函數 max|sum）+ validate / min_nodes / max_ctx
  capacity.py      容量階梯（32GB→24GB 可用、頭尾 embedding 的兩份成本）
  moe.py           EP vs pipeline 的兩筆通訊稅成本模型
  econ.py          容量 / 速度 / 錢三欄
  topology.py      多台怎麼連：鏈（n 台 = n−1 條線）、環 / 全互聯的物理上限、端口區段
  remote.py        gateway / publish / consume / plist / newtoken
  heartbeat 相關   coordinator.deadline_ms / ping、stage 的轉發式心跳
bench/
  m0_baseline.py transport_micro.py transport_only.py m3_compile.py   # 底層量測
  real_model.py batch_decode.py serve_load.py serve_fault.py spec_k.py
  heartbeat_test.py fault_test.py fault_recover.py
  verify_capacity.py cross_machine_rtt.py                             # 容量驗證 / 跨機（未驗）
  chain_topology.py                                                   # 鏈 vs 直連回報
```

## 跑起來

需要 arm64 python（`/usr/bin/python3`）、`mlx`、`numpy`、`tokenizers`、`huggingface_hub`。
模型權重不在這個 repo 裡（956MB）；自行放進 `models/qwen2.5-0.5b/`。

```bash
# 單機基準（greedy vs 切成兩段逐位比對）
PYTHONPATH=. .venv-arm/bin/python bench/real_model.py

# 2 台接力：批次解碼正確性 + B=1/2/4/8 吞吐
PYTHONPATH=. .venv-arm/bin/python bench/batch_decode.py --tokens 12 --batch 1 2 4 8

# 批次服務（吞吐）+ 壓測
.venv-arm/bin/python -m cluster.serve --model-dir models/qwen2.5-0.5b --nodes 2 \
  --stage-port 7801 --head-port 7901 --port 8100 --mode batch
PYTHONPATH=. .venv-arm/bin/python bench/serve_load.py --nodes 2 --clients 1 2 4

# 投機解碼：量 k（先比逐 token 一致，再看速度）
PYTHONPATH=. .venv-arm/bin/python bench/spec_k.py --nodes 4 --tokens 48 --block 8 \
  --stage-port 7641 --head-port 7681

# 4 台鏈（實體就是鏈：4 台 3 條線，每台只接上下家；--chain 是預設）
PYTHONPATH=. .venv-arm/bin/python -m cluster.main run --nodes 4 \
  --stage-port 7101 --head-port 0 --prompt-len 512 --max-tokens 32 --objective max

# 拓撲驗證：鏈式回程 vs 尾巴直連 head，逐 token 必須一致
PYTHONPATH=. .venv-arm/bin/python bench/chain_topology.py --nodes 2 3 4 \
  --stage-port 10301 --prompt-len 512 --tokens 32

# 每台實際要跑什麼指令（含 --ret-port 回程口、端口區段怎麼錯開）
PYTHONPATH=. .venv-arm/bin/python -m cluster.topology --nodes 4

# 掉線 → 重建 → 重放 → 續跑（服務層）
PYTHONPATH=. .venv-arm/bin/python bench/serve_fault.py --mode spec --tokens 96 --after 2.5
```

端口約定：head 與 stage 必須完全錯開（stage 用 `base+1..base+n`，head 用 `base+n+1`）。
殘留進程一律用 `lsof -nP -iTCP:<port> -sTCP:LISTEN -t | xargs kill -9` 清掉——
**殘留端口是本項目最陰的失敗模式：連到舊分片，形狀全合法，只是輸出錯 token。**

## 硬體基線：「一台 Mac mini」具體是哪一台

本項目大部分的數字都是這一台上量出來的，不是泛指「隨便一台小主機」。

| 項目 | 規格 |
|---|---|
| 機型 | **Mac mini（M4，2024 年 10 月發布）** |
| 芯片 | **Apple M4**：10 核 CPU（4 性能核 + 6 能效核）、10 核 GPU、16 核神經網絡引擎 |
| 統一內存 | **32GB**（M4 機型可選 16 / 24 / 32GB，M4 最高就是 32GB） |
| 內存帶寬 | **120 GB/s**（這一點直接決定 decode 速度上限：每 token 要把權重讀一遍） |
| 硬盤 | **512GB SSD**（可選 256GB / 1TB / 2TB） |
| 內建 GPU 顯存 | 與統一內存同一塊，沒有獨立顯存——所以「32GB 運存」就是它能給模型的全部 |
| 雷電 | 背面 **3 × Thunderbolt 4（40Gb/s）**；正面 2 × USB-C（10Gb/s）+ 3.5mm 耳機孔；另有 HDMI + 千兆網（可選 10Gb） |
| 體積重量 | 12.7 × 12.7 × 5.0 cm，0.67 kg |
| 電源 | 規格頁標示最大連續功率 155W（機型上限；跑本項目的 decode 負載遠低於此，實測整機約 30~40W） |
| 價格 | **¥8,999**（Apple 中國官網，32GB + 512GB 這一配置，2026-11 查價） |

### 同系列對照（決定了「能用什麼互聯手段」）

| 機型 | 芯片 / 內存 | 帶寬 | 雷電 | 能不能做 RDMA |
|---|---|---|---|---|
| Mac mini M4 32GB/512GB | M4 10 核 CPU / 10 核 GPU，32GB | 120 GB/s | 3 × TB4（40Gb/s） | **不能**（RDMA 要 TB5 + macOS 26.2） |
| Mac mini M4 Pro | 12 或 14 核 CPU、16 或 20 核 GPU，24~64GB | 273 GB/s | 3 × **TB5（120Gb/s）** | 可以 |
| Mac Studio M3 Ultra 512GB | M3 Ultra（28 或 32 核 CPU、60 或 80 核 GPU），512GB | 819 GB/s | TB5 | 可以 |

兩個直接後果，寫在硬體裡而不是寫在「限制」裡：

1. **基礎版 M4 mini 只有 TB4，沒有 RDMA**。所以本項目的傳輸是 TCP over TB4，
   每一次 hop 都付協議棧成本（實測 0.071ms，原始 3.1ms）。要 RDMA 就得換 M4 Pro / M5。
2. **16 台 32GB 的「512GB」是標稱值，實際可用只有 16 × 24 = 384GB**（每台扣系統 + Metal 約 8GB）。
   要真正打平一台 512GB Studio 的可用容量（約 480GB），需要 **20 台**。
   這也是為什麼下面那張表用「每台 24GB」算，而不是拿標稱 32GB 乘台數。

補一條容易漏的：**每台 512GB 的 SSD 也是分片約束**——4-bit 的 120B 權重約 60GB，
4 台各放 15GB 沒問題；但 planner 目前只核算 RAM，還沒把磁盤容量算進去（見「已知限制」）。

## 幾台？速度大概多少

### 一、最少幾台：由「裝不裝得下」決定，不是由速度決定

前提：**一台 M4 / 32GB / 512GB 的 Mac mini，實際只有約 24GB 給模型**（系統 + Metal 保留約 8GB）。
表裡的數字是單台最重那份的佔用（含該台要背的權重 + KV，node 0 額外背 embed + lm_head），
由 `python -m cluster.capacity --ctx 32768` 算出來的。

| 模型（4-bit 權重） | 4K 上下文 | 8K | 32K | 32GB 單機 |
|---|---|---|---|---|
| 32B dense | 1 台（17.5G） | 1 台 | 需 2 台 | 勉強可以 |
| 70B dense | 2 台 | 2 台 | 2 台（23.7G，幾乎塞滿） | 裝不下 |
| **120B dense**（4-bit ≈ 60GB） | 3 台 | 3 台 | **4 台**（19.9G） | 裝不下 |
| **200B dense**（4-bit ≈ 87GB） | 4 台 | 4 台 | **5 台**（21.1G） | 裝不下 |

- **架構上的最少是 2 台**：pipeline 至少要切成「頭」和「尾」兩段，1 台就不叫叢集。
- **實務上的推薦起點是 4 台**：4×32GB＝96GB 可用，剛好跨過 120B dense 這一檔（32K 上下文 19.9G/台，還有餘裕）。
- **200B dense + 32K 上下文：最少 5 台**。這條是硬門檻，4 台會在最重那台爆掉
  （4 台時 node 0 要背 25.9G > 24G，`裝得下 = NO`）。

### 二、最多幾台

- **實際佈線是「鏈」，不是環、也不是全互聯**：每台只接上下家，**n 台就是 n−1 條線**
  （4 台 3 條、16 台 15 條）。端點只佔 1 個 TB4 口，中間每台佔 2 個，剩下 1 個留給
  電源/管理/外設。這是唯一「不用多拉線」的接法。
- **鏈的代價在回程**：最後一台沒有直達 head 的路，結果要一級一級往上傳，所以每台多開
  一個回程口（`--ret-port`）。`--chain` 是**預設**；只有你真的願意在尾巴和 head 之間
  多拉一條線（成環）時，才用 `--no-chain` 走「尾巴直連 head」。
- **全互聯上限 4 台**：Mac mini M4 只有 3 個 TB4 口，任意兩台直連需要每台 n−1 條線；
  超過 4 台就不可能全互聯（4 台剛好 6 條線、每台 3 個口全占滿，沒有餘量）。
- **鏈可以到 16 台**（15 條線、中間每台 2 個口），16 台 ≈ ¥14.4 萬、滿載約 800W。
- **實測只到 8 節點**（1 層×8 的極端不均切分，正確性通過，hash `3e6516f959cff6a4`）。
  **16 台是物理可行、但本項目沒有實測過的**。
- 注意：**台數變多不會讓 greedy 變快**。接力模式下總時間 ≈ Σ(各段) + (n−1)×hop，
  每多接一台就多賠一整個 hop。這就是為什麼「更多機器」只在**投機解碼**下才換得到速度。

### 三、實測速度（真的量過的，不是估的）

| 台數 | 模型 | 模式 | 速度 |
|---|---|---|---|
| 1 | Qwen2.5-0.5B（真權重） | greedy | **9.8 tok/s**（101.6 ms/token） |
| 2 | 同上 | greedy | **9.35 tok/s** |
| 4 | 同上 | greedy | **8.74 tok/s** |
| 2 | 同上 | 批次併發 4 條 | **34.4 tok/s**（單條延遲只 +9%） |
| 1 | 同上 | 投機（照抄型文字） | **38.5 tok/s**（k=4.2） |
| 4 | 同上 | 投機（照抄型文字） | **35.4 tok/s**（k=4.2） |
| 1 | 玩具 8 層（一台裝得下） | greedy | 248.8 tok/s |
| 2 | 玩具 8 層 | greedy | 123.6 tok/s |
| 3 | 玩具 8 層 | greedy | 116.2 tok/s |
| 4 | 玩具 8 層 | greedy | 91.5 tok/s |
| 2 | 玩具 8 層 | 直連回報 / 鏈式回程 | 178.6 / 148.2 tok/s（同 512 prompt、32 token） |
| 3 | 玩具 8 層 | 直連回報 / 鏈式回程 | 139.1 / 138.9 tok/s |
| 4 | 玩具 8 層 | 直連回報 / 鏈式回程 | 123.7 / 114.9 tok/s；CLI 鏈式 **112.7 tok/s** |

倒數三行是**拓撲驗證**：同一顆模型、同一組切分，「尾巴直連 head」與「鏈式逐級回程」
逐 token 完全一致；鏈式只多付一次回程轉發，實測 2~4 台落在 0.83~1.01 倍之間
（數字抖動主要來自 loopback 與 hop 量測，不是拓撲本身）。

「玩具 8 層 1/2/3/4 台」那四行是**反面教材**：玩具模型一台就裝得下，所以「1 台」那一行存在，而且是**最快**的。
真模型（裝不下）才是這個項目的前提——那時 1 台根本不存在。

### 四、估算速度（成本模型，未實測，假設 k=3）

`python -m cluster.econ --model 120 --ctx 32768 --nodes 3 4 5 6` 這類指令算出來的，
假設每台帶寬 120 GB/s、每跳 0.3ms、每段固定 0.26ms、投機 k=3：

| 台數 | 70B | 120B dense | 200B dense |
|---|---|---|---|
| 2 | 21.0 | — | — |
| 3 | 31.0 | — | — |
| 4 | 41.8 | 24.2 | — |
| 5 | — | 29.4 | 21.0 |
| 6 | — | 34.3 | 23.9 |
| 8 | — | — | 33.5 |

兩條讀法：

- **接力欄幾乎與台數無關**（70B：3.37/3.36/3.36），加機器只換容量；
- **投機欄才隨台數長**，但它的前提是 k=3。自由散文實測 k≈0.2，
  那一欄要打三折看——**k 不是機器的屬性，是文字類型的屬性**。

> 這些數字都是「同一台 Mac mini 級別的機器」估算的。若混入性能不同的機器，
> planner 會自動把慢的機器分到較少的層（驗過 `node_speed=[1.0, 0.55, 1.0]` → 慢機器只拿 2 層）。

## 已知限制

- k 仍受文字類型限制；自由散文的單人加速只有 1.0~1.3×。要更高需要**真的同 tokenizer 的小模型**
  或訓練過的 draft head（early-exit 實測 k=0，不可用）。
- 沒有前綴快取 / 對話續聊：每個批次都從零 prefill。
- **planner 只核算 RAM，沒核算每台 SSD 的容量**：4-bit 的 120B 約 60GB 分給 4 台（各 15GB）
  在 512GB 機器上很寬鬆；但 200B 級 + 多模型共存時，磁盤也會變成硬約束。
- serve 沒有背壓、優先級、超時；只有 `max_batch`、窗口、`--mode` 三個旋鈕。
- 中轉鏈路只有 token 認證，沒有傳輸加密。
- **從未在真 TB4 上跑過**：`--peers`/`--advertise` 的語意在同機兩進程驗過，雲 VM 不可達（本機代理吃連線）。
- 真模型只接了 Qwen2.5-0.5B dense；MoE 只做了成本模型，沒有環內實作。

更完整的推導、失敗案例與量測方法見 [`DESIGN.md`](DESIGN.md) 與 [`RESULTS.md`](RESULTS.md)。

---

### 徵求真機測試

如果有條件的可以移動到多台Mac mini上測試，測試結果反饋給我，然後我進行代碼修改。
