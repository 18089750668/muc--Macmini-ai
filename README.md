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
  topology.py      多台怎麼連：M4 mini 3 個 TB4 口、環 vs 全互聯、端口區段約定
  remote.py        gateway / publish / consume / plist / newtoken
  heartbeat 相關   coordinator.deadline_ms / ping、stage 的轉發式心跳
bench/
  m0_baseline.py transport_micro.py transport_only.py m3_compile.py   # 底層量測
  real_model.py batch_decode.py serve_load.py serve_fault.py spec_k.py
  heartbeat_test.py fault_test.py fault_recover.py
  verify_capacity.py cross_machine_rtt.py                             # 容量驗證 / 跨機（未驗）
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

# 掉線 → 重建 → 重放 → 續跑（服務層）
PYTHONPATH=. .venv-arm/bin/python bench/serve_fault.py --mode spec --tokens 96 --after 2.5
```

端口約定：head 與 stage 必須完全錯開（stage 用 `base+1..base+n`，head 用 `base+n+1`）。
殘留進程一律用 `lsof -nP -iTCP:<port> -sTCP:LISTEN -t | xargs kill -9` 清掉——
**殘留端口是本項目最陰的失敗模式：連到舊分片，形狀全合法，只是輸出錯 token。**

## 已知限制

- k 仍受文字類型限制；自由散文的單人加速只有 1.0~1.3×。要更高需要**真的同 tokenizer 的小模型**
  或訓練過的 draft head（early-exit 實測 k=0，不可用）。
- 沒有前綴快取 / 對話續聊：每個批次都從零 prefill。
- serve 沒有背壓、優先級、超時；只有 `max_batch`、窗口、`--mode` 三個旋鈕。
- 中轉鏈路只有 token 認證，沒有傳輸加密。
- **從未在真 TB4 上跑過**：`--peers`/`--advertise` 的語意在同機兩進程驗過，雲 VM 不可達（本機代理吃連線）。
- 真模型只接了 Qwen2.5-0.5B dense；MoE 只做了成本模型，沒有環內實作。

更完整的推導、失敗案例與量測方法見 [`DESIGN.md`](DESIGN.md) 與 [`RESULTS.md`](RESULTS.md)。
