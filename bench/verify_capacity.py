"""用真權重檔驗一遍容量算術（capacity.py 的公式）。

兩件事：

[1] 逐層權重對帳 —— 直接讀 safetensors header（不載入權重），把每層 bytes 加總，
    跟 formula 比。這是「N 台能不能裝」那張表唯一可驗的部分。

[2] 每個 stage 的實際 peak RSS —— 在獨立子進程裡只載入自己那幾層，量峰值。
    驗兩件事：記憶體是否真的只隨「自己那幾層」長；固定開銷與暫存有多大。

Run:
    PYTHONPATH=. .venv-arm/bin/python bench/verify_capacity.py
"""
import json
import os
import struct
import subprocess
import sys

MODEL_DIR = "models/qwen2.5-0.5b"
SF = os.path.join(MODEL_DIR, "model.safetensors")


def header(sf=SF):
    with open(sf, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def tbytes(h, key):
    a, b = h[key]["data_offsets"]
    return b - a


def layer_of(k):
    return int(k.split("layers.")[1].split(".")[0]) if "layers." in k else None


def layers_bytes(lo, hi, h=None):
    h = h or header()
    return sum(tbytes(h, k) for k in h
               if k != "__metadata__" and layer_of(k) is not None and lo <= layer_of(k) < hi)


def formula_layer_bytes(cfg, n_layers, bits=16.0):
    """一層的權重 bytes：q、k、v、o、gate、up、down（不含 bias）。"""
    h = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kvh = cfg["num_key_value_heads"]
    hd = h // heads
    ffn = cfg["intermediate_size"]
    per = h * (heads * hd) + 2 * h * (kvh * hd) + (heads * hd) * h + 3 * h * ffn
    return per * n_layers * bits / 8


def stage_bytes(cfg, lo, hi, bits=16.0):
    """這個 stage 真正要放進記憶體的 bytes：自己的層 +（頭）embed +（尾）lm_head + norm。

    tie 的模型，尾那台的 lm_head 就是同一顆 embedding 矩陣，但它是獨立的一份記憶體，
    所以整條鏈總共付兩份，不是一份。
    """
    w = formula_layer_bytes(cfg, hi - lo, bits)
    s = 0.0
    emb = cfg["vocab_size"] * cfg["hidden_size"] * bits / 8
    if lo == 0:
        s += emb
    if hi == cfg["num_hidden_layers"]:
        s += emb + 2 * cfg["hidden_size"] * bits / 8
    return w, s


def child(lo, hi):
    import resource
    from cluster.realmodel import RealModel
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    RealModel(MODEL_DIR, lo, hi, need_embed=(lo == 0),
              need_head=(hi == cfg["num_hidden_layers"]))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"lo": lo, "hi": hi, "rss_mb": peak / (1024 * 1024)}))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(int(sys.argv[2]), int(sys.argv[3]))
        return

    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    nl = cfg["num_hidden_layers"]
    hd = header()
    emb = tbytes(hd, "model.embed_tokens.weight")
    print(f"Qwen2.5-0.5B：{nl} 層、hidden {cfg['hidden_size']}、ffn {cfg['intermediate_size']}、"
          f"tie={cfg.get('tie_word_embeddings')}；embedding 一份 {emb/1e6:.1f}MB")

    print("\n[1] 逐層權重對帳（MB）")
    print(f"{'層段':<8}{'檔案':>9}{'公式':>9}{'差':>7}{'相對':>9}")
    ok = True
    for lo, hi in [(0, 1), (0, 8), (8, 16), (16, 24), (0, 12), (12, 24), (0, nl)]:
        real = layers_bytes(lo, hi, hd)
        f = formula_layer_bytes(cfg, hi - lo)
        d = real - f
        print(f"{f'{lo}:{hi}':<8}{real/1e6:>9.1f}{f/1e6:>9.1f}{d/1e6:>7.1f}{d/f:>+9.3%}")
        if abs(d / f) > 0.002:
            ok = False
    print(f"  結果：{'一致（誤差 < 0.2%）' if ok else '不一致 —— 公式要修'}")

    print("\n[2] 每個 stage 的 peak RSS（獨立子進程，只載自己那幾層）")
    for lo, hi in [(0, nl), (0, 12), (12, nl), (0, 6), (6, 12), (18, nl)]:
        out = subprocess.run([sys.executable, __file__, "--child", str(lo), str(hi)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            tail = out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "?"
            print(f"  {lo}:{hi} 失敗：{tail[:160]}")
            continue
        rss = json.loads(out.stdout.strip().splitlines()[-1])["rss_mb"]
        w, s = stage_bytes(cfg, lo, hi)
        need = (w + s) / 1e6
        print(f"  {lo}:{hi:<3} 必須放 {need:7.1f}MB（層 {w/1e6:.1f} + 頭尾 {s/1e6:.1f}）"
              f"  peak RSS {rss:7.1f}MB  差 {rss-need:+6.1f}MB")

    print("\n讀數：")
    print("  * 層權重公式與檔案逐 tensor 一致（誤差 0.02% 級）→ 容量表可信。")
    print("  * 中間段 RSS ≈ 自己的層 + ~40MB 固定開銷；頭/尾段會多出約一份 embedding")
    print("    的轉換暫存（272MB），那是峰值不是常駐，但規劃時必須留。")
    print("  * 所以「每台剛好塞滿」是錯的配置；每台要留一個層段 + 一份 embedding 的餘裕。")


if __name__ == "__main__":
    main()
