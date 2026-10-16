"""Task 2: why checking 5 guesses costs ~the same as checking 1 (roofline, batch 1).

time(k tokens) ~= max(bytes_moved / bandwidth, flops / peak_flops)
At batch 1 every forward pass must stream ALL weights from memory once,
no matter whether it scores 1 token or k tokens. FLOPs grow with k; bytes barely do.
"""
P = 8.0e9                               # parameters (8B, Llama-3.1-8B-shaped)
LAYERS, KV_HEADS, HEAD_DIM, Q_DIM = 32, 8, 128, 4096
CTX = 1024                              # tokens already in the KV cache

HW = {  # name: (bytes per weight, bytes per KV element, mem bandwidth B/s, peak FLOP/s)
    "H100 SXM, BF16":       (2.0,    2, 3.35e12, 989e12),  # NVIDIA datasheet: 3.35 TB/s, 989 TF dense BF16
    "M4 Pro 20c GPU, 4-bit": (0.5625, 2, 273e9,   8.1e12),  # 4 bit + bf16 scale/bias per 64 = 4.5 bit; 273 GB/s; ~8.1 TF FP32 (est.)
}


def step_time(k, bpw, bkv, bw, peak):
    kv_bytes = 2 * LAYERS * KV_HEADS * HEAD_DIM * bkv * (CTX + k)     # read K and V cache
    flops = 2 * P * k + 4 * LAYERS * Q_DIM * (CTX + k) * k              # matmuls + attention
    t_mem, t_cmp = (P * bpw + kv_bytes) / bw, flops / peak
    return max(t_mem, t_cmp), t_mem, t_cmp


for name, (bpw, bkv, bw, peak) in HW.items():
    t1 = step_time(1, bpw, bkv, bw, peak)[0]
    ridge = (P * bpw / bw) / (2 * P / peak)   # k where weight-streaming time == matmul time
    print(f"\n{name}:  weights = {P * bpw / 1e9:.1f} GB,  ridge point ~ {ridge:.0f} tokens per pass")
    print(f"{'tokens verified':>16} {'memory ms':>10} {'compute ms':>11} {'time ms':>8} {'vs 1 token':>11} {'bound':>8}")
    for k in [1, 2, 4, 5, 8, 16, 64]:
        t, tm, tc = step_time(k, bpw, bkv, bw, peak)
        print(f"{k:>16} {tm * 1e3:>10.2f} {tc * 1e3:>11.2f} {t * 1e3:>8.2f} {t / t1:>10.2f}x {'memory' if tm >= tc else 'COMPUTE':>8}")
