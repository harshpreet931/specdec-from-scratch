"""Why isn't verifying k tokens ~free on the Mac? Time ONE big linear layer (an 8B-model MLP
up-projection, 4096 -> 12288) for M = 1..64 rows: 4-bit quantized vs plain bf16 weights."""
import time, statistics
import mlx.core as mx

D_IN, D_OUT, LAYERS = 4096, 12288, 32          # 32 layers in one graph: one eval, no per-layer sync
ws = [mx.random.normal((D_OUT, D_IN)).astype(mx.bfloat16) * 0.02 for _ in range(LAYERS)]
qs = [mx.quantize(w, group_size=64, bits=4) for w in ws]
mx.eval(ws, qs)


def run(x, quant):    # chain layers through a cheap projection so it's ONE graph, one eval
    for i in range(LAYERS):
        y = mx.quantized_matmul(x, *qs[i], transpose=True, group_size=64, bits=4) if quant else x @ ws[i].T
        x = y[:, :D_IN]
    mx.eval(x)


def t(M, quant, iters=12):
    x = mx.random.normal((M, D_IN)).astype(mx.bfloat16)
    for _ in range(3):
        run(x, quant)
    ts = []
    for _ in range(iters):
        s = time.perf_counter(); run(x, quant); ts.append(time.perf_counter() - s)
    return statistics.median(ts) / LAYERS


print(f"one {D_IN}x{D_OUT} linear layer, mlx {mx.__version__}")
print(f"{'M rows':>7} {'4-bit us':>9} {'vs M=1':>7} {'bf16 us':>8} {'vs M=1':>7}")
b4 = b16 = None
for M in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 32, 64]:
    a, b = t(M, True), t(M, False)
    b4, b16 = b4 or a, b16 or b
    print(f"{M:>7} {a * 1e6:9.0f} {a / b4:6.2f}x {b * 1e6:8.0f} {b / b16:6.2f}x")
