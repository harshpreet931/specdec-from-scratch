"""Measure what THIS Mac actually achieves: matmul FLOP/s and memory bandwidth (MLX, GPU)."""
import time
import mlx.core as mx


def bench(fn, iters=20, warmup=3):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


print(f"mlx {mx.__version__}  device: {mx.device_info()['device_name']}")
for dt in [mx.float32, mx.bfloat16, mx.float16]:
    N = 4096
    a, b = mx.random.normal((N, N)).astype(dt), mx.random.normal((N, N)).astype(dt)
    t = bench(lambda: a @ b)
    print(f"matmul {N}x{N} {str(dt):14s}: {2 * N**3 / t / 1e12:5.2f} TFLOP/s")

# bandwidth: read 2 GB, write tiny result (reduction) and read+write copy
x = mx.random.uniform(shape=(512 * 1024 * 1024,)).astype(mx.float32)   # 2 GB
mx.eval(x)
t = bench(lambda: x.sum(), iters=10)
print(f"read  2.0 GB (sum)  : {x.nbytes / t / 1e9:6.1f} GB/s")
t = bench(lambda: x * 2.0, iters=10)
print(f"read+write (x*2)    : {2 * x.nbytes / t / 1e9:6.1f} GB/s")
