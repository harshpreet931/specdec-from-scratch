"""Measured roofline: time of ONE forward pass that scores k tokens, on this Mac.

Prefill a ~300-token context, then repeatedly: run k tokens through the model, eval, trim
the KV cache back. Median of 15 timings per k. Also measures the draft model (cost ratio c).
"""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache  # noqa: E402

KS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 64]


def verify_cost(model, ctx_ids, ks, iters=15):
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array(ctx_ids)[None], cache=cache))
    res = {}
    for k in ks:
        x = mx.array(ctx_ids[:k])[None]
        ts = []
        for i in range(iters + 3):
            t = time.perf_counter()
            mx.eval(model(x, cache=cache))
            dt = time.perf_counter() - t
            trim_prompt_cache(cache, k)
            if i >= 3:
                ts.append(dt)
        res[k] = statistics.median(ts)
    return res


def main():
    target, draft, tok = common.load_pair()
    ctx = common.chat_tokens(tok, common.PROMPTS["rewrite"] + " " + common.PROMPTS["code_edit"])[:300]
    print(f"context = {len(ctx)} tokens")
    for name, m, ks in [(common.TARGET, target, KS), (common.DRAFT, draft, [1, 2, 4, 8])]:
        r = verify_cost(m, ctx, ks)
        print(f"\n{name}")
        print(f"{'k tokens':>9} {'ms/pass':>8} {'vs k=1':>7} {'ms per token':>13}")
        for k, t in r.items():
            print(f"{k:>9} {t * 1e3:8.2f} {t / r[1]:6.2f}x {t * 1e3 / k:13.2f}")
        if m is target:
            t1 = r[1]
        else:
            print(f"\ncost ratio c = draft(1 tok) / target(1 tok) = {r[1] / t1:.3f}")


if __name__ == "__main__":
    main()
