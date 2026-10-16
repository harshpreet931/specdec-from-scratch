"""(A) Headline measurement: cost of ONE forward pass vs tokens per pass, target and draft.

Fixed context of 512 tokens already in the KV cache; time one pass that appends k tokens
(mx.eval forces the GPU work to finish), trim the cache back, repeat.
3 warm-up runs + median of 15 timed runs per k. Saves verify_cost_curve.json.
"""
import json, os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
KS = [1, 2, 3, 4, 5, 6, 8, 16, 32, 64, 128, 256, 512]
CTX, WARMUP, RUNS = 512, 3, 15


def timed_pass(model, x, cache):
    t = time.perf_counter()
    mx.eval(model(x[None], cache=cache))       # logits for all k positions (a real verify needs them)
    return time.perf_counter() - t


def sweep(model, ids):
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array(ids[:CTX])[None], cache=cache))      # 512 tokens of context
    res = {}
    for k in KS:
        x = mx.array(ids[CTX:CTX + k])
        ts = []
        for i in range(WARMUP + RUNS):
            dt = timed_pass(model, x, cache)
            trim_prompt_cache(cache, k)                          # back to exactly 512 cached tokens
            if i >= WARMUP:
                ts.append(dt)
        assert cache[0].offset == CTX
        res[k] = {"median_ms": statistics.median(ts) * 1e3, "min_ms": min(ts) * 1e3,
                  "max_ms": max(ts) * 1e3, "runs": RUNS}
        print(f"  k={k:>3}  {res[k]['median_ms']:8.2f} ms/pass  {res[k]['median_ms'] / k:7.3f} ms/token  "
              f"(min {res[k]['min_ms']:.2f}, max {res[k]['max_ms']:.2f})", flush=True)
    ts = []                                                      # sanity: k=1, NO prior context
    for i in range(WARMUP + RUNS):
        dt = timed_pass(model, mx.array(ids[:1]), make_prompt_cache(model))
        if i >= WARMUP:
            ts.append(dt)
    res["k1_no_context_ms"] = statistics.median(ts) * 1e3
    print(f"  k=1 with empty cache: {res['k1_no_context_ms']:.2f} ms", flush=True)
    return res


def main():
    target, draft, tok = common.load_pair()
    text = " ".join(common.PROMPTS.values()) + " " + common.TYPO_PARAGRAPH
    ids = tok.encode(text)
    while len(ids) < CTX + max(KS):
        ids = ids + ids
    out = {"context_tokens": CTX, "device": mx.device_info()["device_name"], "mlx": mx.__version__}
    for name, m in [(common.TARGET, target), (common.DRAFT, draft)]:
        print(name, flush=True)
        out[name] = sweep(m, ids)
    t, d = out[common.TARGET], out[common.DRAFT]
    out["c_draft_over_target_k1"] = d[1]["median_ms"] / t[1]["median_ms"]
    print(f"\nc = draft/target at k=1: {out['c_draft_over_target_k1']:.3f}")
    print(f"HEADLINE: checking 8 tokens took {t[8]['median_ms']:.1f} ms vs {t[1]['median_ms']:.1f} ms to write 1 "
          f"({t[8]['median_ms'] / t[1]['median_ms']:.2f}x the time for 8x the tokens)")
    json.dump(out, open(os.path.join(HERE, "verify_cost_curve.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
