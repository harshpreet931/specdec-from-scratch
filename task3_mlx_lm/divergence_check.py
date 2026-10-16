"""Are greedy mismatches between baseline and speculative decoding bugs, or bf16 near-ties?

1. For every divergence in results_temp0.0.jsonl, recompute the target's logits at the first
   differing position and report the logit gap between the two competing tokens.
2. Re-run baseline vs speculative (mlx_lm) with the models' float params cast to float32
   (4-bit weights unchanged, but activations/scales in fp32 -> far less rounding noise).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm import stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def first_diff(a, b):
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    return d if d is not None or len(a) == len(b) else min(len(a), len(b))


def gen(target, tok, ids, draft=None, n=0, max_tokens=256):
    kw = dict(max_tokens=max_tokens, sampler=make_sampler(0.0))
    if draft is not None:
        kw.update(draft_model=draft, num_draft_tokens=n)
    return [r.token for r in stream_generate(target, tok, mx.array(ids), **kw)]


def main():
    target, draft, tok = common.load_pair()
    rows = [json.loads(l) for l in open(os.path.join(HERE, "results_temp0.0.jsonl"))]
    base = {r["prompt"]: r["ids"] for r in rows if r["n_draft"] == 0 and r["rep"] == 0}
    print("== 1. logit gap at each divergence (bf16 model, full-prefix forward pass) ==")
    print(f"{'prompt':10s} {'N':>2s} {'pos':>4s}  {'baseline tok':>18s} {'spec tok':>18s} {'logit gap':>9s} {'top1-top2 gap':>13s}")
    seen = set()
    for r in rows:
        if r["n_draft"] == 0 or r["rep"] != 0:
            continue
        b, s = base[r["prompt"]], r["ids"]
        d = first_diff(s, b)
        if d is None or d >= min(len(s), len(b)):
            continue
        ids = common.chat_tokens(tok, common.PROMPTS[r["prompt"]]) + b[:d]
        logits = target(mx.array(ids)[None])[0, -1].astype(mx.float32)
        top = mx.sort(logits)[::-1]
        gap = (logits[b[d]] - logits[s[d]]).item()
        print(f"{r['prompt']:10s} {r['n_draft']:>2d} {d:>4d}  {tok.decode([b[d]])!r:>18s} {tok.decode([s[d]])!r:>18s} "
              f"{gap:9.4f} {(top[0] - top[1]).item():13.4f}")

    print("\n== 2. same comparison with float params cast to float32 ==")
    target.set_dtype(mx.float32); draft.set_dtype(mx.float32)
    for p in ["creative", "math", "code_edit"]:
        ids = common.chat_tokens(tok, common.PROMPTS[p])
        ref = gen(target, tok, ids)
        res = []
        for n in [2, 3, 4, 6, 8]:
            d = first_diff(gen(target, tok, ids, draft, n), ref)
            res.append(f"N={n}: {'identical' if d is None else f'differs at {d}'}")
        print(f"{p:10s} ({len(ref)} tok)  " + " | ".join(res), flush=True)


if __name__ == "__main__":
    main()
