"""Gotcha check: does stock mlx_lm speculative decoding stay correct on hybrid
(linear-attention + attention) models like Qwen3.5, whose recurrent-state caches can't be trimmed?

Method: model drafts for itself at temp 1.0 (so drafts DO get rejected and the cache must be
rolled back). For every emitted token we keep the log-prob the target reported during
speculative decoding, then re-score the same sequence with ONE fresh forward pass (no cache
tricks). If rollback works, the two agree (up to bf16 noise). If it silently fails, they don't.

usage: python3 hybrid_cache_check.py mlx-community/Qwen3-0.6B-4bit mlx-community/Qwen3.5-0.8B-4bit
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache  # noqa: E402


def check(repo, n_draft=3, max_tokens=120, use_draft=True):
    model, tok = load(repo)
    ids = common.chat_tokens(tok, common.PROMPTS["creative"])
    mx.random.seed(0)
    toks, lps, flags = [], [], []
    kw = dict(draft_model=model, num_draft_tokens=n_draft) if use_draft else {}
    for r in stream_generate(model, tok, mx.array(ids), max_tokens=max_tokens,
                             sampler=make_sampler(1.0), **kw):
        toks.append(r.token); lps.append(r.logprobs[r.token].item()); flags.append(r.from_draft)
    logits = model(mx.array(ids + toks[:-1])[None]).astype(mx.float32)
    ref = logits[0, len(ids) - 1:] - mx.logsumexp(logits[0, len(ids) - 1:], axis=-1, keepdims=True)
    ref = [ref[i, t].item() for i, t in enumerate(toks)]
    diffs = [abs(a - b) for a, b in zip(lps, ref)]
    first_bad = next((i for i, d in enumerate(diffs) if d > 0.5), None)
    print(f"{repo:36s} spec={use_draft!s:5s} cache trimmable={can_trim_prompt_cache(make_prompt_cache(model))!s:5s} "
          f"tokens={len(toks):3d} rejections~{len(toks) - sum(flags):3d}  max|dlogp|={max(diffs):6.3f}  "
          f"mean|dlogp|={sum(diffs) / len(diffs):.4f}  first token with |dlogp|>0.5: {first_bad}")


if __name__ == "__main__":
    for repo in sys.argv[1:]:
        check(repo, use_draft=False)   # control: plain decoding
        check(repo, use_draft=True)
