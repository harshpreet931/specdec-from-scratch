"""Task 5: prompt-lookup (n-gram) decoding -- speculative decoding with NO draft model.

The 'draft' is just: find the last n tokens earlier in the text, and guess that what
followed them last time will follow them again. Verify with the target in one pass.
"""
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "task4_scratch"))
from specdec_scratch import summarize  # noqa: E402


# ============================== SLIDE CODE ==============================
def lookup_guesses(tokens, ngram=3, k=8, min_ngram=1):
    """Copy the k tokens that followed the most recent earlier match of the last n-gram."""
    for n in range(ngram, min_ngram - 1, -1):           # try 3-gram, then 2, (then 1)
        tail = tokens[-n:]
        for i in range(len(tokens) - n - 1, -1, -1):    # scan backwards (most recent first)
            if tokens[i:i + n] == tail:
                return tokens[i + n:i + n + k]
    return []


def prompt_lookup_decode(model, prompt, k=8, ngram=3, max_tokens=256, eos=(), min_ngram=1):
    cache = make_prompt_cache(model)
    model(mx.array(prompt[:-1])[None], cache=cache)              # prefill
    tokens, out, log = list(prompt), [], []
    while len(out) < max_tokens:
        guesses = lookup_guesses(tokens, ngram, k, min_ngram)    # free "draft"
        logits = model(mx.array([tokens[-1]] + guesses)[None], cache=cache)
        checks = mx.argmax(logits[0], axis=-1).tolist()          # verify all at once
        n = 0
        while n < len(guesses) and guesses[n] == checks[n]:
            n += 1
        new = guesses[:n] + [checks[n]]
        trim_prompt_cache(cache, len(guesses) - n)               # forget rejected guesses
        tokens, out = tokens + new, out + new
        log.append((n, len(guesses)))
        if any(t in eos for t in new):                           # stop at end-of-sequence
            break
    out = out[:max_tokens]
    if any(t in eos for t in out):
        out = out[: next(i for i, t in enumerate(out) if t in eos) + 1]
    return out, summarize(log)
# ========================================================================
