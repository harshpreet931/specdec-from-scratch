"""Task 4: speculative decoding from scratch on top of mlx_lm model objects.

greedy_decode           - plain one-token-at-a-time reference (the thing we must match)
speculative_decode      - SLIDE CODE: draft gamma tokens, verify in ONE target pass, roll back caches
speculative_sample      - sampling version (Leviathan/Chen rejection sampling, min(1, p/q))
"""
import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache


def greedy_decode(model, prompt, max_tokens=256, eos=()):
    cache = make_prompt_cache(model)
    model(mx.array(prompt[:-1])[None], cache=cache)           # prefill (same split as mlx_lm)
    logits = model(mx.array(prompt[-1:])[None], cache=cache)
    out = []
    while len(out) < max_tokens:
        y = mx.argmax(logits[0, -1]).item()
        out.append(y)
        if y in eos:
            break
        logits = model(mx.array([y])[None], cache=cache)
    return out, {"target_passes": len(out)}


def summarize(log):
    """log = [(accepted n, proposed g) per target pass]. alpha_hat = per-token acceptance
    probability (MLE under i.i.d. acceptance): accepted / (accepted + rejections)."""
    acc = sum(n for n, g in log)
    rej = sum(n < g for n, g in log)
    return {"target_passes": len(log), "accepted": acc, "drafted": sum(g for n, g in log),
            "rejections": rej, "alpha_hat": acc / max(acc + rej, 1)}


# ============================== SLIDE CODE ==============================
def speculative_decode(target, draft, prompt, gamma=4, max_tokens=256, eos=()):
    tc, dc = make_prompt_cache(target), make_prompt_cache(draft)
    target(mx.array(prompt[:-1])[None], cache=tc)          # prefill both models
    draft(mx.array(prompt[:-1])[None], cache=dc)
    last, todo, out, log = prompt[-1], [prompt[-1]], [], []  # todo = tokens draft hasn't seen
    while len(out) < max_tokens:
        # 1. DRAFT: small model guesses gamma tokens, one by one (cheap)
        x, guesses = mx.array(todo), []
        for _ in range(gamma):
            x = mx.argmax(draft(x[None], cache=dc)[0, -1:], axis=-1)
            guesses.append(x)
        guesses = mx.concatenate(guesses)
        # 2. VERIFY: big model scores [last, g1..g_gamma] in ONE forward pass
        logits = target(mx.concatenate([mx.array([last]), guesses])[None], cache=tc)
        checks = mx.argmax(logits[0], axis=-1).tolist()      # target's own pick at each slot
        guesses = guesses.tolist()
        # 3. ACCEPT the longest agreeing prefix + 1 token from the target for free
        n = 0
        while n < gamma and guesses[n] == checks[n]:
            n += 1
        new = guesses[:n] + [checks[n]]
        out += new
        log.append((n, gamma))
        # 4. ROLL BACK both KV caches so they forget the rejected guesses
        trim_prompt_cache(tc, gamma - n)
        trim_prompt_cache(dc, max(gamma - 1 - n, 0))
        todo = [guesses[-1], checks[n]] if n == gamma else [checks[n]]
        last = checks[n]
        if any(t in eos for t in new):                       # stop at end-of-sequence
            break
    out = out[:max_tokens]
    if any(t in eos for t in out):
        out = out[: next(i for i, t in enumerate(out) if t in eos) + 1]
    return out, summarize(log)
# ========================================================================


def speculative_sample(target, draft, prompt, gamma=4, temp=0.7, max_tokens=256, eos=(),
                       track_overlap=False):
    """Rejection-sampling version: output tokens are exact samples from softmax(target/temp)."""
    tc, dc = make_prompt_cache(target), make_prompt_cache(draft)
    target(mx.array(prompt[:-1])[None], cache=tc)
    draft(mx.array(prompt[:-1])[None], cache=dc)
    last, todo, out, log = prompt[-1], [prompt[-1]], [], []
    overlap_min, overlap_dot = [], []
    while len(out) < max_tokens:
        x, guesses, qs = mx.array(todo), [], []
        for _ in range(gamma):                               # draft: x ~ q
            q = mx.softmax(draft(x[None], cache=dc)[0, -1] / temp)
            x = mx.random.categorical(mx.log(q))[None]
            guesses.append(x); qs.append(q)
        guesses, q = mx.concatenate(guesses), mx.stack(qs)
        logits = target(mx.concatenate([mx.array([last]), guesses])[None], cache=tc)
        p = mx.softmax(logits[0] / temp, axis=-1)            # (gamma+1, V)
        idx = mx.arange(gamma)
        ratio = p[idx, guesses] / q[idx, guesses]
        ok = (mx.random.uniform(shape=(gamma,)) < mx.minimum(ratio, 1.0)).tolist()
        n = ok.index(False) if False in ok else gamma
        if track_overlap:   # P(accept) at each TESTED slot: sum min(p,q) (this) vs sum p*q (sample-and-match)
            overlap_min += mx.minimum(p[:gamma], q).sum(-1).tolist()[:n + 1]
            overlap_dot += (p[:gamma] * q).sum(-1).tolist()[:n + 1]
        if n < gamma:                                        # rejected: resample from max(0, p-q)
            r = mx.maximum(p[n] - q[n], 0)
            fix = mx.random.categorical(mx.log(r / r.sum())).item()
        else:                                                # all accepted: bonus token ~ p
            fix = mx.random.categorical(mx.log(p[gamma])).item()
        g = guesses.tolist()
        new = g[:n] + [fix]
        out += new
        log.append((n, gamma))
        trim_prompt_cache(tc, gamma - n)
        trim_prompt_cache(dc, max(gamma - 1 - n, 0))
        todo = [g[-1], fix] if n == gamma else [fix]
        last = fix
        if any(t in eos for t in new):
            break
    out = out[:max_tokens]
    if any(t in eos for t in out):
        out = out[: next(i for i, t in enumerate(out) if t in eos) + 1]
    stats = summarize(log)
    if track_overlap:
        stats["exp_accept_rejection"] = sum(overlap_min) / len(overlap_min)
        stats["exp_accept_match"] = sum(overlap_dot) / len(overlap_dot)
    return out, stats
