"""Task 1: speculative sampling is LOSSLESS -- numerical proof with numpy only.

Run:  python3 spec_sampling.py
"""
import time
import numpy as np
from scipy.stats import chisquare


# ---------------------------------------------------------------- slide code
def speculative_sample(p, q, rng):
    """One position of speculative sampling. Output is distributed EXACTLY as p."""
    x = rng.choice(len(q), p=q)                  # 1. cheap draft proposes x ~ q
    if rng.random() < min(1.0, p[x] / q[x]):     # 2. keep it with prob min(1, p/q)
        return x, True
    residual = np.maximum(p - q, 0)              # 3. else resample from what the
    return rng.choice(len(p), p=residual / residual.sum()), False  # draft under-proposed
# ---------------------------------------------------------------------------


def speculative_sample_batch(p, q, n, rng, resample_from=None):
    """Same algorithm, vectorised over n independent trials (for 1M+ samples).
    resample_from lets us plug in a WRONG fallback distribution for contrast."""
    x = rng.choice(len(q), size=n, p=q)
    accept = rng.random(n) < np.minimum(1.0, p[x] / q[x])
    r = np.maximum(p - q, 0) if resample_from is None else resample_from
    fallback = rng.choice(len(p), size=n, p=r / r.sum()) if r.sum() > 0 else x
    return np.where(accept, x, fallback), accept


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def chi2_pvalue(samples, p):
    V = len(p)
    counts = np.bincount(samples, minlength=V)
    if counts[p == 0].sum() > 0:          # token the target never emits appeared
        return 0.0
    m = p > 0
    return chisquare(counts[m], f_exp=p[m] * len(samples)).pvalue


def make_pairs(rng):
    V8, V20, V50 = 8, 20, 50
    pairs = {}
    z = rng.normal(size=V8)
    pairs["close (q = p + small noise), V=8"] = (softmax(z), softmax(z + 0.3 * rng.normal(size=V8)))
    pairs["random vs random (Dirichlet), V=20"] = (rng.dirichlet(np.ones(V20)), rng.dirichlet(np.ones(V20)))
    p = np.zeros(V50); p[:3] = [0.7, 0.2, 0.1]
    pairs["peaky target, uniform draft, V=50"] = (p, np.full(V50, 1 / V50))
    p = np.array([0.40, 0.30, 0.20, 0.05, 0.02, 0.01, 0.01, 0.01])
    q = p[::-1].copy()
    pairs["reversed (badly mismatched), V=8"] = (p, q)
    pairs["moderate mismatch, V=6 (chart pair)"] = (np.array([0.40, 0.25, 0.15, 0.10, 0.06, 0.04]),
                                                    np.array([0.10, 0.40, 0.10, 0.25, 0.05, 0.10]))
    p = np.array([0.5, 0.3, 0.2, 0, 0, 0, 0, 0]); q = np.array([0, 0, 0.01, 0.39, 0.3, 0.2, 0.05, 0.05])
    pairs["near-disjoint support, V=8"] = (p, q)
    return pairs


def losslessness_table(n=2_000_000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, (p, q) in make_pairs(rng).items():
        out, acc = speculative_sample_batch(p, q, n, rng)
        emp = np.bincount(out, minlength=len(p)) / n
        noise = np.bincount(rng.choice(len(p), size=n, p=p), minlength=len(p)) / n
        wrong, _ = speculative_sample_batch(p, q, n, rng, resample_from=p.copy())
        rows.append(dict(
            pair=name, tv_pq=tv(p, q), alpha_theory=np.minimum(p, q).sum(), alpha_emp=acc.mean(),
            tv_spec=tv(emp, p), tv_noise=tv(noise, p), chi2_p=chi2_pvalue(out, p),
            tv_naive=tv(np.bincount(rng.choice(len(q), size=n, p=q), minlength=len(p)) / n, p),
            tv_wrong=tv(np.bincount(wrong, minlength=len(p)) / n, p), chi2_wrong=chi2_pvalue(wrong, p)))
    return rows


def slide_function_check(n=1_000_000, seed=1):
    """Run the readable 8-line function itself (python loop) for 1M trials."""
    rng = np.random.default_rng(seed)
    p = np.array([0.40, 0.30, 0.20, 0.05, 0.02, 0.01, 0.01, 0.01]); q = p[::-1].copy()
    t = time.time()
    res = [speculative_sample(p, q, rng) for _ in range(n)]
    out = np.array([r[0] for r in res]); acc = np.array([r[1] for r in res])
    emp = np.bincount(out, minlength=len(p)) / n
    return dict(n=n, tv=tv(emp, p), chi2_p=chi2_pvalue(out, p), alpha=acc.mean(),
                alpha_theory=np.minimum(p, q).sum(), secs=time.time() - t, emp=emp, p=p)


# ---------------------------------------------------------------- block level
def tokens_per_pass_iid(alpha, gamma, n, rng):
    """gamma drafted tokens, each accepted i.i.d. with prob alpha; stop at first reject.
    Every target pass yields (#accepted) + 1 tokens (the +1 is the correction/bonus)."""
    ok = rng.random((n, gamma)) < alpha
    accepted = np.cumprod(ok, axis=1).sum(axis=1)
    return (accepted + 1).mean()


def tokens_per_pass_real_sampler(p, q, gamma, n, rng):
    """Same, but each position really runs the speculative sampler on (p, q)."""
    total = 0
    acc_all = np.empty((n, gamma), dtype=bool)
    for j in range(gamma):
        _, acc_all[:, j] = speculative_sample_batch(p, q, n, rng)
    accepted = np.cumprod(acc_all, axis=1).sum(axis=1)
    return (accepted + 1).mean()


def expected_tokens(alpha, gamma):
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def main():
    t0 = time.time()
    print("=" * 100)
    print("TASK 1a  Losslessness: empirical output distribution of speculative sampling vs target p")
    print("         n = 2,000,000 trials per pair (vectorised); 'noise' = TV of 2M direct samples from p")
    print("=" * 100)
    hdr = f"{'(p, q) pair':38s} {'TV(p,q)':>8s} {'a_theory':>8s} {'a_emp':>7s} {'TV(out,p)':>10s} {'noise':>8s} {'chi2 p':>7s} | {'TV naive q':>10s} {'TV wrong':>9s} {'chi2 p':>7s}"
    print(hdr); print("-" * len(hdr))
    for r in losslessness_table():
        print(f"{r['pair']:38s} {r['tv_pq']:8.3f} {r['alpha_theory']:8.3f} {r['alpha_emp']:7.3f} {r['tv_spec']:10.5f} {r['tv_noise']:8.5f} {r['chi2_p']:7.3f} | {r['tv_naive']:10.4f} {r['tv_wrong']:9.4f} {r['chi2_wrong']:7.1e}")
    print("  a = acceptance rate; theory a = sum_i min(p_i, q_i) = 1 - TV(p, q)")
    print("  TV naive q = just using the draft's samples; TV wrong = fallback resamples from p instead of norm(max(0,p-q))")

    s = slide_function_check()
    print(f"\nSlide function itself (python loop), reversed pair, n={s['n']:,}: TV(out,p)={s['tv']:.5f}  chi2 p={s['chi2_p']:.3f}  "
          f"accept={s['alpha']:.4f} (theory {s['alpha_theory']:.4f})  [{s['secs']:.1f}s]")
    print("  target p :", np.round(s["p"], 4))
    print("  empirical:", np.round(s["emp"], 4))

    print("\n" + "=" * 100)
    print("TASK 1b  Tokens per target forward pass with a gamma-token draft block: E = (1 - a^(g+1)) / (1 - a)")
    print("         simulated with 1,000,000 blocks each")
    print("=" * 100)
    rng = np.random.default_rng(2)
    n = 1_000_000
    print(f"{'alpha':>6s} {'gamma':>6s} {'E formula':>10s} {'sim (iid)':>10s} {'speedup c=0.05':>15s} {'speedup c=0.15':>15s}")
    for a in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for g in [1, 2, 4, 8]:
            E = expected_tokens(a, g)
            sim = tokens_per_pass_iid(a, g, n, rng)
            # wall-clock speedup if one draft step costs c x one target step (Leviathan et al. eq.)
            sp = lambda c: E / (g * c + 1)
            print(f"{a:6.1f} {g:6d} {E:10.4f} {sim:10.4f} {sp(0.05):15.2f} {sp(0.15):15.2f}")
    print("  speedup = E / (gamma*c + 1), c = cost(draft step) / cost(target step); ignores verify-cost growth with gamma")

    print("\nSame check but every position runs the REAL sampler on a (p, q) pair (alpha = 1 - TV):")
    pairs = make_pairs(np.random.default_rng(0))
    print(f"{'pair':38s} {'alpha':>6s} {'gamma':>6s} {'E formula':>10s} {'sim':>8s}")
    for name in ["close (q = p + small noise), V=8", "random vs random (Dirichlet), V=20", "reversed (badly mismatched), V=8"]:
        p, q = pairs[name]
        a = np.minimum(p, q).sum()
        for g in [2, 4, 8]:
            print(f"{name:38s} {a:6.3f} {g:6d} {expected_tokens(a, g):10.4f} {tokens_per_pass_real_sampler(p, q, g, 500_000, rng):8.4f}")
    print(f"\n[total {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
