"""(B) What if you get the accept/reject rule slightly wrong? numpy, >= 1M trials per (rule, pair).

Same (p, q) pairs as spec_sampling.py. Draft tokens are SAMPLED from q (temperature 1).
Writes wrong_rules.json (used by plot_wrong_rules.py) and prints a table.
"""
import json, os
import numpy as np
from spec_sampling import make_pairs, tv, chi2_pvalue

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================== SLIDE CODE ==============================
def residual(p, q):
    r = np.maximum(p - q, 0)
    return r / r.sum()

RULES = {   # rule(p, q, x, u) -> (keep the draft token?, distribution to resample from otherwise)
    "1 correct":            lambda p, q, x, u: (u < np.minimum(1, p[x] / q[x]), residual(p, q)),
    "2 accept if p>=q":     lambda p, q, x, u: (p[x] >= q[x],                   residual(p, q)),
    "3 resample from p":    lambda p, q, x, u: (u < np.minimum(1, p[x] / q[x]), p),
    "4 argmax check":       lambda p, q, x, u: (x == p.argmax(),                np.eye(len(p))[p.argmax()]),
}

def simulate(rule, p, q, n, rng):
    x, u = rng.choice(len(q), size=n, p=q), rng.random(n)      # draft proposes x ~ q
    keep, fallback = rule(p, q, x, u)
    return np.where(keep, x, rng.choice(len(p), size=n, p=fallback))
# ========================================================================


def main(n=2_000_000, seed=0):
    pairs = make_pairs(np.random.default_rng(0))
    rng = np.random.default_rng(seed)
    out = {}
    names = list(RULES)
    print(f"n = {n:,} trials per (rule, pair). TV = total-variation distance of output vs target p; "
          f"chi2 p = chi-square goodness-of-fit p-value vs p\n")
    print(f"{'pair':38s} " + " ".join(f"{r:>22s}" for r in names))
    print(f"{'':38s} " + " ".join(f"{'TV':>10s} {'chi2 p':>11s}" for _ in names))
    for pname, (p, q) in pairs.items():
        cells, out[pname] = [], {"p": p.tolist(), "q": q.tolist()}
        for r in names:
            s = simulate(RULES[r], p, q, n, rng)
            emp = np.bincount(s, minlength=len(p)) / n
            d, pv = tv(emp, p), chi2_pvalue(s, p)
            out[pname][r] = {"empirical": emp.tolist(), "tv": d, "chi2_p": pv}
            cells.append(f"{d:10.4f} {pv:11.3g}")
        # bonus: what mlx_lm does at temp>0 -- draft x ~ q, target y ~ p, keep x iff x == y, else emit y
        x, y = rng.choice(len(q), size=n, p=q), rng.choice(len(p), size=n, p=p)
        emp = np.bincount(np.where(x == y, x, y), minlength=len(p)) / n
        out[pname]["5 sample-and-match (mlx_lm)"] = {"empirical": emp.tolist(), "tv": tv(emp, p),
                                                     "accept": float((x == y).mean()),
                                                     "accept_correct_rule": float(np.minimum(p, q).sum())}
        print(f"{pname:38s} " + " ".join(cells))
    print("\nbonus: mlx_lm-style sample-and-match (keep draft iff it equals an independent target sample):")
    for pname, d in out.items():
        b = d["5 sample-and-match (mlx_lm)"]
        print(f"  {pname:38s} TV {b['tv']:.4f} (lossless)  accept {b['accept']:.3f} vs {b['accept_correct_rule']:.3f} for rule 1")
    json.dump(out, open(os.path.join(HERE, "wrong_rules.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
