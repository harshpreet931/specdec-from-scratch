"""Summarise results_temp*.jsonl from bench_mlx_lm.py into markdown tables (medians)."""
import json, os, re, sys, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def per_pass_n(flags):
    """Reconstruct accepted-count n for each verify pass from mlx_lm's from_draft flags."""
    ns, run = [], 0
    for f in flags:
        if f:
            run += 1
        else:
            ns.append(run); run = 0
    return ns, run          # run = trailing accepted tokens of a pass cut off by max_tokens/EOS


def load_verify_costs():
    """Measured ms per target pass for k tokens + draft ms/token (task2_roofline)."""
    p = os.path.join(ROOT, "task2_roofline", "verify_cost_results.txt")
    if not os.path.exists(p):
        return None, None
    txt = open(p).read()
    blocks = txt.split("mlx-community/")
    tv = {int(a): float(b) for a, b in re.findall(r"^\s+(\d+)\s+([\d.]+)\s", blocks[1], re.M)}
    td = {int(a): float(b) for a, b in re.findall(r"^\s+(\d+)\s+([\d.]+)\s", blocks[2], re.M)}
    return tv, td


def interp(table, k):
    """Linear interpolation of measured ms/pass between the measured k values."""
    if k in table:
        return table[k]
    lo = max(x for x in table if x < k); hi = min(x for x in table if x > k)
    return table[lo] + (table[hi] - table[lo]) * (k - lo) / (hi - lo)


def main(path):
    rows = [json.loads(l) for l in open(path)]
    temp = rows[0]["temp"]
    g = defaultdict(list)
    for r in rows:
        g[(r["prompt"], r["n_draft"])].append(r)
    prompts = list(dict.fromkeys(r["prompt"] for r in rows))
    Ns = sorted({r["n_draft"] for r in rows})
    tv, td = load_verify_costs()

    print(f"\n### temp = {temp}  (median of {len(g[(prompts[0], 0)])} runs, max 256 new tokens)\n")
    print("Generation tok/s as reported by mlx_lm (speedup vs baseline in brackets)\n")
    print("| prompt | " + " | ".join("baseline" if n == 0 else f"N={n}" for n in Ns) + " |")
    print("|---" * (len(Ns) + 1) + "|")
    speedups = defaultdict(list)
    for p in prompts:
        base = st.median(r["gen_tps"] for r in g[(p, 0)])
        cells = []
        for n in Ns:
            m = st.median(r["gen_tps"] for r in g[(p, n)])
            speedups[n].append(m / base)
            cells.append(f"{m:.1f}" if n == 0 else f"{m:.1f} ({m / base:.2f}x)")
        print(f"| {p} | " + " | ".join(cells) + " |")
    gm = lambda xs: (st.geometric_mean(xs))
    print("| **geo-mean speedup** | 1.00x | " + " | ".join(f"**{gm(speedups[n]):.2f}x**" for n in Ns if n) + " |")

    print("\nEnd-to-end tok/s (wall clock incl. prompt prefill) speedup vs baseline\n")
    print("| prompt | " + " | ".join(f"N={n}" for n in Ns if n) + " |")
    print("|---" * len(Ns) + "|")
    for p in prompts:
        base = st.median(r["e2e_tps"] for r in g[(p, 0)])
        print(f"| {p} | " + " | ".join(f"{st.median(r['e2e_tps'] for r in g[(p, n)]) / base:.2f}x" for n in Ns if n) + " |")

    print("\nAcceptance: alpha_hat = per-token accept prob = accepted/(accepted+rejections); "
          "'acc/drafted' = fraction of drafted tokens kept; tok/pass = tokens per target forward pass\n")
    print("| prompt | " + " | ".join(f"N={n} a_hat / kept / tok-per-pass" for n in Ns if n) + " |")
    print("|---" * len(Ns) + "|")
    alpha = {}
    for p in prompts:
        cells = []
        for n in Ns:
            if not n:
                continue
            acc = rej = toks = passes = drafted = 0
            for r in g[(p, n)]:
                ns, trail = per_pass_n(r["flags"])
                acc += sum(ns) + trail
                rej += sum(k < n for k in ns)
                passes += len(ns) + (1 if trail else 0)
                toks += r["tokens"]
                drafted += n * (len(ns) + (1 if trail else 0))
            a = acc / max(acc + rej, 1)
            alpha[(p, n)] = a
            cells.append(f"{a:.2f} / {acc / drafted:.2f} / {toks / passes:.2f}")
        print(f"| {p} | " + " | ".join(cells) + " |")

    print("\nTask-1 formula on real data: E(a_hat, N) = (1 - a^(N+1)) / (1 - a)  vs  measured tokens per target pass\n")
    print("| prompt | " + " | ".join(f"N={n} formula / measured" for n in Ns if n) + " |")
    print("|---" * len(Ns) + "|")
    for p in prompts:
        cells = []
        for n in Ns:
            if not n:
                continue
            a = min(alpha[(p, n)], 0.9999)
            toks = sum(r["tokens"] for r in g[(p, n)])
            passes = sum(len(per_pass_n(r["flags"])[0]) + (1 if per_pass_n(r["flags"])[1] else 0) for r in g[(p, n)])
            cells.append(f"{(1 - a ** (n + 1)) / (1 - a):.2f} / {toks / passes:.2f}")
        print(f"| {p} | " + " | ".join(cells) + " |")

    if tv:
        print("\nCost model check: predicted speedup = E(a_hat, N) * t(1) / (N * t_draft + t_verify(N+1)) "
              "using measured pass times; vs measured gen-tps speedup\n")
        print("| prompt | " + " | ".join(f"N={n} pred / meas" for n in Ns if n) + " |")
        print("|---" * len(Ns) + "|")
        for p in prompts:
            base = st.median(r["gen_tps"] for r in g[(p, 0)])
            cells = []
            for n in Ns:
                if not n:
                    continue
                a = min(alpha[(p, n)], 0.999)
                E = (1 - a ** (n + 1)) / (1 - a)
                pred = E * tv[1] / (n * td[1] + interp(tv, n + 1))
                meas = st.median(r["gen_tps"] for r in g[(p, n)]) / base
                cells.append(f"{pred:.2f} / {meas:.2f}")
            print(f"| {p} | " + " | ".join(cells) + " |")

    if temp == 0.0:
        print("\nGreedy token-identity vs baseline (first differing token index; '=' = identical, all reps)\n")
        print("| prompt | baseline reps agree | " + " | ".join(f"N={n}" for n in Ns if n) + " |")
        print("|---" * (len(Ns) + 1) + "|")
        for p in prompts:
            ref = g[(p, 0)][0]["ids"]
            base_ok = all(r["ids"] == ref for r in g[(p, 0)])
            cells = []
            for n in Ns:
                if not n:
                    continue
                ds = set()
                for r in g[(p, n)]:
                    a, b = r["ids"], ref
                    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                    if d is None and len(a) != len(b):
                        d = min(len(a), len(b))
                    ds.add("=" if d is None else str(d))
                cells.append(",".join(sorted(ds)))
            print(f"| {p} | {'yes' if base_ok else 'NO'} | " + " | ".join(cells) + " |")
    pk = max(r["peak_gb"] for r in rows)
    print(f"\npeak memory (max over runs): {pk:.2f} GB")


if __name__ == "__main__":
    for p in sys.argv[1:] or [os.path.join(HERE, f) for f in sorted(os.listdir(HERE)) if f.startswith("results_temp")]:
        main(p)
