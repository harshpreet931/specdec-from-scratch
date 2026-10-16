"""Markdown summary of results_main.jsonl written by bench_scratch.py (medians over reps)."""
import json, os, sys, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def main(path=os.path.join(HERE, "results_main.jsonl")):
    lines = [json.loads(l) for l in open(path)]
    identity = next(l["identity"] for l in lines if "identity" in l)
    rows = [l for l in lines if "method" in l]
    names = list(dict.fromkeys(r["method"] for r in rows))
    prompts = list(dict.fromkeys(r["prompt"] for r in rows))
    reps = len({r["rep"] for r in rows})
    med = lambda n, p, k="tps": st.median(r[k] for r in rows if r["method"] == n and r["prompt"] == p)

    print(f"\nEnd-to-end tok/s (wall clock incl. prefill, median of {reps}); speedup vs from-scratch greedy\n")
    print("| prompt | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    sp = {n: [] for n in names}
    for p in prompts:
        b = med("greedy", p)
        cells = []
        for n in names:
            sp[n].append(med(n, p) / b)
            cells.append(f"{med(n, p):.1f}" if n == "greedy" else f"{med(n, p):.1f} ({med(n, p) / b:.2f}x)")
        print(f"| {p} | " + " | ".join(cells) + " |")
    print("| **geo-mean** | " + " | ".join(f"**{st.geometric_mean(sp[n]):.2f}x**" for n in names) + " |")

    print("\nalpha_hat (per-token acceptance) / tokens per target pass  (pooled over reps)\n")
    others = [n for n in names if n != "greedy"]
    print("| prompt | " + " | ".join(others) + " |")
    print("|---" * (len(others) + 1) + "|")
    for p in prompts:
        cells = []
        for n in others:
            rs = [r for r in rows if r["method"] == n and r["prompt"] == p]
            acc, rej = sum(r["accepted"] for r in rs), sum(r["rejections"] for r in rs)
            cells.append(f"{acc / max(acc + rej, 1):.2f} / {sum(r['tokens'] for r in rs) / sum(r['target_passes'] for r in rs):.2f}")
        print(f"| {p} | " + " | ".join(cells) + " |")

    print("\nToken-for-token identical to greedy? ('=' identical, else index of first differing token)\n")
    greedy_like = [n for n in names if n != "greedy" and not n.startswith("sample")]
    print("| prompt | " + " | ".join(greedy_like) + " |")
    print("|---" * (len(greedy_like) + 1) + "|")
    for p in prompts:
        print(f"| {p} | " + " | ".join("=" if identity.get(f"{n}|{p}") is None else str(identity[f"{n}|{p}"]) for n in greedy_like) + " |")

    s = [r for r in rows if r["method"].startswith("sample")]
    if s:
        print(f"\n{s[0]['method']} (rejection sampling): per prompt  measured alpha_hat | mean sum(min(p,q)) "
              f"[theory for this method] | mean sum(p*q) [what sample-and-match would accept]\n")
        print("| prompt | measured alpha_hat | E[sum min(p,q)] | E[sum p*q] |")
        print("|---|---|---|---|")
        for p in prompts:
            rs = [r for r in s if r["prompt"] == p]
            acc, rej = sum(r["accepted"] for r in rs), sum(r["rejections"] for r in rs)
            print(f"| {p} | {acc / (acc + rej):.3f} | {st.mean(r['exp_accept_rejection'] for r in rs):.3f} | "
                  f"{st.mean(r['exp_accept_match'] for r in rs):.3f} |")


if __name__ == "__main__":
    main(*sys.argv[1:])
