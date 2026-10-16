"""Task 4 + 5 benchmark: from-scratch speculative decoding and prompt lookup vs plain greedy.

Checks token-for-token equality against (a) our greedy_decode and (b) mlx_lm's own greedy
generate_step, then times everything (median of --reps runs, interleaved, after warm-up).

usage: python3 bench_scratch.py --reps 3 --gammas 2 3 4 6 8
       python3 bench_scratch.py --self-test     # 0.6B drafts for itself: must accept 100%
"""
import argparse, json, os, sys, time, statistics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "task5_prompt_lookup"))
import common  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.generate import generate_step  # noqa: E402
from specdec_scratch import greedy_decode, speculative_decode, speculative_sample  # noqa: E402
from prompt_lookup import prompt_lookup_decode  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def mlx_lm_greedy(model, prompt, max_tokens, eos):
    out = []
    for tok, _ in generate_step(mx.array(prompt), model, max_tokens=max_tokens):
        out.append(int(tok))
        if out[-1] in eos:
            break
    return out


def timed(fn, *args, **kw):
    t = time.perf_counter()
    out, stats = fn(*args, **kw)
    return out, stats, time.perf_counter() - t


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gammas", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--lookup", nargs="+", default=["8:1", "8:2", "3:2"], help="k:min_ngram variants")
    ap.add_argument("--prompts", nargs="+", default=["code", "rewrite", "creative", "math", "code_edit"])
    ap.add_argument("--sample-temp", type=float, default=0.7)
    ap.add_argument("--sample-gamma", type=int, default=3)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        target, tok = load(common.DRAFT); draft = target
    else:
        target, draft, tok = common.load_pair()
    eos = set(tok.eos_token_ids)
    prompts = {k: common.chat_tokens(tok, common.PROMPTS[k]) for k in a.prompts}
    M = a.max_tokens

    methods = {"greedy": lambda p: greedy_decode(target, p, M, eos)}
    for g in a.gammas:
        methods[f"spec_g{g}"] = (lambda g: lambda p: speculative_decode(target, draft, p, g, M, eos))(g)
    for v in a.lookup:
        k, mn = map(int, v.split(":"))
        methods[f"look_k{k}n{mn}"] = (lambda k, mn: lambda p: prompt_lookup_decode(target, p, k, 3, M, eos, mn))(k, mn)
    methods[f"sample_g{a.sample_gamma}"] = lambda p: speculative_sample(target, draft, p, a.sample_gamma, a.sample_temp,
                                                                       M, eos, track_overlap=True)

    # ---- correctness: token-for-token vs greedy (ours AND mlx_lm's) ----
    print("== correctness (greedy) ==", flush=True)
    ref = {}
    for pname, ids in prompts.items():
        ours, _ = greedy_decode(target, ids, M, eos)
        theirs = mlx_lm_greedy(target, ids, M, eos)
        ref[pname] = ours
        d = first_diff(ours, theirs)
        print(f"{pname:9s} our greedy vs mlx_lm greedy: {'IDENTICAL' if d is None else f'differ at {d}'} ({len(ours)} tok)")
    identity = {}
    for name in methods:
        if name == "greedy" or name.startswith("sample"):
            continue
        for pname, ids in prompts.items():
            out, st = methods[name](ids)
            d = first_diff(out, ref[pname])
            identity[(name, pname)] = d
            print(f"{name:10s} {pname:9s}: {'IDENTICAL' if d is None else f'DIFFERS at token {d}'}  "
                  f"acc {st['accepted']}/{st['drafted']}  passes {st['target_passes']}", flush=True)

    # ---- timing ----
    for name, fn in methods.items():          # warm-up
        fn(prompts[a.prompts[0]][:])
    rows = []
    for rep in range(a.reps):
        for pname, ids in prompts.items():
            for name, fn in methods.items():
                mx.random.seed(rep)
                out, st, dt = timed(fn, ids)
                r = dict(method=name, prompt=pname, rep=rep, tokens=len(out), secs=dt, tps=len(out) / dt, **st)
                rows.append(r)
                print(f"rep{rep} {pname:9s} {name:10s} {len(out):3d} tok {r['tps']:6.1f} tok/s  "
                      f"passes {st['target_passes']}", flush=True)
    tag = "selftest" if a.self_test else "main"
    with open(os.path.join(HERE, f"results_{tag}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"identity": {f"{k[0]}|{k[1]}": v for k, v in identity.items()}}) + "\n")

    # ---- summary (markdown tables) ----
    import analyze_scratch
    analyze_scratch.main(os.path.join(HERE, f"results_{tag}.jsonl"))


if __name__ == "__main__":
    main()
