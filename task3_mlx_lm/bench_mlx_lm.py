"""Task 3: baseline vs speculative decoding with stock mlx_lm (stream_generate + draft_model).

Interleaved runs (rep -> prompt -> config) so thermal drift hits all configs equally.
Writes raw rows to results_<temp>.jsonl and prints median tables.

usage: python3 bench_mlx_lm.py --temp 0.0 --reps 3 [--prompts code rewrite creative math]
"""
import argparse, json, os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402  (sets HF_HOME before mlx_lm import)
import mlx.core as mx  # noqa: E402
from mlx_lm import stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def run_once(target, draft, tok, prompt_ids, n_draft, temp, max_tokens, seed):
    mx.random.seed(seed)
    kw = dict(max_tokens=max_tokens, sampler=make_sampler(temp))
    if n_draft:
        kw.update(draft_model=draft, num_draft_tokens=n_draft)
    toks, flags = [], []
    t0 = time.perf_counter()
    for r in stream_generate(target, tok, mx.array(prompt_ids), **kw):
        toks.append(r.token); flags.append(r.from_draft); last = r
    wall = time.perf_counter() - t0
    n = len(toks)
    accepted = sum(flags)
    passes = n - accepted                      # every verify pass emits exactly 1 non-draft token
    return dict(tokens=n, gen_tps=last.generation_tps, prompt_tps=last.prompt_tps,
                e2e_tps=n / wall, wall=wall, accepted=accepted, passes=passes,
                drafted=passes * n_draft if n_draft else 0, peak_gb=last.peak_memory,
                finish=last.finish_reason, ids=toks, flags=flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--drafts", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    ap.add_argument("--prompts", nargs="+", default=["code", "rewrite", "creative", "math", "code_edit"])
    a = ap.parse_args()

    target, draft, tok = common.load_pair()
    eos = tok.eos_token_ids
    configs = [0] + a.drafts                   # 0 = baseline (no draft model)
    prompts = {k: common.chat_tokens(tok, common.PROMPTS[k]) for k in a.prompts}
    out_path = os.path.join(HERE, f"results_temp{a.temp}.jsonl")
    f = open(out_path, "w")

    # warm-up: every config once on a short prompt (kernel/library load, allocator)
    for c in configs:
        run_once(target, draft, tok, prompts[a.prompts[0]], c, a.temp, 48, 0)
    print(f"warm-up done; temp={a.temp} reps={a.reps} max_tokens={a.max_tokens}", flush=True)

    rows = []
    for rep in range(a.reps):
        for pname, ids in prompts.items():
            for c in configs:
                r = run_once(target, draft, tok, ids, c, a.temp, a.max_tokens, seed=1000 + rep)
                r.update(prompt=pname, n_draft=c, rep=rep, temp=a.temp, t=time.time())
                rows.append(r)
                f.write(json.dumps(r) + "\n"); f.flush()
                print(f"rep{rep} {pname:9s} N={c}  {r['tokens']:3d} tok  gen {r['gen_tps']:6.1f} tok/s  "
                      f"e2e {r['e2e_tps']:6.1f}  acc {r['accepted']}/{r['drafted']}", flush=True)
    f.close()
    print(f"raw rows -> {out_path}")


if __name__ == "__main__":
    main()
