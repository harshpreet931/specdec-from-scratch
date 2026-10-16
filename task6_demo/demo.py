"""On-camera demo: baseline vs speculative decoding, same prompt, same model, clean table.

  python3 demo.py                       # code prompt, N=2 draft tokens, text streamed live
  python3 demo.py --prompt rewrite      # copy-heavy prompt (biggest win)
  python3 demo.py --prompt "Explain KV caching in 3 sentences."
  python3 demo.py --quiet --repeat 3    # numbers only, median of 3 runs each

During the speculative run, tokens that came from the DRAFT model (and were accepted by
the big model) are printed in green; tokens produced by the big model itself in white.
"""
import argparse, os, statistics, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
import mlx.core as mx  # noqa: E402
from mlx_lm import stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

GREEN, DIM, BOLD, RESET = "\033[32m", "\033[2m", "\033[1m", "\033[0m"


def run(target, tok, ids, draft=None, n_draft=2, max_tokens=256, show=False):
    kw = dict(max_tokens=max_tokens, sampler=make_sampler(0.0))
    if draft is not None:
        kw.update(draft_model=draft, num_draft_tokens=n_draft)
    toks, flags = [], []
    for r in stream_generate(target, tok, mx.array(ids), **kw):
        toks.append(r.token); flags.append(r.from_draft)
        if show:
            print((GREEN + r.text + RESET) if r.from_draft else r.text, end="", flush=True)
        last = r
    if show:
        print("\n")
    passes = len(toks) - sum(flags)
    return dict(tps=last.generation_tps, tokens=len(toks), secs=len(toks) / last.generation_tps,
                accepted=sum(flags), passes=passes, ids=toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="code")
    ap.add_argument("--draft-tokens", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    print(f"{BOLD}Speculative decoding on {mx.device_info()['device_name']}{RESET}")
    print(f"  target: {common.TARGET}\n  draft : {common.DRAFT}   (guesses {a.draft_tokens} tokens ahead)\n")
    t = time.time()
    target, draft, tok = common.load_pair()
    text = common.PROMPTS.get(a.prompt, a.prompt)
    ids = common.chat_tokens(tok, text)
    run(target, tok, ids[:64], max_tokens=16); run(target, tok, ids[:64], draft, a.draft_tokens, 16)  # warm-up
    print(f"{DIM}models loaded + warmed up in {time.time() - t:.1f}s{RESET}\n")
    print(f"{BOLD}Prompt:{RESET} {text[:110]}{'...' if len(text) > 110 else ''}\n")

    base, spec = [], []
    for i in range(a.repeat):
        show = not a.quiet and i == 0
        if show: print(f"{BOLD}[1] Big model alone{RESET}")
        base.append(run(target, tok, ids, max_tokens=a.max_tokens, show=show))
        if show: print(f"{BOLD}[2] Big model + draft model{RESET}  {DIM}(green = draft guess accepted){RESET}")
        spec.append(run(target, tok, ids, draft, a.draft_tokens, a.max_tokens, show=show))

    b = statistics.median(r["tps"] for r in base)
    s = statistics.median(r["tps"] for r in spec)
    r = spec[0]
    same = all(x["ids"] == base[0]["ids"] for x in spec)
    print("+" + "-" * 50 + "+")
    print(f"| {'':24s}{'tokens':>8s}{'tok/s':>9s}{'time':>8s} |")
    print(f"| {'Big model alone':24s}{base[0]['tokens']:>8d}{b:>9.1f}{base[0]['secs']:>7.1f}s |")
    print(f"| {'+ draft (N=%d)' % a.draft_tokens:24s}{r['tokens']:>8d}{s:>9.1f}{r['secs']:>7.1f}s |")
    print("+" + "-" * 50 + "+")
    print(f"| {BOLD}{'Speedup':24s}{s / b:>24.2f}x{RESET} |")
    print(f"| {'Draft tokens accepted':24s}{r['accepted']:>17d} / {r['tokens']:<5d} |")
    print(f"| {'Tokens per big-model pass':26s}{r['tokens'] / r['passes']:>23.2f} |")
    print(f"| {'Output identical':24s}{('yes' if same else 'NO (fp near-tie)'):>25s} |")
    print("+" + "-" * 50 + "+")
    if a.repeat > 1:
        print(f"{DIM}tok/s = median of {a.repeat} runs (mlx_lm generation tps, excludes prompt processing){RESET}")


if __name__ == "__main__":
    main()
