
### temp = 0.0  (median of 3 runs, max 256 new tokens)

Generation tok/s as reported by mlx_lm (speedup vs baseline in brackets)

| prompt | baseline | N=2 | N=3 | N=4 | N=6 | N=8 |
|---|---|---|---|---|---|---|
| code | 50.3 | 71.1 (1.41x) | 67.1 (1.33x) | 63.1 (1.25x) | 57.3 (1.14x) | 49.7 (0.99x) |
| rewrite | 49.7 | 77.4 (1.56x) | 82.3 (1.66x) | 86.1 (1.73x) | 88.1 (1.77x) | 84.2 (1.69x) |
| creative | 49.6 | 47.2 (0.95x) | 38.6 (0.78x) | 31.3 (0.63x) | 23.0 (0.46x) | 16.0 (0.32x) |
| math | 52.8 | 68.9 (1.31x) | 56.7 (1.07x) | 62.9 (1.19x) | 43.7 (0.83x) | 37.2 (0.70x) |
| code_edit | 53.2 | 70.2 (1.32x) | 73.3 (1.38x) | 72.1 (1.36x) | 66.2 (1.25x) | 60.8 (1.14x) |
| **geo-mean speedup** | 1.00x | **1.29x** | **1.21x** | **1.17x** | **0.99x** | **0.85x** |

End-to-end tok/s (wall clock incl. prompt prefill) speedup vs baseline

| prompt | N=2 | N=3 | N=4 | N=6 | N=8 |
|---|---|---|---|---|---|
| code | 1.39x | 1.32x | 1.24x | 1.13x | 0.99x |
| rewrite | 1.34x | 1.41x | 1.45x | 1.48x | 1.38x |
| creative | 0.99x | 0.82x | 0.67x | 0.50x | 0.35x |
| math | 1.27x | 1.07x | 1.18x | 0.84x | 0.72x |
| code_edit | 1.23x | 1.30x | 1.27x | 1.18x | 1.09x |

Acceptance: alpha_hat = per-token accept prob = accepted/(accepted+rejections); 'acc/drafted' = fraction of drafted tokens kept; tok/pass = tokens per target forward pass

| prompt | N=2 a_hat / kept / tok-per-pass | N=3 a_hat / kept / tok-per-pass | N=4 a_hat / kept / tok-per-pass | N=6 a_hat / kept / tok-per-pass | N=8 a_hat / kept / tok-per-pass |
|---|---|---|---|---|---|
| code | 0.87 / 0.81 / 2.61 | 0.85 / 0.72 / 3.16 | 0.87 / 0.68 / 3.71 | 0.86 / 0.60 / 4.57 | 0.86 / 0.54 / 5.33 |
| rewrite | 1.00 / 1.00 / 2.98 | 1.00 / 0.98 / 3.91 | 1.00 / 1.00 / 5.00 | 1.00 / 1.00 / 6.94 | 1.00 / 1.00 / 8.93 |
| creative | 0.47 / 0.35 / 1.70 | 0.47 / 0.27 / 1.80 | 0.46 / 0.20 / 1.82 | 0.46 / 0.14 / 1.84 | 0.44 / 0.10 / 1.78 |
| math | 0.84 / 0.78 / 2.56 | 0.80 / 0.63 / 2.88 | 0.85 / 0.68 / 3.71 | 0.78 / 0.43 / 3.61 | 0.79 / 0.38 / 4.06 |
| code_edit | 0.90 / 0.84 / 2.66 | 0.93 / 0.84 / 3.51 | 0.92 / 0.81 / 4.23 | 0.92 / 0.73 / 5.32 | 0.93 / 0.69 / 6.50 |

Cost model check: predicted speedup = E(a_hat, N) * t(1) / (N * t_draft + t_verify(N+1)) using measured pass times; vs measured gen-tps speedup

| prompt | N=2 pred / meas | N=3 pred / meas | N=4 pred / meas | N=6 pred / meas | N=8 pred / meas |
|---|---|---|---|---|---|
| code | 1.56 / 1.41 | 1.43 / 1.33 | 1.37 / 1.25 | 1.18 / 1.14 | 1.06 / 0.99 |
| rewrite | 1.79 / 1.56 | 1.79 / 1.66 | 1.78 / 1.73 | 1.78 / 1.77 | 1.79 / 1.69 |
| creative | 1.01 / 0.95 | 0.80 / 0.78 | 0.64 / 0.63 | 0.47 / 0.46 | 0.36 / 0.32 |
| math | 1.52 / 1.31 | 1.31 / 1.07 | 1.33 / 1.19 | 0.97 / 0.83 | 0.82 / 0.70 |
| code_edit | 1.62 / 1.32 | 1.61 / 1.38 | 1.52 / 1.36 | 1.40 / 1.25 | 1.37 / 1.14 |

Greedy token-identity vs baseline (first differing token index; '=' = identical, all reps)

| prompt | baseline reps agree | N=2 | N=3 | N=4 | N=6 | N=8 |
|---|---|---|---|---|---|---|
| code | yes | = | = | = | = | = |
| rewrite | yes | = | = | = | = | = |
| creative | yes | 16 | 16 | 16 | 16 | 67 |
| math | yes | 83 | 220 | 84 | 84 | 84 |
| code_edit | yes | = | = | = | = | 3 |

peak memory (max over runs): 5.34 GB
