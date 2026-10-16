
### temp = 0.7  (median of 3 runs, max 256 new tokens)

Generation tok/s as reported by mlx_lm (speedup vs baseline in brackets)

| prompt | baseline | N=2 | N=3 | N=4 | N=6 | N=8 |
|---|---|---|---|---|---|---|
| code | 52.3 | 68.0 (1.30x) | 60.0 (1.15x) | 58.8 (1.12x) | 51.8 (0.99x) | 39.5 (0.75x) |
| rewrite | 52.7 | 79.7 (1.51x) | 79.2 (1.50x) | 86.5 (1.64x) | 88.4 (1.68x) | 86.3 (1.64x) |
| creative | 53.1 | 46.1 (0.87x) | 35.6 (0.67x) | 30.9 (0.58x) | 21.9 (0.41x) | 15.4 (0.29x) |
| math | 53.1 | 63.4 (1.19x) | 57.9 (1.09x) | 58.7 (1.11x) | 45.6 (0.86x) | 36.4 (0.68x) |
| code_edit | 52.6 | 69.4 (1.32x) | 69.7 (1.33x) | 67.5 (1.28x) | 62.0 (1.18x) | 55.2 (1.05x) |
| **geo-mean speedup** | 1.00x | **1.22x** | **1.11x** | **1.09x** | **0.93x** | **0.76x** |

End-to-end tok/s (wall clock incl. prompt prefill) speedup vs baseline

| prompt | N=2 | N=3 | N=4 | N=6 | N=8 |
|---|---|---|---|---|---|
| code | 1.28x | 1.15x | 1.12x | 1.00x | 0.77x |
| rewrite | 1.30x | 1.29x | 1.39x | 1.41x | 1.37x |
| creative | 0.88x | 0.69x | 0.60x | 0.43x | 0.30x |
| math | 1.17x | 1.07x | 1.09x | 0.86x | 0.69x |
| code_edit | 1.23x | 1.27x | 1.20x | 1.12x | 1.02x |

Acceptance: alpha_hat = per-token accept prob = accepted/(accepted+rejections); 'acc/drafted' = fraction of drafted tokens kept; tok/pass = tokens per target forward pass

| prompt | N=2 a_hat / kept / tok-per-pass | N=3 a_hat / kept / tok-per-pass | N=4 a_hat / kept / tok-per-pass | N=6 a_hat / kept / tok-per-pass | N=8 a_hat / kept / tok-per-pass |
|---|---|---|---|---|---|
| code | 0.84 / 0.76 / 2.52 | 0.80 / 0.64 / 2.90 | 0.83 / 0.63 / 3.52 | 0.81 / 0.51 / 4.02 | 0.81 / 0.42 / 4.36 |
| rewrite | 1.00 / 1.00 / 2.98 | 1.00 / 0.98 / 3.91 | 1.00 / 1.00 / 5.00 | 1.00 / 1.00 / 6.94 | 1.00 / 1.00 / 8.93 |
| creative | 0.44 / 0.32 / 1.64 | 0.41 / 0.22 / 1.66 | 0.42 / 0.18 / 1.70 | 0.44 / 0.13 / 1.77 | 0.41 / 0.09 / 1.69 |
| math | 0.78 / 0.68 / 2.35 | 0.76 / 0.58 / 2.73 | 0.81 / 0.60 / 3.38 | 0.80 / 0.47 / 3.80 | 0.78 / 0.36 / 3.90 |
| code_edit | 0.91 / 0.85 / 2.69 | 0.91 / 0.81 / 3.40 | 0.92 / 0.78 / 4.11 | 0.90 / 0.69 / 5.14 | 0.92 / 0.63 / 6.01 |

Cost model check: predicted speedup = E(a_hat, N) * t(1) / (N * t_draft + t_verify(N+1)) using measured pass times; vs measured gen-tps speedup

| prompt | N=2 pred / meas | N=3 pred / meas | N=4 pred / meas | N=6 pred / meas | N=8 pred / meas |
|---|---|---|---|---|---|
| code | 1.51 / 1.30 | 1.31 / 1.15 | 1.28 / 1.12 | 1.05 / 0.99 | 0.90 / 0.75 |
| rewrite | 1.79 / 1.51 | 1.79 / 1.50 | 1.78 / 1.64 | 1.78 / 1.68 | 1.79 / 1.64 |
| creative | 0.98 / 0.87 | 0.74 / 0.67 | 0.61 / 0.58 | 0.45 / 0.41 | 0.34 / 0.29 |
| math | 1.42 / 1.19 | 1.24 / 1.09 | 1.23 / 1.11 | 1.00 / 0.86 | 0.81 / 0.68 |
| code_edit | 1.63 / 1.32 | 1.56 / 1.33 | 1.51 / 1.28 | 1.34 / 1.18 | 1.31 / 1.05 |

peak memory (max over runs): 5.34 GB
