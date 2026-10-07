# Generated experiment tables

Reproduce with `.venv/bin/python benchmarks/latent/summarize_round2.py`. Missing runs are omitted, never imputed.

## Same-GPU matched policy comparison

Negative change means the fork is faster. Intervals bootstrap within-session repeats, not hardware, prompt, or day variability.

| Model | Policy | B | Variant | Fork s | Qwen fork s | Stock s | Change [95% interval] | Overhead vs stock | Exact traces |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.6B | token | 1 | paired-final | 0.4521 | 0.4640 | 0.4568 | -2.5% [-3.1, -2.0] | -1.0% | 1/1 |
| 0.6B | token | 8 | paired-final | 0.5216 | 0.5280 | 0.5252 | -1.2% [-1.6, -0.7] | -0.7% | 8/8 |
| 0.6B | token | 32 | paired-final | 0.6761 | 0.6784 | 0.6767 | -0.4% [-0.5, +0.1] | -0.1% | 32/32 |
| 0.6B | swi512 | 1 | paired-final | 0.4840 | 0.5107 | 0.4568 | -5.2% [-5.4, -4.9] | +6.0% | 1/1 |
| 0.6B | swi512 | 8 | paired-final | 0.5538 | 0.5816 | 0.5252 | -4.8% [-5.3, -4.3] | +5.4% | 8/8 |
| 0.6B | swi512 | 32 | paired-final | 0.7416 | 0.7746 | 0.6767 | -4.3% [-4.4, -3.0] | +9.6% | 32/32 |
| 0.6B | swi8 | 1 | paired-final | 0.5034 | 0.5856 | 0.4568 | -14.0% [-14.2, -13.5] | +10.2% | 1/1 |
| 0.6B | swi8 | 8 | paired-final | 0.6040 | 0.7365 | 0.5252 | -18.0% [-18.3, -17.9] | +15.0% | 7/8 |
| 0.6B | swi8 | 32 | paired-final | 0.8294 | 1.0608 | 0.6767 | -21.8% [-21.9, -21.7] | +22.6% | 22/32 |
| 0.6B | token | 1 | paired-exact | 0.4555 | 0.4640 | 0.4568 | -1.8% [-2.1, -1.3] | -0.3% | 1/1 |
| 0.6B | token | 8 | paired-exact | 0.5241 | 0.5280 | 0.5252 | -0.7% [-0.9, -0.5] | -0.2% | 8/8 |
| 0.6B | token | 32 | paired-exact | 0.6805 | 0.6784 | 0.6767 | +0.3% [-0.2, +0.4] | +0.6% | 32/32 |
| 0.6B | swi512 | 1 | paired-exact | 0.4843 | 0.5107 | 0.4568 | -5.2% [-5.8, -4.7] | +6.0% | 1/1 |
| 0.6B | swi512 | 8 | paired-exact | 0.5688 | 0.5816 | 0.5252 | -2.2% [-3.0, -1.9] | +8.3% | 8/8 |
| 0.6B | swi512 | 32 | paired-exact | 0.7964 | 0.7746 | 0.6767 | +2.8% [+2.3, +3.3] | +17.7% | 25/32 |
| 0.6B | swi8 | 1 | paired-exact | 0.5109 | 0.5856 | 0.4568 | -12.8% [-12.9, -12.5] | +11.8% | 1/1 |
| 0.6B | swi8 | 8 | paired-exact | 0.6183 | 0.7365 | 0.5252 | -16.1% [-16.4, -15.9] | +17.7% | 8/8 |
| 0.6B | swi8 | 32 | paired-exact | 0.8807 | 1.0608 | 0.6767 | -17.0% [-17.3, -16.5] | +30.1% | 8/32 |
| 1.7B | token | 1 | paired-final | 0.8872 | 0.9012 | 0.8889 | -1.6% [-1.7, -1.2] | -0.2% | 1/1 |
| 1.7B | token | 8 | paired-final | 0.9654 | 0.9713 | 0.9629 | -0.6% [-0.8, -0.4] | +0.3% | 8/8 |
| 1.7B | token | 32 | paired-final | 1.1283 | 1.1287 | 1.1309 | -0.0% [-0.3, +0.2] | -0.2% | 32/32 |
| 1.7B | swi512 | 1 | paired-final | 0.9151 | 0.9506 | 0.8889 | -3.7% [-4.0, -3.5] | +3.0% | 1/1 |
| 1.7B | swi512 | 8 | paired-final | 1.0018 | 1.0326 | 0.9629 | -3.0% [-3.1, -2.8] | +4.0% | 8/8 |
| 1.7B | swi512 | 32 | paired-final | 1.2015 | 1.2368 | 1.1309 | -2.9% [-3.1, -2.7] | +6.2% | 32/32 |
| 1.7B | swi8 | 1 | paired-final | 0.9462 | 1.0282 | 0.8889 | -8.0% [-8.2, -7.8] | +6.4% | 1/1 |
| 1.7B | swi8 | 8 | paired-final | 1.1279 | 1.3883 | 0.9629 | -18.8% [-18.9, -18.6] | +17.1% | 7/8 |
| 1.7B | swi8 | 32 | paired-final | 1.3674 | 1.7041 | 1.1309 | -19.8% [-19.9, -19.5] | +20.9% | 32/32 |
| 1.7B | token | 1 | paired-exact | 0.8913 | 0.9012 | 0.8889 | -1.1% [-1.3, -0.7] | +0.3% | 1/1 |
| 1.7B | token | 8 | paired-exact | 0.9674 | 0.9713 | 0.9629 | -0.4% [-0.7, +0.1] | +0.5% | 8/8 |
| 1.7B | token | 32 | paired-exact | 1.1260 | 1.1287 | 1.1309 | -0.2% [-0.4, -0.1] | -0.4% | 32/32 |
| 1.7B | swi512 | 1 | paired-exact | 0.9191 | 0.9506 | 0.8889 | -3.3% [-3.5, -3.2] | +3.4% | 1/1 |
| 1.7B | swi512 | 8 | paired-exact | 1.0121 | 1.0326 | 0.9629 | -2.0% [-2.2, -1.8] | +5.1% | 8/8 |
| 1.7B | swi512 | 32 | paired-exact | 1.2434 | 1.2368 | 1.1309 | +0.5% [+0.3, +0.7] | +10.0% | 32/32 |
| 1.7B | swi8 | 1 | paired-exact | 0.9502 | 1.0282 | 0.8889 | -7.6% [-7.9, -7.4] | +6.9% | 1/1 |
| 1.7B | swi8 | 8 | paired-exact | 1.1417 | 1.3883 | 0.9629 | -17.8% [-17.9, -17.6] | +18.6% | 7/8 |
| 1.7B | swi8 | 32 | paired-exact | 1.4132 | 1.7041 | 1.1309 | -17.1% [-17.2, -16.7] | +25.0% | 32/32 |
| 4B | token | 1 | paired-final | 1.8226 | 1.8296 | 1.8216 | -0.4% [-0.5, -0.1] | +0.1% | 1/1 |
| 4B | token | 8 | paired-final | 1.8880 | 1.8902 | 1.8842 | -0.1% [-0.2, -0.0] | +0.2% | 7/8 |
| 4B | token | 32 | paired-final | 2.1281 | 2.1304 | 2.1203 | -0.1% [-0.2, +0.0] | +0.4% | 23/32 |
| 4B | swi512 | 1 | paired-final | 1.8552 | 1.8830 | 1.8216 | -1.5% [-1.5, -1.4] | +1.8% | 1/1 |
| 4B | swi512 | 8 | paired-final | 1.9269 | 1.9586 | 1.8842 | -1.6% [-1.7, -1.5] | +2.3% | 7/8 |
| 4B | swi512 | 32 | paired-final | 2.2028 | 2.2378 | 2.1203 | -1.6% [-1.7, -1.2] | +3.9% | 25/32 |
| 4B | swi8 | 1 | paired-final | 1.9039 | 2.0124 | 1.8216 | -5.4% [-5.5, -5.2] | +4.5% | 1/1 |
| 4B | swi8 | 8 | paired-final | 2.0214 | 2.3626 | 1.8842 | -14.4% [-14.7, -14.3] | +7.3% | 3/8 |
| 4B | swi8 | 32 | paired-final | 2.3879 | 2.7446 | 2.1203 | -13.0% [-13.1, -12.8] | +12.6% | 11/32 |
| 4B | token | 1 | paired-exact | 1.8285 | 1.8296 | 1.8216 | -0.1% [-0.1, +0.1] | +0.4% | 1/1 |
| 4B | token | 8 | paired-exact | 1.8931 | 1.8902 | 1.8842 | +0.2% [+0.0, +0.2] | +0.5% | 7/8 |
| 4B | token | 32 | paired-exact | 2.1314 | 2.1304 | 2.1203 | +0.0% [-0.0, +0.2] | +0.5% | 21/32 |
| 4B | swi512 | 1 | paired-exact | 1.8654 | 1.8830 | 1.8216 | -0.9% [-1.2, -0.6] | +2.4% | 1/1 |
| 4B | swi512 | 8 | paired-exact | 1.9442 | 1.9586 | 1.8842 | -0.7% [-0.8, -0.4] | +3.2% | 5/8 |
| 4B | swi512 | 32 | paired-exact | 2.2483 | 2.2378 | 2.1203 | +0.5% [+0.3, +0.6] | +6.0% | 26/32 |
| 4B | swi8 | 1 | paired-exact | 1.9086 | 2.0124 | 1.8216 | -5.2% [-5.2, -5.0] | +4.8% | 1/1 |
| 4B | swi8 | 8 | paired-exact | 2.0572 | 2.3626 | 1.8842 | -12.9% [-13.2, -12.9] | +9.2% | 3/8 |
| 4B | swi8 | 32 | paired-exact | 2.4329 | 2.7446 | 2.1203 | -11.4% [-11.4, -11.3] | +14.7% | 15/32 |
| 8B | token | 1 | paired-final | 3.1538 | 3.1570 | 3.1503 | -0.1% [-0.1, -0.0] | +0.1% | 1/1 |
| 8B | token | 8 | paired-final | 3.2530 | 3.2624 | 3.2457 | -0.3% [-0.4, +0.1] | +0.2% | 8/8 |
| 8B | token | 32 | paired-final | 3.6689 | 3.6668 | 3.6654 | +0.1% [-0.1, +0.1] | +0.1% | 31/32 |
| 8B | swi512 | 1 | paired-final | 3.1893 | 3.2187 | 3.1503 | -0.9% [-0.9, -0.9] | +1.2% | 1/1 |
| 8B | swi512 | 8 | paired-final | 3.2971 | 3.3267 | 3.2457 | -0.9% [-0.9, -0.8] | +1.6% | 8/8 |
| 8B | swi512 | 32 | paired-final | 3.7492 | 3.7859 | 3.6654 | -1.0% [-1.0, -0.9] | +2.3% | 32/32 |
| 8B | swi8 | 1 | paired-final | 3.3921 | 3.7276 | 3.1503 | -9.0% [-9.1, -9.0] | +7.7% | 0/1 |
| 8B | swi8 | 8 | paired-final | 3.5589 | 4.0014 | 3.2457 | -11.1% [-11.1, -11.0] | +9.6% | 8/8 |
| 8B | swi8 | 32 | paired-final | 4.1131 | 4.7327 | 3.6654 | -13.1% [-13.1, -13.0] | +12.2% | 32/32 |
| 8B | token | 1 | paired-exact | 3.1465 | 3.1570 | 3.1503 | -0.3% [-0.4, -0.2] | -0.1% | 1/1 |
| 8B | token | 8 | paired-exact | 3.2441 | 3.2624 | 3.2457 | -0.6% [-0.7, -0.2] | -0.0% | 8/8 |
| 8B | token | 32 | paired-exact | 3.6630 | 3.6668 | 3.6654 | -0.1% [-0.3, -0.0] | -0.1% | 32/32 |
| 8B | swi512 | 1 | paired-exact | 3.1844 | 3.2187 | 3.1503 | -1.1% [-1.1, -1.0] | +1.1% | 1/1 |
| 8B | swi512 | 8 | paired-exact | 3.3018 | 3.3267 | 3.2457 | -0.7% [-0.9, -0.7] | +1.7% | 8/8 |
| 8B | swi512 | 32 | paired-exact | 3.7884 | 3.7859 | 3.6654 | +0.1% [-0.0, +0.1] | +3.4% | 32/32 |
| 8B | swi8 | 1 | paired-exact | 3.3850 | 3.7276 | 3.1503 | -9.2% [-9.2, -9.2] | +7.5% | 0/1 |
| 8B | swi8 | 8 | paired-exact | 3.5632 | 4.0014 | 3.2457 | -11.0% [-11.0, -10.9] | +9.8% | 8/8 |
| 8B | swi8 | 32 | paired-exact | 4.1548 | 4.7327 | 3.6654 | -12.2% [-12.2, -12.2] | +13.4% | 32/32 |

## Generic programs: previous release versus combined profile

| Model | Policy | B | Previous s | Combined s | Change | Overhead vs stock |
| --- | --- | --- | --- | --- | --- | --- |
| 0.6B | token | 1 | 0.4533 | 0.4541 | +0.2% | +0.6% |
| 0.6B | token | 8 | 0.5194 | 0.5228 | +0.7% | +0.6% |
| 0.6B | token | 32 | 0.6758 | 0.6776 | +0.3% | +0.4% |
| 0.6B | soft | 1 | 0.4899 | 0.4924 | +0.5% | +9.1% |
| 0.6B | soft | 8 | 0.5611 | 0.5639 | +0.5% | +8.5% |
| 0.6B | soft | 32 | 0.7374 | 0.7346 | -0.4% | +8.8% |
| 0.6B | entropy | 1 | 0.4636 | 0.4674 | +0.8% | +3.6% |
| 0.6B | entropy | 8 | 0.5373 | 0.5403 | +0.6% | +4.0% |
| 0.6B | entropy | 32 | 0.7307 | 0.7167 | -1.9% | +6.2% |
| 0.6B | mixed | 1 | 0.4526 | 0.4548 | +0.5% | +0.8% |
| 0.6B | mixed | 8 | 0.5697 | 0.5788 | +1.6% | +11.4% |
| 0.6B | mixed | 32 | 0.7590 | 0.7687 | +1.3% | +13.9% |
| 1.7B | token | 1 | 0.8924 | 0.8912 | -0.1% | -0.2% |
| 1.7B | token | 8 | 0.9707 | 0.9744 | +0.4% | +0.5% |
| 1.7B | token | 32 | 1.1423 | 1.1447 | +0.2% | +0.3% |
| 1.7B | soft | 1 | 0.9581 | 0.9619 | +0.4% | +7.7% |
| 1.7B | soft | 8 | 1.0422 | 1.0444 | +0.2% | +7.8% |
| 1.7B | soft | 32 | 1.2313 | 1.2285 | -0.2% | +7.7% |
| 1.7B | entropy | 1 | 0.9069 | 0.9051 | -0.2% | +1.3% |
| 1.7B | entropy | 8 | 0.9911 | 0.9902 | -0.1% | +2.2% |
| 1.7B | entropy | 32 | 1.1968 | 1.1856 | -0.9% | +3.9% |
| 1.7B | mixed | 1 | 0.8952 | 0.8973 | +0.2% | +0.4% |
| 1.7B | mixed | 8 | 1.0552 | 1.0624 | +0.7% | +9.6% |
| 1.7B | mixed | 32 | 1.2557 | 1.2710 | +1.2% | +11.4% |
| 4B | token | 1 | 1.8395 | 1.8342 | -0.3% | -0.2% |
| 4B | token | 8 | 1.9007 | 1.9015 | +0.0% | +0.2% |
| 4B | token | 32 | 2.1490 | 2.1571 | +0.4% | +0.5% |
| 4B | soft | 1 | 1.9268 | 1.9201 | -0.3% | +4.5% |
| 4B | soft | 8 | 1.9898 | 1.9920 | +0.1% | +4.9% |
| 4B | soft | 32 | 2.2546 | 2.2472 | -0.3% | +4.6% |
| 4B | entropy | 1 | 1.8609 | 1.8547 | -0.3% | +1.0% |
| 4B | entropy | 8 | 1.9230 | 1.9204 | -0.1% | +1.2% |
| 4B | entropy | 32 | 2.1986 | 2.1944 | -0.2% | +2.2% |
| 4B | mixed | 1 | 1.8479 | 1.8443 | -0.2% | +0.4% |
| 4B | mixed | 8 | 2.0009 | 2.0028 | +0.1% | +5.5% |
| 4B | mixed | 32 | 2.2770 | 2.2839 | +0.3% | +6.4% |
| 8B | token | 1 | 3.1664 | 3.1651 | -0.0% | +0.0% |
| 8B | token | 8 | 3.2797 | 3.2717 | -0.2% | +0.2% |
| 8B | token | 32 | 3.7278 | 3.7106 | -0.5% | +0.2% |
| 8B | soft | 1 | 3.3022 | 3.2956 | -0.2% | +4.2% |
| 8B | soft | 8 | 3.4130 | 3.4031 | -0.3% | +4.2% |
| 8B | soft | 32 | 3.8823 | 3.8575 | -0.6% | +4.2% |
| 8B | entropy | 1 | 3.1942 | 3.1877 | -0.2% | +0.8% |
| 8B | entropy | 8 | 3.3031 | 3.2950 | -0.2% | +0.9% |
| 8B | entropy | 32 | 3.7712 | 3.7509 | -0.5% | +1.3% |
| 8B | mixed | 1 | 3.1804 | 3.1757 | -0.1% | +0.4% |
| 8B | mixed | 8 | 3.4202 | 3.4201 | -0.0% | +4.7% |
| 8B | mixed | 32 | 3.8883 | 3.8911 | +0.1% | +5.1% |

## Natural-termination quality

Same seeded 128-question GSM8K subset. Correct means the final number after a generated end-think matches the gold integer; unfinished thinking fails. Timings include different amounts of generated work and are not fixed-work speedups.

| File | Model | Policy | Seed | Correct/128 | Length limited | Positions | Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adaptive-isolated-base.jsonl.gz | 0.6B | adaptive | 20261006 | 88 | 32 | 163117 | 61.83 |
| adaptive-isolated-gated.jsonl.gz | 0.6B | adaptive | 20261006 | 90 | 32 | 159712 | 59.29 |
| adaptive-strict-base.jsonl.gz | 0.6B | adaptive | 20261006 | 87 | 35 | 164750 | 111.08 |
| adaptive-strict-gated.jsonl.gz | 0.6B | adaptive | 20261006 | 87 | 35 | 164750 | 110.61 |
| exact-compact-quality.jsonl.gz | 0.6B | swi512 | 20261006 | 89 | 32 | 165096 | 63.79 |
| exact-compact-quality.jsonl.gz | 0.6B | dense32 | 20261006 | 87 | 35 | 161195 | 65.09 |
| exact-compact-valid-quality.jsonl.gz | 0.6B | swi512 | 20261006 | 92 | 30 | 159738 | 60.96 |
| exact-compact-valid-quality.jsonl.gz | 0.6B | dense32 | 20261006 | 87 | 35 | 161195 | 65.15 |
| exact-quality1.jsonl.gz | 1.7B | swi512 | 20261006 | 89 | 36 | 195855 | 100.54 |
| exact-quality1.jsonl.gz | 1.7B | dense32 | 20261006 | 80 | 45 | 195424 | 110.15 |
| exact-quality2.jsonl.gz | 4B | swi512 | 20261006 | 86 | 39 | 196003 | 165.08 |
| exact-quality2.jsonl.gz | 4B | dense32 | 20261006 | 79 | 46 | 193463 | 176.28 |
| exact-quality3.jsonl.gz | 8B | swi512 | 20261006 | 78 | 48 | 206646 | 258.52 |
| exact-quality3.jsonl.gz | 8B | dense32 | 20261006 | 83 | 43 | 200672 | 269.35 |
| fork-quality0.jsonl.gz | 0.6B | token | 20261006 | 92 | 27 | 160243 | 54.81 |
| fork-quality0.jsonl.gz | 0.6B | swi512 | 20261006 | 81 | 38 | 164617 | 61.08 |
| fork-quality0.jsonl.gz | 0.6B | dense32 | 20261006 | 90 | 31 | 161052 | 63.17 |
| fork-quality0.jsonl.gz | 0.6B | bf16 | 20261006 | 83 | 35 | 160533 | 60.55 |
| fork-quality0.jsonl.gz | 0.6B | topk | 20261006 | 89 | 28 | 159044 | 59.44 |
| fork-quality0.jsonl.gz | 0.6B | adaptive | 20261006 | 90 | 32 | 159712 | 60.21 |
| fork-quality0.jsonl.gz | 0.6B | lowrank | 20261006 | 1 | 126 | 259565 | 99.22 |
| fork-quality1.jsonl.gz | 1.7B | token | 20261006 | 79 | 47 | 198948 | 93.60 |
| fork-quality1.jsonl.gz | 1.7B | swi512 | 20261006 | 89 | 36 | 196090 | 97.95 |
| fork-quality1.jsonl.gz | 1.7B | dense32 | 20261006 | 78 | 49 | 200345 | 110.15 |
| fork-quality1.jsonl.gz | 1.7B | bf16 | 20261006 | 79 | 40 | 196018 | 102.83 |
| fork-quality1.jsonl.gz | 1.7B | topk | 20261006 | 83 | 43 | 198721 | 101.35 |
| fork-quality1.jsonl.gz | 1.7B | adaptive | 20261006 | 76 | 45 | 194816 | 99.43 |
| fork-quality1.jsonl.gz | 1.7B | lowrank | 20261006 | 46 | 59 | 143279 | 74.92 |
| fork-quality2.jsonl.gz | 4B | token | 20261006 | 83 | 45 | 197280 | 160.33 |
| fork-quality2.jsonl.gz | 4B | swi512 | 20261006 | 84 | 42 | 197836 | 168.16 |
| fork-quality2.jsonl.gz | 4B | dense32 | 20261006 | 80 | 45 | 194171 | 173.56 |
| fork-quality2.jsonl.gz | 4B | bf16 | 20261006 | 86 | 39 | 192972 | 168.54 |
| fork-quality2.jsonl.gz | 4B | topk | 20261006 | 79 | 46 | 194999 | 168.24 |
| fork-quality2.jsonl.gz | 4B | adaptive | 20261006 | 81 | 44 | 195026 | 167.45 |
| fork-quality2.jsonl.gz | 4B | lowrank | 20261006 | 21 | 33 | 147180 | 123.91 |
| fork-quality3.jsonl.gz | 8B | token | 20261006 | 81 | 43 | 200264 | 242.01 |
| fork-quality3.jsonl.gz | 8B | swi512 | 20261006 | 78 | 49 | 206144 | 259.31 |
| fork-quality3.jsonl.gz | 8B | dense32 | 20261006 | 82 | 44 | 201865 | 266.06 |
| fork-quality3.jsonl.gz | 8B | bf16 | 20261006 | 79 | 50 | 201113 | 256.48 |
| fork-quality3.jsonl.gz | 8B | topk | 20261006 | 76 | 50 | 204002 | 254.13 |
| fork-quality3.jsonl.gz | 8B | adaptive | 20261006 | 78 | 50 | 204887 | 254.79 |
| fork-quality3.jsonl.gz | 8B | lowrank | 20261006 | 6 | 121 | 255208 | 295.67 |
| gated-quality0.jsonl.gz | 0.6B | topk | 20261006 | 89 | 28 | 159044 | 58.86 |
| gated-quality0.jsonl.gz | 0.6B | adaptive | 20261006 | 85 | 36 | 165828 | 62.15 |
| gated-quality0.jsonl.gz | 0.6B | lowrank | 20261006 | 1 | 126 | 259565 | 98.71 |
| qwen-quality0.jsonl.gz | 0.6B | token | 20261006 | 52 | 75 | 115602 | 29.91 |
| qwen-quality0.jsonl.gz | 0.6B | swi8 | 20261006 | 1 | 26 | 43124 | 19.42 |
| qwen-quality1.jsonl.gz | 1.7B | token | 20261006 | 24 | 102 | 126202 | 45.63 |
| qwen-quality1.jsonl.gz | 1.7B | swi8 | 20261006 | 6 | 12 | 27489 | 20.71 |
| qwen-quality2.jsonl.gz | 4B | token | 20261006 | 30 | 98 | 125190 | 79.15 |
| qwen-quality2.jsonl.gz | 4B | swi8 | 20261006 | 17 | 23 | 39964 | 43.57 |
| qwen-quality3.jsonl.gz | 8B | token | 20261006 | 15 | 115 | 128692 | 126.53 |
| qwen-quality3.jsonl.gz | 8B | swi8 | 20261006 | 21 | 24 | 41366 | 75.17 |
| qwen-sampled0.jsonl.gz | 0.6B | token | 20261006 | 92 | 27 | 160243 | 54.66 |
| qwen-sampled0.jsonl.gz | 0.6B | swi512 | 20261006 | 92 | 30 | 159738 | 61.33 |
| qwen-sampled0.jsonl.gz | 0.6B | dense32 | 20261006 | 87 | 35 | 161195 | 71.93 |
| qwen-sampled1.jsonl.gz | 1.7B | token | 20261006 | 79 | 47 | 198948 | 93.55 |
| qwen-sampled1.jsonl.gz | 1.7B | swi512 | 20261006 | 79 | 49 | 199516 | 104.02 |
| qwen-sampled1.jsonl.gz | 1.7B | dense32 | 20261006 | 80 | 45 | 195424 | 125.06 |
| qwen-sampled2.jsonl.gz | 4B | token | 20261006 | 81 | 47 | 198475 | 158.07 |
| qwen-sampled2.jsonl.gz | 4B | swi512 | 20261006 | 81 | 46 | 198615 | 167.10 |
| qwen-sampled2.jsonl.gz | 4B | dense32 | 20261006 | 79 | 47 | 197205 | 197.34 |
| qwen-sampled3.jsonl.gz | 8B | token | 20261006 | 75 | 49 | 203184 | 242.96 |
| qwen-sampled3.jsonl.gz | 8B | swi512 | 20261006 | 78 | 47 | 200578 | 254.80 |
| qwen-sampled3.jsonl.gz | 8B | dense32 | 20261006 | 82 | 46 | 201370 | 308.49 |

## Paired quality differences

Question bootstrap, one seed; no multiplicity correction or equivalence claim. Approximate variants compare to the fork's FP32 window-32 policy.

| Model | Policy | Reference | Difference pp [95% interval] | Gains/losses | Exact/128 |
| --- | --- | --- | --- | --- | --- |
| 0.6B | token | qwen:token | +0.0 [+0.0, +0.0] | 0/0 | 128 |
| 0.6B | swi512 | qwen:swi512 | -8.6 [-15.6, -1.6] | 6/17 | 22 |
| 0.6B | dense32 | qwen:dense32 | +2.3 [-4.7, +9.4] | 12/9 | 0 |
| 0.6B | bf16 | fork:dense32 | -5.5 [-12.5, +1.6] | 8/15 | 1 |
| 0.6B | topk | fork:dense32 | -0.8 [-7.8, +6.2] | 10/11 | 0 |
| 0.6B | adaptive | fork:dense32 | +0.0 [-7.0, +7.0] | 10/10 | 0 |
| 0.6B | lowrank | fork:dense32 | -69.5 [-77.3, -61.7] | 0/89 | 0 |
| 1.7B | token | qwen:token | +0.0 [+0.0, +0.0] | 0/0 | 127 |
| 1.7B | swi512 | qwen:swi512 | +7.8 [+2.3, +13.3] | 12/2 | 0 |
| 1.7B | dense32 | qwen:dense32 | -1.6 [-8.6, +5.5] | 9/11 | 0 |
| 1.7B | bf16 | fork:dense32 | +0.8 [-7.0, +8.6] | 14/13 | 1 |
| 1.7B | topk | fork:dense32 | +3.9 [-3.9, +11.7] | 15/10 | 0 |
| 1.7B | adaptive | fork:dense32 | -1.6 [-9.4, +6.2] | 11/13 | 0 |
| 1.7B | lowrank | fork:dense32 | -25.0 [-36.7, -13.3] | 17/49 | 0 |
| 4B | token | qwen:token | +1.6 [-5.5, +8.6] | 11/9 | 0 |
| 4B | swi512 | qwen:swi512 | +2.3 [-3.9, +8.6] | 10/7 | 1 |
| 4B | dense32 | qwen:dense32 | +0.8 [-6.2, +8.6] | 12/11 | 0 |
| 4B | bf16 | fork:dense32 | +4.7 [-1.6, +10.9] | 12/6 | 1 |
| 4B | topk | fork:dense32 | -0.8 [-7.8, +7.0] | 11/12 | 1 |
| 4B | adaptive | fork:dense32 | +0.8 [-5.5, +6.2] | 8/7 | 1 |
| 4B | lowrank | fork:dense32 | -46.1 [-55.5, -36.7] | 2/61 | 0 |
| 8B | token | qwen:token | +4.7 [-1.6, +10.9] | 11/5 | 1 |
| 8B | swi512 | qwen:swi512 | +0.0 [-7.0, +7.0] | 11/11 | 0 |
| 8B | dense32 | qwen:dense32 | +0.0 [-6.2, +6.2] | 8/8 | 0 |
| 8B | bf16 | fork:dense32 | -2.3 [-8.6, +3.9] | 7/10 | 0 |
| 8B | topk | fork:dense32 | -4.7 [-11.7, +1.6] | 7/13 | 1 |
| 8B | adaptive | fork:dense32 | -3.1 [-9.4, +3.1] | 7/11 | 0 |
| 8B | lowrank | fork:dense32 | -59.4 [-68.0, -50.8] | 0/76 | 0 |

## Swi ablations

| Model | Variant | Policy | B | Median seconds |
| --- | --- | --- | --- | --- |
| 0.6B | core | swi512 | 1 | 0.4983 |
| 0.6B | core | swi512 | 32 | 0.7982 |
| 0.6B | core | swi8 | 1 | 0.5349 |
| 0.6B | core | swi8 | 32 | 0.9070 |
| 0.6B | async | swi512 | 1 | 0.4980 |
| 0.6B | async | swi512 | 32 | 0.7999 |
| 0.6B | async | swi8 | 1 | 0.5353 |
| 0.6B | async | swi8 | 32 | 0.9083 |
| 0.6B | tail | swi512 | 1 | 0.4983 |
| 0.6B | tail | swi512 | 32 | 0.7896 |
| 0.6B | tail | swi8 | 1 | 0.5336 |
| 0.6B | tail | swi8 | 32 | 0.8985 |
| 0.6B | head | swi512 | 1 | 0.5011 |
| 0.6B | head | swi512 | 32 | 0.7767 |
| 0.6B | head | swi8 | 1 | 0.5339 |
| 0.6B | head | swi8 | 32 | 0.8880 |
| 0.6B | state | swi512 | 1 | 0.4889 |
| 0.6B | state | swi512 | 32 | 0.7869 |
| 0.6B | state | swi8 | 1 | 0.5204 |
| 0.6B | state | swi8 | 32 | 0.8944 |
| 0.6B | combo | swi512 | 1 | 0.4823 |
| 0.6B | combo | swi512 | 32 | 0.7715 |
| 0.6B | combo | swi8 | 1 | 0.5148 |
| 0.6B | combo | swi8 | 32 | 0.8716 |
| 8B | core | swi512 | 1 | 3.2549 |
| 8B | core | swi512 | 32 | 3.8825 |
| 8B | core | swi8 | 1 | 3.4583 |
| 8B | core | swi8 | 32 | 4.2501 |
| 8B | async | swi512 | 1 | 3.2506 |
| 8B | async | swi512 | 32 | 3.8797 |
| 8B | async | swi8 | 1 | 3.4689 |
| 8B | async | swi8 | 32 | 4.2605 |
| 8B | tail | swi512 | 1 | 3.2473 |
| 8B | tail | swi512 | 32 | 3.8562 |
| 8B | tail | swi8 | 1 | 3.4594 |
| 8B | tail | swi8 | 32 | 4.2440 |
| 8B | head | swi512 | 1 | 3.2477 |
| 8B | head | swi512 | 32 | 3.8537 |
| 8B | head | swi8 | 1 | 3.4640 |
| 8B | head | swi8 | 32 | 4.2377 |
| 8B | state | swi512 | 1 | 3.2421 |
| 8B | state | swi512 | 32 | 3.8549 |
| 8B | state | swi8 | 1 | 3.4495 |
| 8B | state | swi8 | 32 | 4.2474 |
| 8B | combo | swi512 | 1 | 3.2445 |
| 8B | combo | swi512 | 32 | 3.8569 |
| 8B | combo | swi8 | 1 | 3.4620 |
| 8B | combo | swi8 | 32 | 4.2383 |

## Approximation fixed-work timings before gating

| Model | Variant | Policy | B | Median seconds |
| --- | --- | --- | --- | --- |
| 0.6B | candidate | dense32 | 1 | 0.4877 |
| 0.6B | candidate | dense32 | 32 | 0.7830 |
| 0.6B | candidate | bf16 | 1 | 0.4883 |
| 0.6B | candidate | bf16 | 32 | 0.7708 |
| 0.6B | candidate | topk | 1 | 0.4900 |
| 0.6B | candidate | topk | 32 | 0.7934 |
| 0.6B | candidate | adaptive | 1 | 0.4917 |
| 0.6B | candidate | adaptive | 32 | 0.7912 |
| 0.6B | candidate | lowrank | 1 | 0.4908 |
| 0.6B | candidate | lowrank | 32 | 0.7678 |
| 1.7B | candidate | dense32 | 1 | 0.9428 |
| 1.7B | candidate | dense32 | 32 | 1.2475 |
| 1.7B | candidate | bf16 | 1 | 0.9246 |
| 1.7B | candidate | bf16 | 32 | 1.2340 |
| 1.7B | candidate | topk | 1 | 0.9260 |
| 1.7B | candidate | topk | 32 | 1.2547 |
| 1.7B | candidate | adaptive | 1 | 0.9288 |
| 1.7B | candidate | adaptive | 32 | 1.2592 |
| 1.7B | candidate | lowrank | 1 | 0.9279 |
| 1.7B | candidate | lowrank | 32 | 1.2236 |
| 4B | candidate | dense32 | 1 | 1.9238 |
| 4B | candidate | dense32 | 32 | 2.3051 |
| 4B | candidate | bf16 | 1 | 1.9223 |
| 4B | candidate | bf16 | 32 | 2.2927 |
| 4B | candidate | topk | 1 | 1.9203 |
| 4B | candidate | topk | 32 | 2.3048 |
| 4B | candidate | adaptive | 1 | 1.9213 |
| 4B | candidate | adaptive | 32 | 2.3074 |
| 4B | candidate | lowrank | 1 | 1.9178 |
| 4B | candidate | lowrank | 32 | 2.2686 |
| 8B | candidate | dense32 | 1 | 3.2721 |
| 8B | candidate | dense32 | 32 | 3.9243 |
| 8B | candidate | bf16 | 1 | 3.2548 |
| 8B | candidate | bf16 | 32 | 3.8896 |
| 8B | candidate | topk | 1 | 3.2435 |
| 8B | candidate | topk | 32 | 3.8998 |
| 8B | candidate | adaptive | 1 | 3.2441 |
| 8B | candidate | adaptive | 32 | 3.8971 |
| 8B | candidate | lowrank | 1 | 3.2404 |
| 8B | candidate | lowrank | 32 | 3.8402 |

## Approximation fixed-work timings with gating

| Model | Variant | Policy | B | Median seconds |
| --- | --- | --- | --- | --- |
| 0.6B | gated-approximations | dense32 | 1 | 0.4865 |
| 0.6B | gated-approximations | dense32 | 32 | 0.7749 |
| 0.6B | gated-approximations | bf16 | 1 | 0.4808 |
| 0.6B | gated-approximations | bf16 | 32 | 0.7730 |
| 0.6B | gated-approximations | topk | 1 | 0.4840 |
| 0.6B | gated-approximations | topk | 32 | 0.7632 |
| 0.6B | gated-approximations | adaptive | 1 | 0.4815 |
| 0.6B | gated-approximations | adaptive | 32 | 0.7624 |
| 0.6B | gated-approximations | lowrank | 1 | 0.4829 |
| 0.6B | gated-approximations | lowrank | 32 | 0.7664 |
| 1.7B | gated-approximations | dense32 | 1 | 0.9436 |
| 1.7B | gated-approximations | dense32 | 32 | 1.2500 |
| 1.7B | gated-approximations | bf16 | 1 | 0.9267 |
| 1.7B | gated-approximations | bf16 | 32 | 1.2330 |
| 1.7B | gated-approximations | topk | 1 | 0.9194 |
| 1.7B | gated-approximations | topk | 32 | 1.2147 |
| 1.7B | gated-approximations | adaptive | 1 | 0.9214 |
| 1.7B | gated-approximations | adaptive | 32 | 1.2147 |
| 1.7B | gated-approximations | lowrank | 1 | 0.9260 |
| 1.7B | gated-approximations | lowrank | 32 | 1.2164 |
| 4B | gated-approximations | dense32 | 1 | 1.8716 |
| 4B | gated-approximations | dense32 | 32 | 2.2557 |
| 4B | gated-approximations | bf16 | 1 | 1.8693 |
| 4B | gated-approximations | bf16 | 32 | 2.2360 |
| 4B | gated-approximations | topk | 1 | 1.8609 |
| 4B | gated-approximations | topk | 32 | 2.2170 |
| 4B | gated-approximations | adaptive | 1 | 1.8603 |
| 4B | gated-approximations | adaptive | 32 | 2.2185 |
| 4B | gated-approximations | lowrank | 1 | 1.8664 |
| 4B | gated-approximations | lowrank | 32 | 2.2211 |
| 8B | gated-approximations | dense32 | 1 | 3.2231 |
| 8B | gated-approximations | dense32 | 32 | 3.8409 |
| 8B | gated-approximations | bf16 | 1 | 3.2047 |
| 8B | gated-approximations | bf16 | 32 | 3.8106 |
| 8B | gated-approximations | topk | 1 | 3.1901 |
| 8B | gated-approximations | topk | 32 | 3.7684 |
| 8B | gated-approximations | adaptive | 1 | 3.1865 |
| 8B | gated-approximations | adaptive | 32 | 3.7738 |
| 8B | gated-approximations | lowrank | 1 | 3.1949 |
| 8B | gated-approximations | lowrank | 32 | 3.7708 |
