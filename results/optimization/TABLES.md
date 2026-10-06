# Final optimization measurements

128 positions; nine repetitions per engine/policy/batch. Stock is the median of the 18 bracketing control samples.

The intervals below are 95% unpaired bootstrap intervals for medians within one session. They do not cover different GPUs, prompts, or day-to-day variation.

| Model | Batch | Policy | Old (s) | Optimized (s) | Latency reduction [95% interval] | Optimized overhead vs stock |
| --- | --- | --- | --- | --- | --- | --- |
| 0.6B | 1 | token | 0.5312 | 0.4571 | 13.9% [13.5, 14.2] | +0.4% |
| 0.6B | 1 | hidden | 0.5269 | 0.4608 | 12.5% [12.3, 13.1] | +1.2% |
| 0.6B | 1 | soft | 0.5758 | 0.4946 | 14.1% [13.7, 14.4] | +8.7% |
| 0.6B | 1 | entropy | 0.5859 | 0.4661 | 20.5% [20.2, 20.7] | +2.4% |
| 0.6B | 1 | mixed | 0.5329 | 0.4549 | 14.6% [14.4, 14.9] | -0.1% |
| 0.6B | 8 | token | 0.5980 | 0.5263 | 12.0% [11.9, 12.3] | +0.8% |
| 0.6B | 8 | hidden | 0.5970 | 0.5290 | 11.4% [11.2, 11.7] | +1.3% |
| 0.6B | 8 | soft | 0.6504 | 0.5648 | 13.2% [13.0, 13.4] | +8.2% |
| 0.6B | 8 | entropy | 0.6612 | 0.5417 | 18.1% [17.9, 18.6] | +3.7% |
| 0.6B | 8 | mixed | 0.8220 | 0.5731 | 30.3% [30.2, 30.5] | +9.8% |
| 0.6B | 32 | token | 0.7636 | 0.6820 | 10.7% [10.2, 10.8] | +0.5% |
| 0.6B | 32 | hidden | 0.7634 | 0.6909 | 9.5% [8.8, 9.7] | +1.8% |
| 0.6B | 32 | soft | 0.8358 | 0.7355 | 12.0% [11.8, 13.0] | +8.4% |
| 0.6B | 32 | entropy | 0.8845 | 0.7339 | 17.0% [16.4, 17.3] | +8.1% |
| 0.6B | 32 | mixed | 0.9977 | 0.7623 | 23.6% [23.4, 24.2] | +12.3% |
| 1.7B | 1 | token | 0.9552 | 0.8865 | 7.2% [7.2, 7.7] | -0.5% |
| 1.7B | 1 | hidden | 0.9571 | 0.8947 | 6.5% [6.3, 6.8] | +0.4% |
| 1.7B | 1 | soft | 1.0653 | 0.9585 | 10.0% [9.8, 10.3] | +7.6% |
| 1.7B | 1 | entropy | 1.0739 | 0.9041 | 15.8% [15.6, 16.0] | +1.5% |
| 1.7B | 1 | mixed | 0.9599 | 0.8956 | 6.7% [6.5, 6.9] | +0.5% |
| 1.7B | 8 | token | 1.0400 | 0.9638 | 7.3% [7.0, 7.6] | -0.4% |
| 1.7B | 8 | hidden | 1.0401 | 0.9762 | 6.1% [5.9, 6.3] | +0.9% |
| 1.7B | 8 | soft | 1.1516 | 1.0396 | 9.7% [9.5, 9.9] | +7.5% |
| 1.7B | 8 | entropy | 1.1679 | 0.9861 | 15.6% [15.3, 15.7] | +2.0% |
| 1.7B | 8 | mixed | 1.3713 | 1.0535 | 23.2% [23.0, 23.6] | +8.9% |
| 1.7B | 32 | token | 1.2133 | 1.1324 | 6.7% [6.3, 6.9] | -0.3% |
| 1.7B | 32 | hidden | 1.2152 | 1.1483 | 5.5% [5.4, 5.8] | +1.1% |
| 1.7B | 32 | soft | 1.3501 | 1.2240 | 9.3% [9.1, 9.4] | +7.8% |
| 1.7B | 32 | entropy | 1.3960 | 1.1939 | 14.5% [14.2, 15.1] | +5.2% |
| 1.7B | 32 | mixed | 1.5607 | 1.2526 | 19.7% [19.6, 19.9] | +10.3% |
| 4B | 1 | token | 1.9005 | 1.8270 | 3.9% [3.7, 4.0] | -0.0% |
| 4B | 1 | hidden | 1.9037 | 1.8395 | 3.4% [3.3, 3.5] | +0.7% |
| 4B | 1 | soft | 2.0394 | 1.9123 | 6.2% [6.1, 6.4] | +4.6% |
| 4B | 1 | entropy | 2.0474 | 1.8428 | 10.0% [9.9, 10.0] | +0.8% |
| 4B | 1 | mixed | 1.9061 | 1.8332 | 3.8% [3.7, 4.0] | +0.3% |
| 4B | 8 | token | 1.9678 | 1.8889 | 4.0% [3.9, 4.1] | +0.0% |
| 4B | 8 | hidden | 1.9721 | 1.8979 | 3.8% [3.6, 3.8] | +0.5% |
| 4B | 8 | soft | 2.1090 | 1.9766 | 6.3% [6.1, 6.3] | +4.7% |
| 4B | 8 | entropy | 2.1246 | 1.9114 | 10.0% [10.0, 10.2] | +1.2% |
| 4B | 8 | mixed | 2.3626 | 1.9862 | 15.9% [15.9, 15.9] | +5.2% |
| 4B | 32 | token | 2.2214 | 2.1355 | 3.9% [3.7, 4.0] | +0.0% |
| 4B | 32 | hidden | 2.2327 | 2.1454 | 3.9% [3.6, 4.3] | +0.5% |
| 4B | 32 | soft | 2.3781 | 2.2409 | 5.8% [5.7, 5.9] | +5.0% |
| 4B | 32 | entropy | 2.4289 | 2.1812 | 10.2% [10.1, 10.3] | +2.2% |
| 4B | 32 | mixed | 2.6260 | 2.2626 | 13.8% [13.7, 13.9] | +6.0% |
| 8B | 1 | token | 3.2214 | 3.1461 | 2.3% [2.3, 2.4] | -0.1% |
| 8B | 1 | hidden | 3.2210 | 3.1579 | 2.0% [1.9, 2.0] | +0.3% |
| 8B | 1 | soft | 3.4460 | 3.2749 | 5.0% [4.9, 5.0] | +4.0% |
| 8B | 1 | entropy | 3.4485 | 3.1622 | 8.3% [8.2, 8.4] | +0.5% |
| 8B | 1 | mixed | 3.2227 | 3.1516 | 2.2% [2.1, 2.2] | +0.1% |
| 8B | 8 | token | 3.3211 | 3.2477 | 2.2% [1.7, 2.3] | +0.1% |
| 8B | 8 | hidden | 3.3184 | 3.2553 | 1.9% [1.8, 2.0] | +0.3% |
| 8B | 8 | soft | 3.5459 | 3.3749 | 4.8% [4.7, 4.9] | +4.0% |
| 8B | 8 | entropy | 3.5582 | 3.2675 | 8.2% [8.1, 8.2] | +0.7% |
| 8B | 8 | mixed | 3.8816 | 3.3830 | 12.8% [12.8, 12.9] | +4.3% |
| 8B | 32 | token | 3.7475 | 3.6681 | 2.1% [2.1, 2.2] | +0.1% |
| 8B | 32 | hidden | 3.7460 | 3.6753 | 1.9% [1.8, 1.9] | +0.3% |
| 8B | 32 | soft | 3.9952 | 3.8307 | 4.1% [4.1, 4.5] | +4.5% |
| 8B | 32 | entropy | 4.0406 | 3.7123 | 8.1% [8.0, 8.1] | +1.3% |
| 8B | 32 | mixed | 4.3168 | 3.8400 | 11.0% [11.0, 11.1] | +4.8% |

## Arithmetic smoke check

Last integer equals the answer; 64 fixed questions, 64 positions, four latent-prefix positions. This is not a general reasoning evaluation.

| Model | Engine | Policy | Correct / 64 | Strict integer format correct |
| --- | --- | --- | --- | --- |
| 0.6B | stock-start | token | 53 | 0 |
| 0.6B | baseline | token | 53 | 0 |
| 0.6B | baseline | soft | 53 | 0 |
| 0.6B | baseline | hidden | 52 | 0 |
| 0.6B | optimized | token | 53 | 0 |
| 0.6B | optimized | soft | 53 | 0 |
| 0.6B | optimized | hidden | 52 | 0 |
| 0.6B | optimized | topk | 52 | 0 |
| 1.7B | stock-start | token | 54 | 0 |
| 1.7B | baseline | token | 54 | 0 |
| 1.7B | baseline | soft | 55 | 0 |
| 1.7B | baseline | hidden | 46 | 0 |
| 1.7B | optimized | token | 54 | 0 |
| 1.7B | optimized | soft | 55 | 0 |
| 1.7B | optimized | hidden | 46 | 0 |
| 1.7B | optimized | topk | 55 | 0 |
| 4B | stock-start | token | 57 | 22 |
| 4B | baseline | token | 57 | 22 |
| 4B | baseline | soft | 47 | 0 |
| 4B | baseline | hidden | 31 | 2 |
| 4B | optimized | token | 57 | 22 |
| 4B | optimized | soft | 47 | 0 |
| 4B | optimized | hidden | 31 | 2 |
| 4B | optimized | topk | 47 | 0 |
| 8B | stock-start | token | 57 | 32 |
| 8B | baseline | token | 58 | 31 |
| 8B | baseline | soft | 56 | 0 |
| 8B | baseline | hidden | 43 | 1 |
| 8B | optimized | token | 58 | 32 |
| 8B | optimized | soft | 56 | 0 |
| 8B | optimized | hidden | 43 | 1 |
| 8B | optimized | topk | 55 | 0 |
