# Multi-turn evaluation

EX after each turn (%). Fixed = wrong at turn 1, right at the end. Broken = right at turn 1, wrong at the end.

| Benchmark | Mode | Model | Questions | Turn 1 | Turn 2 | Turn 3 | Fixed | Broken | Mean turns |
|---|---|---|---|---|---|---|---|---|---|
| spider_dev | all | grpo | 1034 | 71.5 | 71.5 | 71.5 | 6 | 6 | 2.09 |
| spider_dev | all | rloo | 1034 | 72.1 | 72.1 | 71.9 | 6 | 9 | 2.11 |
| spider_dev | all | mt-rloo | 1034 | 71.5 | 71.8 | 71.7 | 5 | 3 | 2.04 |
| spider_dev | all | mt-grpo | 1034 | 71.8 | 72.2 | 72.2 | 5 | 0 | 2.02 |
| val | all | mt-rloo-50 | 409 | 72.4 | 73.8 | 73.8 | 6 | 0 | 2.03 |
| val | all | mt-rloo-100 | 409 | 72.6 | 73.6 | 73.6 | 4 | 0 | 2.02 |
| val | all | mt-rloo-150 | 409 | 73.6 | 74.1 | 74.1 | 2 | 0 | 2.02 |
| val | all | rloo-600 | 409 | 74.3 | 74.3 | 74.6 | 3 | 2 | 2.05 |
| val | all | grpo-400 | 409 | 74.3 | 74.1 | 74.1 | 0 | 1 | 2.04 |
| val | all | mt-grpo-100 | 409 | 75.6 | 75.6 | 75.6 | 0 | 0 | 2.03 |
| val | all | mt-grpo-150 | 409 | 76.0 | 76.3 | 76.3 | 1 | 0 | 2.03 |
| val | all | mt-grpo-50 | 409 | 73.1 | 73.6 | 73.6 | 2 | 0 | 2.03 |