# Significance tests for the multi-turn evaluation

Exact McNemar tests on paired per-question results (the same questions for both models). b = questions where only the first is true, c = only the second. Two-sided, p < 0.05 marked *.

## spider_dev

| Comparison | b | c | p |
|---|---|---|---|
| grpo: correct at turn 1 vs after feedback | 6 | 6 | 1.0000 |
| mt-grpo: correct at turn 1 vs after feedback | 0 | 5 | 0.0625 |
| grpo vs mt-grpo: correct at turn 1 | 19 | 22 | 0.7552 |
| grpo vs mt-grpo: correct after feedback | 20 | 28 | 0.3123 |
| grpo vs mt-grpo: broken by feedback | 6 | 0 | 0.0312 * |
| rloo: correct at turn 1 vs after feedback | 9 | 6 | 0.6072 |
| mt-rloo: correct at turn 1 vs after feedback | 3 | 5 | 0.7266 |
| rloo vs mt-rloo: correct at turn 1 | 36 | 29 | 0.4570 |
| rloo vs mt-rloo: correct after feedback | 39 | 37 | 0.9088 |
| rloo vs mt-rloo: broken by feedback | 7 | 1 | 0.0703 |

## val

| Comparison | b | c | p |
|---|---|---|---|
| grpo-400: correct at turn 1 vs after feedback | 1 | 0 | 1.0000 |
| mt-grpo-150: correct at turn 1 vs after feedback | 0 | 1 | 1.0000 |
| grpo-400 vs mt-grpo-150: correct at turn 1 | 8 | 15 | 0.2100 |
| grpo-400 vs mt-grpo-150: correct after feedback | 7 | 16 | 0.0931 |
| grpo-400 vs mt-grpo-150: broken by feedback | 1 | 0 | 1.0000 |
| rloo-600: correct at turn 1 vs after feedback | 2 | 3 | 1.0000 |
| mt-rloo-150: correct at turn 1 vs after feedback | 0 | 2 | 0.5000 |
| rloo-600 vs mt-rloo-150: correct at turn 1 | 16 | 13 | 0.7111 |
| rloo-600 vs mt-rloo-150: correct after feedback | 15 | 13 | 0.8506 |
| rloo-600 vs mt-rloo-150: broken by feedback | 2 | 0 | 0.5000 |

## Broken answers, both methods pooled (Spider dev)

Questions broken only by the single-turn model: 13, only by the multi-turn model: 1, exact sign test p = 0.0018. (Pooling treats the GRPO and RLOO comparisons as one set of paired observations; it is a summary, not a separate experiment.)
