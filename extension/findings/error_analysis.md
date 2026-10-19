# Error analysis of the saved predictions

Every prediction from the final evaluation (greedy decoding, best checkpoints: DPO step 100, GRPO step 400, RLOO step 600) is put into one category. Categories are checked in the order of the table and the first that applies is used. They come from simple pattern rules that compare the predicted SQL with the gold SQL (tables after FROM / JOIN, number of SELECTs and set operations, aggregate functions, GROUP BY / HAVING / ORDER BY / LIMIT / DISTINCT), so they are an approximation, not a full SQL parse.

## Spider dev (1034 questions): share of questions per category (%)

| Category | Untrained | DPO | GRPO | RLOO |
|---|---|---|---|---|
| Correct | 61.7 | 59.6 | 71.5 | 72.1 |
| SQL error | 16.3 | 10.6 | 7.1 | 6.1 |
| Wrong tables | 7.3 | 9.4 | 6.4 | 6.6 |
| Nesting / set operation | 3.7 | 4.7 | 2.8 | 3.9 |
| Aggregation | 1.8 | 2.8 | 2.4 | 1.8 |
| Grouping / ordering | 1.7 | 3.7 | 1.4 | 1.8 |
| Columns / conditions | 7.4 | 9.2 | 8.5 | 7.6 |

## Spider-Syn (1034 questions): share of questions per category (%)

| Category | Untrained | DPO | GRPO | RLOO |
|---|---|---|---|---|
| Correct | 49.8 | 46.7 | 58.8 | 58.5 |
| SQL error | 21.2 | 16.3 | 9.9 | 10.3 |
| Wrong tables | 13.1 | 15.3 | 14.5 | 14.0 |
| Nesting / set operation | 3.4 | 3.7 | 3.8 | 3.5 |
| Aggregation | 1.6 | 3.6 | 2.1 | 1.6 |
| Grouping / ordering | 2.5 | 3.3 | 1.7 | 2.2 |
| Columns / conditions | 8.4 | 11.1 | 9.2 | 9.8 |

## Spider-DK (535 questions): share of questions per category (%)

| Category | Untrained | DPO | GRPO | RLOO |
|---|---|---|---|---|
| Correct | 54.8 | 54.2 | 62.2 | 61.7 |
| SQL error | 16.1 | 11.4 | 8.2 | 6.0 |
| Wrong tables | 9.2 | 10.5 | 9.9 | 10.5 |
| Nesting / set operation | 5.4 | 4.9 | 4.7 | 5.6 |
| Aggregation | 2.4 | 3.7 | 2.2 | 2.8 |
| Grouping / ordering | 2.6 | 3.4 | 1.7 | 2.2 |
| Columns / conditions | 9.5 | 12.0 | 11.0 | 11.2 |

## BIRD dev (1534 questions): share of questions per category (%)

| Category | Untrained | DPO | GRPO | RLOO |
|---|---|---|---|---|
| Correct | 26.8 | 24.4 | 33.2 | 33.7 |
| SQL error | 44.9 | 34.4 | 32.2 | 31.6 |
| Wrong tables | 11.4 | 16.4 | 12.6 | 13.4 |
| Nesting / set operation | 3.7 | 4.8 | 7.2 | 6.8 |
| Aggregation | 2.5 | 4.6 | 2.9 | 3.1 |
| Grouping / ordering | 4.8 | 6.9 | 5.3 | 4.6 |
| Columns / conditions | 6.0 | 8.4 | 6.6 | 6.8 |

## What training changes, question by question

Fixed = wrong before training, right after. Broken = right before, wrong after.

| Benchmark | Model | Fixed | Broken | Net gain (questions) | Net gain (points) |
|---|---|---|---|---|---|
| Spider dev | DPO | 110 | 132 | -22 | -2.1 |
| Spider dev | GRPO | 125 | 24 | 101 | +9.8 |
| Spider dev | RLOO | 124 | 16 | 108 | +10.4 |
| Spider-Syn | DPO | 102 | 134 | -32 | -3.1 |
| Spider-Syn | GRPO | 130 | 37 | 93 | +9.0 |
| Spider-Syn | RLOO | 118 | 28 | 90 | +8.7 |
| Spider-DK | DPO | 60 | 63 | -3 | -0.6 |
| Spider-DK | GRPO | 53 | 13 | 40 | +7.5 |
| Spider-DK | RLOO | 47 | 10 | 37 | +6.9 |
| BIRD dev | DPO | 136 | 173 | -37 | -2.4 |
| BIRD dev | GRPO | 170 | 71 | 99 | +6.5 |
| BIRD dev | RLOO | 167 | 61 | 106 | +6.9 |

## Spider dev: which of the untrained model's errors are fixed

For each category of error the untrained model makes: how many of those questions each trained model gets right.

| Untrained model's error | Questions | fixed by DPO | fixed by GRPO | fixed by RLOO |
|---|---|---|---|---|
| SQL error | 169 | 61 (36.1%) | 77 (45.6%) | 80 (47.3%) |
| Wrong tables | 75 | 21 (28.0%) | 18 (24.0%) | 17 (22.7%) |
| Nesting / set operation | 38 | 6 (15.8%) | 11 (28.9%) | 8 (21.1%) |
| Aggregation | 19 | 2 (10.5%) | 2 (10.5%) | 2 (10.5%) |
| Grouping / ordering | 18 | 5 (27.8%) | 5 (27.8%) | 6 (33.3%) |
| Columns / conditions | 77 | 15 (19.5%) | 12 (15.6%) | 11 (14.3%) |

## BIRD dev: which of the untrained model's errors are fixed

For each category of error the untrained model makes: how many of those questions each trained model gets right.

| Untrained model's error | Questions | fixed by DPO | fixed by GRPO | fixed by RLOO |
|---|---|---|---|---|
| SQL error | 688 | 96 (14.0%) | 122 (17.7%) | 118 (17.2%) |
| Wrong tables | 175 | 19 (10.9%) | 29 (16.6%) | 27 (15.4%) |
| Nesting / set operation | 56 | 1 (1.8%) | 3 (5.4%) | 6 (10.7%) |
| Aggregation | 39 | 5 (12.8%) | 2 (5.1%) | 1 (2.6%) |
| Grouping / ordering | 73 | 4 (5.5%) | 5 (6.8%) | 6 (8.2%) |
| Columns / conditions | 92 | 11 (12.0%) | 9 (9.8%) | 9 (9.8%) |

## BIRD dev: accuracy by difficulty (%)

| Model | simple (925) | moderate (464) | challenging (145) |
|---|---|---|---|
| Untrained | 34.3 | 15.3 | 15.9 |
| DPO | 32.1 | 14.7 | 6.2 |
| GRPO | 41.9 | 19.8 | 20.7 |
| RLOO | 43.6 | 19.4 | 16.6 |
