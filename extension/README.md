# Extension: what does RL fine-tuning change in a text-to-SQL agent?

Branch `extension`, made from `text2sql-rl`. Nothing in `text2sql-rl/` is changed: everything here only
reads the saved evaluation results, the training logs and the existing code (`common.py`, `reward.py`,
`generation.py`).

| Script | What it does | GPU |
| --- | --- | --- |
| `error_analysis.py` | Puts every saved prediction into an error category (SQL error, wrong tables, nesting / set operation, aggregation, grouping / ordering, columns / conditions) and shows which errors each trained model fixes | No |
| `dpo_analysis.py` | Uses the DPO training log to show why DPO did not help (likelihood displacement) | No |
| `multiturn_eval.py` | Multi-turn environment: after each answer the SQL is run and the model sees the error or the first rows, and can fix its query (up to 3 turns) | Yes |

Results go to `extension/findings/`.

## 1. Error analysis and DPO analysis (done, no GPU)

```bash
python extension/error_analysis.py     # -> findings/error_analysis.md, error_categories.png
python extension/dpo_analysis.py       # -> findings/dpo_analysis.md, dpo_analysis.png
```

Main findings:
- GRPO and RLOO mostly fix queries that did not run: on Spider dev, SQL errors drop from 16.3% of
  questions (untrained) to 7.1% (GRPO) and 6.1% (RLOO). Wrong tables, wrong columns and wrong
  conditions change very little.
- On Spider dev RLOO fixes 124 questions and breaks 16; GRPO fixes 125 and breaks 24; DPO fixes 110 but
  breaks 132 (same counts as the McNemar test in `statistics/compare_spider_dev.txt`).
- On BIRD about 31% of answers are still SQL errors after RL, so the multi-turn feedback has room to help.
- DPO: preference accuracy rises from 0.51 to 0.89, but the log-probability of the correct (chosen)
  answers also falls (-27 to -45) while validation accuracy drops from 61.9% to 41.1%.

## 2. Multi-turn evaluation (GPU)

The environment becomes multi-step: state = the conversation so far (prompt, the model's answers and
the execution feedback), action = the next answer, reward at the end = whether the final SQL returns the
gold result. The gold SQL is never shown to the model.

- Turn 1 reuses the saved evaluation answer (same greedy decoding as in `evaluation/results/`), so only
  turns 2 and 3 need the GPU. `--regenerate` makes turn 1 again.
- `--mode errors`: only answers that fail to run get feedback (cheap).
- `--mode all`: every answer sees its result; a question stops when the model repeats its query.

On the T4 space (ml.g4dn.xlarge), from the repository root, with the same adapters as the demo:

```bash
source ~/SageMaker/.bashrc_t2s
cd ~/Marl-SQL && git fetch origin && git checkout extension && git pull
nvidia-smi                       # the GPU must be free

# quick test first (50 questions, one model)
python extension/multiturn_eval.py --tag rloo --adapter ~/demo_models/outputs/rloo-1.5b-s0/checkpoint-600 \
    --split spider_dev --mode all --limit 50
```

If the quick test works, run all four models (one after another, never two at the same time):

```bash
M=~/demo_models/outputs
for MODE in errors all; do
  python extension/multiturn_eval.py --tag base --split spider_dev --mode $MODE
  python extension/multiturn_eval.py --tag dpo  --adapter $M/dpo-1.5b-s0/checkpoint-100  --split spider_dev --mode $MODE
  python extension/multiturn_eval.py --tag grpo --adapter $M/grpo-1.5b-s0/checkpoint-400 --split spider_dev --mode $MODE
  python extension/multiturn_eval.py --tag rloo --adapter $M/rloo-1.5b-s0/checkpoint-600 --split spider_dev --mode $MODE
done
python extension/multiturn_eval.py --summary     # -> findings/multiturn/summary.md
```

Then the same for BIRD dev if there is time (`--split bird_dev`). Run long jobs with `nohup ... &` so a
closed browser tab does not stop them, and keep every `findings/multiturn/` folder: commit them to this
branch afterwards (`per_question.jsonl` and `metrics.json` are small text files).
