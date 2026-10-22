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

## 3. Multi-turn RLOO and GRPO training (GPU)

`train_multiturn.py` trains the policy inside the multi-turn environment, so the agent learns to use the
execution feedback instead of only seeing it at test time.

- Episode: the model answers, the SQL is run, and (in `--mode errors`) a failed query gets the error
  message back and the model answers again, up to 3 turns.
- Reward only at the end of the episode: score of the final query (+1 / 0 / -0.1) minus 0.05 for every
  extra turn, so getting it right the first time is still the best outcome.
- RLOO: 4 episodes per question, advantage = reward minus the mean of the other 3 episodes, KL to the
  starting policy subtracted from the reward (beta 0.04, same as before). The gradient goes through every
  token the model wrote in every turn; the prompt and the feedback are masked out.
- `--method grpo` uses the GRPO advantage instead ((reward minus group mean) / group std) and adds the KL
  to the loss per token, as TRL's GRPO trainer did in the single-turn runs. It starts from the GRPO step
  400 adapter. Everything else (episodes, reward, settings) is the same for both.
- Starts from the single-turn RLOO step 600 adapter, so the only difference from plain RLOO is the
  multi-turn training. Same LoRA, learning rate, temperature and questions per step as the main runs.

```bash
source ~/SageMaker/.bashrc_t2s
cd ~/Marl-SQL && git pull
# 3 step test (check memory and seconds per step)
python extension/train_multiturn.py --init_adapter ~/demo_models/outputs/rloo-1.5b-s0/checkpoint-600 \
    --run_name mt-test --max_steps 3 --save_steps 3 --wandb 0
# full run in the background
nohup python extension/train_multiturn.py --init_adapter ~/demo_models/outputs/rloo-1.5b-s0/checkpoint-600 \
    --run_name mt-rloo-1.5b-s0 --max_steps 150 > mt_train.log 2>&1 &
tail -f mt_train.log
```

GRPO (can run at the same time on a second space, since each run needs the whole T4):

```bash
nohup python extension/train_multiturn.py --method grpo \
    --init_adapter ~/demo_models/outputs/grpo-1.5b-s0/checkpoint-400 \
    --run_name mt-grpo-1.5b-s0 --max_steps 150 > mt_grpo_train.log 2>&1 &
```

Checkpoints go to `~/outputs/mt-rloo-1.5b-s0/checkpoint-50, -100, -150`; the step log is in
`~/outputs/mt-rloo-1.5b-s0/logs/steps.jsonl`. Then evaluate it the same way as the other models (turn 1
is generated because there is no saved single-turn answer for it):

```bash
for MODE in errors all; do
  python extension/multiturn_eval.py --tag mt-rloo --adapter ~/outputs/mt-rloo-1.5b-s0/checkpoint-150 \
      --split spider_dev --mode $MODE --regenerate
done
# and the same with --tag mt-grpo --adapter ~/outputs/mt-grpo-1.5b-s0/checkpoint-150
python extension/multiturn_eval.py --summary
mkdir -p extension/findings/mt_training && cp ~/outputs/mt-rloo-1.5b-s0/logs/*.jsonl extension/findings/mt_training/
git add -f extension/findings && git commit -m "Multi-turn training results" && git push
```
