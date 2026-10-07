# Text-to-SQL with RL: DPO vs GRPO vs RLOO

Qwen2.5-Coder-1.5B-Instruct + LoRA, trained on Spider with DPO, GRPO and RLOO, evaluated on
Spider dev (EX, TS), Spider-Syn/DK/Realistic, and BIRD dev (EX, Soft-F1, R-VES).
Who does what is in [TEAM_PLAN.md](TEAM_PLAN.md).

## Files

| File | What it does | Plan phase |
| --- | --- | --- |
| `common.py` | Paths, prompt + schema format, SQL extraction, model/LoRA loading, shared training config | all |
| `reward.py` | **The shared reward** (+1 / 0 / -0.1), read-only SQLite, 5 s timeout, 8 workers. Freeze after Phase 1 | 1 |
| `download_data.sh` | Spider, test-suite DBs, BIRD dev, Spider-Syn/DK, official eval repos | 1 |
| `prepare_data.py` | Filters train, holds out ~400 val questions by DB, caches gold results, builds prompts | 1 |
| `generation.py` | vLLM or HF generation (with or without a LoRA adapter) | 1, 4 |
| `sample.py` | Feasibility run (8 samples/question, pass@k, mixed tags); also the pass@k add-on | 1 |
| `make_dpo_pairs.py` | DPO pairs from the feasibility samples | 2 |
| `train_dpo.py` | DPO | 2-3 |
| `train_rl.py` | GRPO **and** RLOO (`--method grpo/rloo`), same settings for both | 2-3 |
| `launch_sagemaker.py` | Optional: runs training as a SageMaker Training Job (needs S3). Not used: we train inside Studio | - |
| `val_curve.py` | Evaluates every checkpoint on the val slice and plots the shared curve | 3 |
| `evaluate.py` | Greedy eval: EX, valid-SQL rate, EX by difficulty, prediction files for the official scripts | 4 |
| `run_official_eval.sh` | Official Spider EX + Test-Suite; how to run the BIRD EX/Soft-F1/R-VES scripts | 4 |
| `analysis.py` | Bootstrap CIs, McNemar, commonly-correct efficiency analysis | 4, add-ons |
| `app.py` | Gradio demo (4 models side by side) | 8 |
| `monitor.py` | Telegram messages (start, progress, checkpoints, errors) + log files of every step | all |
| `collect_proof.sh` | Collects proof of a training run done in Studio (step logs, checkpoint histories, Studio space, AWS identity, code commit, versions) | 3 |

## Order to run things

All commands run from this folder in a SageMaker notebook/JupyterLab terminal (ml.g4dn.2xlarge).

```bash
# 0. setup
pip install --no-cache-dir -r requirements.txt vllm==0.10.2 matplotlib scipy gradio pandas "sagemaker<3" gdown
wandb login

# 1. data (Phase 1)
bash download_data.sh
python prepare_data.py                 # prints how many train examples were dropped and why
python reward.py --split val           # sanity check: gold-vs-gold must be 1.000

# 2. feasibility + DPO data (Phase 1)
python sample.py --split train --limit 50 --n 8   # 2-minute test first
python sample.py --split train --n 8 --temperature 0.8
python make_dpo_pairs.py

# 3. smoke tests, ~50 steps each (Phase 3) - note s/step and peak memory (nvidia-smi)
python train_dpo.py --max_steps 50 --lr 5e-5
python train_rl.py --method grpo --max_steps 50 --lr 5e-5
python train_rl.py --method rloo --max_steps 50 --lr 5e-5

# 4. full runs inside the Studio space, in the background (each person their own method)
nohup python train_dpo.py --lr 5e-5 --seed 0 > logs_dpo.txt 2>&1 &
nohup python train_rl.py --method grpo --seed 0 --max_steps 600 --lr 5e-5 > logs_grpo.txt 2>&1 &
nohup python train_rl.py --method rloo --seed 0 --max_steps 600 --lr 5e-5 > logs_rloo.txt 2>&1 &
bash collect_proof.sh dpo-1.5b-s0      # after each run finishes

# 5. evaluation (Phase 4); GRPO / RLOO adapters arrive via Google Drive (readme.txt section 14)
python val_curve.py --runs outputs/dpo-1.5b-s0 outputs/grpo-1.5b-s0 outputs/rloo-1.5b-s0
python evaluate.py --split spider_dev --tag base
python evaluate.py --split spider_dev --tag grpo --adapter outputs/grpo-1.5b-s0/checkpoint-XXX   # best on val
# ... repeat for every model x {spider_dev, spider_syn, spider_dk, spider_realistic, bird_dev}
bash run_official_eval.sh grpo
python analysis.py compare --split spider_dev
python analysis.py efficiency --split bird_dev
```

## Things that will probably need fixing the first time

- **vLLM on the T4.** If `--engine vllm` crashes, add `--engine hf` to `sample.py` / `evaluate.py` /
  `val_curve.py` (slower but works). Keep `--use_vllm 0` for training unless it fits in memory.
- **Out of memory in GRPO/RLOO.** Keep `--num_generations 4` (the plan fixes it). Try `--load_4bit 1`,
  or switch all three methods to `--model Qwen/Qwen2.5-Coder-0.5B-Instruct`.
- **NaN loss in fp16.** Lower `--lr` (e.g. 5e-6), then try `--load_4bit 1` (QLoRA).
- **Download links.** If a Google Drive link hits its quota, download in a browser and unzip into `data/`.
- **Changed model size?** Redo `sample.py` + `make_dpo_pairs.py` with the new model so the tags and
  DPO pairs come from the model you actually train.

## Settings worth stating in the report

- Same prompt, LoRA (r=16, alpha=32, all attention + MLP projections), lr, fp16 and completion length (256) for all methods.
- GRPO and RLOO: 4 completions/question, temperature 0.8, KL beta = 0.04 (set explicitly; TRL's GRPO default is 0),
  4 prompts (16 completions) per step, same `--max_steps` -> same number of generated completions.
- TRL 0.24 GRPO uses `loss_type="dapo"` and `scale_rewards="group"` by default; RLOO uses no std normalisation.
- DPO beta = 0.1 is the DPO temperature, not a KL penalty coefficient.
- A second reward function, `valid_sql_reward`, has weight 0. It does not affect training; it is only there so
  W&B logs the valid-SQL rate. Other things to watch in W&B: `reward`, `kl`, `completions/mean_length`,
  `frac_reward_zero_std` (zero-variance groups) and `entropy`.
- Timeout uses SQLite's progress handler inside a thread pool, not one process per query (safe inside the training process).
- BIRD prompts include the evidence hints.
- SQL is taken from `<answer>` tags, a ``` block, or plain SQL starting a line (the untrained model mostly ignores the tags), so the reward checks the result, not the format.
