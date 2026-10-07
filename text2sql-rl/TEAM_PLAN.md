# Team plan (3 people, 3 AWS accounts)

Updated 8 Oct. Each method is trained on a different AWS account at the same time, with
exactly the same code (this branch), data, library versions and settings. All evaluation
happens on one machine (Eby's), so every model is evaluated the same way.

| | Eby | GRPO teammate | Aditya |
| --- | --- | --- | --- |
| Trains | **DPO** | **GRPO** | **RLOO** |
| Also | Data prep, reward check, feasibility run, `tags.json`, DPO pairs, **all evaluation**, results tables | | Gradio demo (`app.py`) |
| Instructions | `how-to-run/readme.txt` | `GRPO_RETRAIN_PLAN.txt` (sent by Eby) | `RLOO_PLAN.txt` (sent by Eby) |
| Report | Data & filtering, reward, DPO, evaluation setup, benchmark results | GRPO method + training dynamics | RLOO method + training dynamics, demo |
| Together | Intro, related work, comparison / discussion, conclusion, demo rehearsal | | |

## Shared settings (all three methods)

Qwen2.5-Coder-1.5B-Instruct (unless all switch to 0.5B after the smoke tests), LoRA r=16 / alpha=32 on all
attention + MLP layers, fp16, `--lr 5e-5`, seed 0, max completion 256 tokens, prompts over 2048 tokens skipped
(899 of 6631 training questions, 5732 remain). GRPO and RLOO: 4 completions per question, 4 questions per step,
temperature 0.8, beta 0.04, only "mixed" questions from `tags.json`, the **same** `--max_steps`.

## How things move between us

| What | How |
| --- | --- |
| Code and fixes | This GitHub branch. Eby pushes, the others `git pull`. Nobody else pushes to `text2sql-rl`. |
| `tags.json` | Eby commits it after the feasibility run; the others `git pull`. |
| Code check | Before launching, each trainer sends Eby `git log -1` (must be the latest commit) and a clean `git status`. |
| Training proof | Each trainer runs `collect_proof.sh` and pushes `proof/<run>/` to their own branch (`grpo-run`, `rloo-run`). |
| Checkpoints | A Google Drive link to a .tar.gz of the adapter files (~0.5 GB, too big for git). Eby checks them with section 22 of their plan. |
| W&B | One team project `text2sql-rl` (or public personal projects if a team isn't possible). |

## Order

1. Eby: feasibility run, go/no-go, DPO pairs, push `tags.json`.
2. Teammates (meanwhile): setup, data prep (numbers must match: 7040 / 6631 / 409), reward check, 50-step smoke test.
3. All: compare smoke-test timings, agree model size and `max_steps`.
4. All: start DPO, GRPO, RLOO inside your own Studio space (nohup) at the same time. No S3, no training jobs.
5. Teammates: proof to their branch, checkpoint links to Eby. Aditya: demo while RLOO trains.
6. Eby: validation curve, final evaluation, statistics, results tables. Then the report together.

If someone drops out, `how-to-run/readme.txt` section 17 is the plan for doing it alone.
