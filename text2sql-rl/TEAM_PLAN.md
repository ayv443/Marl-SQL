# Team plan for 2 people

The third member is out, so the work is re-split between two people. The code is already
written, so the job now is to run it, fix whatever breaks, train, evaluate and write the report.

- **Person A: online RL (GRPO + RLOO).** `train_rl.py` trains both methods, so one person can own them.
- **Person B: data, DPO and evaluation.**

Both of you have AWS, so **each person runs their own experiments on their own account** and
neither of you has to wait for the other's GPU.

## Who owns what

| | Person A | Person B |
| --- | --- | --- |
| Methods | **GRPO**, **RLOO** | **DPO** |
| Code they own (debug + explain in report) | `reward.py`, `train_rl.py`, `launch_sagemaker.py`, `app.py` | `prepare_data.py`, `sample.py`, `make_dpo_pairs.py`, `train_dpo.py`, `evaluate.py`, `val_curve.py`, `analysis.py` |
| Runs on their AWS account | GRPO + RLOO smoke tests and full jobs | Feasibility run, DPO jobs, **all** evaluation (one place = same timing conditions for R-VES) |
| Add-ons | pass@k (Tier 1), compute cost table (Tier 1), zero-variance analysis (Tier 2), Gradio demo | Bootstrap CI + McNemar (Tier 1), commonly-correct efficiency, error analysis |
| Report sections | GRPO + RLOO method & results, training dynamics (KL, entropy, zero-variance) | Data & filtering, reward, DPO method & results, evaluation setup, benchmark results |
| Together | Intro, related work, final comparison/discussion, conclusion, demo rehearsal | |

## What gets handed over (via git)

Data prep is deterministic (seed 42), so **both** of you run `download_data.sh` + `prepare_data.py`
and get identical splits. Only these small files travel through git (`.gitignore` already allows them):

| File | From → To | Needed for |
| --- | --- | --- |
| `data/processed/tags.json` | B → A | GRPO/RLOO train on "mixed" questions only |
| `data/processed/dpo_pairs.jsonl` | B (for DPO) | |
| `data/processed/filter_log.json`, `passk_train.json` | B → report | Phase 1 numbers |
| Final LoRA adapters (`final/` or best checkpoint, ~70 MB each) | A → B (Google Drive or S3 link) | B evaluates all four models in one place |
| W&B runs | both, one shared W&B **team** project | Training curves for the report |

## Day 1 (together, ~2 hours, today)

1. **Both:** request SageMaker quotas in *Service Quotas → Amazon SageMaker*: `ml.g4dn.2xlarge for training job usage`,
   `ml.g4dn.2xlarge for spot training job usage` and `ml.g4dn.2xlarge for notebook instance usage` (or Studio
   JupyterLab apps). Set each to 1. Approval can take days, so do this first.
2. **A:** create a private GitHub repo, push this `text2sql-rl` folder, add B as a collaborator.
3. **B:** create a W&B team, invite A, create project `text2sql-rl`.
4. **Both:** start a notebook (g4dn.2xlarge, 50 GB disk), clone the repo, run the setup line from the README.
   If GPU quota isn't approved yet, use an `ml.t3.xlarge` CPU notebook for data prep (it needs no GPU).

## Week 1: get everything running

| Day | Person A | Person B |
| --- | --- | --- |
| 1–2 | `download_data.sh`, `prepare_data.py`, `python reward.py --split val`. Try the reward on a few hand-written wrong queries. **Freeze `reward.py`** once you both agree it's right. | `download_data.sh`, `prepare_data.py`. Note the filter counts. `python evaluate.py --split val --limit 50` to check vLLM works (else `--engine hf`). |
| 2–3 | Upload data to S3 and launch a 20-step test job with `launch_sagemaker.py` to prove the job pipeline works. Smoke test runs with `--only_mixed 0` until B's tags arrive. | **Feasibility run:** `sample.py --split train --n 8`. Report pass@1, pass@8, share of mixed questions. **Go/no-go decision together.** Commit `tags.json`. |
| 3–4 | Smoke test GRPO + RLOO (50 steps each): s/step, peak memory (`nvidia-smi`). | `make_dpo_pairs.py`, commit pairs. DPO smoke test (50 steps). Baseline eval of the untrained model on spider_dev + bird_dev. |
| 5 | **Together (30 min call):** compare smoke-test timings and **lock the model size** (1.5B or 0.5B for all three) and `max_steps`, so GRPO and RLOO generate the same number of completions and everything fits the deadline. | |

Rough time maths for the call: `full-run hours = s/step × max_steps / 3600`. If a single run is over
~8–10 h, reduce `max_steps` or move to 0.5B **now**, not halfway through.

## Weeks 2–3: full training

- **A:** launch GRPO seed 0 and RLOO seed 0 as SageMaker jobs (spot). Both can run at the same time if
  the quota allows 2 instances, otherwise one after the other. Watch W&B: reward going up, `kl` not exploding,
  `completions/mean_length` stable, `rewards/valid_sql_reward/mean` high, `frac_reward_zero_std` not stuck near 1.
  More seeds if time/budget allows.
- **B:** launch DPO seed 0 (+ more seeds if there's budget). While jobs run: get `run_official_eval.sh` working on the
  base model (Spider EX + TS, BIRD EX / Soft-F1 / R-VES), so evaluation is ready when the models arrive.
- **A, while jobs run:** build the Gradio demo against the base model + first checkpoints.

## Week 4: evaluation

- **A:** send final checkpoints to B. Pull the cost numbers (GPU-hours, s/step, peak memory) from W&B/SageMaker.
  pass@k: `sample.py --split spider_dev --n 16 --adapter ... --tag grpo` (and rloo).
- **B:** `val_curve.py` for all three methods → choose the best checkpoint of each **on val** → evaluate base, DPO,
  GRPO and RLOO on all 5 benchmarks → official scripts → `analysis.py compare` + `efficiency` (on an idle instance) → fill in both results tables.

## Week 5: add-ons + report

- **Both:** error analysis, ~50 failures for each of your own methods (B also does base).
- **A:** zero-variance plot, and the DPO→GRPO run (`train_rl.py --init_adapter <dpo adapter>`) if there's time.
- **Both:** write your own sections (table above), then do intro/discussion/conclusion together; rehearse the demo.

If the due date is sooner, squash weeks 2–5 together. The must-haves are: Phase 1 numbers, one seed of each method,
val curve, both results tables. Add-ons only after those are done.

## Rules so the comparison stays fair

- Nobody edits `reward.py`, the prompt in `common.py`, or the data split after the freeze. If something has to change, tell the other person, because every method then has to be re-run.
- Same model size, LoRA, lr, completion length for all methods. GRPO and RLOO use the same `--max_steps`.
- Spider dev and its variants are **never** used to pick checkpoints. Use the val slice only.
- Name runs `<method>-<size>-s<seed>` (the scripts do this automatically) so W&B and S3 stay tidy.
