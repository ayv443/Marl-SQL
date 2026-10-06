# marl-sql

Credit assignment in a three-agent text-to-SQL pipeline. UTS 43008 Reinforcement Learning.

A question in English plus a database schema goes in; SQL comes out, produced by three
agents in sequence (Selector -> Decomposer -> Refiner). The agents are three LoRA adapters
over one frozen `Qwen2.5-Coder-1.5B-Instruct`. Reward comes from **executing** the generated
query against the real SQLite database and comparing returned rows to a gold query's rows.

One scalar reward arrives at the end of a three-agent pipeline. **How that reward should be
split between the three agents is the research question.** We compare three answers:

| Condition | Credit basis | Role |
|---|---|---|
| `independent` | identical team reward, group-normalised | baseline |
| `counterfactual` | marginal contribution via default substitution (COMA) | our contribution |
| `monolith` | n/a, single agent, no decomposition | ablation |

---

## The rule that matters most

> **The credit assignment method is the only thing allowed to differ between runs.**

Same base model, LoRA rank, learning rate, group size, temperature, step count, seed,
training subset, reward function and prompts. A better accuracy number obtained by varying
anything else is a failed experiment, not a better result.

Five files are therefore **contract files**. Never push a change to them directly to `main` —
branch, open a PR, and tell the other person:

```
configs/base.yaml              # shared hyperparameters
src/execution/harness.py       # how SQL is executed
src/execution/compare.py       # how correctness is defined
src/agents/prompts.py          # all prompt templates
src/rollout.py                 # episode structure
```

They are frozen at the end of week 6. After that, changing one means re-running every
condition already trained.

## Attribution

Commits and pull requests carry the repository owner's name only. No co-author trailers, no
"generated with" lines, no coding agent named in any commit message, PR, branch name or code
comment. See `CONVENTIONS.md`.

---

## Layout

```
configs/      base.yaml + one file per condition (credit_method only)
src/
  execution/  harness.py (pooled, timed, read-only SQL), compare.py (correctness)
  data/       filter.py (the gating preprocessing pass), schema.py, loader.py
  agents/     prompts.py (ALL prompts), parse.py, selector/decomposer/refiner
  credit/     independent.py, counterfactual.py, monolith.py
  rollout.py  train.py  evaluate.py
scripts/      download_data.py, check_pass_at_k.py, plot_run.py, audit_invariants.py
tests/        test_compare.py  <- the one mandatory test file
results/      one subdirectory per run: config.yaml + metrics.csv (both committed)
```

The planning documents — spec, execution plan, task breakdown — are **not in this
repository**. They live in `docs/` one level up and are shared out of band.

## Setup

### First time on a new SageMaker space

```bash
git clone https://github.com/ayv443/Marl-SQL.git
cd Marl-SQL
git checkout <your-branch>

GIT_NAME="Your Name" GIT_EMAIL="you@example.com" bash scripts/dev-setup.sh

pip install -r requirements.txt
pip freeze > environment/freeze-<yourname>.txt     # commit this
```

The first `git push` asks for a username and password. Use your GitHub username and a
**personal access token** with `repo` scope as the password -- an account password is
rejected. `dev-setup.sh` enables the credential store, so you are asked once per space.

### Every session after that

```bash
bash scripts/dev-setup.sh
```

Idempotent and quick. It matters because a fresh space has no git config and would
otherwise author your commits as `sagemaker-user`. It also installs the contract-file
warning hook, and reports Python version, free disk and GPU.

To avoid passing the two variables every time, either put them in `~/.bashrc`:

```bash
echo 'export GIT_NAME="Your Name"'          >> ~/.bashrc
echo 'export GIT_EMAIL="you@example.com"'   >> ~/.bashrc
```

or keep a `my-setup.sh` wrapper at the repo root that exports them and calls the script.
`dev-setup.sh` excludes that filename from git for you, so it stays local -- which also
means **it does not come down with a clone**. Recreate it, or use the `~/.bashrc` route.

Everything runs as a module from this directory:

```bash
python scripts/download_data.py --spider-only
python -m src.data.filter --report
pytest tests/test_compare.py
```

Long runs go in `tmux`, never a notebook cell — the SageMaker space idle-shuts after 60
minutes:

```bash
tmux new -s train
python -m src.train --config configs/independent.yaml 2>&1 | tee results/run.log
# Ctrl-b d detaches; tmux attach -t train returns
```

## Status

Skeleton and Gate 0 components written, **nothing has been executed yet**. Every target
number in `NOTES.md` is from the spec, not a measurement.

| Gate | Due | State |
|---|---|---|
| 0 — data, harness, pass@8 | end week 2 | in progress |
| 1 — Refiner-only GRPO rising | end week 4 | |
| 2 — full pipeline, overhead ~1.5x | end week 6 | |
| 3 — three conditions trained, seed 42 | end week 8 | |
| 4 — eval, error analysis, demo, report | end week 10 | |

## People

| | Person A | Person B |
|---|---|---|
| Name | | |
| Owns | data, execution, scoring, evaluation, error analysis | model, adapters, prompts, rollout, training loop, demo |

Weekly sync: _____________. Agenda: did last week's gate pass; what is blocked; who owns
what this week. Outcome goes in `NOTES.md`.
