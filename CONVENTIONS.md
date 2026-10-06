# CONVENTIONS

Rules for working in this repository. Read before writing code.

## Commits

Commits and pull requests carry the author's own name and nothing else. No co-author
trailers, no tool-generated "generated with" lines, no tooling named in a commit message,
pull request, branch name or code comment.

Messages describe the change:

```
harness: pool connections per db_id
filter: report removal counts per reason
compare: fix empty-vs-empty returning True
```

Check your identity once per machine before the first commit — a fresh SageMaker space has
no git config and will otherwise author commits as `sagemaker-user`:

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email"
git log -1 --format='%an <%ae>%n%b'
```

## The rule everything else serves

> **The credit assignment method is the only thing allowed to differ between runs.**

Same base model, LoRA rank, learning rate, group size, temperature, step count, seed,
training subset, reward function and prompts. A better accuracy number obtained by varying
anything else is a failed experiment, not a better result. A clean comparison at 35%
execution accuracy is a success; 45% from an uncontrolled change is not.

When a choice is between "more accurate" and "more controlled", choose controlled and say
so in `NOTES.md`.

## Contract files

A change to any of these can silently invalidate the whole experiment:

```
configs/base.yaml              # shared hyperparameters
src/execution/harness.py       # how SQL is executed
src/execution/compare.py       # how correctness is defined
src/agents/prompts.py          # all prompt templates
src/rollout.py                 # episode structure
```

Never push to these directly on `main` — branch, open a PR, tell the other person. They are
frozen at end of week 6; after that, changing one means re-running every condition already
trained. Install the warning hook once per clone: `bash scripts/install-hooks.sh`

## Do not

1. **Change the reward function.** `-0.1` on error or timeout, `1.0` on result-set match,
   `0.0` otherwise. No partial credit, no length penalty, no format bonus, no stepwise
   shaping. Partial-match rewards would be a separate experiment re-run across all
   conditions.
2. **Change a shared hyperparameter** in `configs/base.yaml`. Condition configs set
   `credit_method` and that method's own parameters only.
3. **Use `bfloat16`.** The T4 is Turing (SM 7.5): fp16 everywhere, `vllm_dtype: half`, no
   Flash Attention 2, `attn_implementation="sdpa"`.
4. **Train, tune or select on dev or test.** Spider's splits have disjoint databases and
   that is the point.
5. **Judge correctness by comparing SQL strings.** Executed result sets only.
6. **Retry generation on a parse failure.** A malformed output is a legitimate low-reward
   sample; retrying corrupts the reward distribution.
7. **Drop an agent** to save time or memory. Two agents is a different architecture, not a
   smaller one.
8. **Add** a fourth agent, a multi-turn Refiner loop, retrieval, or a schema-linking model
   beyond the Selector.
9. **Reimplement PPO clipping or KL penalties.** Use `trl`; the only custom code in the RL
   loop is the advantage computation.
10. **Prompt-engineer for absolute accuracy.** It confounds the comparison. Prompts are
    fixed once and shared across conditions.
11. **Download BIRD train databases** — 8 GB, never used. Dev only, in week 9.

## Always

- **Seed everything**: Python `random`, `numpy`, `torch`, and the generation sampler. One
  `set_seed()` covering all four, called once.
- **Dump the full resolved config** to `results/<run_id>/config.yaml` on run start. An
  unlogged run did not happen.
- **`eps = 1e-4`** in every advantage normalisation denominator. Without it, a group where
  every candidate scores the same gives NaN advantages.
- **Record acceptance numbers in `NOTES.md`**, dated and initialled.
- **Report measurements faithfully.** If a benchmark misses its target, say so and
  investigate — do not relax the target or report the best of several runs.

## Ask rather than assume

Decisions that are not the code's to make: 1.5B versus 0.5B, how many seeds, whether a
condition gets cut, the demonstration database, and anything that would weaken a rule above.

## Running things

From the repository root, as modules, relative paths only:

```bash
python scripts/download_data.py --spider-only
python -m src.data.filter --report
pytest tests/test_compare.py
python scripts/benchmark_harness.py --n 100
```

Long runs go in `tmux`, never a notebook cell — the space idle-shuts after 60 minutes.
Checkpoint every 50 steps.

Notebooks are for exploration only. Anything that produces a number in the report lives in
`src/` or `scripts/` and is called from the notebook. Keep them personal: `notebooks/A-*.ipynb`
and `notebooks/B-*.ipynb`, never edit the other person's.

## Scope

Three conditions: `independent`, `counterfactual`, `monolith`.
Not built: `src/credit/mappo.py`, `src/credit/dpo.py`.

The specification, execution plan and task breakdown are not in this repository. They are
shared separately.
