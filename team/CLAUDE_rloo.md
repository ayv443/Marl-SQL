# RLOO run for the Text-to-SQL RL project

I am training the RLOO model for a team project (DPO vs GRPO vs RLOO on Spider, Qwen2.5-Coder + LoRA,
AWS SageMaker, T4 GPU). Eby trains DPO and evaluates all three models; the other two methods are
trained by teammates on their own AWS accounts. My RLOO model goes into the same results table, so it
must be trained with exactly the same code, data, library versions and settings as the others.

The step-by-step plan is ~/Marl-SQL/team/RLOO_PLAN.txt (in the repo, updated with git pull). Follow it
in order.
The general guide is ~/Marl-SQL/how-to-run/readme.txt.
The code is the GitHub branch text2sql-rl of ayv443/Marl-SQL.

## Rules for Claude

- Do NOT edit or delete any file tracked by git in this repo (the code and docs in text2sql-rl/,
  how-to-run/ and team/). No "small fixes", no refactors, no formatting, no new .py files in the repo. If code
  needs to change, the fix comes from Eby and I get it with `git pull`. The scripts themselves
  creating data/, outputs/, logs/ and proof/ inside text2sql-rl/ is expected and fine.
- Exception (only AFTER my RLOO training run has started): edits to text2sql-rl/app.py on my own
  branch `demo` for the Gradio demo (plan section 23.2). Nothing else, and never on text2sql-rl.
  Never RUN app.py while the training is running on this space (shared GPU, could crash training).
- `git status --untracked-files=no` must show no modified files on branch text2sql-rl. If it lists files,
  undo with `git checkout -- .` and tell me.
- Never push to text2sql-rl or main. The only pushes allowed: the proof folder to my own branch
  rloo-run (plan section 15.2) and app.py to my own branch demo (section 23.2). Don't commit the CLAUDE.md link.
- Do NOT pip install, upgrade or downgrade packages beyond the install line in the plan (section 4.3).
  Required versions: trl 0.24.0, transformers 4.56.2, peft 0.17.1, accelerate 1.10.1, datasets 4.0.0.
  Never install TRL 1.x or Transformers 5.x in this environment.
- Do NOT change training settings. The only flags I pass are the ones in the plan
  (`--method rloo --seed 0 --lr 5e-5 --max_steps <agreed>`, plus `--model` / `--instance_type`
  only if agreed with Eby). Don't add, remove or "tune" anything else.
- If the run stops, restart it with the SAME command (it resumes from the last checkpoint); never
  start a second copy while one is running.
- Do NOT start the full training run (plan section 12, nohup python train_rl.py ...) until the plan's
  section 10 is done: tags.json pulled from git, and model size + max_steps agreed with Eby.
- Training runs inside this Studio space (no S3, no SageMaker training jobs, no launch_sagemaker.py).
- Do NOT evaluate on Spider dev or BIRD dev, and don't pick checkpoints. Eby does all evaluation.
- Do NOT delete checkpoints, outputs/ or logs/ (except the smoke-test outputs in plan section 9).
- When something fails: don't work around it in the code. Show me the full error / traceback and
  the command that caused it, so I can send it to Eby. Known issues are in plan section 17.
- Keep WANDB_API_KEY out of any file in the repo and out of messages I send to others. It lives in
  ~/SageMaker/.bashrc_t2s.
- I don't use the Telegram feature. Don't set it up or suggest it; the code works without it.

## Things Claude can do

- Run the commands from the plan, read logs and outputs, and explain them.
- Read-only checks: git status / git log / git fetch, git pull when I say there's an update, the
  version check, reading outputs/*/logs/*_steps.jsonl to get seconds per step, peak GPU memory,
  reward and KL.
- Watch the running training (logs_<method>.txt, W&B numbers, nvidia-smi) and tell me if the signals in
  plan section 13 look wrong.
- Fill in the message templates (plan section 20) and the numbers table (plan section 16).
- Create my own notes or helper files OUTSIDE the repo (e.g. in ~/notes/).
