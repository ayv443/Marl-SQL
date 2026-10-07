HOW TO RUN THE TEXT-TO-SQL RL PROJECT (DPO vs GRPO vs RLOO)
=============================================================

The code is in the folder text2sql-rl/. This file explains, step by step, how to run
all of it on AWS SageMaker.

Who does what (since 8 Oct), three AWS accounts training at the same time:
  - Eby: data, reward check, feasibility run, tags.json, DPO, and ALL evaluation.
  - GRPO teammate: GRPO, on her own AWS account.
  - Aditya: RLOO on his own AWS account, plus the Gradio demo.
  Both teammates use this GitHub branch (git pull for updates) and follow their own plan,
  which is in the repo too: team/GRPO_RETRAIN_PLAN.txt and team/RLOO_PLAN.txt, with
  team/CLAUDE_grpo.md and team/CLAUDE_rloo.md for Claude Code. When a plan changes, push it
  and tell them to git pull: no more sending files. Section 22 of each plan explains how
  Eby receives and checks their model. (The plans are generated from one template on Eby's
  laptop, plan-generator/gen_plans.py, so both stay identical apart from the method.)
  (Where this guide says "Person A" / "Person B", read it with this split in mind.
  Section 17 is the plan if you end up doing everything alone.)

Learning rate for ALL three methods: --lr 5e-5 (agreed 7 Oct; top of the plan's
1e-5 to 5e-5 range, after a pilot run showed 1e-5 barely changed the model).


=============================================================
WHERE WE ARE (updated as we go)
=============================================================
  [x] Data downloaded and prepared (section 4): 8659 -> 7040 kept, 6631 train / 409 val,
      Spider-DK 535 after merging its databases. Spider-Realistic not downloaded (optional).
  [x] Reward check passed (section 5), also after the extract_sql() fix.
  [x] vLLM works on the T4. 50-question feasibility test after the fix: pass@1 0.615,
      pass@8 0.90, 62% mixed, valid SQL 0.76.
  [x] GRPO plan, CLAUDE.md and code bundle sent to the teammate.
  [x] 8 Oct: Aditya added her to GitHub, so no more bundles: both teammates use git pull.
      Aditya trains RLOO (+ demo). Their plans + Claude files are now in team/ in the repo
      (no more WhatsApp files); each links team/CLAUDE_<method>.md as ~/Marl-SQL/CLAUDE.md.
  [x] 6.1 failure check: only 3 of 400 attempts "no SQL found", 94 real SQL errors
      (wrong tables/columns). Reward is fine.
  [x] First full feasibility attempt crashed: a few training prompts are ~8,850 tokens
      (huge schemas), more than vLLM's 8,192 limit. Fixed: sample.py now skips training
      questions with prompts > 2048 tokens (training drops those anyway) and vLLM allows
      32,768 tokens (for BIRD evaluation). git pull, then restart 6.2.
  [x] 899 of 6631 training questions have prompts > 2048 tokens (a few databases with very
      big schemas) and are skipped by sampling and by all three trainers: 5,732 training
      questions remain. Same subset for every method; eval sets are not cut.
  [x] S3 checked (8 Oct): shared course AWS account, bucket writable.
  [x] Decided 8 Oct: NO S3 / training jobs for anyone. Everyone trains inside their own
      Studio space with nohup (section 8), proof with collect_proof.sh (section 19),
      teammates share adapters via Google Drive (section 14).
  [x] 6.2 full feasibility run finished 7 Oct (3h 58m on the T4 space, 5732 questions x 8):
      pass@1 0.573, pass@4 0.786, pass@8 0.845, all_wrong 15.5%, mixed 60.3% (~3460
      questions), all_right 24.2%, valid SQL 0.760. Clear GO (6.3).
      For max_steps: ~3460 mixed questions / 4 per step = ~865 steps per pass over the data.
  [ ] NOW: git pull, set git identity (6.5), make_dpo_pairs.py (6.4), push tags.json (6.5),
      tell both teammates to git pull. Then the DPO smoke test (section 7). (Second attempt failed because
      the crashed first run's vLLM process was still holding the GPU: see section 15,
      "Free memory on device ... is less than desired".)
  [ ] Then: section 7 DPO smoke test (they do GRPO / RLOO smoke tests), agree model size and
      max_steps with both (GRPO and RLOO the same), section 8 launch DPO.
  [ ] Then: get GRPO and RLOO models (section 22 of their plans), sections 10-13.

Contents
  0. What you need
  1. AWS setup (both people, do this first)
  2. GitHub + Weights & Biases setup
  3. Start SageMaker Studio and install everything
  4. Download and prepare the data
  5. Check the reward function
  6. Feasibility run (go / no-go) and DPO pairs
  7. Smoke tests (50 steps of each method)
  8. Full training inside your Studio space (no S3)
  9. Getting the trained models together
  10. Validation curve and picking the best checkpoint
  11. Final evaluation (Spider, variants, BIRD)
  12. Statistics, pass@k and efficiency analysis
  13. Gradio demo
  14. Sharing files between us
  15. Common problems
  16. Saving money
  17. SOLO GUIDE: doing the whole project by yourself
  18. Telegram updates (training progress and errors on your phone)
  19. Proof of training on AWS (log files)


-------------------------------------------------------------
0. WHAT YOU NEED
-------------------------------------------------------------
- An AWS account with SageMaker (both of us have one).
- A GitHub account.
- A Weights & Biases (wandb.ai) account, free is fine.
- About 50 GB of disk on the notebook (datasets + models + checkpoints). 30-40 GB also
  works if you follow the space-saving tips in section 16.

Instance used everywhere: ml.g4dn.2xlarge = 1 x NVIDIA T4 (16 GB), 8 vCPUs, 32 GB RAM.
The T4 has no bf16, so everything runs in fp16.


-------------------------------------------------------------
1. AWS SETUP (BOTH PEOPLE, DO THIS FIRST)
-------------------------------------------------------------
1.1 Pick one region and stay in it the whole project (e.g. us-east-1 or ap-southeast-2).
    (Our course account uses ap-southeast-2, Sydney.)

1.2 Request GPU quota. In the AWS console:
      Service Quotas -> AWS services -> Amazon SageMaker
    Search for and request an increase to 1 for:
      - Studio JupyterLab Apps running on ml.g4dn.2xlarge instance
        (called "ml.g4dn.2xlarge for notebook instance usage" if you use notebook instances)
    Training job quotas are not needed: we train inside Studio (section 8).
    This can take a few days. While waiting, you can do steps 2-5 on a cheap CPU
    space (ml.t3.xlarge). Data prep does not need a GPU.

1.3 IAM role. Studio runs with the course's execution role
    (43008-RL-Spr26-sagemaker-execution-role). Nothing to change: we don't use S3 or
    training jobs, so no extra permissions are needed.


-------------------------------------------------------------
2. GITHUB + WEIGHTS & BIASES SETUP
-------------------------------------------------------------
2.1 The code is on GitHub in the repo ayv443/Marl-SQL, branch text2sql-rl:
      https://github.com/ayv443/Marl-SQL/tree/text2sql-rl
    The repo owner adds the other person as a collaborator (repo Settings -> Collaborators).

2.2 If the repo is private, cloning on SageMaker asks for a password: use a GitHub personal
    access token (GitHub -> Settings -> Developer settings -> Personal access tokens ->
    classic, tick "repo") instead of your GitHub password.

2.3 W&B is free. Sign up with your UTS student email and apply for the free academic
    plan (more storage, and teams are allowed). Then Person B creates a team, invites
    Person A and creates a project called text2sql-rl.
    If the academic plan isn't approved in time, each person just uses their own free
    account (the scripts log to a project called text2sql-rl in your own account) and you
    share run links with each other. Free accounts can't make teams, but that's fine.
    Both get your API key from https://wandb.ai/authorize


-------------------------------------------------------------
3. START SAGEMAKER AND INSTALL EVERYTHING
-------------------------------------------------------------
We use SageMaker Studio (JupyterLab). Your terminal prompt looks like
"sagemaker-user@default:~$" and your home folder is /home/sagemaker-user.
In Studio the WHOLE home folder is kept when the space is stopped, so nothing gets lost.
(If you use a classic notebook instance instead, see 3.8.)

3.1 Create the space:
      SageMaker console -> Studio -> Open Studio -> JupyterLab -> Create JupyterLab space
      Name:            text2sql
      Instance:        ml.g4dn.2xlarge   (ml.t3.xlarge is fine for steps 3-5, no GPU needed)
      Storage:         50 GB (30-40 GB works with the tips in section 16)
    Click "Run space", wait, then "Open JupyterLab".
    You can change the instance type later: stop the space, change it, run it again.

3.2 Open a terminal (File -> New -> Terminal) and get the code:

      cd ~
      git clone https://github.com/ayv443/Marl-SQL.git
      cd Marl-SQL
      git checkout text2sql-rl
      cd text2sql-rl

    Check you're on the right branch:
      git branch              (the line with * must say text2sql-rl)

    Later, to get the newest code:
      cd ~/Marl-SQL && git pull && cd text2sql-rl

3.3 Make the Python environment (once, takes ~10 minutes):

      conda create -p ~/SageMaker/envs/t2s python=3.11 -y
      source activate ~/SageMaker/envs/t2s
      pip install --no-cache-dir -r requirements.txt vllm==0.10.2 matplotlib scipy gradio pandas "sagemaker<3" gdown

    --no-cache-dir stops pip keeping a second copy of every download, which saves a few GB.
    "(t2s)" at the start of the prompt means the environment is switched on.

3.4 Get your keys ready:
      - W&B API key: https://wandb.ai/authorize
      - Telegram bot token and chat id: section 18 (5 minutes, recommended)

3.5 Save your keys in a settings file (once). Paste this whole block into the terminal,
    with your own values in place of the three "paste_..." parts:

mkdir -p ~/SageMaker
cat > ~/SageMaker/.bashrc_t2s <<'EOF'
export WANDB_API_KEY=paste_your_wandb_key_here
export TELEGRAM_BOT_TOKEN=paste_your_bot_token_here
export TELEGRAM_CHAT_ID=paste_your_chat_id_here
cd ~/Marl-SQL/text2sql-rl
[ "$CONDA_PREFIX" = "$HOME/SageMaker/envs/t2s" ] || source activate ~/SageMaker/envs/t2s
EOF

    (The block above is not indented on purpose: the last line must be exactly EOF with
    no spaces in front, otherwise the terminal keeps waiting for more input. If that
    happens, press Ctrl+C and paste it again.)

    What it does: "cat > file <<'EOF' ... EOF" writes the lines in between into the file.
    The last two lines go to the code folder and switch on the environment (only if it
    isn't on already).

    Check it:
      cat ~/SageMaker/.bashrc_t2s          (shows your 5 lines)

    To change something later:
      nano ~/SageMaker/.bashrc_t2s         (edit, Ctrl+O, Enter to save, Ctrl+X to exit)

    This file has your keys in it. It is outside the repo, never copy it into the repo.

3.6 Load the settings. In EVERY new terminal run:
      source ~/SageMaker/.bashrc_t2s

    Or make every new terminal do it automatically (run this ONCE only, running it twice
    adds the line twice):
      echo 'source ~/SageMaker/.bashrc_t2s' >> ~/.bashrc

    Check it worked:
      echo $TELEGRAM_CHAT_ID               (prints your chat id)
      which python                         (prints /home/sagemaker-user/SageMaker/envs/t2s/bin/python)
      wandb login                          (once; it uses WANDB_API_KEY, or paste the key)
      python monitor.py --test             (you get a Telegram message)

    If you see "bash: activate: No such file or directory": the environment was already
    switched on, it's harmless. The [ "$CONDA_PREFIX" ... ] line in 3.5 stops it happening.

    If you see "python: can't open file '.../monitor.py'": you're not in the code folder.
    Run: cd ~/Marl-SQL/text2sql-rl

3.7 Check the GPU (only when the space runs on ml.g4dn.2xlarge):
      nvidia-smi             (should show a Tesla T4)

    Studio idle shutdown: Studio can stop a JupyterLab space that looks idle, and that
    kills anything running in its terminals. Long notebook runs (feasibility sampling,
    evaluation, val_curve, and training itself, section 8) are started with nohup and you
    should keep the browser tab open while they run. If a run gets stopped anyway, start
    the same command again: training resumes from its last checkpoint.

3.8 Using a classic notebook instance instead of Studio (only if you don't use Studio):
    - SageMaker console -> Notebooks -> Notebook instances -> Create, ml.g4dn.2xlarge,
      volume 50 GB, then "Open JupyterLab".
    - Only ~/SageMaker survives a stop/start there, so clone into it:
        cd ~/SageMaker && git clone https://github.com/ayv443/Marl-SQL.git
      and in 3.5 use  cd ~/SageMaker/Marl-SQL/text2sql-rl  instead.
    - ~/.bashrc is reset on every restart, so run the "source" line yourself each time.


-------------------------------------------------------------
4. DOWNLOAD AND PREPARE THE DATA   (both people run this)
-------------------------------------------------------------
4.1 Download:
      bash download_data.sh

    This creates:
      data/spider_data/            Spider train/dev json + databases
      data/testsuite_databases/    Spider test-suite databases (for TS accuracy)
      data/bird_dev/               BIRD dev json, dev.sql, dev_databases/
      data/spider_variants/        Spider-Syn and Spider-DK json
      eval_repos/                  official evaluation code (test-suite, BIRD mini_dev)

    Spider-DK uses 3 extra databases (new_concert_singer, new_orchestra, new_pets_1) that
    come with its repo in eval_repos/Spider-DK/database/. prepare_data.py copies them into
    data/spider_data/test_database/ by itself and prints "copied Spider-DK database ...".
    If it still prints "WARNING spider_dk: skipped ...", check the folder exists:
      ls eval_repos/Spider-DK/database
    If it doesn't, run download_data.sh again (it clones the Spider-DK repo).

    Spider-Realistic must be downloaded by hand: get spider-realistic.json from
    https://zenodo.org/record/5205322 and save it as
    data/spider_variants/spider_realistic.json  (upload it through the JupyterLab file browser).

    If gdown says "too many users have viewed or downloaded this file", open the Google
    Drive link in your browser, download the zip, upload it to the notebook and unzip it
    into data/ (Spider should end up as data/spider_data/...).

4.2 Prepare:
      python prepare_data.py

    Our actual run printed (yours should be the same, same data + same seed):
      Spider train: 8659 questions
      filter: {'empty_result': 1616, 'gold_error': 3, 'kept': 7040}
      train: 6631 questions | val: 409 questions from 10 DBs
    and the eval sets: Spider dev 1034, Spider-Syn 1034, Spider-DK 535, BIRD dev 1534,
    Spider-Realistic 508 (if downloaded). The first time, you also see 3 lines
    "copied Spider-DK database ...".
    No query hit the 5 s timeout. Note for the report: 1616 of 8659 (about 19%) training
    questions were dropped because the gold query returns no rows.
    These numbers are also saved in data/processed/filter_log.json.

    "skipping spider_realistic: ... not found" means you haven't downloaded it yet (4.1).
    It is optional; without it you just have one robustness set fewer.

    Output folder data/processed/ now has:
      train.jsonl, val.jsonl, spider_dev.jsonl, bird_dev.jsonl,
      spider_syn.jsonl, spider_dk.jsonl, spider_realistic.jsonl (if the files existed),
      gold_cache.pkl, filter_log.json

    The split uses a fixed seed (42), so both people get exactly the same train/val split.
    Takes under a minute.


-------------------------------------------------------------
5. CHECK THE REWARD FUNCTION   (Person A)
-------------------------------------------------------------
      python reward.py --split val

    Expected:
      gold vs gold mean reward on val: 1.000 (should be 1.000)
      wrong result  -> 0.0
      syntax error  -> -0.1
      not a SELECT  -> -0.1
      no answer tag -> -0.1
      timeout       -> -0.1  (takes ~5 s)

    If all of these match, the reward is FROZEN. Don't edit reward.py, the prompt in
    common.py or prepare_data.py after this point, or all methods have to be re-run.


-------------------------------------------------------------
6. FEASIBILITY RUN (GO / NO-GO) AND DPO PAIRS   (Person B)
-------------------------------------------------------------
6.1 Quick test on 50 questions first (about 3 minutes, the first time also downloads the
    3 GB model):
      python sample.py --split train --limit 50 --n 8

    vLLM works on the T4. You will see this line, it is harmless (vLLM just uses a
    different attention method, "Using FlexAttention backend"):
      ERROR ... Cannot use FA version 2 ... compute capability >= 8
    Only if vLLM really crashes: add --engine hf to every sample.py / evaluate.py /
    val_curve.py command from now on (slower, but works).

    Check valid_sql_rate in the output. It should be high (roughly 0.8-0.9). If it is low,
    the model is probably writing SQL in a format extract_sql() in common.py doesn't
    recognise. See why attempts failed with:

python - <<'EOF'
import json
from collections import Counter
from common import extract_sql
rows = [json.loads(l) for l in open("data/processed/samples_train.jsonl")]
bad = [c for r in rows for c, rw in zip(r["completions"], r["rewards"]) if rw < 0]
print(len(bad), "failed attempts out of", sum(len(r["completions"]) for r in rows))
print(Counter("no SQL found" if extract_sql(c) is None else "SQL found but errored" for c in bad))
for c in bad[:6]:
    print("-----\n" + c)
EOF

    What happened to us (7 Oct): the first test gave valid_sql_rate 0.29 because the
    untrained model answers with plain "SELECT ..." and ignores the <answer> tags, and the
    first version of extract_sql() only looked inside tags or ``` blocks (245 of 284
    failures were "no SQL found"). extract_sql() now also accepts plain SQL that starts a
    line. This was fixed before any training, so all three methods use the same version.
    Mention it in the report: the reward checks the SQL result, not the answer format.

    Our 50-question test (old extraction): pass@1 0.245, pass@8 0.60, 58% mixed, about
    6.3 attempts per second on the T4. Re-run the test after the fix (git pull first); with
    the new extraction pass@1 and valid_sql_rate go up and the mixed share may change.

6.2 Full run. It first prints "skipping N of 6631 questions with prompt > 2048 tokens":
    those questions have huge database schemas and are dropped by all three trainers too,
    so they are not sampled either. Write N down for the report.
    6631 questions x 8 attempts = about 53,000 attempts, roughly 2-2.5 hours
    with vLLM on the T4 (several hours more with --engine hf),
    so run it in the background so closing the browser doesn't kill it:
      nohup python sample.py --split train --n 8 --temperature 0.8 > logs_sample.txt 2>&1 &
      tail -f logs_sample.txt          (Ctrl+C stops watching, not the job)

    At the end it prints:
      pass@1, pass@4, pass@8, share_all_wrong, share_mixed, share_all_right, valid_sql_rate
    and writes:
      data/processed/samples_train.jsonl   every attempt with its reward
      data/processed/tags.json             each question tagged all_wrong / mixed / all_right
      data/processed/passk_train.json      the numbers above

6.3 GO / NO-GO (decide together):
      GO    if pass@8 is clearly above 0 and share_mixed is a decent chunk (roughly 15%+).
      NO-GO if almost nothing is mixed -> talk about an SFT warm start (see the plan).

6.4 Build the DPO pairs:
      python make_dpo_pairs.py
    Prints e.g. "9000 pairs from 4800 questions (1200 questions used the gold SQL as 'chosen')".
    Output: data/processed/dpo_pairs.jsonl

6.5 Share tags.json with both teammates through git (the .gitignore allows it).
    First time pushing from the Studio space: set your name for commits, and use a GitHub
    personal access token (GitHub -> Settings -> Developer settings -> Personal access
    tokens -> classic, tick "repo") as the password when git push asks:
      git config --global user.name "Eby Kurian"
      git config --global user.email "ebyyykurian@gmail.com"
      git config --global credential.helper store     (remembers the token after the first push)
    Then:
      cd ~/Marl-SQL
      git add text2sql-rl/data/processed/tags.json
      git commit -m "Feasibility tags"
      git push
    Then tell them: "tags.json is pushed, git pull". Commit only tags.json now: they already
    have their own filter_log.json from prepare_data.py, and a committed copy would make
    their git pull complain. filter_log.json, passk_train.json and dpo_pairs.jsonl can be
    committed at the end of the project as a record.


-------------------------------------------------------------
7. SMOKE TESTS (50 STEPS OF EACH METHOD)
-------------------------------------------------------------
Run these directly in the notebook terminal (not as jobs). Open a second terminal and
run "watch -n 2 nvidia-smi" to see GPU memory.

  You (Eby):
      python train_dpo.py --max_steps 50 --lr 5e-5

  GRPO teammate (her plan section 9) and Aditya (his plan section 9), on their accounts:
      python train_rl.py --method grpo --max_steps 50 --lr 5e-5 --only_mixed 0
      python train_rl.py --method rloo --max_steps 50 --lr 5e-5 --only_mixed 0
  They send you their s/step and peak GPU memory, so the three of you can decide model
  size and max_steps together (GRPO and RLOO must use the same max_steps).

For each one write down:
  - seconds per step (shown in the progress bar, e.g. "25.3s/it")
  - peak GPU memory from nvidia-smi
  - that the loss / reward isn't NaN (check the W&B run)

Then work out the time for a full run:
      hours = seconds_per_step x max_steps / 3600
Example: 25 s/step x 600 steps = ~4.2 hours.

DECIDE TOGETHER NOW:
  - model size: if 1.5B is too slow or runs out of memory, switch ALL THREE methods to
    0.5B by adding  --model Qwen/Qwen2.5-Coder-0.5B-Instruct  to every training command,
    and change MODEL_NAME in common.py so sampling/eval use it too.
    If you switch, redo step 6 with the 0.5B model.
  - max_steps for GRPO and RLOO (must be the SAME for both, so they generate the same
    number of completions). Each step = 4 questions x 4 completions = 16 completions.
    A rough guide: number of mixed questions / 4 = one pass over the data.

Smoke test output goes to outputs/<method>-1.5b-s0/. Delete it before the real runs:
      rm -rf outputs/


-------------------------------------------------------------
8. FULL TRAINING (INSIDE YOUR STUDIO SPACE, NO S3)
-------------------------------------------------------------
Decided 8 Oct: we don't use SageMaker training jobs or S3. Each person trains their method
in their own Studio JupyterLab space on a T4, in the background with nohup, exactly like
the feasibility run. (launch_sagemaker.py is still in the code in case jobs + S3 are ever
needed, but nobody uses it now.)

8.1 Before starting:
      - The space runs on ml.g4dn.2xlarge (or ml.g4dn.xlarge) and the GPU is empty:
          nvidia-smi                 (no processes, memory near 0 MiB)
      - Settings loaded and code up to date:
          source ~/SageMaker/.bashrc_t2s
          cd ~/Marl-SQL && git pull && git status --untracked-files=no && cd text2sql-rl
        (git status: "up to date with 'origin/text2sql-rl'", no modified files)
      - The smoke-test output is deleted (rm -rf outputs/), and wandb login works.

8.2 Start the run (keep the command exactly like this):

    You (Eby):
      nohup python train_dpo.py --lr 5e-5 --seed 0 > logs_dpo.txt 2>&1 &

    GRPO teammate and Aditya (section 12 of their plans, on their accounts, SAME max_steps):
      nohup python train_rl.py --method grpo --seed 0 --max_steps 600 --lr 5e-5 > logs_grpo.txt 2>&1 &
      nohup python train_rl.py --method rloo --seed 0 --max_steps 600 --lr 5e-5 > logs_rloo.txt 2>&1 &

    Change 600 to whatever was agreed in section 7. For extra seeds use --seed 1, 2.
    Telegram (if set up) sends progress every 50 steps (--notify_every to change it).

8.3 Watch it:
      tail -f logs_dpo.txt         (Ctrl+C stops watching, not the run)
    One line per step:  step 120/600 | 29.8 s/step | loss=..., reward=..., kl=...
    and "checkpoint saved: checkpoint-100" every 100 steps. W&B shows the charts
    (project text2sql-rl, runs dpo-1.5b-s0, grpo-1.5b-s0, rloo-1.5b-s0).

    GRPO / RLOO, what to look at in W&B:
      train/reward                          should go up
      train/rewards/valid_sql_reward/mean   valid SQL rate, should go up and stay high
      train/kl                              should rise slowly, not explode
      train/completions/mean_length         should stay roughly stable
      train/frac_reward_zero_std            share of groups where all 4 got the same reward;
                                            if it climbs toward 1.0, learning has stalled
      train/entropy                         a sudden crash = no more exploration
    DPO:
      train/rewards/margins, train/rewards/accuracies should go up, train/loss down.

    If several of these go bad at once, stop the run (kill <PID>, find it with
    ps aux | grep train_) and tell the others before changing anything.

8.4 If it stops (space stopped, Studio idle shutdown, a crash, you closed it by mistake):
    checkpoints are saved every 100 steps in outputs/<run>/checkpoint-N on the space's
    disk, which survives a stop. Make sure nothing is still running (nvidia-smi, then
    ps aux | grep -E "train_|EngineCore"), then run the SAME command again. It continues
    from the last checkpoint (the log says "training started at step 300 ..."). You lose at
    most the steps since the last checkpoint. Never run two copies at the same time.

    To avoid stops: keep the browser tab open with tail -f running, and don't stop the
    space until the log says FINISHED.

8.5 While it trains the GPU is busy: don't start evaluation or other GPU work on the same
    space. CPU work (writing, analysis of earlier results) is fine.

8.6 When it finishes (log: "FINISHED after ..."), the run is in:
      outputs/<run>/checkpoint-100, checkpoint-200, ..., final/, logs/
    Keep all checkpoints (needed for the validation curve). About 1.5 GB per run.
    Then collect the proof (section 19): bash collect_proof.sh <run>


-------------------------------------------------------------
9. GETTING THE TRAINED MODELS TOGETHER
-------------------------------------------------------------
Your DPO run is already on your space in outputs/dpo-1.5b-s0.

GRPO and RLOO are on the teammates' spaces. Each of them:
  - pushes their proof folder to their own branch (grpo-run, rloo-run), and
  - shares a Google Drive link to a .tar.gz of their adapters (section 14).
Get and check them with section 22 of their plans:
      cd ~/Marl-SQL
      git fetch origin grpo-run
      git checkout origin/grpo-run -- text2sql-rl/proof/grpo-1.5b-s0
      cd text2sql-rl && mkdir -p outputs
      gdown --fuzzy "<Google Drive link>" -O /tmp/grpo.tar.gz
      tar xzf /tmp/grpo.tar.gz -C outputs
    (same with rloo), then the checks in 22.4 and 22.5 of their plans.

The folder structure you should end up with:
      outputs/dpo-1.5b-s0/checkpoint-100/  ...  final/
      outputs/grpo-1.5b-s0/checkpoint-100/ ...  final/
      outputs/rloo-1.5b-s0/checkpoint-100/ ...  final/


-------------------------------------------------------------
10. VALIDATION CURVE AND PICKING THE BEST CHECKPOINT   (Person B)
-------------------------------------------------------------
      nohup python val_curve.py --runs outputs/dpo-1.5b-s0 outputs/grpo-1.5b-s0 outputs/rloo-1.5b-s0 \
            > logs_curve.txt 2>&1 &

It evaluates the untrained model plus every checkpoint on the val split (~400 questions,
a few minutes each) and writes:
      results/val_curve.csv
      results/val_curve.png     <- the shared training graph for the report

It skips checkpoints it has already done, so it's safe to re-run.

Best checkpoint = the row with the highest val_EX for each method in val_curve.csv.
NEVER pick checkpoints using Spider dev or BIRD dev.


-------------------------------------------------------------
11. FINAL EVALUATION   (Person B, on one machine, with nothing else running)
-------------------------------------------------------------
11.1 Our own EX + valid SQL rate. Replace checkpoint-XXX with the best one from step 10.
     Run for every model and every split:

      for SPLIT in spider_dev spider_syn spider_dk spider_realistic bird_dev; do
        python evaluate.py --split $SPLIT --tag base
        python evaluate.py --split $SPLIT --tag dpo  --adapter outputs/dpo-1.5b-s0/checkpoint-XXX
        python evaluate.py --split $SPLIT --tag grpo --adapter outputs/grpo-1.5b-s0/checkpoint-XXX
        python evaluate.py --split $SPLIT --tag rloo --adapter outputs/rloo-1.5b-s0/checkpoint-XXX
      done

     (Put that in a file run_eval.sh and start it with nohup, it takes a while.)

     Each run writes results/<tag>/<split>/:
       metrics.json        EX, valid_sql_rate, EX by difficulty (BIRD: simple/moderate/challenging)
       per_question.jsonl  every question with predicted SQL and correct/valid
       pred.txt            for the Spider official script
       predict_dev.json    for the BIRD official scripts

11.2 Official Spider EX and Test-Suite accuracy (also gives EX by easy/medium/hard/extra):
      bash run_official_eval.sh base
      bash run_official_eval.sh dpo
      bash run_official_eval.sh grpo
      bash run_official_eval.sh rloo
     Results are printed and saved in results/<tag>/<split>/official_ex.txt and official_ts.txt.
     Use the "execution" row, "all" column.
     Spider-DK only gets EX, no TS: it has 3 extra databases (new_concert_singer,
     new_orchestra, new_pets_1) that the test-suite databases don't include. Say so in the report.

11.3 BIRD official EX, Soft-F1 and R-VES:
     Open eval_repos/mini_dev/evaluation/run_evaluation.sh and set the paths at the top to:
       predicted sql json : results/<tag>/bird_dev/predict_dev.json
       ground truth sql   : data/bird_dev/dev.sql
       db root            : data/bird_dev/dev_databases/
       difficulty json    : data/bird_dev/dev.json
       dialect            : SQLite
     then run it once per model:
       cd eval_repos/mini_dev/evaluation && sh run_evaluation.sh && cd -
     R-VES measures speed, so run it when NO training or other eval is running on
     that machine, and do all four models in the same session.

11.4 Fill in the two results tables from the plan using these numbers.
     Save the final table as results/results_table.md (the Gradio demo shows it).


-------------------------------------------------------------
12. STATISTICS, PASS@K AND EFFICIENCY
-------------------------------------------------------------
Bootstrap 95% CIs + McNemar tests (Person B):
      python analysis.py compare --split spider_dev --tags base dpo grpo rloo
      python analysis.py compare --split bird_dev   --tags base dpo grpo rloo
  p < 0.05 (marked with *) means the difference between two models is probably real.

Commonly-correct efficiency analysis (Person B, idle machine):
      python analysis.py efficiency --split bird_dev --tags base dpo grpo rloo
  Prints how many questions every model got right (report this number) and the
  average gold-time / model-time ratio (above 1 = faster than the gold query).

pass@k for k = 1, 4, 8, 16 (Person A, for each model):
      python sample.py --split spider_dev --n 16 --tag base
      python sample.py --split spider_dev --n 16 --tag grpo --adapter outputs/grpo-1.5b-s0/checkpoint-XXX
      (same for dpo and rloo)
  Results: data/processed/passk_spider_dev_<tag>.json

Compute cost table: take seconds/step, total time and peak GPU memory from each run's
proof/<run>/summary.txt (and W&B), plus the Studio instance type and its hourly price.


-------------------------------------------------------------
13. GRADIO DEMO   (Person A)
-------------------------------------------------------------
      python app.py --dpo  outputs/dpo-1.5b-s0/checkpoint-XXX \
                    --grpo outputs/grpo-1.5b-s0/checkpoint-XXX \
                    --rloo outputs/rloo-1.5b-s0/checkpoint-XXX

It prints a public link (https://xxxx.gradio.live) that works for 72 hours. Open it in
your browser. Pick a database, type a question, optionally paste a gold SQL to get the
check/cross marks. The "Results" tab shows results/val_curve.png and results/results_table.md
if they exist.


-------------------------------------------------------------
14. SHARING FILES BETWEEN US
-------------------------------------------------------------
Code, fixes, tags.json: GitHub. Eby pushes to text2sql-rl, the others git pull.
Training proof: each trainer pushes proof/<run>/ to their own branch (grpo-run, rloo-run).

Model checkpoints (on different Studio spaces, no S3): Google Drive.
  The trainer packs only the adapter files (no optimizer state, about 0.5 GB per run):
      cd ~/Marl-SQL/text2sql-rl/outputs
      tar czf ~/grpo-1.5b-s0-adapters.tar.gz --exclude=optimizer.pt --exclude=scheduler.pt \
          --exclude=rng_state.pth --exclude=scaler.pt --exclude=training_args.bin grpo-1.5b-s0
  Then in the JupyterLab file browser (home folder): right-click the .tar.gz -> Download,
  upload it to Google Drive, Share -> General access "Anyone with the link" -> Copy link,
  and send the link.

  Eby (on the evaluation space):
      cd ~/Marl-SQL/text2sql-rl && mkdir -p outputs
      gdown --fuzzy "<link>" -O /tmp/grpo.tar.gz
      tar xzf /tmp/grpo.tar.gz -C outputs
  If gdown fails, download it in the browser and upload it with the file browser instead.
  When done, the trainer can switch the Drive link back to "Restricted".


-------------------------------------------------------------
15. COMMON PROBLEMS
-------------------------------------------------------------
"CUDA out of memory" in GRPO/RLOO
    -> add --load_4bit 1, or switch all methods to the 0.5B model (step 7).
       Keep --num_generations 4, it's fixed in the plan.

Loss is NaN
    -> lower the learning rate: --lr 5e-6. If still NaN, add --load_4bit 1 (QLoRA).

vLLM error on the T4
    -> use --engine hf in sample.py, evaluate.py and val_curve.py.

"dropped N examples with prompt > 2048 tokens"  /  "skipping N of 6631 questions ..."
    -> normal, a few Spider databases have huge schemas. Same cut-off everywhere.

"AttributeError: module 'sagemaker' has no attribute 'Session'"
    -> you are not in the t2s environment, so Studio's default Python (SageMaker SDK v3)
       ran the command. Run "source ~/SageMaker/.bashrc_t2s" first; t2s has SDK 2.x
       (check: python -c "import sagemaker; print(sagemaker.__version__)" -> 2.257.x).
       The "SageMakerV2DeprecationWarning" lines are harmless; hide them with:
         echo 'export SAGEMAKER_SUPPRESS_V2_WARNING=1' >> ~/SageMaker/.bashrc_t2s

"I can't access S3" / checking S3 works
    -> our AWS account is the university's shared course account (role
       43008-RL-Spr26-sagemaker-execution-role): we can't change IAM, but S3 works.
       Test writing to our folder in the SageMaker bucket:
         echo test > /tmp/t.txt
         aws s3 cp /tmp/t.txt s3://sagemaker-ap-southeast-2-443142193439/text2sql/test.txt
         aws s3 rm s3://sagemaker-ap-southeast-2-443142193439/text2sql/test.txt
       Other buckets in "aws s3 ls" belong to other students / the course: never touch them,
       and only use the text2sql/ folder in sagemaker-ap-southeast-2-443142193439.
       GPU quota in a shared account is shared with the whole class, so capacity can run out.
       (Since 8 Oct we don't use S3 at all: training runs inside Studio, section 8.)

"ValueError: Free memory on device (0.97/14.56 GiB) on startup is less than desired
 GPU memory utilization"
    -> something else is still using the GPU, usually a leftover vLLM "EngineCore" process
       from a run that crashed, or the same command started twice. Check and stop it:
         nvidia-smi
         ps aux | grep -E "sample.py|evaluate.py|EngineCore" | grep -v grep
         pkill -f sample.py; pkill -f evaluate.py; pkill -f EngineCore
         sleep 5; nvidia-smi        (should now show 14+ GB free)
       If memory is still used, kill -9 <PID> for each process ps showed. Then start again.
       Make it a habit: check nvidia-smi before starting any GPU run.
       (Since 7 Oct the scripts stop vLLM's EngineCore themselves when they crash, so this
       should not happen any more with the current code.)

"ValueError: The decoder prompt (length ...) is longer than the maximum model length"
    -> you have old code (vLLM limit 8192). git pull: the limit is now 32768 and sample.py
       skips over-long training prompts.

"wandb: ERROR api_key not configured" or no run in W&B
    -> run "wandb login" once in the space (or source ~/SageMaker/.bashrc_t2s, which sets
       WANDB_API_KEY), then start again. Check: echo ${WANDB_API_KEY:0:6}

"telegram message failed after 3 tries: ... Connection reset by peer"
    -> a short network problem between AWS and Telegram. The script tries 3 times and then
       carries on; the run itself is not affected. If it happens every time, run
       "python monitor.py --test" to check the token and chat id.

No Telegram messages
    -> run "python monitor.py --test". If that works but a run sends nothing, the terminal
       you started it from didn't have the Telegram settings: source ~/SageMaker/.bashrc_t2s
       before starting the next run.

Training stopped in the middle (space stopped, idle shutdown, crash)
    -> run the same nohup command again, it resumes from the last checkpoint (section 8.4).

Can't start the space on ml.g4dn.2xlarge (no capacity)
    -> try ml.g4dn.xlarge (same T4 GPU), or try again later; the course account's GPU quota
       is shared with other students.

pip install fails because of versions
    -> make sure you are in the t2s conda environment (step 3.3), not the default one.

The notebook was stopped and the environment is gone
    -> the conda env is in ~/SageMaker/envs/t2s, just run
       "source activate ~/SageMaker/envs/t2s" again.

KeyError: 'train_1234' (or similar) from reward.py
    -> data/processed is out of date. Re-run prepare_data.py.


-------------------------------------------------------------
16. SAVING MONEY
-------------------------------------------------------------
- STOP the Studio space whenever you're not using it (Studio -> JupyterLab -> your
  space -> Stop), but NEVER while a training or evaluation run is going (section 8.4).
  A running g4dn.2xlarge space costs money every hour even when idle. Files are kept.
- Use a cheap ml.t3.xlarge space for anything that doesn't need a GPU (data prep,
  analysis.py, writing). Stop the space, change the instance, run it again.
- Set an AWS budget alert if the course account allows it: Billing -> Budgets.

Disk space in the Studio space (rough sizes):
    Python environment ............ 8-10 GB
    Qwen 1.5B model ............... ~3 GB
    Spider + test-suite + BIRD .... ~6-8 GB
    Your training run ............. ~1.5 GB (checkpoints with optimizer state)
    Teammates' adapters ........... ~0.5 GB each
    Everything else ............... ~1-2 GB
    Total ......................... ~25-30 GB
  To save space:
    - always pip install with --no-cache-dir
    - delete checkpoints only after the validation curve and final evaluation are done
    - check free space with: df -h ~

=============================================================
17. SOLO GUIDE: DOING THE WHOLE PROJECT BY YOURSELF
=============================================================
Use this if the others drop out. Everything runs in YOUR Studio space. The steps are
the same as sections 1-13, just in one order, with nothing to send to anyone. Section 14
(sharing) doesn't apply: all checkpoints are already on your space.

Without S3 / training jobs, the three trainings run one after another on your one GPU
(each roughly 4-8 hours for 1.5B), so plan for 1-2 days of GPU time and use the waiting
time for the report.


17.1 QUOTA
----------
You only need the Studio JupyterLab quota for ml.g4dn.2xlarge (or xlarge). Training jobs
are not used.


17.2 IF YOU ARE TAKING OVER HALFWAY
-----------------------------------
Check what already exists before redoing anything:

  a) git pull. If data/processed/tags.json and dpo_pairs.jsonl are in the repo,
     the feasibility run is done: skip section 6. You still need to run
     download_data.sh and prepare_data.py yourself (same seed = same split).

  b) Ask your partner (or check W&B) whether any training finished. If they can send
     checkpoints (section 14), put them in outputs/<run name>/ and skip that training.
     If you can't get them, retrain that method on your account.
     The run names stay the same (e.g. grpo-1.5b-s0) so nothing else changes.

  c) If the model size was already decided (1.5B or 0.5B), keep it. Mixing sizes
     makes the comparison invalid.

  d) Results from your partner's evaluation can't be mixed with yours for R-VES (timing
     depends on the machine). Re-run section 11 for all four models on your notebook.
     EX / TS / Soft-F1 don't depend on the machine, but re-running is simpler.


17.3 ORDER OF WORK (ONE PERSON)
-------------------------------
Each line gives the section to follow. [GPU] = needs the g4dn space,
[CPU] = a cheap t3 space is fine.

  Step 1   [CPU]  Section 1: Studio GPU quota.
  Step 2   [CPU]  Section 2: your own private GitHub repo + W&B project (no team needed).
  Step 3   [CPU]  Section 3: notebook + environment. Use t3.xlarge until GPU quota arrives.
  Step 4   [CPU]  Section 4: download_data.sh, prepare_data.py.
  Step 5   [CPU]  Section 5: python reward.py --split val. Freeze reward.py.
  Step 6   [GPU]  Section 6: sample.py feasibility run (nohup), go/no-go, make_dpo_pairs.py.
                  While it runs: start the report (intro, related work, data section).
  Step 7   [GPU]  Section 7: smoke tests, one after another:
                    python train_dpo.py --max_steps 50 --lr 5e-5
                    python train_rl.py --method grpo --max_steps 50 --lr 5e-5
                    python train_rl.py --method rloo --max_steps 50 --lr 5e-5
                  Pick the model size and max_steps. Then rm -rf outputs/
  Step 8   [GPU]  Section 8: train DPO in the space (nohup), then GRPO, then RLOO, one after
                  another (same command style, the same --max_steps for GRPO and RLOO):
                    nohup python train_dpo.py --lr 5e-5 --seed 0 > logs_dpo.txt 2>&1 &
                    nohup python train_rl.py --method grpo --seed 0 --max_steps <N> --lr 5e-5 > logs_grpo.txt 2>&1 &
                    nohup python train_rl.py --method rloo --seed 0 --max_steps <N> --lr 5e-5 > logs_rloo.txt 2>&1 &
                  Start the next one only after the previous log says FINISHED.
  Step 9   [GPU]  Between trainings (or before the first): evaluate the untrained model once:
                    for SPLIT in val spider_dev spider_syn spider_dk spider_realistic bird_dev; do
                      python evaluate.py --split $SPLIT --tag base
                    done
                    bash run_official_eval.sh base
  Step 10  [CPU]  While training runs: check the log and W&B a few times (section 8.3), write
                  the method sections for DPO, GRPO and RLOO.
  Step 11  [CPU]  After each run: bash collect_proof.sh <run> (section 19).
  Step 12         (no download step: all checkpoints are already in outputs/)
  Step 13  [GPU]  Section 10: val_curve.py, pick the best checkpoint of each method.
  Step 14  [GPU]  Section 11: evaluate dpo, grpo, rloo on all splits (base is already done),
                  official Spider script, BIRD scripts (all four models in one session for R-VES).
  Step 15  [CPU]  Section 12: analysis.py compare (both splits) + efficiency. Fill the tables.
  Step 16         Extras only if there's time left (see 17.5).
  Step 17  [GPU]  Section 13: Gradio demo, record a short screen video as a backup.
  Step 18         Write results + discussion, final check of the report.


17.4 ROUGH TIMELINE ALONE
-------------------------
These are rough guesses for 1.5B. Use your own smoke test numbers.

  Day 1      Steps 1-5 (most of it is waiting for downloads / quota).
  Day 2      Step 6 feasibility run (a few hours, mostly waiting) + start report.
  Day 3      Step 7 smoke tests, decide size + max_steps, Steps 8-10.
  Days 4-6   The three trainings, one after another in your space (Steps 8-11).
  Day 7      Steps 12-14 (evaluation, mostly waiting on the GPU).
  Day 8      Step 15 + tables + plots.
  Day 9+     Extras, demo, report.

If you have less time than this, use the cut list below.


17.5 WHAT TO CUT IF YOU'RE SHORT ON TIME
----------------------------------------
Must have (this is the actual assignment):
  - Phase 1 numbers: filter counts, pass@1, pass@8, share of mixed questions
  - DPO, GRPO and RLOO, ONE seed each, same model size
  - The validation curve (results/val_curve.png)
  - Spider dev EX + TS and BIRD dev EX for base, DPO, GRPO and RLOO
  - Valid SQL rate

Cut first (in this order, top = cut first):
  1. DPO -> GRPO run (Tier 2)
  2. Zero-variance analysis with 8 samples (Tier 2). Just show the frac_reward_zero_std
     chart from W&B instead, which you already have.
  3. Extra seeds. Say "one seed per method, bootstrap CIs instead" in the report.
  4. pass@k at k = 16. Use --n 8 instead, which is half the time.
  5. Spider-DK / Spider-Syn / Spider-Realistic. Keep at least one (Spider-Syn) if possible.
  6. Error analysis: do 20 failures per model instead of 50.

Keep even when short on time (they're cheap and add marks):
  - analysis.py compare (CIs + McNemar), takes a few seconds
  - The compute cost table, from the runs' summary.json / W&B
  - Soft-F1 and R-VES, which come out of the same BIRD script run as EX

Make the runs shorter if needed:
  - Lower --max_steps (same value for GRPO and RLOO!)
  - DPO: --epochs 0.5
  - Switch to 0.5B at the smoke test stage (never halfway through)
  - --save_steps 200 means fewer checkpoints, so val_curve.py is quicker


17.6 SOLO CHECKLIST (tick these off)
------------------------------------
  [ ] Quota approved for a g4dn Studio space
  [ ] Repo on GitHub, W&B logged in, Telegram test message received
  [ ] data/processed created, filter numbers written down
  [ ] reward.py check prints 1.000, reward frozen
  [ ] Feasibility done, GO decision, tags.json + dpo_pairs.jsonl made
  [ ] Smoke tests done, model size + max_steps decided
  [ ] DPO run finished (log: FINISHED)
  [ ] GRPO run finished
  [ ] RLOO run finished
  [ ] Base model evaluated on all splits
  [ ] val_curve.png made, best checkpoints picked
  [ ] DPO / GRPO / RLOO evaluated on all splits
  [ ] Official Spider EX + TS done for all four
  [ ] BIRD EX / Soft-F1 / R-VES done for all four (one session, idle machine)
  [ ] analysis.py compare + efficiency done
  [ ] Results tables filled in
  [ ] Demo works (and a backup video recorded)
  [ ] Proof collected for every run (section 19)
  [ ] Report written
  [ ] Space STOPPED



-------------------------------------------------------------
18. TELEGRAM UPDATES (TRAINING PROGRESS AND ERRORS ON YOUR PHONE)
-------------------------------------------------------------
Every script sends you plain text messages when it starts, finishes, fails or gets stopped.
Training also sends progress every 50 steps and a message for every checkpoint.
It is optional: if the two settings below are not set, nothing is sent and everything
else works the same.

18.1 Create a bot (once):
     1. In Telegram, search for @BotFather and open the chat.
     2. Send:  /newbot
     3. Give it a name (e.g. text2sql training) and a username ending in "bot"
        (e.g. eby_text2sql_bot).
     4. BotFather replies with a token like 1234567890:AAH...  This is TELEGRAM_BOT_TOKEN.
        Keep it private (anyone with it can send messages as your bot).

18.2 Get your chat id (once):
     1. Open a chat with your new bot and send it any message, e.g. "hi".
     2. In a browser open:
          https://api.telegram.org/bot<TOKEN>/getUpdates
        (replace <TOKEN> with your token, keep the word "bot" in front of it)
     3. Find "chat":{"id":123456789 ...  That number is TELEGRAM_CHAT_ID.
        If the page shows "result":[] send the bot another message and refresh.

18.3 Set them in the notebook terminal (and in ~/SageMaker/.bashrc_t2s):
      export TELEGRAM_BOT_TOKEN=1234567890:AAH...
      export TELEGRAM_CHAT_ID=123456789

18.4 Test:
      python monitor.py --test
     You should get "test message from SageMaker notebook ... gpu: Tesla T4 ...".

18.5 What you will receive (example):
      [grpo-1.5b-s0] STARTED
      where: <hostname of your Studio space>
      gpu: Tesla T4
      [grpo-1.5b-s0] training started at step 0 of 600
      [grpo-1.5b-s0] step 50/600 (8%)
      24.8 s/step, elapsed 0h 20m 40s, remaining about 3h 47m 20s
      gpu peak memory: 11.2 GB
      loss=0.01, reward=0.31, rewards/valid_sql_reward/mean=0.92, kl=0.002, ...
      [grpo-1.5b-s0] checkpoint saved at step 100
      [grpo-1.5b-s0] WARNING: loss is NaN at step 230. Consider stopping the run.
      [grpo-1.5b-s0] FAILED after 1h 2m 3s ... error: <last part of the error message>
      [grpo-1.5b-s0] STOPPED: SIGTERM received (job stopped or spot instance taken back)
                     (in Studio: the run was killed or the space stopped -> section 8.4)
      [grpo-1.5b-s0] FINISHED after 4h 1m 10s ... last metrics: ...

     Scripts that send messages: train_rl.py, train_dpo.py, sample.py, evaluate.py,
     val_curve.py, prepare_data.py.
     val_curve.py runs evaluate.py many times; those inner runs don't message you, only
     the final result does.

     STOPPED: start the same command again, it resumes from the last checkpoint (8.4).


-------------------------------------------------------------
19. PROOF OF TRAINING ON AWS (LOG FILES)
-------------------------------------------------------------
Every run writes three files into outputs/<run name>/logs/ (training runs) or
text2sql-rl/logs/ (sample.py, evaluate.py, val_curve.py, prepare_data.py):

  <run>_<time>.log            readable log with a timestamp on every line:
                              - where it ran: hostname, the SageMaker Studio space
                                (domain, space, app), python / torch versions, the command
                              - full nvidia-smi output (shows the Tesla T4)
                              - all settings / hyperparameters
                              - one line per training step: step, seconds/step, loss,
                                reward, KL, valid SQL rate, ...
                              - every checkpoint saved, errors with the full traceback,
                                and how long the run took
  <run>_<time>_steps.jsonl    one JSON line per training step with ALL metrics, time,
                              seconds per step and peak GPU memory (good for plots)
  <run>_<time>_summary.json   start/end time, duration, status (FINISHED / FAILED /
                              STOPPED), environment, settings and final metrics

A run that was stopped and resumed has one set of files per start, all in the same folder.
Training logs every single step (logging_steps=1), in these files and in W&B.

19.1 Collect everything for one run into proof/<run name>/ (in text2sql-rl/, after the run
     has finished):
      bash collect_proof.sh dpo-1.5b-s0

     The folder then contains:
       logs/                      the three files above (all starts of the run)
       checkpoints/<name>/        trainer_state.json (the trainer's own record of every
                                  logged step) + adapter_config.json, for every checkpoint
       studio_space.json          which SageMaker Studio domain / space / app it ran in
       aws_identity.json          the AWS account and role (the course account)
       nvidia_smi.txt             the GPU
       git_commit.txt             the exact code commit (and that no code was modified)
       versions.txt               trl / transformers / peft / torch versions
       summary.txt                short readable summary of all of the above

     The teammates run the same for grpo-1.5b-s0 / rloo-1.5b-s0 and push it to their own
     branch; Eby takes it from there (section 9).

19.2 Extra proof you can screenshot for the report / appendix:
     - Studio -> JupyterLab -> your space running on ml.g4dn.2xlarge (instance shown)
     - the terminal with the log showing the steps and "FINISHED after ..."
     - W&B run page -> System tab (GPU usage over time) and Overview (host name, command)
     - Billing (if the course account lets you see it): SageMaker Studio charges

19.3 Keep the proof safe: commit it to the repo (small text files, allowed by .gitignore):
      cd ~/Marl-SQL
      git add text2sql-rl/proof/
      git commit -m "Training proof for dpo, grpo, rloo"
      git push
