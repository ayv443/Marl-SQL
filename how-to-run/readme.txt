HOW TO RUN THE TEXT-TO-SQL RL PROJECT (DPO vs GRPO vs RLOO)
=============================================================

The code is in the folder text2sql-rl/. This file explains, step by step, how to run
all of it on AWS SageMaker. Person A = GRPO + RLOO, Person B = data + DPO + evaluation
(see text2sql-rl/TEAM_PLAN.md for who does what and when).

Contents
  0. What you need
  1. AWS setup (both people, do this first)
  2. GitHub + Weights & Biases setup
  3. Start the notebook and install everything
  4. Download and prepare the data
  5. Check the reward function
  6. Feasibility run (go / no-go) and DPO pairs
  7. Smoke tests (50 steps of each method)
  8. Full training as SageMaker training jobs
  9. Getting the trained models back
  10. Validation curve and picking the best checkpoint
  11. Final evaluation (Spider, variants, BIRD)
  12. Statistics, pass@k and efficiency analysis
  13. Gradio demo
  14. Sharing files between the two of us
  15. Common problems
  16. Saving money
  17. SOLO GUIDE: doing the whole project by yourself


-------------------------------------------------------------
0. WHAT YOU NEED
-------------------------------------------------------------
- An AWS account with SageMaker (both of us have one).
- A GitHub account.
- A Weights & Biases (wandb.ai) account, free is fine.
- About 100 GB of disk on the notebook (datasets + models + checkpoints).

Instance used everywhere: ml.g4dn.2xlarge = 1 x NVIDIA T4 (16 GB), 8 vCPUs, 32 GB RAM.
The T4 has no bf16, so everything runs in fp16.


-------------------------------------------------------------
1. AWS SETUP (BOTH PEOPLE, DO THIS FIRST)
-------------------------------------------------------------
1.1 Pick one region and stay in it the whole project (e.g. us-east-1 or ap-southeast-2).
    Everything (notebook, S3 bucket, training jobs) must be in the same region.

1.2 Request GPU quota. In the AWS console:
      Service Quotas -> AWS services -> Amazon SageMaker
    Search for and request an increase to 1 (or 2 if you want two jobs at once) for:
      - ml.g4dn.2xlarge for notebook instance usage
      - ml.g4dn.2xlarge for training job usage
      - ml.g4dn.2xlarge for spot training job usage
    This can take a few days. While waiting, you can do steps 2-4 on a cheap CPU
    notebook (ml.t3.xlarge). Data prep does not need a GPU.

1.3 IAM role. When you create the notebook instance, choose
    "Create a new role" (AmazonSageMaker-ExecutionRole-...). Give it access to
    "Any S3 bucket". This role already has permission to launch training jobs.


-------------------------------------------------------------
2. GITHUB + WEIGHTS & BIASES SETUP
-------------------------------------------------------------
2.1 Person A: create a PRIVATE GitHub repo (e.g. text2sql-rl), then on your laptop:
      cd text2sql-rl
      git init
      git add .
      git commit -m "initial code"
      git branch -M main
      git remote add origin https://github.com/<you>/text2sql-rl.git
      git push -u origin main
    Add Person B as a collaborator (repo Settings -> Collaborators).

2.2 To clone a private repo on SageMaker you need a GitHub personal access token
    (GitHub -> Settings -> Developer settings -> Personal access tokens -> classic, tick "repo").
    Use the token as the password when git asks.

2.3 W&B is free. Sign up with your UTS student email and apply for the free academic
    plan (more storage, and teams are allowed). Then Person B creates a team, invites
    Person A and creates a project called text2sql-rl.
    If the academic plan isn't approved in time, each person just uses their own free
    account (the scripts log to a project called text2sql-rl in your own account) and you
    share run links with each other. Free accounts can't make teams, but that's fine.
    Both get your API key from https://wandb.ai/authorize


-------------------------------------------------------------
3. START THE NOTEBOOK AND INSTALL EVERYTHING
-------------------------------------------------------------
3.1 SageMaker console -> Notebooks -> Notebook instances -> Create notebook instance
      Name:            text2sql-<yourname>
      Instance type:   ml.g4dn.2xlarge   (ml.t3.xlarge while waiting for quota)
      Volume size:     100 GB
      IAM role:        the one from step 1.3
    Wait until it says InService, then click "Open JupyterLab".

3.2 Open a Terminal in JupyterLab (File -> New -> Terminal).
    IMPORTANT: only the ~/SageMaker folder survives when the notebook is stopped.
    Keep everything inside it.

      cd ~/SageMaker
      git clone https://github.com/<you>/text2sql-rl.git
      cd text2sql-rl

3.3 Make a Python environment inside ~/SageMaker so it survives restarts:

      conda create -p ~/SageMaker/envs/t2s python=3.11 -y
      source activate ~/SageMaker/envs/t2s
      pip install -r requirements.txt vllm==0.10.2 matplotlib scipy gradio pandas "sagemaker<3" gdown

    Every time you open a new terminal later, run:
      cd ~/SageMaker/text2sql-rl
      source activate ~/SageMaker/envs/t2s

3.4 Log in to W&B:
      wandb login            (paste your API key)
      export WANDB_API_KEY=<your key>
    Put the export line at the end of ~/SageMaker/.bashrc_t2s and run
    "source ~/SageMaker/.bashrc_t2s" in each new terminal, or just paste it each time.

3.5 Check the GPU:
      nvidia-smi             (should show a Tesla T4)


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

    Spider-Realistic must be downloaded by hand: get spider-realistic.json from
    https://zenodo.org/record/5205322 and save it as
    data/spider_variants/spider_realistic.json  (upload it through the JupyterLab file browser).

    If gdown says "too many users have viewed or downloaded this file", open the Google
    Drive link in your browser, download the zip, upload it to the notebook and unzip it
    into data/ (Spider should end up as data/spider_data/...).

4.2 Prepare:
      python prepare_data.py

    It prints something like:
      Spider train: 8659 questions
      filter: {'empty_result': ..., 'gold_error': ..., 'timeout_over_5s': ..., 'kept': ...}
      train: ~7800 questions | val: ~400 questions from ~10 DBs
    Write these numbers down, they go in the report (also saved in data/processed/filter_log.json).

    Output folder data/processed/ now has:
      train.jsonl, val.jsonl, spider_dev.jsonl, bird_dev.jsonl,
      spider_syn.jsonl, spider_dk.jsonl, spider_realistic.jsonl (if the files existed),
      gold_cache.pkl, filter_log.json

    The split uses a fixed seed (42), so both people get exactly the same train/val split.
    Takes ~5-15 minutes (BIRD prompts take the longest).


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
6.1 Quick test on 50 questions first (couple of minutes):
      python sample.py --split train --limit 50 --n 8
    If vLLM crashes, add --engine hf to every sample.py / evaluate.py / val_curve.py
    command from now on (slower, but works).

6.2 Full run. This takes a while (roughly 1 h with vLLM, several hours with --engine hf),
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

6.5 Share with Person A through git (the .gitignore already allows these files):
      git add data/processed/tags.json data/processed/dpo_pairs.jsonl \
              data/processed/filter_log.json data/processed/passk_train.json
      git commit -m "feasibility results"
      git push
    Person A then runs "git pull".


-------------------------------------------------------------
7. SMOKE TESTS (50 STEPS OF EACH METHOD)
-------------------------------------------------------------
Run these directly in the notebook terminal (not as jobs). Open a second terminal and
run "watch -n 2 nvidia-smi" to see GPU memory.

  Person B:
      python train_dpo.py --max_steps 50

  Person A (if tags.json hasn't arrived yet, add --only_mixed 0):
      python train_rl.py --method grpo --max_steps 50
      python train_rl.py --method rloo --max_steps 50

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
8. FULL TRAINING AS SAGEMAKER TRAINING JOBS
-------------------------------------------------------------
Training jobs run on their own machine, so you can close the notebook and they keep going.
They use spot instances (much cheaper). If AWS takes the instance back, launch the same
command again and it resumes from the last checkpoint.

8.1 Find your default bucket (do this once):
      python -c "import sagemaker; print(sagemaker.Session().default_bucket())"
    It looks like sagemaker-<region>-<account id>. Below this is called BUCKET.

8.2 Upload the data the jobs need (once, and again if data/processed changes, e.g. after
    pulling tags.json):
      aws s3 sync data/processed            s3://BUCKET/text2sql/data/processed
      aws s3 sync data/spider_data/database s3://BUCKET/text2sql/data/spider_data/database

8.3 Make sure the W&B key is set in this terminal:
      export WANDB_API_KEY=<your key>

8.4 Launch (any extra --arguments are passed straight to the training script):

    Person A:
      python launch_sagemaker.py --script train_rl.py --method grpo --seed 0 --max_steps 600
      python launch_sagemaker.py --script train_rl.py --method rloo --seed 0 --max_steps 600

    Person B:
      python launch_sagemaker.py --script train_dpo.py --seed 0

    Change 600 to whatever you agreed in step 7. For extra seeds, change --seed 1, 2.
    If your quota is only 1 instance, launch the second job after the first finishes.
    Use --spot 0 if spot jobs keep getting stuck waiting for capacity.

8.5 Watch the jobs:
    - SageMaker console -> Training -> Training jobs -> click the job -> "View logs"
      (CloudWatch). The first ~10 minutes is installing packages and downloading the model.
    - W&B project text2sql-rl. Runs are named grpo-1.5b-s0, rloo-1.5b-s0, dpo-1.5b-s0.

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

    If several of these go bad at once, stop the job (console -> Stop) and use the last
    good checkpoint.

8.6 Each job saves a checkpoint every 100 steps to
      s3://BUCKET/text2sql/checkpoints/<run name>/checkpoint-100, checkpoint-200, ...
    and the final adapter to .../<run name>/final


-------------------------------------------------------------
9. GETTING THE TRAINED MODELS BACK
-------------------------------------------------------------
On the notebook of whoever does the evaluation (Person B):
      aws s3 sync s3://BUCKET/text2sql/checkpoints/dpo-1.5b-s0  outputs/dpo-1.5b-s0

Person A's models live in Person A's account, so Person A sends them (see section 14), and
Person B puts them in outputs/grpo-1.5b-s0 and outputs/rloo-1.5b-s0 with the
checkpoint-XXX folders inside.

The folder structure Person B should end up with:
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

11.3 BIRD official EX, Soft-F1 and R-VES:
     Open eval_repos/mini_dev/evaluation/run_evaluation.sh and set the paths at the top to:
       predicted sql json : results/<tag>/bird_dev/predict_dev.json
       ground truth sql   : data/bird_dev/dev.sql
       db root            : data/bird_dev/dev_databases/
       difficulty json    : data/bird_dev/dev.json
       dialect            : SQLite
     then run it once per model:
       cd eval_repos/mini_dev/evaluation && sh run_evaluation.sh && cd -
     R-VES measures speed, so run it when NO training job or other eval is running on
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

Compute cost table (Person A): take seconds/step, total time and GPU memory from
W&B (System tab) and the SageMaker job page ("Billable seconds").


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
14. SHARING FILES BETWEEN THE TWO OF US
-------------------------------------------------------------
Small files (tags.json, dpo_pairs.jsonl, filter_log.json): git push / git pull.

Model checkpoints (different AWS accounts): make a temporary download link.
  Person A:
      cd outputs   (or download from S3 first with aws s3 sync)
      tar czf grpo-1.5b-s0.tar.gz grpo-1.5b-s0
      aws s3 cp grpo-1.5b-s0.tar.gz s3://BUCKET/share/
      aws s3 presign s3://BUCKET/share/grpo-1.5b-s0.tar.gz --expires-in 604800
  Send Person B the printed link (it works for 7 days). Person B:
      mkdir -p outputs && cd outputs
      curl -L -o grpo.tar.gz "<link>"
      tar xzf grpo.tar.gz

  A LoRA checkpoint is small (~100-300 MB with optimizer state), so this is quick.
  If the tar is too big, only send the checkpoint folders you need for the val curve.


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

"dropped N examples with prompt > 2048 tokens"
    -> normal, a few Spider databases have huge schemas.

Training job fails straight away
    -> check the CloudWatch log. Usually: data not uploaded to S3 (step 8.2),
       tags.json missing from s3://BUCKET/text2sql/data/processed, or quota not approved.

Training job stuck at "Starting" / "Waiting for spot capacity"
    -> wait, or stop it and relaunch with --spot 0.

pip install fails because of versions
    -> make sure you are in the t2s conda environment (step 3.3), not the default one.

The notebook was stopped and the environment is gone
    -> the conda env is in ~/SageMaker/envs/t2s, just run
       "source activate ~/SageMaker/envs/t2s" again.

KeyError: 'train_1234' (or similar) from reward.py
    -> data/processed is out of date. Re-run prepare_data.py (and re-upload to S3).


-------------------------------------------------------------
16. SAVING MONEY
-------------------------------------------------------------
- STOP the notebook instance whenever you're not using it (Notebook instances ->
  select -> Actions -> Stop). A running g4dn.2xlarge notebook costs money every hour
  even when idle. Files in ~/SageMaker are kept.
- Use a cheap ml.t3.xlarge notebook for anything that doesn't need a GPU (data prep,
  launching jobs, analysis.py, writing). You can change the instance type of a stopped
  notebook (Edit).
- Training jobs use spot by default, which is a lot cheaper than on-demand.
- Set an AWS budget alert: Billing -> Budgets -> Create budget.
- At the end of the project, delete the S3 checkpoints you don't need.


=============================================================
17. SOLO GUIDE: DOING THE WHOLE PROJECT BY YOURSELF
=============================================================
Use this if the other person drops out. Everything runs on YOUR AWS account. The steps are
the same as sections 1-13, just in one order, with nothing to send to anyone. Section 14
(sharing) doesn't apply: the files are already on your notebook and the checkpoints are
already in your own S3 bucket.

The trick to finishing alone: let SageMaker training jobs do the long GPU work in the
background while you use the notebook for everything else. You're never waiting on one
thing at a time.


17.1 FIRST: ASK FOR MORE QUOTA
------------------------------
In Service Quotas (section 1.2), ask for 3 instead of 1 for:
  - ml.g4dn.2xlarge for training job usage
  - ml.g4dn.2xlarge for spot training job usage
With 3 you can train DPO, GRPO and RLOO at the SAME time, which saves days.
With only 1 you have to run them one after another (see the timeline in 17.4).
Keep notebook instance usage at 1.


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
Each line gives the section to follow. [GPU] = needs the g4dn notebook,
[JOB] = runs as a SageMaker job (keeps going after you close the notebook),
[CPU] = a cheap t3 notebook is fine.

  Step 1   [CPU]  Section 1: quotas (ask for 3 training instances), IAM role.
  Step 2   [CPU]  Section 2: your own private GitHub repo + W&B project (no team needed).
  Step 3   [CPU]  Section 3: notebook + environment. Use t3.xlarge until GPU quota arrives.
  Step 4   [CPU]  Section 4: download_data.sh, prepare_data.py.
  Step 5   [CPU]  Section 5: python reward.py --split val. Freeze reward.py.
  Step 6   [GPU]  Section 6: sample.py feasibility run (nohup), go/no-go, make_dpo_pairs.py.
                  While it runs: start the report (intro, related work, data section).
  Step 7   [GPU]  Section 7: smoke tests, one after another:
                    python train_dpo.py --max_steps 50
                    python train_rl.py --method grpo --max_steps 50
                    python train_rl.py --method rloo --max_steps 50
                  Pick the model size and max_steps. Then rm -rf outputs/
  Step 8   [CPU]  Section 8.1-8.3: upload data to S3, export WANDB_API_KEY.
  Step 9   [JOB]  Section 8.4: launch all three:
                    python launch_sagemaker.py --script train_dpo.py --seed 0
                    python launch_sagemaker.py --script train_rl.py --method grpo --seed 0 --max_steps <N>
                    python launch_sagemaker.py --script train_rl.py --method rloo --seed 0 --max_steps <N>
                  With quota 1: launch DPO first (shortest), then GRPO, then RLOO.
  Step 10  [GPU]  While the jobs run: evaluate the untrained model now, so it's done:
                    for SPLIT in val spider_dev spider_syn spider_dk spider_realistic bird_dev; do
                      python evaluate.py --split $SPLIT --tag base
                    done
                    bash run_official_eval.sh base
                  and set up the BIRD mini_dev paths (section 11.3) and run it for base.
                  Then STOP the GPU notebook or switch it to t3 while you wait.
  Step 11  [CPU]  While the jobs run: check W&B once or twice a day (section 8.5). Write the
                  method sections for DPO, GRPO and RLOO.
  Step 12  [GPU]  When the jobs finish, download all three from S3 (no sharing needed):
                    aws s3 sync s3://BUCKET/text2sql/checkpoints/dpo-1.5b-s0  outputs/dpo-1.5b-s0
                    aws s3 sync s3://BUCKET/text2sql/checkpoints/grpo-1.5b-s0 outputs/grpo-1.5b-s0
                    aws s3 sync s3://BUCKET/text2sql/checkpoints/rloo-1.5b-s0 outputs/rloo-1.5b-s0
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
  Days 4-6   Training jobs running (Step 11). With quota 1, add ~2 extra days because they
             run one after another.
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
  - The compute cost table, copied from W&B / SageMaker
  - Soft-F1 and R-VES, which come out of the same BIRD script run as EX

Make the runs shorter if needed:
  - Lower --max_steps (same value for GRPO and RLOO!)
  - DPO: --epochs 0.5
  - Switch to 0.5B at the smoke test stage (never halfway through)
  - --save_steps 200 means fewer checkpoints, so val_curve.py is quicker


17.6 SOLO CHECKLIST (tick these off)
------------------------------------
  [ ] Quota approved for g4dn.2xlarge (notebook + training + spot training)
  [ ] Repo on GitHub, W&B logged in
  [ ] data/processed created, filter numbers written down
  [ ] reward.py check prints 1.000, reward frozen
  [ ] Feasibility done, GO decision, tags.json + dpo_pairs.jsonl made
  [ ] Smoke tests done, model size + max_steps decided
  [ ] Data uploaded to S3
  [ ] DPO job finished
  [ ] GRPO job finished
  [ ] RLOO job finished
  [ ] Base model evaluated on all splits
  [ ] Checkpoints downloaded, val_curve.png made, best checkpoints picked
  [ ] DPO / GRPO / RLOO evaluated on all splits
  [ ] Official Spider EX + TS done for all four
  [ ] BIRD EX / Soft-F1 / R-VES done for all four (one session, idle machine)
  [ ] analysis.py compare + efficiency done
  [ ] Results tables filled in
  [ ] Demo works (and a backup video recorded)
  [ ] Report written
  [ ] Notebook STOPPED, unneeded S3 files deleted
