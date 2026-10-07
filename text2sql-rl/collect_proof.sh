#!/usr/bin/env bash
# Collects proof that a run was trained on AWS (inside a SageMaker Studio space) into proof/<run>/
#   bash collect_proof.sh grpo-1.5b-s0
set -e
RUN=$1
if [ -z "$RUN" ]; then echo "usage: bash collect_proof.sh <run name, e.g. grpo-1.5b-s0>"; exit 1; fi
SRC=outputs/$RUN
OUT=proof/$RUN
if [ ! -d "$SRC" ]; then echo "no folder $SRC (run this in text2sql-rl/ after training)"; exit 1; fi
mkdir -p $OUT/checkpoints

# 1. our step-by-step logs: every step with time, metrics, GPU memory, plus where it ran
cp -r $SRC/logs $OUT/

# 2. the training history the trainer saved in every checkpoint (log_history = every step)
for ck in $SRC/checkpoint-* $SRC/final; do
  [ -d "$ck" ] || continue
  name=$(basename $ck)
  mkdir -p $OUT/checkpoints/$name
  cp $ck/trainer_state.json $ck/adapter_config.json $OUT/checkpoints/$name/ 2>/dev/null || true
done

# 3. where it ran: the Studio space, the AWS account/role, the GPU
cp /opt/ml/metadata/resource-metadata.json $OUT/studio_space.json 2>/dev/null || echo "not a SageMaker Studio space" > $OUT/studio_space.json
aws sts get-caller-identity > $OUT/aws_identity.json 2>/dev/null || echo "aws cli not available" > $OUT/aws_identity.json
nvidia-smi > $OUT/nvidia_smi.txt 2>&1 || true

# 4. which code and library versions were used
git log -1 --format='%H  %ad  %s' > $OUT/git_commit.txt
git status --short --untracked-files=no >> $OUT/git_commit.txt
pip freeze 2>/dev/null | grep -iE "^(trl|transformers|peft|accelerate|datasets|torch|vllm)==" > $OUT/versions.txt || true

# 5. short summary
python - "$OUT" "$RUN" > $OUT/summary.txt <<'EOF'
import glob, json, os, sys
out, run = sys.argv[1], sys.argv[2]
print("run:", run)
for f in sorted(glob.glob(f"{out}/logs/*_summary.json")):
    s = json.load(open(f))
    print(f"{os.path.basename(f)}: {s['status']}, {s['start']} -> {s['end']} ({s['duration']})")
steps = [json.loads(l) for f in sorted(glob.glob(f"{out}/logs/*_steps.jsonl")) for l in open(f)]
steps = [r for r in steps if "reward" in r or "loss" in r]
if steps:
    print("steps logged:", len(steps), "| last step:", steps[-1].get("step"), "of", steps[-1].get("max_steps"),
          "| avg s/step:", steps[-1].get("seconds_per_step"), "| peak GPU GB:", max(r.get("gpu_mem_peak_gb", 0) for r in steps))
cks = sorted(glob.glob(f"{out}/checkpoints/checkpoint-*"), key=lambda p: int(p.split("-")[-1]))
print("checkpoints:", ", ".join(os.path.basename(c) for c in cks))
space = json.load(open(f"{out}/studio_space.json")) if open(f"{out}/studio_space.json").read(1) == "{" else {}
print("studio space:", space.get("SpaceName", ""), "| domain:", space.get("DomainId", ""), "| app:", space.get("ResourceName", ""))
ident = open(f"{out}/aws_identity.json").read()
print("aws:", json.loads(ident).get("Arn", "") if ident.startswith("{") else ident.strip())
EOF

echo "saved to $OUT:"
ls $OUT
cat $OUT/summary.txt
