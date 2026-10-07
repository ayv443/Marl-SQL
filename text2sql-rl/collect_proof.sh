#!/usr/bin/env bash
# Collects proof that a run was trained on AWS into proof/<run name>/
#   bash collect_proof.sh grpo-1.5b-s0
# Uses the most recent SageMaker training job for that run (or pass the job name as 2nd argument).
set -e
RUN=$1
if [ -z "$RUN" ]; then echo "usage: bash collect_proof.sh <run name, e.g. grpo-1.5b-s0> [job name]"; exit 1; fi
PREFIX=$(echo "$RUN" | tr '.' '-')
JOB=${2:-$(aws sagemaker list-training-jobs --name-contains "$PREFIX" --sort-by CreationTime --sort-order Descending \
            --max-results 1 --query 'TrainingJobSummaries[0].TrainingJobName' --output text)}
BUCKET=$(python -c "import sagemaker; print(sagemaker.Session().default_bucket())" 2>/dev/null)
OUT=proof/$RUN
mkdir -p $OUT
echo "run: $RUN   job: $JOB   bucket: $BUCKET"

# 1. the job record from AWS: instance type, start/end time, billable seconds, status, hyperparameters
aws sagemaker describe-training-job --training-job-name "$JOB" > $OUT/training_job.json
python - "$OUT/training_job.json" > $OUT/training_job_summary.txt <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
keys = ["TrainingJobName", "TrainingJobArn", "TrainingJobStatus", "CreationTime", "TrainingStartTime",
        "TrainingEndTime", "TrainingTimeInSeconds", "BillableTimeInSeconds", "EnableManagedSpotTraining"]
for k in keys:
    print(f"{k}: {j.get(k)}")
print(f"InstanceType: {j['ResourceConfig']['InstanceType']}")
print(f"InstanceCount: {j['ResourceConfig']['InstanceCount']}")
print(f"HyperParameters: {json.dumps(j.get('HyperParameters', {}), indent=2)}")
for t in j.get("SecondaryStatusTransitions", []):
    print(f"{t.get('StartTime')}  {t.get('Status')}: {t.get('StatusMessage')}")
EOF

# 2. the full console output of the job from CloudWatch
aws logs filter-log-events --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name-prefix "$JOB" \
    --query 'events[*].[timestamp,message]' --output text > $OUT/cloudwatch_log.txt || echo "could not read CloudWatch logs"

# 3. our own step-by-step log files (saved next to the checkpoints in S3)
aws s3 sync s3://$BUCKET/text2sql/checkpoints/$RUN/logs $OUT/logs --quiet || echo "no logs folder in S3 yet"

echo "saved to $OUT:"
ls -la $OUT $OUT/logs 2>/dev/null
cat $OUT/training_job_summary.txt | head -15
