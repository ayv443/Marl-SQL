# Launch a training script as a SageMaker training job.
# python launch_sagemaker.py --script train_rl.py --method grpo --seed 0 --max_steps 600
import argparse
import os
import shutil
import tempfile

import sagemaker
from sagemaker.pytorch import PyTorch

CODE_FILES = ["common.py", "reward.py", "train_rl.py", "train_dpo.py", "requirements.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, choices=["train_rl.py", "train_dpo.py"])
    ap.add_argument("--method", default="dpo", help="grpo / rloo for train_rl.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--bucket", default=None)
    ap.add_argument("--role", default=None)
    ap.add_argument("--spot", type=int, default=1)
    ap.add_argument("--max_hours", type=int, default=24)
    args, extra = ap.parse_known_args()  # extra args are passed to the script

    sess = sagemaker.Session()
    bucket = args.bucket or sess.default_bucket()
    role = args.role or sagemaker.get_execution_role()

    method = args.method if args.script == "train_rl.py" else "dpo"
    size = "0.5b" if "0.5B" in args.model else "1.5b"
    run = f"{method}-{size}-s{args.seed}"

    hp = {"seed": args.seed, "model": args.model}
    if args.script == "train_rl.py":
        hp["method"] = method
    for k, v in zip(extra[::2], extra[1::2]):
        hp[k.lstrip("-")] = v

    # upload only the code, not the data folder
    src = tempfile.mkdtemp()
    for f in CODE_FILES:
        shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), src)

    est = PyTorch(
        entry_point=args.script,
        source_dir=src,
        role=role,
        instance_type="ml.g4dn.2xlarge",
        instance_count=1,
        framework_version="2.8.0",
        py_version="py312",
        hyperparameters=hp,
        environment={"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                     "WANDB_PROJECT": "text2sql-rl"},
        checkpoint_s3_uri=f"s3://{bucket}/text2sql/checkpoints/{run}",
        checkpoint_local_path="/opt/ml/checkpoints",
        use_spot_instances=bool(args.spot),
        max_run=args.max_hours * 3600,
        max_wait=(args.max_hours + 12) * 3600 if args.spot else None,
        base_job_name=run.replace(".", "-"),
        disable_profiler=True,
    )
    est.fit({"data": f"s3://{bucket}/text2sql/data"}, wait=False)
    print(f"launched {est.latest_training_job.name}  (run={run}, hyperparameters={hp})")


if __name__ == "__main__":
    main()
