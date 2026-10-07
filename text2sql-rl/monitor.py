# Telegram messages + log files for every run.
# Telegram is optional: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to turn it on.
# Test: python monitor.py --test
import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.parse
import urllib.request

from transformers import TrainerCallback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

KEY_METRICS = [
    "loss", "reward", "rewards/execution_reward/mean", "rewards/valid_sql_reward/mean",
    "kl", "entropy", "frac_reward_zero_std", "completions/mean_length",
    "rewards/accuracies", "rewards/margins", "grad_norm", "learning_rate",
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h {seconds % 3600 // 60}m {seconds % 60}s"


def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text[:4000]}).encode()
    for attempt in range(3):
        try:
            urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=10)
            return
        except Exception as e:
            error = e
            time.sleep(2 * (attempt + 1))
    print(f"telegram message failed after 3 tries: {error}", flush=True)


def _ec2_metadata():
    # works on notebook instances, usually blocked inside training jobs
    try:
        req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(req, timeout=2).read().decode()
        out = {}
        for key in ("instance-id", "instance-type", "placement/region"):
            req = urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{key}",
                                         headers={"X-aws-ec2-metadata-token": token})
            out[key] = urllib.request.urlopen(req, timeout=2).read().decode()
        return out
    except Exception:
        return {}


def environment_info():
    info = {
        "time": now(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "command": " ".join(sys.argv),
        "sagemaker_training_job": os.environ.get("TRAINING_JOB_NAME", ""),
        "sagemaker_training_job_arn": os.environ.get("TRAINING_JOB_ARN", ""),
        "sagemaker_instance_type": os.environ.get("SM_CURRENT_INSTANCE_TYPE", ""),
        "aws_region": os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "")),
        "ec2": _ec2_metadata(),
    }
    nb = "/opt/ml/metadata/resource-metadata.json"  # notebook instances only
    if os.path.exists(nb):
        with open(nb) as f:
            info["sagemaker_notebook"] = json.load(f)
    try:
        import torch
        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        info["gpu"] = "unknown"
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        info["nvidia_smi"] = "nvidia-smi not available"
    return info


def where(env):
    if env["sagemaker_training_job"]:
        return f"SageMaker training job {env['sagemaker_training_job']} ({env['sagemaker_instance_type']})"
    if env.get("sagemaker_notebook"):
        return f"SageMaker notebook {env['sagemaker_notebook'].get('ResourceName', '')} ({env['ec2'].get('instance-type', '')})"
    return env["hostname"]


# usage: with Monitor(name, log_dir, settings) as mon: ...
# writes <name>_<time>.log, _steps.jsonl and _summary.json
class Monitor:
    def __init__(self, name, log_dir=LOG_DIR, settings=None, notify=True):
        self.name = name.replace("/", "-")
        self.settings = settings or {}
        self.notify = notify
        self.result = {}
        self.last_metrics = {}
        os.makedirs(log_dir, exist_ok=True)
        base = os.path.join(log_dir, f"{self.name}_{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}")
        self.log_path = base + ".log"
        self.steps_path = base + "_steps.jsonl"
        self.summary_path = base + "_summary.json"

    def log(self, msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def log_step(self, record):
        with open(self.steps_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def message(self, msg):
        if self.notify:
            send_telegram(f"[{self.name}] {msg}")

    def __enter__(self):
        self.t0 = time.time()
        self.start_time = now()
        self.env = environment_info()
        self.log(f"START {self.name}")
        self.log("environment: " + json.dumps({k: v for k, v in self.env.items() if k != "nvidia_smi"}, indent=2))
        self.log("nvidia-smi:\n" + self.env["nvidia_smi"])
        self.log("settings: " + json.dumps(self.settings, default=str))
        signal.signal(signal.SIGTERM, self._on_sigterm)
        self.message(f"STARTED\nwhere: {where(self.env)}\ngpu: {self.env['gpu']}\nlog: {self.log_path}")
        return self

    def _on_sigterm(self, signum, frame):
        raise SystemExit("SIGTERM received (job stopped or spot instance taken back)")

    def __exit__(self, exc_type, exc, tb):
        duration = time.time() - self.t0
        if exc_type is None:
            status = "FINISHED"
        elif exc_type is KeyboardInterrupt:
            status = "STOPPED by user"
        elif exc_type is SystemExit:
            status = f"STOPPED: {exc}"
        else:
            status = "FAILED"
        error = "".join(traceback.format_exception(exc_type, exc, tb)) if status == "FAILED" else ""

        summary = {"run": self.name, "status": status, "start": self.start_time, "end": now(),
                   "duration_seconds": round(duration), "duration": fmt_time(duration),
                   "environment": self.env, "settings": self.settings,
                   "last_metrics": self.last_metrics, "result": self.result,
                   "log_file": self.log_path, "steps_file": self.steps_path, "error": error}
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        if error:
            self.log("ERROR:\n" + error)
        self.log(f"{status} after {fmt_time(duration)}")

        msg = f"{status} after {fmt_time(duration)}\nwhere: {where(self.env)}"
        if self.last_metrics:
            msg += "\nlast metrics: " + short_metrics(self.last_metrics)
        if self.result:
            msg += "\nresult: " + json.dumps(self.result, default=str)[:1500]
        if error:
            msg += "\nerror:\n" + error[-1500:]
        self.message(msg)
        stop_child_processes()
        return False


def stop_child_processes():
    # vLLM's EngineCore process can stay alive after the run (crash or not) and hold the GPU
    import multiprocessing
    for child in multiprocessing.active_children():
        child.terminate()
        child.join(timeout=10)
        if child.is_alive():
            child.kill()


def short_metrics(logs):
    parts = []
    for k in KEY_METRICS:
        v = logs.get(k)
        if isinstance(v, (int, float)):
            parts.append(f"{k}={v:.4g}")
    return ", ".join(parts)


# logs every step, sends a Telegram update every `every` steps
class TrainingMonitorCallback(TrainerCallback):
    def __init__(self, mon, every=50):
        self.mon = mon
        self.every = every
        self.last_sent = 0
        self.nan_warned = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
        self.start_step = state.global_step
        self.last_sent = state.global_step
        self.mon.log(f"training started at step {state.global_step}, max_steps={state.max_steps}")
        self.mon.message(f"training started at step {state.global_step} of {state.max_steps}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        step = state.global_step
        done = step - self.start_step
        elapsed = time.time() - self.t0
        sec_per_step = elapsed / done if done else 0.0
        eta = sec_per_step * (state.max_steps - step)
        record = {"time": now(), "step": step, "max_steps": state.max_steps, "epoch": state.epoch,
                  "elapsed_seconds": round(elapsed, 1), "seconds_per_step": round(sec_per_step, 2), **logs}
        try:
            import torch
            if torch.cuda.is_available():
                record["gpu_mem_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        except Exception:
            pass
        self.mon.log_step(record)
        self.mon.last_metrics = record
        self.mon.log(f"step {step}/{state.max_steps} | {sec_per_step:.1f} s/step | {short_metrics(logs)}")

        loss = logs.get("loss")
        if isinstance(loss, float) and math.isnan(loss) and not self.nan_warned:
            self.nan_warned = True
            self.mon.log("WARNING: loss is NaN")
            self.mon.message(f"WARNING: loss is NaN at step {step}. Consider stopping the run.")

        if step - self.last_sent >= self.every:
            self.last_sent = step
            self.mon.message(
                f"step {step}/{state.max_steps} ({100 * step / max(state.max_steps, 1):.0f}%)\n"
                f"{sec_per_step:.1f} s/step, elapsed {fmt_time(elapsed)}, remaining about {fmt_time(eta)}\n"
                f"gpu peak memory: {record.get('gpu_mem_peak_gb', 'n/a')} GB\n"
                f"{short_metrics(logs)}")

    def on_save(self, args, state, control, **kwargs):
        self.mon.log(f"checkpoint saved: checkpoint-{state.global_step}")
        self.mon.message(f"checkpoint saved at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        self.mon.log(f"training ended at step {state.global_step}")


if __name__ == "__main__":
    if "--test" in sys.argv:
        if not os.environ.get("TELEGRAM_BOT_TOKEN") or not os.environ.get("TELEGRAM_CHAT_ID"):
            print("set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID first")
            sys.exit(1)
        env = environment_info()
        send_telegram(f"test message from {where(env)}, gpu: {env['gpu']}, time: {now()}")
        print("sent, check Telegram")
