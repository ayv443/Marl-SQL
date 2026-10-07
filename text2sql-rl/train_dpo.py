# DPO training on the pairs from make_dpo_pairs.py
# python train_dpo.py --epochs 1
import argparse

from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from common import (MAX_COMPLETION_LEN, MAX_PROMPT_LEN, PROCESSED_DIR, add_common_args,
                    common_training_kwargs, drop_long_prompts, last_checkpoint, load_jsonl,
                    load_model, lora_config, save_final, setup_run)


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    run_name, out = setup_run("dpo", args)
    model, tok = load_model(args.model, bool(args.load_4bit))

    pairs = load_jsonl(f"{PROCESSED_DIR}/dpo_pairs.jsonl")
    dataset = Dataset.from_list([{k: p[k] for k in ("prompt", "chosen", "rejected")} for p in pairs])
    dataset = drop_long_prompts(dataset, tok)

    config = DPOConfig(
        **common_training_kwargs(args, run_name, out),
        beta=args.beta,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.batch_size,
        max_prompt_length=MAX_PROMPT_LEN,
        max_completion_length=MAX_COMPLETION_LEN,
        max_length=MAX_PROMPT_LEN + MAX_COMPLETION_LEN,
    )
    # with LoRA the reference model is just the base model with the adapter off
    trainer = DPOTrainer(model=model, ref_model=None, args=config, train_dataset=dataset,
                         processing_class=tok, peft_config=lora_config(args.lora_rank))
    trainer.train(resume_from_checkpoint=last_checkpoint(out))
    save_final(trainer, out)


if __name__ == "__main__":
    main()
