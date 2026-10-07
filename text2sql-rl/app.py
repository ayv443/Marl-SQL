# Gradio demo, compares all four models.
# python app.py --dpo <adapter> --grpo <adapter> --rloo <adapter>
import argparse
import os
import time

import gradio as gr
import pandas as pd
import torch
from peft import PeftModel

from common import DATA_DIR, MAX_COMPLETION_LEN, build_prompt, extract_sql, load_jsonl, load_model, PROCESSED_DIR
from reward import execute, has_order_by, results_match

MODELS = ["untrained", "dpo", "grpo", "rloo"]


def list_dbs():
    dbs = {}
    for split, root in [("spider_dev", "spider_data/database"), ("bird_dev", "bird_dev/dev_databases")]:
        p = f"{PROCESSED_DIR}/{split}.jsonl"
        if os.path.exists(p):
            for r in load_jsonl(p):
                dbs[f"{split.split('_')[0]}: {r['db_id']}"] = os.path.join(DATA_DIR, r["db_path"])
    return dbs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", required=True)
    ap.add_argument("--grpo", required=True)
    ap.add_argument("--rloo", required=True)
    args = ap.parse_args()

    base, tok = load_model()
    model = PeftModel.from_pretrained(base.to("cuda"), args.dpo, adapter_name="dpo")
    model.load_adapter(args.grpo, adapter_name="grpo")
    model.load_adapter(args.rloo, adapter_name="rloo")
    model.eval()
    dbs = list_dbs()

    def generate(name, prompt):
        text = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            if name == "untrained":
                with model.disable_adapter():
                    out = model.generate(**enc, max_new_tokens=MAX_COMPLETION_LEN, do_sample=False)
            else:
                model.set_adapter(name)
                out = model.generate(**enc, max_new_tokens=MAX_COMPLETION_LEN, do_sample=False)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def answer(db_name, question, evidence, gold_sql):
        db = dbs[db_name]
        prompt = build_prompt(db, question, evidence)
        gold_rows = execute(db, gold_sql, timeout=30)[0] if gold_sql.strip() else None
        outputs = []
        for name in MODELS:
            sql = extract_sql(generate(name, prompt))
            t = time.perf_counter()
            rows, err = execute(db, sql)
            runtime = time.perf_counter() - t
            if err:
                status, table = f"error: {err}", pd.DataFrame()
            else:
                table = pd.DataFrame(rows[:50])
                status = f"{len(rows)} rows in {runtime * 1000:.1f} ms"
                if gold_rows is not None:
                    status += ", correct" if results_match(rows, gold_rows, has_order_by(gold_sql)) else ", wrong"
            outputs += [sql or "(no SQL found)", status, table]
        return outputs

    with gr.Blocks(title="Text-to-SQL: DPO vs GRPO vs RLOO") as demo:
        with gr.Tab("Try it"):
            db_in = gr.Dropdown(sorted(dbs), label="Database")
            q_in = gr.Textbox(label="Question")
            ev_in = gr.Textbox(label="Evidence / hint (BIRD, optional)")
            gold_in = gr.Textbox(label="Gold SQL (optional, shows correct / wrong)")
            btn = gr.Button("Run all four models")
            outs = []
            with gr.Row():
                for name in MODELS:
                    with gr.Column():
                        gr.Markdown(f"### {name}")
                        outs += [gr.Code(language="sql"), gr.Textbox(label="result"), gr.Dataframe()]
            btn.click(answer, [db_in, q_in, ev_in, gold_in], outs)
        with gr.Tab("Results"):
            if os.path.exists("results/val_curve.png"):
                gr.Image("results/val_curve.png", label="Validation EX during training")
            if os.path.exists("results/results_table.md"):
                gr.Markdown(open("results/results_table.md").read())
    demo.launch(share=True)


if __name__ == "__main__":
    main()
