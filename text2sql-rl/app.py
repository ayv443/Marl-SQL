# Gradio demo, compares all four models.
# python app.py --dpo <adapter> --grpo <adapter> --rloo <adapter>
import argparse
import html
import inspect
import os
import random
import time

import gradio as gr

from common import DATA_DIR, MAX_COMPLETION_LEN, build_prompt, extract_sql, load_jsonl, load_model, PROCESSED_DIR, schema_text
from reward import execute, has_order_by, results_match

MODELS = ["untrained", "dpo", "grpo", "rloo"]
NAMES = {"untrained": "Untrained", "dpo": "DPO", "grpo": "GRPO", "rloo": "RLOO"}
KINDS = {"untrained": "Base model", "dpo": "Offline, preference pairs",
         "grpo": "Online, group baseline", "rloo": "Online, leave-one-out"}
SPLIT_NAMES = {"spider_dev": "Spider dev", "bird_dev": "BIRD dev"}

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_CURVE = os.path.join(HERE, "evaluation", "results", "val_curve.png")
RESULTS_MD = os.path.join(HERE, "..", "RESULTS.md")


def list_dbs():
    dbs = {}
    for split, root in [("spider_dev", "spider_data/database"), ("bird_dev", "bird_dev/dev_databases")]:
        p = f"{PROCESSED_DIR}/{split}.jsonl"
        if os.path.exists(p):
            for r in load_jsonl(p):
                dbs[f"{split.split('_')[0]}: {r['db_id']}"] = os.path.join(DATA_DIR, r["db_path"])
    return dbs


def load_examples():
    # questions with their database, gold SQL and BIRD evidence, for the "Random example" button
    examples = {}
    for split in SPLIT_NAMES:
        p = f"{PROCESSED_DIR}/{split}.jsonl"
        if os.path.exists(p):
            examples[split] = load_jsonl(p)
    return examples


def results_sections():
    # sections 4, 5 and 6 of RESULTS.md (our EX with CIs, official Spider, official BIRD)
    if not os.path.exists(RESULTS_MD):
        return "RESULTS.md not found."
    text = open(RESULTS_MD).read()
    start = text.find("## 4.")
    end = text.find("## Summary")
    if start == -1:
        return text
    return text[start:end if end != -1 else len(text)]


# ---------------------------------------------------------------- page pieces (HTML)
E = html.escape
ICONS = {
    "ok": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>',
    "bad": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>',
    "warn": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3 2 20h20L12 3z"/><path d="M12 10v4"/><path d="M12 17h.01"/></svg>',
    "plain": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16h.01"/></svg>',
}
BADGES = {"correct": ("ok", "Correct"), "wrong": ("bad", "Wrong result"),
          "error": ("warn", "SQL error"), "ran": ("plain", "Ran (no gold SQL to compare)")}

HEADER = """
<div class="t2s-header">
  <div class="t2s-brand">
    <div class="t2s-logo"><svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="1.8"
      stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><ellipse cx="12" cy="5.5" rx="7" ry="2.5"/>
      <path d="M5 5.5v6c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-6"/><path d="M5 11.5v6c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-6"/></svg></div>
    <div>
      <div class="t2s-title">Text-to-SQL with Reinforcement Learning</div>
      <div class="t2s-sub">Qwen2.5-Coder-1.5B + LoRA, trained on Spider with DPO, GRPO and RLOO</div>
    </div>
  </div>
</div>
"""


def rows_html(rows):
    if not rows:
        return ""
    body = "".join("<tr>" + "".join(f"<td>{E(str(v))}</td>" for v in r) + "</tr>" for r in rows[:50])
    more = f'<div class="t2s-more">first 50 of {len(rows)} rows</div>' if len(rows) > 50 else ""
    return f'<div class="t2s-rows"><table><tbody>{body}</tbody></table></div>{more}'


def card_html(name, sql, status, result_line, rows):
    cls, label = BADGES[status]
    return (f'<article class="t2s-card"><div class="t2s-card-head"><div class="t2s-card-title">'
            f'<h3>{NAMES[name]}</h3><span>{KINDS[name]}</span></div>'
            f'<div class="t2s-badge {cls}">{ICONS[cls]}{label}</div></div>'
            f'<pre class="t2s-sql">{E(sql or "(no SQL found)")}</pre>'
            f'<div class="t2s-section"><span class="t2s-label">Result</span>'
            f'<span class="t2s-result">{E(result_line)}</span>{rows_html(rows)}</div></article>')


def who_got_it_right(statuses, marked):
    # one plain line above the four columns
    if not marked:
        return "No gold SQL given, so the answers are not marked right or wrong."
    right = [NAMES[m] for m in MODELS if statuses[m] == "correct"]
    wrong = [NAMES[m] for m in MODELS if statuses[m] != "correct"]
    if not right:
        return "None of the four models got it right."
    if not wrong:
        return "All four models got it right."
    return f"Right: {', '.join(right)}. Not right: {', '.join(wrong)}."


def question_html(db_name, question, evidence, gold_sql, statuses):
    hint = f'<div class="t2s-hint"><b>Hint:</b> {E(evidence)}</div>' if evidence.strip() else ""
    gold = (f'<div class="t2s-gold"><span class="t2s-eyebrow">Gold SQL</span><code>{E(gold_sql)}</code></div>'
            if gold_sql.strip() else "")
    marked = gold_sql.strip() != ""
    n_right = sum(s == "correct" for s in statuses.values())
    score = (f'<div class="t2s-big">{n_right}<small> / 4</small></div><div>models return the same result as the gold SQL</div>'
             if marked else '<div class="t2s-big">?</div><div>add a gold SQL to mark the answers</div>')
    return (f'<div class="t2s-qrow"><div class="t2s-q"><div><span class="t2s-eyebrow">Question</span>'
            f'<span class="t2s-chip">{E(db_name)}</span></div><div class="t2s-qtext">{E(question)}</div>{hint}{gold}</div>'
            f'<div class="t2s-score"><span class="t2s-eyebrow dark">This question</span>{score}</div></div>'
            f'<p class="t2s-verdict">{E(who_got_it_right(statuses, marked))}</p>')


# ---------------------------------------------------------------- the app
def build_app(generate, dbs, examples):
    """generate(name, prompt) -> model text. Kept separate from the model loading in main()."""

    def answer(db_name, question, evidence, gold_sql):
        if not db_name or not question.strip():
            return '<p class="t2s-verdict">Pick a database and type a question first.</p>', ""
        db = dbs[db_name]
        prompt = build_prompt(db, question, evidence)
        gold_rows = execute(db, gold_sql, timeout=30)[0] if gold_sql.strip() else None
        cards, statuses = [], {}
        for name in MODELS:
            sql = extract_sql(generate(name, prompt))
            t = time.perf_counter()
            rows, err = execute(db, sql)
            runtime = time.perf_counter() - t
            if err:
                status, line, rows = "error", f"error: {err}", None
            else:
                line = f"{len(rows)} row{'' if len(rows) == 1 else 's'} in {runtime * 1000:.1f} ms"
                if gold_rows is None:
                    status = "ran"
                elif results_match(rows, gold_rows, has_order_by(gold_sql)):
                    status = "correct"
                else:
                    status = "wrong"
            statuses[name] = status
            cards.append(card_html(name, sql, status, line, rows))
        top = question_html(db_name, question, evidence, gold_sql, statuses)
        return top, f'<div class="t2s-grid">{"".join(cards)}</div>'

    def random_example(split):
        rows = [r for r in examples.get(split, []) if f"{split.split('_')[0]}: {r['db_id']}" in dbs]
        if not rows:
            return gr.update(), "", "", ""
        r = random.choice(rows)
        return f"{split.split('_')[0]}: {r['db_id']}", r["question"], r.get("evidence", ""), r["gold_sql"]

    def show_schema(db_name):
        if not db_name:
            return ""
        try:
            return schema_text(dbs[db_name])
        except Exception as e:
            return f"-- could not read the schema: {e}"

    css = CSS
    blocks_kw = {"title": "Text-to-SQL: DPO vs GRPO vs RLOO"}
    if "css" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kw["css"] = css          # Gradio 4 and 5
    with gr.Blocks(**blocks_kw) as demo:
        gr.HTML(HEADER)
        with gr.Tab("Try it"):
            with gr.Row(elem_classes="t2s-panel"):
                split_in = gr.Radio([(SPLIT_NAMES[s], s) for s in examples] or [("Spider dev", "spider_dev")],
                                    value=next(iter(examples), "spider_dev"), label="Example questions from", scale=2)
                ex_btn = gr.Button("Random example", elem_classes="t2s-btn-light", scale=1)
            with gr.Group(elem_classes="t2s-panel"):
                db_in = gr.Dropdown(sorted(dbs), label="Database")
                q_in = gr.Textbox(label="Question")
                ev_in = gr.Textbox(label="Evidence / hint (BIRD, optional)")
                gold_in = gr.Textbox(label="Gold SQL (optional, shows correct / wrong)")
                with gr.Accordion("Database schema (what the models see)", open=False):
                    schema_out = gr.Code(language="sql", label="Schema")
            btn = gr.Button("Run all four models", elem_classes="t2s-btn")
            top = gr.HTML()
            cards = gr.HTML()
            ex_btn.click(random_example, split_in, [db_in, q_in, ev_in, gold_in])
            db_in.change(show_schema, db_in, schema_out)
            btn.click(answer, [db_in, q_in, ev_in, gold_in], [top, cards])
        with gr.Tab("Results"):
            if os.path.exists(VAL_CURVE):
                gr.Image(VAL_CURVE, label="Validation EX during training", show_label=True)
            gr.Markdown(results_sections(), elem_classes="t2s-results")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpo", required=True)
    ap.add_argument("--grpo", required=True)
    ap.add_argument("--rloo", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel

    base, tok = load_model()
    model = PeftModel.from_pretrained(base.to("cuda"), args.dpo, adapter_name="dpo")
    model.load_adapter(args.grpo, adapter_name="grpo")
    model.load_adapter(args.rloo, adapter_name="rloo")
    model.eval()

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

    demo = build_app(generate, list_dbs(), load_examples())
    launch_kw = {"share": True}
    if "css" not in inspect.signature(gr.Blocks.__init__).parameters:
        launch_kw["css"] = CSS          # Gradio 6 takes the css in launch()
    demo.launch(**launch_kw)


CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
body, gradio-app, .gradio-container { background: #F6F6F3 !important; }
.gradio-container { font-family: 'IBM Plex Sans', system-ui, sans-serif !important; color: #15171C;
  max-width: 1320px !important; margin: 0 auto !important; }
.gradio-container code, .gradio-container pre { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
.t2s-header { display: flex; align-items: center; gap: 16px; background: #FFFFFF; border: 1px solid #E2E2DC;
  border-radius: 14px; padding: 16px 20px; }
.t2s-brand { display: flex; align-items: center; gap: 14px; }
.t2s-logo { width: 40px; height: 40px; border-radius: 10px; background: #15171C; display: flex; align-items: center; justify-content: center; }
.t2s-title { font-weight: 700; font-size: 18px; color: #15171C; }
.t2s-sub { color: #555A64; font-size: 13px; }
.t2s-panel { background: #FFFFFF !important; border: 1px solid #E2E2DC !important; border-radius: 14px !important; padding: 12px !important; }
.t2s-btn { background: #2456C7 !important; color: #FFFFFF !important; border: 0 !important; border-radius: 9px !important;
  font-weight: 600 !important; min-height: 46px !important; font-size: 16px !important; }
.t2s-btn:hover { background: #173F99 !important; }
.t2s-btn-light { background: #FFFFFF !important; color: #2456C7 !important; border: 1px solid #2456C7 !important;
  border-radius: 9px !important; font-weight: 600 !important; min-height: 44px !important; }
.t2s-qrow { display: flex; flex-wrap: wrap; gap: 16px; align-items: stretch; margin-top: 8px; }
.t2s-q { flex: 3 1 520px; background: #15171C; color: #FFFFFF; border-radius: 14px; padding: 24px 28px;
  display: flex; flex-direction: column; gap: 12px; }
.t2s-eyebrow { font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #C9CCD3; }
.t2s-eyebrow.dark { color: #555A64 !important; }
.t2s-chip { font-family: 'JetBrains Mono', monospace; font-size: 12px; padding: 3px 8px; border-radius: 6px;
  background: #2A2D35; color: #E4E6EB; margin-left: 8px; }
.t2s-qtext { font-size: 24px; font-weight: 600; line-height: 1.3; color: #FFFFFF; }
.t2s-hint { font-size: 15px; color: #E4E6EB; }
.t2s-hint b { color: #C9CCD3 !important; }
.t2s-gold { border-top: 1px solid #2E323B; padding-top: 12px; display: flex; flex-direction: column; gap: 6px; }
.t2s-gold code { color: #E4E6EB !important; background: transparent !important; font-size: 14px; white-space: pre-wrap; word-break: break-word; }
.t2s-score { flex: 1 1 220px; background: #FFFFFF; border: 1px solid #E2E2DC; border-radius: 14px; padding: 24px;
  display: flex; flex-direction: column; justify-content: center; gap: 6px; color: #3A3E46; }
.t2s-big { font-size: 44px; font-weight: 700; line-height: 1; color: #15171C; }
.t2s-big small { font-size: 22px; color: #555A64; font-weight: 500; }
.t2s-verdict { font-size: 17px; font-weight: 600; color: #15171C; margin: 16px 0 0; }
.t2s-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-top: 12px; }
@media (max-width: 1000px) { .t2s-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 560px) { .t2s-grid { grid-template-columns: minmax(0, 1fr); } }
.t2s-card { background: #FFFFFF; border: 1px solid #E2E2DC; border-radius: 14px; overflow: hidden; display: flex; flex-direction: column; }
.t2s-card-head { padding: 16px 18px 12px; display: flex; flex-direction: column; gap: 10px; border-bottom: 1px solid #EDEDE8; }
.t2s-card-title { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.t2s-card-title h3 { margin: 0; font-size: 19px; font-weight: 700; color: #15171C; }
.t2s-card-title span { font-size: 12px; color: #555A64; }
.t2s-badge { display: inline-flex; align-self: flex-start; align-items: center; gap: 7px; padding: 6px 11px;
  border-radius: 999px; font-weight: 600; font-size: 14px; }
.t2s-badge.ok { background: #E3F1E8; color: #145C38; }
.t2s-badge.bad { background: #FBE9E2; color: #8F2F10; }
.t2s-badge.warn { background: #FBF1D6; color: #6E4A00; }
.t2s-badge.plain { background: #F1F1EC; color: #3A3E46; }
.t2s-sql { margin: 0 !important; padding: 16px 18px !important; background: #F7F7F4 !important; font-size: 13px; line-height: 1.6;
  white-space: pre-wrap; word-break: break-word; color: #15171C; flex: 1 1 auto; min-height: 110px; border-radius: 0 !important; }
.t2s-section { padding: 12px 18px; border-top: 1px solid #EDEDE8; display: flex; flex-direction: column; gap: 6px; }
.t2s-label { font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #555A64; }
.t2s-result { font-size: 14px; color: #3A3E46; }
.t2s-more { font-size: 12px; color: #555A64; }
.t2s-rows { max-height: 180px; overflow: auto; border: 1px solid #EDEDE8; border-radius: 8px; }
.t2s-rows table { width: 100%; border-collapse: collapse; font-size: 13px; border-style: hidden !important; border: 0 !important; }
.t2s-rows td { padding: 5px 8px; border: 0 !important; border-bottom: 1px solid #EDEDE8 !important; color: #15171C; }
.t2s-rows table, .t2s-rows tbody, .t2s-rows tr { border-left: 0 !important; border-right: 0 !important; border-top: 0 !important; }
.t2s-results { background: #FFFFFF; border: 1px solid #E2E2DC; border-radius: 14px; padding: 8px 20px 16px; }
.t2s-results table { font-variant-numeric: tabular-nums; }
"""


if __name__ == "__main__":
    main()
