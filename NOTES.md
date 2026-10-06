# NOTES

Append-only. **Newest entry at the top.** Every entry dated and initialled.

Every surprising number, every "why is it like this", every decision. This is what gets
mined when the report is written, and it costs 30 seconds per entry. Chat is not searchable
in week 10; this file is.

---

## Numbers to record (targets from the spec, not yet measured)

| Measurement | Target | Actual | Date |
|---|---|---|---|
| Python version (both machines must match) | same | | |
| Storage available (`df -h`) | 50 GB | | |
| Spider database folders | ~166 | | |
| Filtered out, total | ~1,700 | | |
| &nbsp;&nbsp;of which: gold returned zero rows | majority | | |
| &nbsp;&nbsp;of which: gold timed out | | | |
| &nbsp;&nbsp;of which: gold errored | | | |
| &nbsp;&nbsp;of which: db file missing | | | |
| Harness: 100 queries end to end | < 60 s | | |
| `pytest tests/test_compare.py` | green | | |
| pass@8 (200 questions, untrained base) | 10-60% | | |
| GPU memory at model load | | | |
| Counterfactual overhead vs independent | ~1.5x | | |
| Idle shutdown killed a detached tmux run? | | | |

---

## Result comparison semantics (paste the final version here)

These change every number in the report and must be documented in the write-up.
Copy them out of `src/execution/compare.py` once the implementation is settled.

---

<!-- new entries above this line -->
