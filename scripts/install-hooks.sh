#!/usr/bin/env bash
# Install the local pre-commit hook.
#
# Git hooks are not version controlled, so each person runs this once per clone:
#   bash scripts/install-hooks.sh
#
# The hook WARNS when a contract file is staged. It does not block -- it is a reminder
# that the change needs a branch, a PR and a ping, because an unreviewed edit to one of
# these five files can silently invalidate the whole experiment.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK="$REPO_ROOT/.git/hooks/pre-commit"

cat > "$HOOK" <<'HOOK_BODY'
#!/usr/bin/env bash
CONTRACT_FILES=(
  "configs/base.yaml"
  "src/execution/harness.py"
  "src/execution/compare.py"
  "src/agents/prompts.py"
  "src/rollout.py"
)

staged="$(git diff --cached --name-only)"
hits=()
for f in "${CONTRACT_FILES[@]}"; do
  if grep -qxF "$f" <<< "$staged"; then
    hits+=("$f")
  fi
done

if [ ${#hits[@]} -gt 0 ]; then
  echo ""
  echo "  ============================================================"
  echo "   CONTRACT FILE STAGED"
  echo "  ============================================================"
  for f in "${hits[@]}"; do echo "     $f"; done
  echo ""
  echo "   These define the experimental control. Changing one after"
  echo "   the week-6 freeze means re-running every trained condition."
  echo ""
  echo "   Branch, open a PR, and tell the other person."
  echo "   Log the reason in NOTES.md."
  echo "  ============================================================"
  echo ""
fi

exit 0
HOOK_BODY

chmod +x "$HOOK"
echo "installed $HOOK"
