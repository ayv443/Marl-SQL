#!/usr/bin/env bash
#
# Per-session development setup. Safe to run every time you open a SageMaker space,
# and safe to run twice -- everything here is idempotent.
#
# A fresh space has no git config and will otherwise author your commits as
# "sagemaker-user", which breaks the repository's commit convention.
#
# Usage:
#
#   GIT_NAME="Your Name" GIT_EMAIL="you@example.com" bash scripts/dev-setup.sh
#
# or set them once in your shell profile and just run:
#
#   bash scripts/dev-setup.sh
#
# Contains no secrets. Your access token is handled separately -- see the notes this
# script prints at the end.

set -euo pipefail

say()  { printf '  %s\n' "$*"; }
head2() { printf '\n%s\n' "$*"; }
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; }

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

printf '\n==============================================\n'
printf ' marl-sql session setup\n'
printf '==============================================\n'
say "repo: $REPO_ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  bad "not a git repository -- run this from inside the clone"
  exit 1
fi

# ---------------------------------------------------------------- identity ----
head2 "1. git identity"

GIT_NAME="${GIT_NAME:-$(git config --global user.name 2>/dev/null || true)}"
GIT_EMAIL="${GIT_EMAIL:-$(git config --global user.email 2>/dev/null || true)}"

if [ -z "$GIT_NAME" ] || [ -z "$GIT_EMAIL" ]; then
  bad "GIT_NAME and GIT_EMAIL are not set and no global config exists"
  say ""
  say "Run it like this:"
  say '  GIT_NAME="Your Name" GIT_EMAIL="you@example.com" bash scripts/dev-setup.sh'
  say ""
  say "Or persist them so you never pass them again:"
  say '  echo '"'"'export GIT_NAME="Your Name"'"'"' >> ~/.bashrc'
  say '  echo '"'"'export GIT_EMAIL="you@example.com"'"'"' >> ~/.bashrc'
  exit 1
fi

git config --global user.name  "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"
git config --local  user.name  "$GIT_NAME"
git config --local  user.email "$GIT_EMAIL"
ok "author set to $GIT_NAME <$GIT_EMAIL>"

# Never let tooling append co-author or "generated with" trailers: commits in this
# repository carry the author's own name and nothing else.
git config --local commit.cleanup strip
ok "commit trailers: cleanup=strip"

# ------------------------------------------------------------- credentials ----
head2 "2. push credentials"

if [ -f "$HOME/.git-credentials" ]; then
  chmod 600 "$HOME/.git-credentials" 2>/dev/null || true
  ok "stored credentials found (~/.git-credentials, mode 600)"
else
  git config --global credential.helper store
  warn "no stored credentials yet"
  say "Your FIRST push will prompt for username and password."
  say "Use your GitHub username, and a personal access token as the password"
  say "(a real password will be rejected). Create one at:"
  say "  github.com -> Settings -> Developer settings -> Tokens (classic)"
  say "  scope needed: repo"
  say "It gets saved to ~/.git-credentials so you are only asked once."
fi

# ------------------------------------------------------------- local files ----
head2 "3. local-only files"

# A one-line pointer at the committed conventions, for any coding assistant used in
# this clone. Deliberately NOT committed: it is excluded through .git/info/exclude,
# which is local to this clone and leaves no trace in the repository.
if [ ! -f CLAUDE.md ]; then
  {
    echo "Read CONVENTIONS.md in the repository root and follow it."
    echo ""
    echo "The specification, execution plan and task breakdown are shared separately"
    echo "and are not in this repository. If they are not available to you, say so"
    echo "rather than inventing their contents."
    echo ""
    echo "Do not edit files owned by the other person without saying so first."
  } > CLAUDE.md
  ok "created CLAUDE.md (local only)"
else
  ok "CLAUDE.md already present"
fi

EXCLUDE=".git/info/exclude"
mkdir -p "$(dirname "$EXCLUDE")"
touch "$EXCLUDE"
if ! grep -qxF "CLAUDE.md" "$EXCLUDE"; then
  printf '\n# local-only, never pushed\nCLAUDE.md\n' >> "$EXCLUDE"
  ok "excluded CLAUDE.md from git"
else
  ok "CLAUDE.md already excluded"
fi

if ! grep -qxF "my-setup.sh" "$EXCLUDE"; then
  printf 'my-setup.sh\n' >> "$EXCLUDE"
  ok "excluded my-setup.sh from git"
fi

# ------------------------------------------------------------------ hooks -----
head2 "4. contract-file warning hook"

if [ -f scripts/install-hooks.sh ]; then
  bash scripts/install-hooks.sh >/dev/null 2>&1 && ok "pre-commit hook installed" \
    || warn "hook install failed -- not fatal"
else
  warn "scripts/install-hooks.sh not found on this branch"
fi

# ---------------------------------------------------------------- machine -----
head2 "5. machine check"

PY="$(python --version 2>&1 || true)"
say "python     $PY"
say "record this in NOTES.md -- both spaces must match"

AVAIL="$(df -h "$HOME" | awk 'NR==2 {print $4}')"
say "disk free  $AVAIL on $HOME"
case "$AVAIL" in
  *G)
    NUM="${AVAIL%G}"
    NUM="${NUM%%.*}"
    if [ "${NUM:-0}" -lt 20 ] 2>/dev/null; then
      warn "under 20 GB free. Spider plus the model cache will not fit."
      warn "If the space was never resized from the 5 GB default, raise it with the"
      warn "admin now -- that has someone else's turnaround time attached."
    else
      ok "enough room for Spider and the model cache"
    fi
    ;;
  *) warn "could not parse free space: $AVAIL" ;;
esac

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  say "gpu        ${GPU:-unknown}"
  case "$GPU" in
    *T4*) ok "T4 confirmed: Turing, so fp16 only -- no bf16, no Flash Attention 2" ;;
    *)    warn "not a T4. The fp16-only constraint assumes Turing; re-check before training" ;;
  esac
else
  say "gpu        nvidia-smi not available"
fi

# ----------------------------------------------------------------- verify -----
head2 "6. verify"

say "branch     $(git rev-parse --abbrev-ref HEAD)"
say "remote     $(git remote get-url origin 2>/dev/null || echo 'none')"
printf '  last commit\n'
git log -1 --format='             %h  %an <%ae>  %s' 2>/dev/null || say "   (no commits yet)"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [ "$BRANCH" = "main" ]; then
  printf '\n'
  warn "you are on main."
  warn "The five contract files reach main through a pull request, never directly:"
  warn "  configs/base.yaml  src/execution/harness.py  src/execution/compare.py"
  warn "  src/agents/prompts.py  src/rollout.py"
  warn "Work on a branch: git checkout -b <yourname>/<what>"
fi

printf '\n==============================================\n'
printf ' ready\n'
printf '==============================================\n'
say "pip install -r requirements.txt           # once per space"
say "pytest tests/test_compare.py              # no GPU or data needed"
say "python scripts/download_data.py --spider-only"
say "python -m src.data.filter --report"
say "python scripts/benchmark_harness.py --n 100"
printf '\n'
say "Long runs go in tmux, never a notebook cell -- the space idle-shuts after 60 min:"
say "  tmux new -s train"
say "  python -m src.train --config configs/independent.yaml 2>&1 | tee results/run.log"
say "  Ctrl-b d detaches, tmux attach -t train returns"
printf '\n'
