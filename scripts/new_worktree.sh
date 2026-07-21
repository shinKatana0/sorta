#!/usr/bin/env bash
# Create a worktree for a feature: ./scripts/new_worktree.sh geo
set -euo pipefail
NAME="${1:?usage: new_worktree.sh <feature-name>}"
ROOT="$(git rev-parse --show-toplevel)"
DIR="$ROOT/../sorta-worktrees/$NAME"
git -C "$ROOT" worktree add -b "feature/$NAME" "$DIR" main
echo "Worktree: $DIR (branch feature/$NAME)"
echo "Start the worker session:  cd '$DIR' && claude"
