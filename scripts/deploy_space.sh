#!/usr/bin/env bash
# Upload the demo to a Hugging Face Space: scripts/deploy_space.sh <hf-user>/<space-name>
# Needs the hf CLI logged in (`hf auth login`). Set NEBIUS_API_KEY as a secret in the Space settings yourself.
set -euo pipefail
SPACE="${1:?usage: scripts/deploy_space.sh <hf-user>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "$ROOT/space/Dockerfile" "$ROOT/space/README.md" "$ROOT/pyproject.toml" "$STAGE/"
mkdir -p "$STAGE/evals"
cp "$ROOT/evals/email_thresholds.json" "$ROOT/evals/demo_judge_cache.json" "$STAGE/evals/"
# demo.py is the coding-agent demo: unused here, and its fake .env key trips secret scanners
rsync -a --exclude '__pycache__' --exclude '.DS_Store' --exclude 'clearance/demo.py' "$ROOT/src" "$STAGE/"
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy Clearance demo $(git -C "$ROOT" rev-parse --short HEAD)"
