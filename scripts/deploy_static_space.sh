#!/usr/bin/env bash
# Publish the static demo to a free Hugging Face Space: scripts/deploy_static_space.sh <hf-user>/<space-name>
# Needs the hf CLI logged in (`hf auth login`). Creates the Space (sdk: static) if it does not exist yet.
set -euo pipefail
SPACE="${1:?usage: scripts/deploy_static_space.sh <hf-user>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
(cd "$ROOT" && uv run python -m clearance static-demo --out "$STAGE")
cp "$ROOT/space-static/README.md" "$STAGE/"
hf repos create "$SPACE" --type space --sdk static --no-private --exist-ok
hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy Clearance static demo $(git -C "$ROOT" rev-parse --short HEAD)"
