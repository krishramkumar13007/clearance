"""Kaggle: fine-tune Laya with RLCD on Clearance's email permission questions and score email-v1.

Accelerator: one GPU is enough (Laya is a 421M ModernBERT); T4 x2 also works. Internet must be on.
Input: the clearance-email-cases dataset (cases.jsonl from `python -m clearance clm-export`).
Output: /kaggle/working/email_laya_rlcd_probs.json,
  {"zero": {state_key: probs}, "seed0": {...}, "seed1": {...}, "meta": {...}}.
Copy it to evals/ and run `email-calibrate --judge laya-rlcd --judge laya-rlcd-s1`, then email-eval.

Uses Laya's own loop (`laya.train.finetune`): RLCD loss, temperature fit on a slice held out before
training, a checkpoint `laya.load` reads. Takeoff is seed-dependent per Laya's docs, so two seeds train and
both are reported; neither is chosen using email-v1.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default=None)
ap.add_argument("--work", default="/kaggle/working")
ap.add_argument("--epochs", type=int, default=6)  # 366 updates on 1,428 rows; Laya's reference run spends 352
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--limit", type=int, default=None, help="train on the first N rows only (smoke test)")
args, _ = ap.parse_known_args()

if os.path.isdir("/kaggle"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "laya==0.4.7"], check=True)  # the version results used

import laya  # noqa: E402
from laya.train import TrainConfig, finetune, resolve_checkpoint_dir  # noqa: E402

WORK = args.work
os.makedirs(WORK, exist_ok=True)
CASES = args.cases or glob.glob("/kaggle/input/**/cases.jsonl", recursive=True)[0]
rows = [json.loads(l) for l in open(CASES) if l.strip()]
train = [r for r in rows if r["split"] == "train"][:args.limit]
held = [r for r in rows if r["split"] != "train"]
print(f"[cases] {len(rows)} from {CASES}: train {len(train)}, email-v1 {len(held)} | laya {laya.__version__}",
      flush=True)

train_path = os.path.join(WORK, "train.jsonl")
with open(train_path, "w") as f:
    for r in train:
        f.write(json.dumps({"state": r["state"], "questions": r["questions"],
                            "expected": {q: bool(v) for q, v in r["labels"].items()}}) + "\n")

BASE_REPO, BASE_REVISION = "convaiinnovations/laya", "7b928d828b7b0e022f929d9bd2e44165aa270148"
from huggingface_hub import snapshot_download  # noqa: E402

base = resolve_checkpoint_dir(snapshot_download(BASE_REPO, revision=BASE_REVISION, allow_patterns=[
    "rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]))
QUESTIONS = held[0]["questions"]


def score(agent) -> dict:
    out = agent.predict_batch([r["state"] for r in held], QUESTIONS)
    return {r["key"]: {q: round(float(o["answers"][q]["noul"]), 6) for q in QUESTIONS} for r, o in zip(held, out)}


train_sha = hashlib.sha256(json.dumps([[r["key"], r["labels"]] for r in train], sort_keys=True).encode()).hexdigest()
result = {"zero": score(laya.load(base)),
          "meta": {"epochs": args.epochs, "train_rows": len(train), "runs": {}, "laya": laya.__version__,
                   "base_revision": BASE_REVISION, "train_sha": train_sha}}
for seed in args.seeds:
    out_dir = os.path.join(WORK, f"laya_email_seed{seed}")
    rep = finetune(train_path, base, out_dir, TrainConfig(epochs=args.epochs, seed=seed))
    keep = ("train_items", "calibration_items", "skipped", "epoch_loss", "temperature")
    result["meta"]["runs"][f"seed{seed}"] = {k: rep.get(k) for k in keep}
    print(f"[train] seed {seed}: " + json.dumps(result["meta"]["runs"][f"seed{seed}"], default=str), flush=True)
    result[f"seed{seed}"] = score(laya.load(out_dir))

json.dump(result, open(os.path.join(WORK, "email_laya_rlcd_probs.json"), "w"), default=str)

# quick look: mean P(yes) on email-v1 by expected outcome (the real scoring happens in Clearance)
for head in [k for k in result if k != "meta"]:
    for exp in ("proceed", "ask"):
        ks = [r["key"] for r in held if r.get("expected") == exp]
        m = {q: round(sum(result[head][k][q] for k in ks) / len(ks), 3) for q in QUESTIONS}
        print(f"[score] {head:<6} expected {exp:<7} n={len(ks)} mean P(yes) {m}", flush=True)
print("[done] wrote email_laya_rlcd_probs.json", flush=True)
