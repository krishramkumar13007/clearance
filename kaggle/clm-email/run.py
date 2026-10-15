"""Kaggle: fine-tune CLM's heads on Clearance's email permission questions and score email-v1.

Accelerator: GPU T4 x2 (Qwen3-8B in fp16 is ~16 GB, split across both cards). Internet must be on.
Input: the clearance-email-cases dataset (cases.jsonl from `python -m clearance clm-export`).
Output: /kaggle/working/email_clm_probs.json, {"zero": {state_key: probs}, "finetuned": {...}, "meta": {...}}.
Copy it to evals/email_clm_probs.json and run `email-calibrate --judge clm --judge clm-zero`, then email-eval.

Steps: embed every distinct text once (Qwen3-8B, last-token pooling, L2-normalised, like CLM's vLLM recipe),
save them in finetune.py's cache format so it never needs vLLM, train the heads warm-started from the
released CLM-v0.1-8B head with one whole scenario held out for validation, then score both heads.
`--dry-run` swaps the encoder for random vectors to test the pipeline on a CPU.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

CLM_REPO = "https://github.com/Contrastive-LM/CLM"
CLM_COMMIT = "d5f9ef0"
MODEL = "Qwen/Qwen3-8B"
MAX_LEN = 2048
HEAD_REPO, HEAD_REVISION = "Contrastive-LM/CLM-v0.1-8B", "e939398d4556fcd9400c76fa8c5a513202f42b0a"

ap = argparse.ArgumentParser()
ap.add_argument("--cases", default=None, help="cases.jsonl (default: found under /kaggle/input)")
ap.add_argument("--work", default="/kaggle/working")
ap.add_argument("--clm-dir", default=None, help="existing CLM checkout (default: clone into WORK/CLM)")
ap.add_argument("--dry-run", action="store_true", help="random embeddings, no encoder (pipeline test)")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--loss", choices=["softce", "infonce"], default="softce",
                help="softce: softmax over each question's own yes/no, which is how the judge is scored")
args, _ = ap.parse_known_args()  # notebooks pass extra argv

WORK = args.work
os.makedirs(WORK, exist_ok=True)
CASES = args.cases or glob.glob("/kaggle/input/**/cases.jsonl", recursive=True)[0]
CLM = args.clm_dir or os.path.join(WORK, "CLM")
if not os.path.isdir(CLM):
    subprocess.run(["git", "clone", "-q", CLM_REPO, CLM], check=True)
    subprocess.run(["git", "-C", CLM, "checkout", "-q", CLM_COMMIT], check=True)
sys.path[:0] = [os.path.join(CLM, "src"), os.path.join(CLM, "train")]
from clm.schema import build_pairs  # noqa: E402

rows = [json.loads(l) for l in open(CASES) if l.strip()]
print(f"[cases] {len(rows)} from {CASES}", flush=True)

# ---- 1. typed-decision parquet for finetune.py: id = scenario, so the val split holds out whole scenarios ----
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

data_dir = os.path.join(WORK, "data", "email")
os.makedirs(data_dir, exist_ok=True)


def typed(r, with_gold):
    gold = {q: {"label": bool(v)} for q, v in r["labels"].items()} if with_gold else {}
    return {"id": r["group"], "workflow": "email", "state": json.dumps(r["state"]),
            "questions": json.dumps(r["questions"]), "gold": json.dumps(gold)}


train = [r for r in rows if r["split"] == "train"]
held = [r for r in rows if r["split"] != "train"]
pq.write_table(pa.Table.from_pylist([typed(r, True) for r in train]), os.path.join(data_dir, "train-0.parquet"))
# email-v1 has no per-question labels; it is scored separately below, never trained or selected on.
pq.write_table(pa.Table.from_pylist([typed(r, False) for r in held]), os.path.join(data_dir, "test-0.parquet"))

# ---- 2. embed every distinct text once ----
pairs = {r["key"]: build_pairs(r["state"], r["questions"]) for r in rows}
texts = list(dict.fromkeys(t for p in pairs.values() for st, _, cands in p.values() for t in (st, *cands)))
print(f"[embed] {len(texts)} distinct texts", flush=True)


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


if args.dry_run:
    rng = np.random.default_rng(0)
    vecs = l2(rng.standard_normal((len(texts), 4096)).astype(np.float32))
else:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto").eval()
    ids = [tok(t, add_special_tokens=False)["input_ids"][-(MAX_LEN - 1):] or [tok(" ")["input_ids"][0]]
           for t in texts]  # tail truncation, as finetune.py's Recipe.text_ids(keep="tail")
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    out = np.zeros((len(ids), model.config.hidden_size), dtype=np.float32)
    first = next(model.parameters()).device
    with torch.no_grad():
        for b in range(0, len(order), 16):
            idx = order[b:b + 16]
            n = max(len(ids[i]) for i in idx)
            x = torch.full((len(idx), n), tok.pad_token_id or 0, dtype=torch.long)
            m = torch.zeros((len(idx), n), dtype=torch.long)
            for j, i in enumerate(idx):
                x[j, :len(ids[i])] = torch.tensor(ids[i]); m[j, :len(ids[i])] = 1
            h = model(input_ids=x.to(first), attention_mask=m.to(first)).last_hidden_state  # after final norm
            last = m.sum(1).to(h.device) - 1
            out[idx] = h[torch.arange(len(idx), device=h.device), last].float().cpu().numpy()
            if b % 320 == 0:
                print(f"[embed] {b + len(idx)}/{len(order)}", flush=True)
    if not np.isfinite(out).all():
        raise SystemExit("[embed] non-finite embeddings (fp16 overflow); rerun on a bf16-capable GPU")
    vecs = l2(out)
    del model
    torch.cuda.empty_cache()

emb_dir = os.path.join(WORK, "embeddings")
os.makedirs(emb_dir, exist_ok=True)
sha = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
np.savez(os.path.join(emb_dir, f"choice_{MODEL.replace('/', '_')}_{MAX_LEN}.npz"),
         keys=np.array(sha), vecs=vecs.astype(np.float16))
vec_of = dict(zip(texts, vecs.astype(np.float16).astype(np.float32)))

# ---- 3. released head ----
from huggingface_hub import hf_hub_download  # noqa: E402

init = hf_hub_download(HEAD_REPO, "CLM_v0.1-8B.pt", revision=HEAD_REVISION, local_dir=os.path.join(WORK, "ckpt"))

# ---- 4. fine-tune ----
run_dir = os.path.join(WORK, "run")
subprocess.run([sys.executable, os.path.join(CLM, "train", "finetune.py"), "--task", "choice",
                "--data", os.path.join(WORK, "data"), "--workflow", "email", "--init-ckpt", init,
                "--targets", "hard", "--loss", args.loss, "--embed-model", MODEL, "--max-len", str(MAX_LEN),
                "--embed-cache", emb_dir, "--out-dir", run_dir, "--epochs", str(args.epochs),
                "--lr", str(args.lr), "--batch", str(args.batch), "--patience", "8"], check=True)

# ---- 5. score both heads on every case ----
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from clm.heads import make_head  # noqa: E402


def load_head(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)  # tensors and plain dicts only, no pickled code
    c = ck["cfg"]
    hs = []
    for k in ("state_head", "action_head"):
        h = make_head(c["width"], c.get("depth", 2), c.get("projection_dim", 512), c.get("activation", "gelu"),
                      c.get("layernorm", False), c.get("residual", False))
        h.load_state_dict(ck[k]); hs.append(h.eval())
    return hs[0], hs[1], float(torch.as_tensor(ck["logit_scale"]).exp().clamp(max=100.0))


@torch.no_grad()
def score(path):
    sh, ah, scale = load_head(path)
    table = {}
    for key, p in pairs.items():
        out = {}
        for q, (st, keys, cands) in p.items():
            zs = F.normalize(sh(torch.tensor(vec_of[st])[None]), dim=-1)
            zc = F.normalize(ah(torch.tensor(np.stack([vec_of[t] for t in cands]))), dim=-1)
            pr = torch.softmax(scale * (zs @ zc.T)[0], -1)
            out[q] = round(float(pr[keys.index("true")]), 6)
        table[key] = out
    return table


summary = json.load(open(os.path.join(run_dir, "finetune_summary.json")))
hist = json.load(open(os.path.join(run_dir, "history.json")))
result = {"zero": score(init), "finetuned": score(os.path.join(run_dir, "best_head.pt")),
          "meta": {"clm_commit": CLM_COMMIT, "model": MODEL, "dry_run": args.dry_run,
                   "best_epoch": summary["best_epoch"], "val_acc": summary["val_acc"],
                   "init_val_acc": hist[0]["val"]["acc"], "train_questions": summary["train_questions"],
                   "minutes": summary["minutes"], "lr": args.lr, "batch": args.batch, "loss": args.loss,
                   "head_revision": HEAD_REVISION,
                   "train_sha": hashlib.sha256(json.dumps([[r["key"], r["labels"]] for r in train],
                                                          sort_keys=True).encode()).hexdigest()}}
json.dump(result, open(os.path.join(WORK, "email_clm_probs.json"), "w"))

# quick look: mean P(yes) on email-v1 cases by expected outcome (the real scoring happens in Clearance)
for head in ("zero", "finetuned"):
    for exp in ("proceed", "ask"):
        ks = [r["key"] for r in held if r.get("expected") == exp]
        m = {q: round(float(np.mean([result[head][k][q] for k in ks])), 3) for q in rows[0]["questions"]}
        print(f"[score] {head:<9} expected {exp:<7} n={len(ks)} mean P(yes) {m}", flush=True)
print(f"[done] wrote {os.path.join(WORK, 'email_clm_probs.json')} | val acc init "
      f"{result['meta']['init_val_acc']:.3f} -> best {result['meta']['val_acc']:.3f}", flush=True)
