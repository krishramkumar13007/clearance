"""CLM (Contrastive-LM, Apache 2.0) as an email judge, trained and scored offline on Kaggle.

`export` writes every case the judge will see (the training split plus email-v1's calib and eval content cases)
in CLM's System One shape: the judge state plus one `noul` question per entry in QUESTIONS. kaggle/clm-email/run.py
embeds them with Qwen3-8B, fine-tunes CLM's heads on the training split, and writes per-case probabilities
for the released head ("clm-zero") and the fine-tuned one ("clm"). ClmJudge looks those up by state, so
email-eval scores CLM exactly like the live judges. An unknown state raises, which the assistant treats as an ASK.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from .evals import _permission, load
from .judge import QUESTIONS, judge_state
from .train_data import OUT as TRAIN

ROOT = Path(__file__).resolve().parents[3]
KAGGLE_DIR = ROOT / "kaggle" / "clm-email"
PROBS = ROOT / "evals" / "email_clm_probs.json"
LAYA_RLCD_PROBS = ROOT / "evals" / "email_laya_rlcd_probs.json"  # kaggle/laya-email/run.py, same layout
NOUL_QUESTIONS = {q: {"type": "noul", "instructions": text} for q, text in QUESTIONS.items()}


def state_key(state: dict) -> str:
    return hashlib.sha1(json.dumps(state, sort_keys=True).encode()).hexdigest()


def cases() -> list[dict]:
    out = []
    for t in (json.loads(l) for l in TRAIN.read_text().splitlines() if l.strip()):
        perm = SimpleNamespace(**t["permission"])
        st = judge_state(perm, t["email"], t["replying_to"])
        out.append({"id": t["id"], "split": "train", "group": t["scenario"], "key": state_key(st), "state": st,
                    "labels": t["labels"]})
    for c in load():
        if c["id"].startswith("r"):  # recipient cases: rules decide before any judge runs
            continue
        st = judge_state(_permission(), c["email"], {"cc": [], **c["replying_to"]})
        out.append({"id": c["id"], "split": c["split"], "group": "pilot", "key": state_key(st), "state": st,
                    "expected": c["expected"]})
    return out


def train_sha(train_rows) -> str:
    """Fingerprint of the training rows (state key + labels), stored in the probs meta so stale results show."""
    return hashlib.sha256(json.dumps([[r["key"], r["labels"]] for r in train_rows], sort_keys=True).encode()).hexdigest()


def export(kaggle_user: str | None = None, out_dir: Path = KAGGLE_DIR) -> dict:
    """Write the Kaggle dataset (cases.jsonl) and, given a username, the dataset and kernel metadata."""
    data = out_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    cs = cases()
    (data / "cases.jsonl").write_text("".join(json.dumps({**c, "questions": NOUL_QUESTIONS}) + "\n" for c in cs))
    if kaggle_user:
        (data / "dataset-metadata.json").write_text(json.dumps(
            {"title": "clearance-email-cases", "id": f"{kaggle_user}/clearance-email-cases",
             "licenses": [{"name": "Apache 2.0"}]}, indent=1) + "\n")
        (out_dir / "kernel-metadata.json").write_text(json.dumps(
            {"id": f"{kaggle_user}/clearance-clm-email", "title": "clearance-clm-email", "code_file": "run.py",
             "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
             "enable_internet": True, "machine_shape": "NvidiaTeslaT4",  # GPU T4 x2
             "dataset_sources": [f"{kaggle_user}/clearance-email-cases"]}, indent=1) + "\n")
    by = {s: sum(c["split"] == s for c in cs) for s in ("train", "calib", "eval")}
    return {"dir": str(out_dir), "cases": len(cs), **by}


@dataclass
class ClmJudge:
    """Offline lookup of probabilities computed on Kaggle. For CLM, head is "finetuned" or "zero" (released head);
    the Laya RLCD file (kaggle/laya-email) uses the same layout with heads "zero", "seed0" and "seed1"."""
    head: str = "finetuned"
    path: Path = PROBS
    name: str = "clm"
    table: dict = field(default_factory=dict)

    def __post_init__(self):
        self.table = json.loads(self.path.read_text())[self.head]

    def __call__(self, state: dict) -> dict[str, float]:
        p = self.table[state_key(state)]
        return {q: float(p[q]) for q in QUESTIONS}
