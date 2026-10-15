"""Per-surface autonomy, computed from DecisionRecords (never stored)."""
from __future__ import annotations

from collections import defaultdict

from .models import DecisionRecord

MIN_ACTIONS = 20
MAX_DENIAL_HIGH = 0.05   # at most 1 in 20 human-reviewed ASKs denied
MAX_DENIAL_MEDIUM = 0.20
_MISCONDUCT = ("rule.hard_policy", "rule.secret_in_payload")


def autonomy_levels(records: list[DecisionRecord]) -> dict[str, dict]:
    """UNVALIDATED until a surface has MIN_ACTIONS decisions. After that, trust is earned by humans approving
    what was asked: HIGH when almost nothing reviewed was denied and the agent never tripped a hard rule
    anywhere; LOW when reviewers keep denying or it tried something a hard rule blocked."""
    by_surface: dict[str, list[DecisionRecord]] = defaultdict(list)
    for r in records:
        by_surface[r.surface].append(r)
    misconduct = sum(r.decided_by in _MISCONDUCT for r in records)
    out = {}
    for surface in ("fs", "shell", "git", "pkg", "net"):
        rs = by_surface.get(surface, [])
        n = len(rs)
        asks = [r for r in rs if r.decision == "ASK"]
        denied = [r for r in asks if r.human_resolution == "denied"]
        resolved = [r for r in asks if r.human_resolution]
        ask_rate = len(asks) / n if n else 0.0
        override_rate = len(denied) / len(resolved) if resolved else 0.0
        if n < MIN_ACTIONS:
            level = "UNVALIDATED"
        elif misconduct or override_rate > MAX_DENIAL_MEDIUM:
            level = "LOW"
        elif override_rate <= MAX_DENIAL_HIGH:
            level = "HIGH"
        else:
            level = "MEDIUM"
        out[surface] = {"level": level, "actions": n, "ask_rate": round(ask_rate, 3),
                        "override_rate": round(override_rate, 3), "reviewed": len(resolved),
                        "misconduct": misconduct}
    return out
