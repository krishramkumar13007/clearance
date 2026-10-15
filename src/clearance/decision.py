"""Decision engine: rules first (escalate only), then minimum expected cost."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass

from .models import ActionEvent, DecisionRecord, RiskEstimate, SessionLedger
from .policy import Policy, hard_policy_violation
from .secrets import SecretIndex

# Tie-break toward the safer choice: ASK, then BLOCK, then ALLOW.
_TIE_RANK = {"ASK": 0, "BLOCK": 1, "ALLOW": 2}


@dataclass(frozen=True)
class CostMatrix:
    ask: float = 1.0
    false_block: float = 5.0
    false_allow_destructive: float = 200.0
    false_allow_exfil: float = 500.0

    def costs(self, p_d: float, p_e: float) -> dict[str, float]:
        return {
            "ALLOW": p_d * self.false_allow_destructive + p_e * self.false_allow_exfil,
            "ASK": self.ask,
            "BLOCK": (1 - p_d) * (1 - p_e) * self.false_block,
        }


def valid_probability(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and 0.0 <= x <= 1.0


def signature(a: ActionEvent) -> str:
    """Normalized signature: surface, op, full command template (digits collapsed), destination."""
    template = ""
    if a.surface == "shell":
        template = re.sub(r"\d+", "N", a.args.get("command", "").strip())
    elif a.surface == "pkg":
        template = a.args.get("package", "")
    return f"{a.surface}:{a.op}:{hashlib.sha256(template.encode()).hexdigest()[:10] if template else ''}:{a.destination or ''}"


def is_reversible(a: ActionEvent) -> bool:
    """Gateway-computed. Agent-supplied hints (e.g. args["snapshotted"]) are ignored."""
    if a.surface == "net":
        return False
    if a.surface == "git" and a.op == "push":
        return False
    if a.surface == "shell":
        return False  # no gateway-taken restore point covers arbitrary shell effects yet
    return True  # fs edits, commits, sandbox installs are snapshot-covered


NOVELTY_THRESHOLD = 3
EARNED_MAX_P_DESTRUCTIVE = 0.3
EARNED_MAX_P_EXFIL = 0.05


@dataclass
class History:
    """Per-fingerprint counts of EXECUTED action signatures (ASK/BLOCK never count)."""
    counts: Counter

    def novelty(self, a: ActionEvent) -> float:
        return 1.0 if self.counts[signature(a)] < NOVELTY_THRESHOLD else 0.0

    def record(self, a: ActionEvent) -> None:
        self.counts[signature(a)] += 1


def outbound_payload(a: ActionEvent) -> str | None:
    """Everything the agent supplied that could carry data out. Scans all args, not a chosen field."""
    if a.surface in ("net", "shell", "git", "pkg"):
        return json.dumps(a.args, default=str, sort_keys=True) + "\n" + (a.destination or "")
    return None


def decide(
    a: ActionEvent,
    ledger: SessionLedger,
    policy: Policy,
    secrets: SecretIndex,
    history: History,
    risk: RiskEstimate | None,
    costs: CostMatrix = CostMatrix(),
    autonomy: str | None = None,
) -> DecisionRecord:
    novelty = history.novelty(a)
    reversible = is_reversible(a)
    if a.surface == "shell" and a.args.get("_sandboxed"):
        reversible = True  # workspace restore point taken before execution; container state is disposable
    if (a.surface == "net" and str(a.args.get("method", "GET")).upper() in ("GET", "HEAD")
            and policy.is_trusted(a.destination)):
        reversible = True  # read-only fetch from a trusted registry has no outside side effect

    def rec(decision, by, reason="", evidence=None, cost=None):
        return DecisionRecord(
            action_id=a.id, decision=decision, decided_by=by, costs=cost,
            evidence=evidence or [], novelty=novelty, reversible=reversible, reason=reason, surface=a.surface,
        )

    # 1. Hard policy
    reason = hard_policy_violation(policy, a.surface, a.op, a.args)
    if reason:
        return rec("BLOCK", "rule.hard_policy", reason)

    # 2. Secret content in any outbound-capable payload, trusted destination or not
    hits = secrets.scan(outbound_payload(a))
    if hits:
        return rec("BLOCK", "rule.secret_in_payload", "outbound payload contains workspace secret material", evidence=hits)

    # 3. Tainted session reaching beyond the sandbox
    if ledger.taint:
        egress = (
            (a.surface == "net" and not policy.is_trusted(a.destination))
            or (a.surface == "git" and a.op == "push" and not policy.is_trusted(a.destination))
            or (a.surface == "pkg" and a.args.get("registry") and not policy.is_trusted(a.args.get("registry")))
            # Unsandboxed shell has no egress control, so it is default-ASK when tainted. In the sandbox every
            # request the command makes is held at the proxy and decided as net egress instead.
            or (a.surface == "shell" and not a.args.get("_sandboxed"))
        )
        if egress:
            return rec("ASK", "rule.taint_egress", "session read secrets; this action may send data out",
                       evidence=sorted(ledger.taint))

    # 4. Novel and irreversible
    if novelty >= 1.0 and not reversible:
        return rec("ASK", "rule.novel_irreversible", "novel action that cannot be undone")

    # 5. Expected cost (arm A has no estimate: rules only)
    if risk is None:
        decision, by, cost = "ALLOW", "rules_only", None
    elif not (valid_probability(risk.p_destructive) and valid_probability(risk.p_exfiltration)):
        return rec("ASK", "rule.invalid_risk", "risk estimate unavailable or invalid")
    else:
        cost = costs.costs(risk.p_destructive, risk.p_exfiltration)
        decision, by = min(cost, key=lambda k: (cost[k], _TIE_RANK[k])), "expected_cost"

    # 5b. Earned autonomy: a fingerprint with a clean reviewed history skips the interruption for actions the
    #     gateway can roll back, when the model is only mildly unsure. Rules above still escalate first.
    if (decision == "ASK" and by == "expected_cost" and autonomy == "HIGH" and reversible
            and risk.p_destructive < EARNED_MAX_P_DESTRUCTIVE and risk.p_exfiltration < EARNED_MAX_P_EXFIL):
        return rec("ALLOW_AUDIT", "earned_autonomy", cost=cost)

    # 6. Audit trail for allowed actions in a tainted session
    if decision == "ALLOW" and ledger.taint:
        decision = "ALLOW_AUDIT"
    return rec(decision, by, cost=cost)
