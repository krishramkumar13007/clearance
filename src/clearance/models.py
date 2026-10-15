from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

Surface = Literal["fs", "shell", "git", "pkg", "net", "email", "permission"]
DecisionKind = Literal["ALLOW", "ALLOW_AUDIT", "ASK", "BLOCK"]


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class AgentFingerprint:
    model: str
    tool_manifest_hash: str
    environment_id: str
    policy_version: str
    system_prompt_hash: str = ""  # minor: recorded, does not reset trust

    @property
    def id(self) -> str:
        major = "|".join([self.model, self.tool_manifest_hash, self.environment_id, self.policy_version])
        return hashlib.sha256(major.encode()).hexdigest()[:12]


@dataclass
class ActionEvent:
    session_id: str
    seq: int
    surface: Surface
    op: str
    args: dict
    fingerprint_id: str
    destination: str | None = None
    payload_digest: str | None = None
    id: str = field(default_factory=new_id)
    ts: datetime = field(default_factory=now)


@dataclass
class NetAttempt:
    host: str
    port: int
    bytes: int
    matched_secret_ids: list[str] = field(default_factory=list)


@dataclass
class Observation:
    action_id: str
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    net_attempts: list[NetAttempt] = field(default_factory=list)
    exit_code: int | None = None
    snapshot_id: str | None = None


@dataclass
class TaintSource:
    path: str
    action_id: str
    secret_ids: list[str]


@dataclass
class SessionLedger:
    session_id: str
    fingerprint_id: str
    taint: set[str] = field(default_factory=set)
    taint_sources: list[TaintSource] = field(default_factory=list)
    files_touched: set[str] = field(default_factory=set)
    destinations: set[str] = field(default_factory=set)
    decisions: list[str] = field(default_factory=list)  # action ids, in order
    recent_surfaces: list[str] = field(default_factory=list)


@dataclass
class RiskEstimate:
    action_id: str
    p_destructive: float
    p_exfiltration: float
    mode: str  # prompted | calibrated | verdict | laya | error
    model: str
    rationale: str = ""  # display only, never a decision input
    latency_ms: int = 0


@dataclass
class DecisionRecord:
    action_id: str
    decision: DecisionKind
    decided_by: str  # rule id, or "expected_cost"
    costs: dict[str, float] | None = None
    evidence: list[str] = field(default_factory=list)
    novelty: float = 0.0
    reversible: bool = True
    human_resolution: Literal["approved_once", "denied"] | None = None
    reason: str = ""  # short, safe to show the agent
    surface: str = ""
    shadow: dict | None = None  # a second estimator's probabilities, recorded but never acted on
    detail: dict | None = None  # surface-specific evidence (permission id, scope problems, judge output)
