"""Permissions: what the user authorized once, in explicit, inspectable, revocable form.

A permission names the action, its purpose, who it may reach, what may be shared, what it must not do,
how many times and until when. Clearance only proceeds silently inside an active permission; model
confidence never creates or widens one. Repeated approvals can produce a *suggestion*, never a grant.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from ..models import new_id


@dataclass
class Permission:
    action: str                       # e.g. "email.send"
    purpose: str                      # "Routine follow-ups on the Q4 pilot"
    recipients: list[str]             # exact addresses, or "@domain.com" for a whole domain
    may_share: list[str] = field(default_factory=list)   # "meeting notes", "next steps"
    must_not: list[str] = field(default_factory=list)    # "pricing", "new deadlines or delivery dates"
    max_uses: int = 10
    expires_at: float = 0.0           # epoch seconds; 0 = no expiry
    id: str = field(default_factory=new_id)
    created_at: float = field(default_factory=time.time)
    uses: int = 0
    revoked: bool = False
    source: str = "user"              # user | proposal (approved by the user from an agent's request)

    def active(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return not self.revoked and self.uses < self.max_uses and (not self.expires_at or now < self.expires_at)

    def covers_recipient(self, address: str) -> bool:
        address = address.strip().lower()
        for r in self.recipients:
            r = r.strip().lower()
            if r.startswith("@") and address.endswith(r):
                return True
            if address == r:
                return True
        return False

    def public(self) -> dict:
        d = asdict(self)
        d["active"] = self.active()
        return d


def _recipients(args: dict) -> list[str]:
    out = []
    for key in ("to", "cc", "bcc"):
        v = args.get(key) or []
        out += [v] if isinstance(v, str) else list(v)
    return [x.strip().lower() for x in out if x and x.strip()]


@dataclass
class ScopeResult:
    permission: Permission | None
    problems: list[tuple[str, str]]   # (code, human-readable reason)


class PermissionStore:
    def __init__(self, clock=time.time):
        self.clock = clock
        self.items: dict[str, Permission] = {}
        self.suggestions: list[dict] = []

    def add(self, p: Permission) -> Permission:
        self.items[p.id] = p
        return p

    def revoke(self, pid: str) -> Permission:
        p = self.items[pid]
        p.revoked = True
        return p

    def list(self) -> list[Permission]:
        return sorted(self.items.values(), key=lambda p: p.created_at)

    def check(self, action: str, args: dict) -> ScopeResult:
        """Deterministic scope check. Picks the active permission for this action that fits best and lists
        every way the action falls outside it. No permission at all is itself a problem."""
        now = self.clock()
        candidates = [p for p in self.items.values() if p.action == action and not p.revoked]
        if not candidates:
            return ScopeResult(None, [("scope.no_permission", "No permission covers this kind of action yet.")])
        recips = _recipients(args)
        best, best_problems = None, None
        for p in candidates:
            problems = []
            if p.expires_at and now >= p.expires_at:
                problems.append(("scope.expired", "The permission for this has expired."))
            if p.uses >= p.max_uses:
                problems.append(("scope.limit", f"The permission's limit of {p.max_uses} has been used up."))
            outside = [r for r in recips if not p.covers_recipient(r)]
            if outside:
                problems.append(("scope.recipient", "Adds a recipient you didn't approve: " + ", ".join(outside)))
            if not recips and action.startswith("email.send"):
                problems.append(("scope.recipient", "The email has no recipient."))
            if best is None or len(problems) < len(best_problems):
                best, best_problems = p, problems
        return ScopeResult(best, best_problems)

    def note_approval(self, permission_id: str | None, problems: list[tuple[str, str]], args: dict) -> dict | None:
        """After the user approves an out-of-scope recipient more than once, suggest widening the permission.
        This only proposes; the user must accept the suggestion for anything to change."""
        if not permission_id or not any(c == "scope.recipient" for c, _ in problems):
            return None
        p = self.items.get(permission_id)
        if p is None:
            return None
        for r in _recipients(args):
            if p.covers_recipient(r):
                continue
            key = (permission_id, r)
            count = 1 + sum(1 for s in self.suggestions if (s["permission_id"], s["recipient"]) == key)
            entry = {"id": new_id(), "permission_id": permission_id, "recipient": r, "approvals": count,
                     "status": "pending" if count >= 2 else "counting"}
            self.suggestions.append(entry)
            if count >= 2:
                return entry
        return None

    def accept_suggestion(self, suggestion_id: str) -> Permission:
        s = next(x for x in self.suggestions if x["id"] == suggestion_id and x["status"] == "pending")
        p = self.items[s["permission_id"]]
        if not p.covers_recipient(s["recipient"]):
            p.recipients.append(s["recipient"])
        s["status"] = "accepted"
        return p
