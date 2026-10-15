"""Personal-assistant surfaces for the gateway: permission requests and email.

Decision order for every email action:
  1. Is there an active permission for this kind of action?            no  -> ASK (offer to create one)
  2. Is the action inside it (recipients, limits, expiry)?               no  -> ASK, saying exactly what changed
  3. Judge (optional): new commitment / outside sharing / off purpose?   yes -> ASK, saying which
  4. Otherwise proceed silently and log it, citing the permission.
The judge can only add an ASK. Nothing here can create or widen a permission without the user.
"""
from __future__ import annotations

import json
import time

from ..models import ActionEvent, DecisionRecord, new_id
from .judge import REASONS, JudgeCosts, evaluate
from .mailbox import MockMailbox
from .permissions import Permission, PermissionStore, _recipients

SIDE_EFFECTS = {("email", "send")}


class Assistant:
    def __init__(self, store: PermissionStore | None = None, mailbox: MockMailbox | None = None, judge=None,
                 costs: JudgeCosts = JudgeCosts()):
        self.store = store or PermissionStore()
        self.mailbox = mailbox or MockMailbox()
        self.judge = judge
        self.costs = costs

    # ---------- decisions ----------
    def decide(self, a: ActionEvent) -> DecisionRecord:
        def rec(decision, by, reason="", detail=None):
            return DecisionRecord(action_id=a.id, decision=decision, decided_by=by, reason=reason,
                                  surface=a.surface, reversible=False, detail=detail or {})

        if a.surface == "permission":
            return rec("ASK", "permission.request", "Your assistant is asking for a standing permission.",
                       {"proposal": self._proposal(a.args)})
        if a.surface != "email":
            return rec("BLOCK", "assistant.unknown_surface", "Unsupported action.")

        sends = [p for p in self.store.items.values() if p.action == "email.send" and p.active()]
        if a.op == "list":
            if sends:
                return rec("ALLOW", f"permission:{sends[0].id}", detail={"permission_id": sends[0].id})
            return rec("ASK", "scope.no_permission", "No permission covers working in your inbox yet.")
        if a.op == "read":
            sender = self.mailbox.sender_of(str(a.args.get("id", "")))
            p = next((p for p in sends if sender and p.covers_recipient(sender)), None)
            if p:
                return rec("ALLOW", f"permission:{p.id}", detail={"permission_id": p.id})
            return rec("ASK", "scope.read", f"Reading mail from {sender or 'an unknown sender'} is outside your "
                                            "permission.")
        if a.op != "send":
            return rec("BLOCK", "assistant.unknown_op", "Unsupported email action.")

        scope = self.store.check("email.send", a.args)
        detail = {"permission_id": scope.permission.id if scope.permission else None,
                  "problems": [c for c, _ in scope.problems]}
        if scope.problems:
            return rec("ASK", scope.problems[0][0], " ".join(r for _, r in scope.problems), detail)
        if self.judge is not None:
            replying_to = self._replying_to(a.args)
            j = evaluate(self.judge, scope.permission, a.args, replying_to, self.costs)
            detail["judge"] = {"name": getattr(self.judge, "name", "judge"), **j}
            if "error" in j:
                return rec("ASK", "judge.unavailable", "Couldn't check this draft automatically.", detail)
            if j["flags"]:
                return rec("ASK", "judge." + j["flags"][0], REASONS[j["flags"][0]] + ".", detail)
        return rec("ALLOW", f"permission:{scope.permission.id}", detail=detail)

    # ---------- execution ----------
    def execute(self, a: ActionEvent, d: DecisionRecord) -> str:
        if a.surface == "permission":
            p = self.store.add(Permission(**self._proposal(a.args), source="proposal"))
            return json.dumps({"permission_id": p.id, "status": "active"})
        if a.op == "list":
            return json.dumps(self.mailbox.list())
        if a.op == "read":
            return json.dumps(self.mailbox.read(str(a.args["id"])))
        msg = self.mailbox.send(_as_list(a.args.get("to")), _as_list(a.args.get("cc")),
                                str(a.args.get("subject", "")), str(a.args.get("body", "")), a.args.get("in_reply_to"))
        pid = (d.detail or {}).get("permission_id")
        if pid and pid in self.store.items:
            self.store.items[pid].uses += 1
        return json.dumps({"sent": msg["id"]})

    def on_approved(self, d: DecisionRecord, a: ActionEvent) -> dict | None:
        """Approving the same out-of-scope recipient twice suggests widening the permission (never silently)."""
        if a.surface != "email" or a.op != "send":
            return None
        problems = [(c, "") for c in (d.detail or {}).get("problems", [])]
        return self.store.note_approval((d.detail or {}).get("permission_id"), problems, a.args)

    # ---------- helpers ----------
    def _replying_to(self, args: dict) -> dict | None:
        mid = args.get("in_reply_to")
        try:
            return self.mailbox.read(str(mid)) if mid else None
        except KeyError:
            return None

    @staticmethod
    def _proposal(args: dict) -> dict:
        days = float(args.get("days", 7))
        return {"action": str(args.get("action", "email.send")), "purpose": str(args.get("purpose", ""))[:300],
                "recipients": [str(r).lower() for r in args.get("recipients", [])][:50],
                "may_share": [str(x) for x in args.get("may_share", [])][:20],
                "must_not": [str(x) for x in args.get("must_not", [])][:20],
                "max_uses": max(1, min(int(args.get("max_uses", 10)), 100)),
                "expires_at": time.time() + days * 86400 if days > 0 else 0.0}

    @staticmethod
    def describe(a: ActionEvent) -> str:
        if a.surface == "permission":
            return f"permission for {a.args.get('action', 'email.send')}: {a.args.get('purpose', '')}"[:200]
        if a.op == "send":
            to = ", ".join(_recipients(a.args))
            return f"to {to}: {a.args.get('subject', '')}"[:200]
        if a.op == "read":
            return f"message {a.args.get('id', '')}"
        return "inbox"


def _as_list(v) -> list[str]:
    if not v:
        return []
    return [v] if isinstance(v, str) else [str(x) for x in v]


def interruptions(records: list[DecisionRecord]) -> dict:
    """With Clearance: every approval card shown. Without: a confirm-each-action host asks once per email sent
    (send decisions are the ones carrying a scope check in their detail). Also splits the sends into the ones that
    went out quietly inside a permission and the ones that stopped for you."""
    sends = [r for r in records if r.surface == "email" and "problems" in (r.detail or {})]
    asked = sum(r.decision == "ASK" for r in sends)
    return {"with_clearance": sum(r.decision == "ASK" for r in records), "without_clearance": len(sends),
            "emails": len(sends), "emails_asked": asked, "emails_quiet": len(sends) - asked,
            "permission_cards": sum(r.decision == "ASK" and r.surface == "permission" for r in records)}
