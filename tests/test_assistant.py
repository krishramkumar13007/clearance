import time

import pytest

from clearance.assistant.core import Assistant, interruptions
from clearance.assistant.judge import JudgeCosts, evaluate
from clearance.assistant.permissions import Permission, PermissionStore
from clearance.gateway import Gateway
from clearance.models import AgentFingerprint
from clearance.policy import Policy

PILOT = ["nadia@contoso.test", "tom@contoso.test", "lena@contoso.test", "omar@contoso.test"]
GRANT = {"action": "email.send", "purpose": "Routine follow-ups on the Q4 pilot", "recipients": PILOT,
         "may_share": ["meeting notes", "next steps", "pilot scope"],
         "must_not": ["pricing", "new deadlines or delivery dates"], "max_uses": 10, "days": 5}


class FakeJudge:
    name = "fake"

    def __init__(self, flag_words=("commit",)):
        self.flag_words = flag_words

    def __call__(self, state):
        body = state["email"]["body"].lower()
        hit = any(w in body for w in self.flag_words)
        return {"new_commitment": 0.9 if hit else 0.01, "outside_sharing": 0.001, "off_purpose": 0.02}


@pytest.fixture
def gw(tmp_path):
    g = Gateway(Policy(), assistant=Assistant(judge=FakeJudge()))
    s = g.start_session(tmp_path, AgentFingerprint("nemotron", "mcp-assistant", "mock-mail", "p1"))
    return g, s


def grant(g, s):
    r = g.submit(s.id, "permission", "request", GRANT)
    assert r.decision.decision == "ASK" and r.decision.detail["proposal"]["recipients"] == PILOT
    out = g.resolve_ask(r.ask_id, approve=True)
    assert out.error is None
    return next(iter(g.assistant.store.items.values()))


def send(g, s, to, body="Here are the notes and next steps.", cc=None, reply="m1"):
    return g.submit(s.id, "email", "send", {"to": to, "cc": cc or [], "subject": "Re: pilot", "body": body,
                                            "in_reply_to": reply})


def test_nothing_proceeds_without_a_permission(gw):
    g, s = gw
    assert g.submit(s.id, "email", "list", {}).decision.decision == "ASK"
    r = send(g, s, ["nadia@contoso.test"])
    assert r.decision.decision == "ASK" and r.decision.decided_by == "scope.no_permission"
    assert g.assistant.mailbox.sent == []


def test_permission_once_then_quiet_execution(gw):
    g, s = gw
    p = grant(g, s)
    assert g.submit(s.id, "email", "list", {}).decision.decision == "ALLOW"
    assert g.submit(s.id, "email", "read", {"id": "m1"}).decision.decision == "ALLOW"
    for who in ("nadia@contoso.test", "tom@contoso.test", "lena@contoso.test"):
        r = send(g, s, [who])
        assert r.decision.decision == "ALLOW" and r.decision.decided_by == f"permission:{p.id}"
    assert len(g.assistant.mailbox.sent) == 3 and p.uses == 3
    assert interruptions(g.records[s.id]) == {"with_clearance": 1, "without_clearance": 3, "emails": 3,
                                              "emails_asked": 0, "emails_quiet": 3, "permission_cards": 1}


def test_exceptions_ask_with_a_clear_reason(gw):
    g, s = gw
    grant(g, s)
    r = send(g, s, ["lena@contoso.test"], cc=["finance@contoso.test"], reply="m3")
    assert r.decision.decision == "ASK" and "finance@contoso.test" in r.decision.reason
    r = send(g, s, ["omar@contoso.test"], body="Yes, we commit to Monday.", reply="m4")
    assert r.decision.decision == "ASK" and r.decision.decided_by == "judge.new_commitment"
    assert "commitment" in r.decision.reason
    assert g.submit(s.id, "email", "read", {"id": "m5"}).decision.decision == "ASK"  # vendor thread: not in scope
    assert g.assistant.mailbox.sent == []


def test_judge_cannot_grant_and_errors_ask(tmp_path):
    class Broken:
        name = "broken"

        def __call__(self, state):
            raise TimeoutError

    g = Gateway(Policy(), assistant=Assistant(judge=Broken()))
    s = g.start_session(tmp_path, AgentFingerprint("m", "t", "e", "p1"))
    # confident judge, no permission: still ASK
    g.assistant.judge = lambda st: {"new_commitment": 0.0, "outside_sharing": 0.0, "off_purpose": 0.0}
    assert send(g, s, ["nadia@contoso.test"]).decision.decided_by == "scope.no_permission"
    g.assistant.judge = Broken()
    grant(g, s)
    assert send(g, s, ["nadia@contoso.test"]).decision.decided_by == "judge.unavailable"


def test_limits_expiry_and_revocation(tmp_path):
    now = [1000.0]
    store = PermissionStore(clock=lambda: now[0])
    p = store.add(Permission("email.send", "x", ["a@x.test"], max_uses=1, expires_at=2000.0))
    assert store.check("email.send", {"to": ["a@x.test"]}).problems == []
    p.uses = 1
    assert store.check("email.send", {"to": ["a@x.test"]}).problems[0][0] == "scope.limit"
    p.uses = 0
    now[0] = 3000.0
    assert store.check("email.send", {"to": ["a@x.test"]}).problems[0][0] == "scope.expired"
    store.revoke(p.id)
    assert store.check("email.send", {"to": ["a@x.test"]}).problems[0][0] == "scope.no_permission"


def test_domain_recipients():
    p = Permission("email.send", "x", ["@contoso.test"])
    assert p.covers_recipient("anyone@contoso.test") and not p.covers_recipient("x@contoso.test.evil")


def test_repeated_approvals_suggest_but_never_widen(gw):
    g, s = gw
    p = grant(g, s)
    for _ in range(2):
        r = send(g, s, ["lena@contoso.test"], cc=["finance@contoso.test"], reply="m3")
        g.resolve_ask(r.ask_id, approve=True)
    assert not p.covers_recipient("finance@contoso.test")  # not silently widened
    sug = [e for e in g.events if e["type"] == "suggestion"]
    assert sug and sug[-1]["recipient"] == "finance@contoso.test"
    g.assistant.store.accept_suggestion(sug[-1]["id"])
    assert p.covers_recipient("finance@contoso.test")


def test_cost_thresholds():
    c = JudgeCosts()
    assert c.threshold("new_commitment") == pytest.approx(0.05)
    p = Permission("email.send", "x", ["a@x.test"])
    out = evaluate(lambda st: {"new_commitment": 0.04, "outside_sharing": 0.03, "off_purpose": 0.0}, p,
                   {"to": ["a@x.test"], "body": "hi"}, None, c)
    assert out["flags"] == ["outside_sharing"]  # 0.03 > 1/50, 0.04 < 1/20
    assert evaluate(lambda st: {"new_commitment": 1.5, "outside_sharing": 0, "off_purpose": 0}, p,
                    {"to": ["a@x.test"]}, None, c) == {"error": "invalid probabilities"}
