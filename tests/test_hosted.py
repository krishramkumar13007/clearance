import json

from fastapi.testclient import TestClient

from clearance.assistant.core import Assistant
from clearance.assistant.demo import DRAFTS, PERMISSION, PLAN
from clearance.gateway import Gateway
from clearance.hosted import CACHE, Limits, SavedAnswersJudge, create_hosted_app
from clearance.models import AgentFingerprint
from clearance.policy import Policy


def test_saved_answers_cover_every_scripted_reply(tmp_path):
    # the replay must see exactly the states the assistant builds, or every send would turn into an ASK
    g = Gateway(Policy(), assistant=Assistant(judge=SavedAnswersJudge(json.loads(CACHE.read_text()))))
    s = g.start_session(tmp_path, AgentFingerprint("nemotron", "mcp-assistant", "mock-mailbox", "p1"))
    r = g.submit(s.id, "permission", "request", PERMISSION)
    assert g.resolve_ask(r.ask_id, approve=True).error is None
    out = {}
    for mid in PLAN:
        subject, to, body = DRAFTS[mid]
        d = g.submit(s.id, "email", "send", {"to": to, "cc": [], "subject": subject, "body": body,
                                             "in_reply_to": mid}).decision
        assert d.decided_by != "judge.unavailable", mid
        out[mid] = d.decision
    assert out["m4"] == "ASK" and all(v != "ASK" for k, v in out.items() if k != "m4")  # the one exception


def test_limits_rate_limit_and_daily_cap():
    lim = Limits(daily_cap=2, per_ip_hour=2)
    assert lim.admit("1.1.1.1", 0) and lim.admit("1.1.1.1", 10) and not lim.admit("1.1.1.1", 20)
    assert lim.admit("1.1.1.1", 3700)  # an hour later
    assert lim.take_live(0) and lim.take_live(1) and not lim.take_live(2) and lim.remaining(2) == 0
    assert lim.take_live(86400) and lim.remaining(86400) == 1  # next UTC day


def test_each_visitor_gets_a_private_gateway():
    calls = []

    def live():
        calls.append(1)
        return SavedAnswersJudge({}, name="live")

    app = create_hosted_app(port=1, limits=Limits(daily_cap=1, per_ip_hour=5), live_judge_factory=live, pause=0)
    c = TestClient(app)
    assert "Start the demo" in c.get("/").text
    a, b = c.post("/demo/start").json(), c.post("/demo/start").json()
    assert a["live"] and a["judge"] == "live" and not b["live"] and b["judge"] == "nemotron (saved answers)"
    assert len(calls) == 1  # the daily cap holds the live judge to one run
    base_a, tok_a = a["dashboard"].split("dashboard#token=")
    _, tok_b = b["dashboard"].split("dashboard#token=")
    assert c.get(base_a + "sessions", headers={"Authorization": f"Bearer {tok_a}"}).status_code == 200
    assert c.get(base_a + "sessions", headers={"Authorization": f"Bearer {tok_b}"}).status_code == 401
    assert c.get(base_a, follow_redirects=False).headers["location"] == base_a + "dashboard"


def test_per_address_limit():
    app = create_hosted_app(port=1, limits=Limits(per_ip_hour=1), live_judge_factory=lambda: None, pause=0)
    c = TestClient(app)
    h = {"x-forwarded-for": "9.9.9.9, 10.0.0.1"}
    assert c.post("/demo/start", headers=h).status_code == 200
    assert c.post("/demo/start", headers=h).status_code == 429
    assert c.post("/demo/start", headers={"x-forwarded-for": "8.8.8.8"}).status_code == 200


def test_without_a_key_every_run_replays_saved_answers(monkeypatch):
    import clearance.hosted as h

    monkeypatch.setattr(h, "_has_key", lambda: False)
    c = TestClient(create_hosted_app(port=1, limits=Limits(daily_cap=40), pause=0))
    assert c.get("/demo/status").json()["live_runs_left_today"] == 0
    r = c.post("/demo/start").json()
    assert not r["live"] and r["judge"] == "nemotron (saved answers)"
