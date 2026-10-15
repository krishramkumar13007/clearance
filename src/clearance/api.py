"""Gateway HTTP API. Two credentials: the agent token (actions, ask polling) and the human token
(session creation, ask resolution, stream, summary, eval). The agent never holds the human token."""
from __future__ import annotations

import asyncio
import json
import os
import secrets as pysecrets
from collections import Counter
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from .autonomy import autonomy_levels
from .gateway import ActionResult, Gateway
from .models import AgentFingerprint
from .policy import POLICY_VERSION, Policy


class SessionRequest(BaseModel):
    workspace: str
    model: str = "nvidia/nemotron-3-super-120b-a12b"
    tool_manifest_hash: str = "mcp-v1"
    environment_id: str = "local"
    system_prompt_hash: str = ""


class ActionRequest(BaseModel):
    surface: str
    op: str
    args: dict = {}
    destination: str | None = None


class ResolveRequest(BaseModel):
    approve: bool


class EgressRequest(BaseModel):
    client_ip: str
    method: str
    url: str
    headers: dict = {}
    body: str = ""
    body_truncated: bool = False


def _public(r: ActionResult) -> dict:
    """What the agent may see: decision, a short safe reason, output. Never matched secrets or rule internals."""
    d = r.decision
    body = {"decision": d.decision, "reason": d.reason, "output": r.output, "error": r.error}
    if r.ask_id:
        body["ask_id"] = r.ask_id
        body["status"] = "pending"
    return body


def create_app(gateway: Gateway | None = None, agent_token: str | None = None, human_token: str | None = None,
               workspace_root: Path | None = None, proxy_token: str | None = None,
               egress_ask_wait: float | None = None) -> FastAPI:
    gw = gateway or Gateway(Policy())
    agent_tok = agent_token or os.environ.get("CLEARANCE_AGENT_TOKEN") or pysecrets.token_urlsafe(24)
    human_tok = human_token or os.environ.get("CLEARANCE_HUMAN_TOKEN") or pysecrets.token_urlsafe(24)
    proxy_tok = proxy_token or os.environ.get("CLEARANCE_PROXY_TOKEN") or pysecrets.token_urlsafe(24)
    ask_wait = egress_ask_wait if egress_ask_wait is not None else float(os.environ.get("CLEARANCE_EGRESS_ASK_WAIT", "120"))
    root = (workspace_root or Path(os.environ.get("CLEARANCE_WORKSPACE_ROOT", "."))).resolve()
    app = FastAPI(title="Clearance gateway")
    app.state.gateway, app.state.agent_token, app.state.human_token = gw, agent_tok, human_tok

    def _check(expected: str, authorization: str | None) -> None:
        got = (authorization or "").removeprefix("Bearer ").strip()
        if not pysecrets.compare_digest(got, expected):
            raise HTTPException(401, "unauthorized")

    def agent(authorization: str | None = Header(None)) -> None:
        _check(agent_tok, authorization)

    def human(authorization: str | None = Header(None), token: str | None = None) -> None:
        # EventSource cannot send headers, so the dashboard stream passes the human token as ?token=
        _check(human_tok, authorization or (f"Bearer {token}" if token else None))

    def proxy(authorization: str | None = Header(None)) -> None:
        _check(proxy_tok, authorization)

    def session(session_id: str):
        s = gw.sessions.get(session_id)
        if s is None:
            raise HTTPException(404, "unknown session")
        return s

    @app.post("/sessions", dependencies=[Depends(human)])
    def start_session(req: SessionRequest):
        ws = Path(req.workspace).resolve()
        if not ws.is_dir() or not ws.is_relative_to(root):
            raise HTTPException(400, "workspace must be an existing directory under the allowed root")
        fp = AgentFingerprint(req.model, req.tool_manifest_hash, req.environment_id, POLICY_VERSION,
                              req.system_prompt_hash)
        s = gw.start_session(ws, fp)
        return {"session_id": s.id, "fingerprint_id": fp.id, "secrets_indexed": len(s.secrets.entries)}

    @app.get("/", include_in_schema=False)
    def index(request: Request):
        return RedirectResponse(request.url_for("dashboard").path)  # keeps a mount prefix (hosted demo)

    @app.get("/dashboard", include_in_schema=False)
    def dashboard():
        return HTMLResponse((Path(__file__).parent / "static" / "dashboard.html").read_text())

    @app.get("/sessions", dependencies=[Depends(human)])
    def list_sessions():
        return [{"session_id": s.id, "fingerprint_id": s.fingerprint.id, "model": s.fingerprint.model,
                 "workspace": s.workspace.name, "actions": len(gw.records.get(s.id, [])),
                 "taint": sorted(s.ledger.taint), "sandboxed": s.container_ip is not None}
                for s in gw.sessions.values()]

    @app.get("/sessions/{session_id}/events", dependencies=[Depends(human)])
    def session_events(session_id: str):
        session(session_id)
        return [e for e in gw.events if e.get("session") == session_id]

    @app.post("/sessions/{session_id}/actions", dependencies=[Depends(agent)])
    def submit_action(session_id: str, req: ActionRequest):
        session(session_id)
        if req.surface not in ("fs", "shell", "git", "pkg", "net", "email", "permission"):
            raise HTTPException(422, "unknown surface")
        return _public(gw.submit(session_id, req.surface, req.op, req.args, req.destination))

    @app.get("/asks/{ask_id}", dependencies=[Depends(agent)])
    async def poll_ask(ask_id: str, timeout: float = 25.0):
        ask = gw.asks.get(ask_id)
        if ask is None:
            raise HTTPException(404, "unknown ask")
        deadline = asyncio.get_event_loop().time() + min(timeout, 60.0)
        while ask.result is None and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.2)
        if ask.result is None:
            return {"status": "pending", "ask_id": ask_id}
        return {"status": "resolved", **_public(ask.result)}

    @app.get("/asks", dependencies=[Depends(human)])
    def list_asks():
        return [{"ask_id": a.id, "session_id": a.session_id, "surface": a.action.surface, "op": a.action.op,
                 "summary": gw.describe(gw.sessions[a.session_id], a.action), "reason": a.record.reason,
                 "detail": a.record.detail,
                 "destination": a.action.destination, "why": a.record.decided_by,
                 "evidence_ids": a.record.evidence, "expires_in": max(0, int(a.expires_at - gw.clock()))}
                for a in gw.asks.values() if not a.resolved]

    @app.post("/asks/{ask_id}/resolve", dependencies=[Depends(human)])
    def resolve(ask_id: str, req: ResolveRequest):
        try:
            return _public(gw.resolve_ask(ask_id, req.approve))
        except KeyError:
            raise HTTPException(404, "unknown or already-resolved ask")

    @app.get("/sessions/{session_id}/stream", dependencies=[Depends(human)])
    async def stream(session_id: str):
        session(session_id)

        async def gen():
            i = 0
            while True:
                evs = [e for e in gw.events[i:]]
                i += len(evs)
                for e in evs:
                    if e.get("session") == session_id:
                        yield f"data: {json.dumps(e)}\n\n"
                if not evs:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.5)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/sessions/{session_id}/summary", dependencies=[Depends(human)])
    def summary(session_id: str):
        session(session_id)
        recs = gw.records[session_id]
        counts = Counter(r.decision for r in recs)
        total = len(recs)
        extra = {}
        if gw.assistant is not None:
            from .assistant.core import interruptions
            extra["interruptions"] = interruptions(recs)
        return {**extra, "total": total, "counts": dict(counts),
                "intervention_rate": round((counts["ASK"] + counts["BLOCK"]) / total, 3) if total else 0.0,
                "taint": sorted(gw.sessions[session_id].ledger.taint)}

    @app.get("/fingerprints/{fingerprint_id}/autonomy", dependencies=[Depends(human)])
    def autonomy(fingerprint_id: str):
        return autonomy_levels(gw.fp_records.get(fingerprint_id, []))

    # ---- personal-assistant permissions (human only) ----
    def _assistant():
        if gw.assistant is None:
            raise HTTPException(404, "assistant surfaces not enabled")
        return gw.assistant

    @app.get("/permissions", dependencies=[Depends(human)])
    def list_permissions():
        return [p.public() for p in _assistant().store.list()]

    @app.post("/permissions", dependencies=[Depends(human)])
    def create_permission(req: dict):
        from .assistant.permissions import Permission
        a = _assistant()
        return a.store.add(Permission(**a._proposal(req), source="user")).public()

    @app.post("/permissions/{pid}/revoke", dependencies=[Depends(human)])
    def revoke_permission(pid: str):
        try:
            return _assistant().store.revoke(pid).public()
        except KeyError:
            raise HTTPException(404, "unknown permission")

    @app.get("/suggestions", dependencies=[Depends(human)])
    def list_suggestions():
        return [x for x in _assistant().store.suggestions if x["status"] == "pending"]

    @app.post("/suggestions/{sid}/accept", dependencies=[Depends(human)])
    def accept_suggestion(sid: str):
        try:
            return _assistant().store.accept_suggestion(sid).public()
        except StopIteration:
            raise HTTPException(404, "unknown or already-handled suggestion")

    @app.get("/mailbox/sent", dependencies=[Depends(human)])
    def sent_mail():
        return _assistant().mailbox.sent

    @app.post("/internal/egress", dependencies=[Depends(proxy)])
    async def egress(req: EgressRequest):
        """The egress proxy holds each sandbox request here until it is decided. An ASK waits for a human
        (bounded), then denies. Unknown sources and truncated bodies are denied."""
        if req.body_truncated:
            return {"allow": False, "reason": "request body too large to inspect"}
        r = await asyncio.to_thread(gw.egress, req.client_ip, req.method, req.url, req.headers, req.body)
        if r is None:
            return {"allow": False, "reason": "unknown source"}
        if r.ask_id:
            ask = gw.asks[r.ask_id]
            deadline = asyncio.get_event_loop().time() + ask_wait
            while ask.result is None and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.2)
            if ask.result is None:
                try:
                    gw.resolve_ask(r.ask_id, approve=False)
                except KeyError:
                    pass
                return {"allow": False, "reason": "awaiting approval timed out"}
            r = ask.result
        allowed = r.decision.decision in ("ALLOW", "ALLOW_AUDIT") or (
            r.decision.decision == "ASK" and r.decision.human_resolution == "approved_once")
        return {"allow": allowed and not r.error, "reason": r.decision.reason or r.error or ""}

    @app.post("/eval/runs", dependencies=[Depends(human)])
    def eval_run():
        raise HTTPException(501, "eval harness not wired yet")

    return app
