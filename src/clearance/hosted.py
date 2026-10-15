"""Public demo server: every visitor gets their own gateway, tokens, permissions and mock mailbox.

`clearance serve-demo` serves a landing page. "Start the demo" creates a private gateway mounted at /v/<id>/,
starts the scripted email assistant against it and returns that visitor's dashboard link; the human token rides in
the URL fragment, which browsers never send to the server. The Token Factory key never leaves the server: up to
`daily_cap` runs a day (and `per_ip_hour` per address) use live Nemotron, after that runs replay Nemotron's saved
answers for the same fixed drafts (`clearance demo-cache` writes them). Visitors expire after `ttl` seconds.
"""
from __future__ import annotations

import json
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from starlette.routing import Mount

from .api import create_app
from .assistant.clm_judge import state_key
from .assistant.core import Assistant
from .assistant.demo import DRAFTS, PERMISSION, PLAN, run
from .assistant.judge import JudgeCosts, NemotronJudge, judge_state
from .assistant.mailbox import INBOX
from .evals import ROOT_EVALS
from .gateway import Gateway
from .models import AgentFingerprint
from .policy import Policy
from .risk import load_api_key

CACHE = ROOT_EVALS / "demo_judge_cache.json"
STATIC = Path(__file__).parent / "static"


def demo_states() -> list[dict]:
    """The judge state for every scripted reply, exactly as the assistant builds it inside the permission."""
    perm = SimpleNamespace(**{k: PERMISSION[k] for k in ("purpose", "may_share", "must_not")})
    inbox = {m["id"]: m for m in INBOX}
    return [judge_state(perm, {"to": DRAFTS[mid][1], "cc": [], "subject": DRAFTS[mid][0], "body": DRAFTS[mid][2]},
                        inbox[mid]) for mid in PLAN]


def write_cache(judge=None) -> dict:
    """Ask the live judge once per scripted reply and save the answers the demo replays after its daily cap."""
    judge = judge or NemotronJudge()
    cache = {state_key(st): judge(st) for st in demo_states()}
    CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    return {"path": str(CACHE), "states": len(cache)}


@dataclass
class SavedAnswersJudge:
    """Replays the saved Nemotron answers; an unknown state raises, which the assistant turns into an ASK."""
    table: dict
    name: str = "nemotron (saved answers)"

    def __call__(self, state: dict) -> dict[str, float]:
        return self.table[state_key(state)]


@dataclass
class Limits:
    daily_cap: int = 40
    per_ip_hour: int = 3
    max_visitors: int = 25
    ttl: float = 1800.0
    day: str = ""
    live_today: int = 0
    by_ip: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def admit(self, ip: str, now: float) -> bool:
        """Per-address rate limit for starting a demo at all (live or replay)."""
        with self.lock:
            recent = [t for t in self.by_ip.get(ip, []) if now - t < 3600]
            if len(recent) >= self.per_ip_hour:
                self.by_ip[ip] = recent
                return False
            self.by_ip[ip] = recent + [now]
            return True

    def take_live(self, now: float) -> bool:
        """One live-judge run from today's budget, if any is left."""
        with self.lock:
            today = time.strftime("%Y-%m-%d", time.gmtime(now))
            if today != self.day:
                self.day, self.live_today = today, 0
            if self.live_today >= self.daily_cap:
                return False
            self.live_today += 1
            return True

    def remaining(self, now: float) -> int:
        with self.lock:
            return self.daily_cap if time.strftime("%Y-%m-%d", time.gmtime(now)) != self.day \
                else max(0, self.daily_cap - self.live_today)


def _has_key() -> bool:
    try:
        load_api_key()
        return True
    except RuntimeError:
        return False


def create_hosted_app(port: int, limits: Limits | None = None, live_judge_factory=None,
                      pause: float = 2.0, scale: float | None = None) -> FastAPI:
    limits = limits or Limits()
    if live_judge_factory is None:
        live_judge_factory = NemotronJudge
        if not _has_key():  # no Token Factory key on this server: every run replays the saved answers
            limits = replace(limits, daily_cap=0)
    saved = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if scale is None:
        th = ROOT_EVALS / "email_thresholds.json"
        scale = json.loads(th.read_text()).get("nemotron", {}).get("scale", 1.0) if th.exists() else 1.0
    costs = replace(JudgeCosts(), scale=scale)  # Nemotron's threshold scale, fitted on the calib split
    app = FastAPI(title="Clearance demo")
    visitors: dict[str, float] = {}  # visitor id -> created at
    vlock = threading.Lock()

    def expire(now: float) -> None:
        with vlock:
            gone = {v for v, t in visitors.items() if now - t > limits.ttl}
            for v in gone:
                del visitors[v]
        if gone:
            app.router.routes[:] = [r for r in app.router.routes
                                    if not (isinstance(r, Mount) and r.path.removeprefix("/v/") in gone)]

    @app.get("/", include_in_schema=False)
    def landing():
        return HTMLResponse((STATIC / "landing.html").read_text())

    @app.get("/demo/status")
    def status():
        now = time.time()
        expire(now)
        return {"live_runs_left_today": limits.remaining(now), "saved_answers": bool(saved),
                "visitors": len(visitors), "max_visitors": limits.max_visitors}

    @app.post("/demo/start")
    def start(request: Request):
        now = time.time()
        expire(now)
        ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?"))
        ip = ip.split(",")[0].strip()
        with vlock:
            if len(visitors) >= limits.max_visitors:
                raise HTTPException(503, "The demo is busy right now. Try again in a few minutes.")
        if not limits.admit(ip, now):
            raise HTTPException(429, "You've started a few demos this hour. Try again later.")
        live = limits.take_live(now)
        if not live and not saved:
            raise HTTPException(503, "Today's live demos are used up. Try again tomorrow.")
        judge = live_judge_factory() if live else SavedAnswersJudge(saved)
        vid, agent_tok, human_tok = secrets.token_urlsafe(9), secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        gw = Gateway(Policy(), assistant=Assistant(judge=judge, costs=costs))
        root = Path(tempfile.mkdtemp(prefix="clearance-visitor-"))
        app.router.routes.append(Mount(f"/v/{vid}", app=create_app(gw, agent_tok, human_tok, root)))
        with vlock:
            visitors[vid] = now
        s = gw.start_session(root, AgentFingerprint("nvidia/nemotron-3-super-120b-a12b", "mcp-assistant",
                                                    "mock-mailbox", "p1"))

        def drive():
            try:
                run(f"http://127.0.0.1:{port}/v/{vid}", agent_tok, s.id, pause=pause, log=lambda *_: None)
            except (httpx.HTTPError, KeyError):
                pass  # the visitor left or expired; nothing to clean up beyond the mount

        threading.Thread(target=drive, daemon=True).start()
        return {"dashboard": f"/v/{vid}/dashboard#token={human_tok}", "judge": getattr(judge, "name", "nemotron"),
                "live": live}

    return app
