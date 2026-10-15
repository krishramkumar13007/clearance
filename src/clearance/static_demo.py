"""Static demo: the email follow-up demo as one self-contained page, for free static hosting.

`clearance static-demo --out DIR` runs the scripted assistant through the real gateway and assistant code, with the
judge replaying Nemotron's saved answers for these drafts (evals/demo_judge_cache.json), records every decision,
and writes DIR/index.html with that trace embedded. The page replays it in the browser: the visitor still decides
the two approval cards, and nothing calls Token Factory. Live judging stays in `assistant-demo` and `serve-demo`.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from .assistant.core import Assistant
from .assistant.demo import DRAFTS, PERMISSION, PLAN
from .assistant.judge import JudgeCosts
from .evals import ROOT_EVALS
from .gateway import Gateway
from .hosted import CACHE, SavedAnswersJudge
from .models import AgentFingerprint
from .policy import Policy

TEMPLATE = Path(__file__).parent / "static" / "replay.html"


def record_trace() -> dict:
    """Every step of the scripted run with the decision the real code made. The permission card is approved while
    recording (nothing happens without it); the exception is recorded as pending and decided by the visitor."""
    th = json.loads((ROOT_EVALS / "email_thresholds.json").read_text())
    costs = replace(JudgeCosts(), scale=th["nemotron"]["scale"])
    g = Gateway(Policy(), assistant=Assistant(judge=SavedAnswersJudge(json.loads(CACHE.read_text())), costs=costs))
    s = g.start_session(Path(tempfile.mkdtemp(prefix="clearance-static-")),
                        AgentFingerprint("nvidia/nemotron-3-super-120b-a12b", "mcp-assistant", "mock-mailbox", "p1"))
    steps = []

    def act(title: str, surface: str, op: str, args: dict, what: str, approve: bool = False) -> None:
        r = g.submit(s.id, surface, op, args)
        d = r.decision
        step = {"title": title, "surface": surface, "op": op, "what": what, "decision": d.decision,
                "by": d.decided_by, "reason": d.reason}
        if surface == "permission":
            step["proposal"] = (d.detail or {}).get("proposal")
        if op == "send":
            step["to"], step["subject"], step["body"] = args["to"], args["subject"], args["body"]
            step["probs"] = ((d.detail or {}).get("judge") or {}).get("probs")
        steps.append(step)
        if r.ask_id:
            g.resolve_ask(r.ask_id, approve=approve)

    act("Ask for permission (once)", "permission", "request", PERMISSION,
        f"permission for email.send: {PERMISSION['purpose']}", approve=True)
    act("Check the inbox", "email", "list", {}, "list inbox")
    for mid in PLAN:
        act(f"Read {mid}", "email", "read", {"id": mid}, f"read message {mid}")
        subject, to, body = DRAFTS[mid]
        act(f"Reply to {to[0].split('@')[0]}", "email", "send",
            {"to": to, "cc": [], "subject": subject, "body": body, "in_reply_to": mid},
            f"send to {to[0]}: {subject}")
    return {"steps": steps, "judge": "Nemotron 3 Super (recorded answers)"}


def write(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    trace = record_trace()
    html = TEMPLATE.read_text().replace("/*TRACE*/null", json.dumps(trace).replace("</", "<\\/"))
    (out / "index.html").write_text(html)
    asks = [st["what"] for st in trace["steps"] if st["decision"] == "ASK"]
    return {"path": str(out / "index.html"), "steps": len(trace["steps"]), "asks": asks}
