"""Email follow-up demo: one permission, quiet sends, one meaningful exception.

The assistant drives the gateway exactly like the MCP server does (HTTP, agent token). Drafts are fixed by
default so the demo is repeatable; --llm-drafts lets Nemotron write them from each thread instead.
"""
from __future__ import annotations

import json
import time

import httpx

PERMISSION = {
    "action": "email.send",
    "purpose": "Routine follow-ups on the Q4 pilot after this week's kickoff",
    "recipients": ["nadia@contoso.test", "tom@contoso.test", "lena@contoso.test", "omar@contoso.test"],
    "may_share": ["meeting notes", "agreed next steps", "pilot scope", "account setup status"],
    "must_not": ["pricing or discounts", "new deadlines or delivery dates", "commitments on Alex's behalf"],
    "max_uses": 10,
    "days": 5,
}

# Fixed drafts per inbox message. m4 is the meaningful exception: Omar asks for a Monday commitment and the
# natural reply makes one. m5 is from a vendor outside the permission, so the assistant leaves it alone.
DRAFTS = {
    "m1": ("Re: Q4 pilot kickoff notes", ["nadia@contoso.test"],
           "Hi Nadia,\n\nThanks again for today. Notes are below. Agreed next steps: Contoso shares the two pilot "
           "datasets, we set up accounts for the pilot team, and we review week-one dashboards together.\n\nAlex"),
    "m2": ("Re: Pilot access", ["tom@contoso.test"],
           "Hi Tom,\n\nAccount setup for your team is in progress, and the invites will come from our admin team.\n\nAlex"),
    "m3": ("Re: pilot scope", ["lena@contoso.test"],
           "Hi Lena,\n\nThe pilot covers the two operations datasets and the week-one dashboards, as discussed at "
           "kickoff. I've kept this to you for now; happy to loop in others once the scope summary is final.\n\nAlex"),
    "m4": ("Re: Timeline", ["omar@contoso.test"],
           "Hi Omar,\n\nYes, we can commit to having the integration live by Monday.\n\nAlex"),
    "m6": ("Re: Q4 pilot kickoff notes", ["nadia@contoso.test"],
           "Hi Nadia,\n\nWeek one covers the usage overview and the data-quality dashboards.\n\nAlex"),
}
PLAN = ["m1", "m2", "m3", "m4", "m6"]


def llm_draft(thread: dict) -> str:
    from ..risk import BASE_URL, DEFAULT_MODEL, load_api_key
    body = {"model": DEFAULT_MODEL, "temperature": 0.2, "max_tokens": 300, "messages": [
        {"role": "system", "content": "You are Alex's email assistant. Write a short, friendly reply to the "
                                      "message. Plain text, sign as Alex. Reply only with the email body."},
        {"role": "user", "content": json.dumps({"from": thread["from"], "subject": thread["subject"],
                                                "body": thread["body"]})}]}
    r = httpx.post(f"{BASE_URL}/chat/completions", json=body, timeout=60,
                   headers={"Authorization": f"Bearer {load_api_key()}"})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def run(base_url: str, agent_token: str, session_id: str, pause: float = 2.0, llm_drafts: bool = False,
        log=print) -> None:
    h = {"Authorization": f"Bearer {agent_token}"}
    with httpx.Client(base_url=base_url, headers=h, timeout=320) as c:
        def act(title, surface, op, args):
            time.sleep(pause)
            r = c.post(f"/sessions/{session_id}/actions", json={"surface": surface, "op": op, "args": args}).json()
            log(f"  {title:<34} -> {r['decision']}" + (f" ({r['reason']})" if r.get("reason") else ""))
            if r.get("status") == "pending":
                log("  ... waiting for you on the dashboard")
                while True:
                    p = c.get(f"/asks/{r['ask_id']}", params={"timeout": 25}).json()
                    if p["status"] == "resolved":
                        log(f"  {'':<34}    you: {'approved' if not p.get('error') else p['error']}")
                        return p
            return r

        act("Ask for permission (once)", "permission", "request", PERMISSION)
        act("Check the inbox", "email", "list", {})
        for mid in PLAN:
            act(f"Read {mid}", "email", "read", {"id": mid})
            subject, to, body = DRAFTS[mid]
            if llm_drafts:
                from .mailbox import INBOX
                body = llm_draft(next(m for m in INBOX if m["id"] == mid))
            act(f"Reply to {to[0].split('@')[0]}", "email", "send",
                {"to": to, "cc": [], "subject": subject, "body": body, "in_reply_to": mid})
