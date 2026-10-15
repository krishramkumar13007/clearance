"""mitmproxy addon: every request from a sandbox is held until the Clearance gateway decides.

The gateway sees method, URL, headers and body (TLS is intercepted with a CA the sandbox trusts).
ALLOW -> forwarded. BLOCK or unresolved ASK -> 403 with a short reason. Gateway unreachable -> 403 (fail closed).
"""
import asyncio
import json
import os
import urllib.request

from mitmproxy import http

GATEWAY = os.environ.get("CLEARANCE_GATEWAY", "http://host.docker.internal:8750")
TOKEN = os.environ["CLEARANCE_PROXY_TOKEN"]
MAX_BODY = 1_000_000


def _ask_gateway(payload: dict) -> dict:
    req = urllib.request.Request(
        f"{GATEWAY}/internal/egress", data=json.dumps(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


class Clearance:
    async def request(self, flow: http.HTTPFlow) -> None:
        body = flow.request.get_content(strict=False) or b""
        payload = {
            "client_ip": flow.client_conn.peername[0],
            "method": flow.request.method,
            "url": flow.request.pretty_url,
            "headers": {k: v for k, v in flow.request.headers.items()},
            "body": body[:MAX_BODY].decode("utf-8", "replace"),
            "body_truncated": len(body) > MAX_BODY,
        }
        try:
            verdict = await asyncio.to_thread(_ask_gateway, payload)
        except Exception as e:  # fail closed
            verdict = {"allow": False, "reason": f"gateway unavailable ({type(e).__name__})"}
        if not verdict.get("allow"):
            flow.response = http.Response.make(
                403, f"Blocked by Clearance: {verdict.get('reason', 'denied')}\n", {"Content-Type": "text/plain"})


addons = [Clearance()]
