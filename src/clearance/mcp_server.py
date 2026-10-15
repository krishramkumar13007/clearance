"""MCP server: the agent's only tools. Every call becomes an action submitted to the gateway HTTP API.

An ASK is a pending result the tool call waits on (long-poll) until a human resolves it or it expires.
"""
from __future__ import annotations

import os
import time

import httpx
from mcp.server.fastmcp import FastMCP

URL = os.environ.get("CLEARANCE_URL", "http://127.0.0.1:8750")
ASK_WAIT = float(os.environ.get("CLEARANCE_ASK_WAIT", "300"))

mcp = FastMCP("clearance")


def _client() -> httpx.Client:
    return httpx.Client(base_url=URL, headers={"Authorization": f"Bearer {os.environ['CLEARANCE_AGENT_TOKEN']}"},
                        timeout=70.0)


def _render(r: dict) -> str:
    if r["decision"] == "BLOCK":
        return f"BLOCKED: {r['reason']}"
    if r.get("error"):
        return f"DENIED or FAILED: {r['error']}" if r["decision"] == "ASK" else f"ERROR: {r['error']}"
    return r.get("output") or "ok"


def submit(surface: str, op: str, args: dict, destination: str | None = None) -> str:
    sid = os.environ["CLEARANCE_SESSION_ID"]
    with _client() as c:
        resp = c.post(f"/sessions/{sid}/actions", json={"surface": surface, "op": op, "args": args,
                                                        "destination": destination})
        resp.raise_for_status()
        r = resp.json()
        if r.get("status") == "pending":
            deadline = time.monotonic() + ASK_WAIT
            while time.monotonic() < deadline:
                poll = c.get(f"/asks/{r['ask_id']}", params={"timeout": 25}).json()
                if poll["status"] == "resolved":
                    return _render(poll)
            return "DENIED or FAILED: no human response before timeout"
        return _render(r)


@mcp.tool()
def fs_read(path: str) -> str:
    """Read a text file inside the workspace."""
    return submit("fs", "read", {"path": path})


@mcp.tool()
def fs_write(path: str, content: str) -> str:
    """Write a text file inside the workspace."""
    return submit("fs", "write", {"path": path, "content": content})


@mcp.tool()
def fs_list(path: str = ".") -> str:
    """List a directory inside the workspace."""
    return submit("fs", "list", {"path": path})


@mcp.tool()
def shell_exec(command: str) -> str:
    """Run a shell command in the sandbox workspace."""
    return submit("shell", "exec", {"command": command})


@mcp.tool()
def git_status() -> str:
    """Show git status."""
    return submit("git", "status", {})


@mcp.tool()
def git_diff() -> str:
    """Show the working-tree diff."""
    return submit("git", "diff", {})


@mcp.tool()
def git_commit(message: str) -> str:
    """Commit staged changes."""
    return submit("git", "commit", {"message": message})


@mcp.tool()
def git_push(branch: str) -> str:
    """Push a branch to the configured remote (the gateway holds the credentials)."""
    return submit("git", "push", {"branch": branch})


@mcp.tool()
def pkg_install(package: str, registry: str | None = None) -> str:
    """Install a package inside the sandbox."""
    args = {"package": package}
    if registry:
        args["registry"] = registry
    return submit("pkg", "install", args)


@mcp.tool()
def http_request(url: str, method: str = "GET", body: str = "") -> str:
    """Make an outbound HTTP request."""
    return submit("net", "request", {"url": url, "method": method, "body": body})


# ---- personal-assistant tools ----
@mcp.tool()
def request_permission(purpose: str, recipients: list[str], may_share: list[str], must_not: list[str],
                       max_uses: int = 10, days: float = 7) -> str:
    """Ask the user once for a standing permission to send email for a purpose, to listed recipients, sharing
    only what is listed. Waits until the user approves or denies."""
    return submit("permission", "request", {"action": "email.send", "purpose": purpose, "recipients": recipients,
                                            "may_share": may_share, "must_not": must_not, "max_uses": max_uses,
                                            "days": days})


@mcp.tool()
def email_list() -> str:
    """List inbox messages (id, sender, subject)."""
    return submit("email", "list", {})


@mcp.tool()
def email_read(id: str) -> str:
    """Read one inbox message."""
    return submit("email", "read", {"id": id})


@mcp.tool()
def email_send(to: list[str], subject: str, body: str, cc: list[str] | None = None,
               in_reply_to: str | None = None) -> str:
    """Send an email. Inside an approved permission it goes out without asking; otherwise the user is asked."""
    return submit("email", "send", {"to": to, "cc": cc or [], "subject": subject, "body": body,
                                    "in_reply_to": in_reply_to})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
