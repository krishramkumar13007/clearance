"""In-process gateway: ties rules, ledger, secret index and a local executor together.

The container sandbox and egress proxy replace LocalExecutor later; the decision path stays the same.
Trust inputs (destination, reversibility, session ids, fingerprints) are derived here, never taken from the agent.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .decision import CostMatrix, History, decide
from .models import (ActionEvent, AgentFingerprint, DecisionRecord, NetAttempt, Observation,
                     RiskEstimate, SessionLedger, TaintSource, new_id)
from .policy import Policy
from .secrets import SECRET_FILE_PATTERNS, SecretIndex

ASK_TTL_SECONDS = 300


@dataclass
class ActionResult:
    decision: DecisionRecord
    output: str | None = None
    observation: Observation | None = None
    ask_id: str | None = None
    error: str | None = None


@dataclass
class PendingAsk:
    id: str
    session_id: str
    action: ActionEvent
    record: DecisionRecord
    action_hash: str
    expires_at: float
    resolved: bool = False
    result: ActionResult | None = None


@dataclass
class Session:
    id: str
    workspace: Path
    fingerprint: AgentFingerprint
    ledger: SessionLedger
    secrets: SecretIndex
    seq: int = 0
    net_log: list[NetAttempt] = field(default_factory=list)  # what actually left the sandbox
    lock: threading.Lock = field(default_factory=threading.Lock)
    container_ip: str | None = None


def action_hash(a: ActionEvent) -> str:
    blob = json.dumps([a.surface, a.op, a.args, a.destination], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class Gateway:
    def __init__(self, policy: Policy | None = None, risk_fn=None, costs: CostMatrix = CostMatrix(),
                 clock=time.monotonic, executor=None, earned_autonomy: bool = False, shadow_fn=None,
                 assistant=None):
        self.policy = policy or Policy()
        self.executor = executor  # DockerExecutor, or None for the in-process simulator (tests, arm evals)
        self.earned_autonomy = earned_autonomy
        self.shadow_fn = shadow_fn  # e.g. LayaRisk: scored on every action, never changes the decision
        self.assistant = assistant  # personal-assistant surfaces (permissions, email); None = coding only
        self.fp_records: dict[str, list[DecisionRecord]] = {}  # fingerprint id -> every decision it received
        self.ip_sessions: dict[str, str] = {}  # sandbox container IP -> session id, to attribute proxy egress
        self.risk_fn = risk_fn  # (action, ledger) -> RiskEstimate | None; no risk_fn = arm A (rules only)
        self.costs = costs
        self.clock = clock
        self.sessions: dict[str, Session] = {}
        self.history: dict[str, History] = {}  # keyed by fingerprint id
        self.records: dict[str, list[DecisionRecord]] = {}
        self.asks: dict[str, PendingAsk] = {}
        self.events: list[dict] = []  # feed for the dashboard stream

    # ---- sessions: ids are server-generated; the fingerprint is operator-supplied, not agent-claimed ----
    def start_session(self, workspace: Path, fp: AgentFingerprint) -> Session:
        idx = SecretIndex()
        idx.scan_workspace(workspace)
        sid = new_id()
        s = Session(sid, workspace.resolve(), fp, SessionLedger(sid, fp.id), idx)
        if self.executor is not None:
            s.container_ip = self.executor.start(sid, s.workspace)
            self.ip_sessions[s.container_ip] = sid
        self.sessions[sid] = s
        self.history.setdefault(fp.id, History(Counter()))
        self.records[sid] = []
        return s

    def end_session(self, session_id: str) -> None:
        s = self.sessions.get(session_id)
        if s and self.executor is not None:
            self.executor.stop(session_id)
            self.ip_sessions.pop(s.container_ip or "", None)

    def egress(self, client_ip: str, method: str, url: str, headers: dict, body: str) -> ActionResult | None:
        """Called by the egress proxy for every request leaving a sandbox. None = unknown source (deny)."""
        sid = self.ip_sessions.get(client_ip)
        if sid is None:
            return None
        return self.submit(sid, "net", "egress", {"url": url, "method": method.upper(), "headers": headers,
                                                  "body": body})

    # ---- action path ----
    def submit(self, session_id: str, surface: str, op: str, args: dict,
               destination: str | None = None) -> ActionResult:
        s = self.sessions[session_id]
        with s.lock:
            # Agent-supplied trust hints are ignored: underscore keys are gateway-derived only.
            args = {k: v for k, v in args.items() if not k.startswith("_") and k != "snapshotted"}
            if self.executor is not None:
                args["_sandboxed"] = True
            self._derive_trusted_fields(s, surface, op, args, destination)
            destination = args.pop("_destination", destination)
            s.seq += 1
            a = ActionEvent(session_id, s.seq, surface, op, args, s.fingerprint.id, destination=destination)
            hist = self.history[s.fingerprint.id]
            level = None
            # Earned trust is suspended for the rest of a session once something in it was blocked, is awaiting
            # review or was denied: a follow-up to an action that needed stopping should not be waved through.
            unreviewed = any(r.decision == "BLOCK" or (r.decision == "ASK" and r.human_resolution != "approved_once")
                             for r in self.records.get(session_id, []))
            if self.earned_autonomy and not unreviewed:
                from .autonomy import autonomy_levels
                level = autonomy_levels(self.fp_records.get(s.fingerprint.id, []))[surface]["level"] \
                    if surface in ("fs", "shell", "git", "pkg", "net") else None
            if surface in ("email", "permission"):
                if self.assistant is None:
                    raise ValueError("assistant surfaces are not enabled on this gateway")
                d = self.assistant.decide(a)
            else:
                d = decide(a, s.ledger, self.policy, s.secrets, hist, self._risk(a, s), self.costs, autonomy=level)
            if self.shadow_fn is not None and surface not in ("email", "permission"):
                try:
                    sh = self.shadow_fn(a, s.ledger)
                    d.shadow = {"model": sh.model, "p_destructive": round(sh.p_destructive, 4),
                                "p_exfiltration": round(sh.p_exfiltration, 4), "latency_ms": sh.latency_ms}
                except Exception as e:  # a shadow failure must never affect the real decision
                    d.shadow = {"error": type(e).__name__}
            self._record(s, a, d)

            if d.decision == "BLOCK":
                self._discard_push_clone(a)
                return ActionResult(d)
            if d.decision == "ASK":
                ask = PendingAsk(new_id(), session_id, a, d, action_hash(a), self.clock() + ASK_TTL_SECONDS)
                self.asks[ask.id] = ask
                self.events.append({"type": "ask", "session": session_id, "ask": ask.id, "seq": a.seq,
                                    "summary": self.describe(s, a), "ts": time.time()})
                return ActionResult(d, ask_id=ask.id)
            if not self._runs_in_sandbox(a):
                return self._execute(s, a, d, hist)
        # Sandbox commands run outside the session lock: their outbound requests come back through the proxy
        # and must be decided while the command is still running.
        return self._execute(s, a, d, hist)

    def _runs_in_sandbox(self, a: ActionEvent) -> bool:
        return self.executor is not None and a.surface in ("shell", "git", "pkg")

    def resolve_ask(self, ask_id: str, approve: bool) -> ActionResult:
        """Human resolution. Single-use, expires to deny, bound to the exact stored action."""
        ask = self.asks.get(ask_id)
        if ask is None or ask.resolved:
            raise KeyError("unknown or already-resolved ask")
        ask.resolved = True
        s = self.sessions[ask.session_id]
        expired = self.clock() > ask.expires_at
        approved = approve and not expired and action_hash(ask.action) == ask.action_hash
        self.events.append({"type": "resolved", "session": s.id, "seq": ask.action.seq, "ask": ask.id,
                            "resolution": "approved_once" if approved else ("expired" if expired else "denied"),
                            "ts": time.time()})
        if not approve or expired or action_hash(ask.action) != ask.action_hash:
            ask.record.human_resolution = "denied"
            self._discard_push_clone(ask.action)
            ask.result = ActionResult(ask.record, error="expired" if expired else "denied")
            return ask.result
        ask.record.human_resolution = "approved_once"
        if self._runs_in_sandbox(ask.action):
            ask.result = self._execute(s, ask.action, ask.record, self.history[s.fingerprint.id])
            return ask.result
        with s.lock:
            ask.result = self._execute(s, ask.action, ask.record, self.history[s.fingerprint.id])
            if self.assistant is not None and not ask.result.error:
                suggestion = self.assistant.on_approved(ask.record, ask.action)
                if suggestion:
                    self.events.append({"type": "suggestion", "session": s.id, **suggestion, "ts": time.time()})
        return ask.result

    # ---- internals ----
    def _risk(self, a: ActionEvent, s: Session) -> RiskEstimate | None:
        if self.risk_fn is None:
            return None
        if (a.surface == "net" and str(a.args.get("method", "GET")).upper() in ("GET", "HEAD")
                and self.policy.is_trusted(a.destination)):
            return None  # read-only fetch from a trusted registry: rules decide, no model call per request
        try:
            return self.risk_fn(a, s.ledger)
        except Exception:  # model outage, 429, bad JSON: fail to ASK, never to ALLOW
            return RiskEstimate(a.id, float("nan"), float("nan"), "prompted", "error")

    def _derive_trusted_fields(self, s: Session, surface: str, op: str, args: dict, claimed: str | None) -> None:
        if surface == "net":
            host = urlparse(args.get("url", "")).hostname
            args["_destination"] = host or "unknown"  # the URL decides, not the claimed destination
        elif surface == "git" and op == "push":
            args["_destination"] = args.get("remote") or self.policy.git_remote or "unknown"
            remote_ok = self.policy.git_remote and args.get("remote", self.policy.git_remote) == self.policy.git_remote
            if self.executor is not None and remote_ok:
                from .policy import normalize_branch
                try:
                    clone, outgoing = self.executor.prepare_push(s.workspace, normalize_branch(args.get("branch", "")),
                                                                 self.policy.git_remote)
                    args["_push_clone"], args["_outgoing"] = clone, outgoing  # scanned by rule 2
                except Exception as e:
                    args["_push_error"] = type(e).__name__
        elif surface == "pkg":
            npm = args.get("manager") == "npm"
            args["_destination"] = args.get("registry") or ("registry.npmjs.org" if npm else "pypi.org")
        if surface == "fs":
            path = args.get("path", "")
            try:
                resolved = (s.workspace / path).resolve()
                if path.startswith(("/", "~")) or not resolved.is_relative_to(s.workspace):
                    args["_escapes_workspace"] = True  # also catches symlinks that leave the workspace
            except (OSError, ValueError):
                args["_escapes_workspace"] = True

    def _record(self, s: Session, a: ActionEvent, d: DecisionRecord) -> None:
        s.ledger.decisions.append(a.id)
        s.ledger.recent_surfaces = (s.ledger.recent_surfaces + [a.surface])[-10:]
        self.records[s.id].append(d)
        self.fp_records.setdefault(s.fingerprint.id, []).append(d)
        self.events.append({"type": "decision", "session": s.id, "seq": a.seq, "surface": a.surface,
                            "op": a.op, "decision": d.decision, "by": d.decided_by, "reason": d.reason,
                            "shadow": d.shadow,
                            "summary": self.describe(s, a), "ts": time.time()})

    @staticmethod
    def describe(s: Session, a: ActionEvent) -> str:
        """One-line, human-facing description of an action with any secret material masked."""
        if a.surface in ("email", "permission"):
            from .assistant.core import Assistant
            return Assistant.describe(a)
        args = a.args
        text = (args.get("command") or args.get("path") or args.get("url") or args.get("package")
                or args.get("message") or args.get("branch") or "")
        if a.surface == "net":
            text = f"{args.get('method', 'GET')} {text}"
        text = str(text)
        for sid in s.secrets.scan(text):
            for v in s.secrets.entries[sid]._variants:
                text = text.replace(v, f"[secret {sid}]")
        return text[:200]

    def _execute(self, s: Session, a: ActionEvent, d: DecisionRecord, hist: History) -> ActionResult:
        obs = Observation(action_id=a.id)
        out: str | None = None
        try:
            if a.surface in ("email", "permission"):
                out = self.assistant.execute(a, d)
            elif a.surface == "fs":
                path = s.workspace / a.args["path"]
                if a.op == "read":
                    out = path.read_text()
                    obs.files_read.append(a.args["path"])
                    self._observe_read(s, a, path, out)
                elif a.op == "write":
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(a.args["content"])
                    obs.files_written.append(a.args["path"])
                    if any(path.match(g) for g in SECRET_FILE_PATTERNS):
                        s.secrets.scan_workspace(s.workspace)  # secret files created mid-session
                elif a.op == "list":
                    out = "\n".join(sorted(p.name for p in path.iterdir()))
                else:
                    raise ValueError(f"unsupported fs op: {a.op}")
                s.ledger.files_touched.add(a.args["path"])
            elif self.executor is not None and a.surface in ("shell", "git", "pkg"):
                out, obs = self._execute_sandboxed(s, a, obs)
            elif a.surface == "net" and a.op == "egress":
                body = str(a.args.get("body", ""))
                attempt = NetAttempt(a.destination or "", 443, len(body), [])
                s.net_log.append(attempt)
                obs.net_attempts.append(attempt)
                s.ledger.destinations.add(a.destination or "")
                out = "forwarded by proxy"
            elif a.surface == "net" and self.executor is not None:
                out = self._http_request(s, a, obs)
            elif a.surface == "net":
                body = str(a.args.get("body", ""))
                attempt = NetAttempt(a.destination or "", 443, len(body), s.secrets.scan(body))
                s.net_log.append(attempt)
                obs.net_attempts.append(attempt)
                s.ledger.destinations.add(a.destination or "")
                out = "200 OK (simulated)"
            elif a.surface == "shell":
                self._observe_shell(s, a)
                out = "(shell execution arrives with the container sandbox)"
        except Exception as e:  # execution failed: record the failure, do not count as executed history
            return ActionResult(d, error=f"{type(e).__name__}: {e}", observation=obs)
        hist.record(a)  # only executed actions build trust
        return ActionResult(d, out, obs)

    def _execute_sandboxed(self, s: Session, a: ActionEvent, obs: Observation) -> tuple[str, Observation]:
        ex = self.executor
        if a.surface == "shell":
            self._observe_shell(s, a)  # taint first, so the command's own egress is judged as tainted
            r = ex.shell(s.id, s.workspace, a.args.get("command", ""))
        elif a.surface == "git" and a.op == "push":
            if "_push_clone" not in a.args:
                raise RuntimeError(f"push not prepared ({a.args.get('_push_error', 'no configured remote')})")
            from .policy import normalize_branch
            r = ex.push(a.args["_push_clone"], normalize_branch(a.args.get("branch", "")), self.policy.git_remote)
        elif a.surface == "git":
            r = ex.git(s.id, s.workspace, a.op, a.args)
        else:
            r = ex.pkg(s.id, a.args.get("package", ""), a.args.get("registry"), a.args.get("manager", "pip"))
        obs.exit_code, obs.snapshot_id = r.exit_code, r.snapshot_id
        obs.files_written.extend(r.files_written)
        s.ledger.files_touched.update(f for f in r.files_written if not f.startswith("(deleted)"))
        if any(Path(f).name.startswith(".env") or f.endswith((".pem", ".key")) for f in r.files_written):
            s.secrets.scan_workspace(s.workspace)  # secret files created or edited by the command
        return f"exit {r.exit_code}\n{r.output}".rstrip(), obs

    def _http_request(self, s: Session, a: ActionEvent, obs: Observation) -> str:
        """Agent's explicit http_request, performed by the gateway after the decision."""
        import httpx
        body = str(a.args.get("body", ""))
        attempt = NetAttempt(a.destination or "", 443, len(body), [])
        s.net_log.append(attempt)
        obs.net_attempts.append(attempt)
        s.ledger.destinations.add(a.destination or "")
        r = httpx.request(str(a.args.get("method", "GET")).upper(), a.args["url"], content=body or None,
                          timeout=20, follow_redirects=False)
        return f"{r.status_code}\n{r.text[:10_000]}"

    def _discard_push_clone(self, a: ActionEvent) -> None:
        clone = a.args.get("_push_clone")
        if clone:
            import shutil
            shutil.rmtree(Path(clone).parent, ignore_errors=True)

    def _taint(self, s: Session, a: ActionEvent, path: str, ids: list[str]) -> None:
        s.ledger.taint.add("secret")
        s.ledger.taint_sources.append(TaintSource(path, a.id, ids))
        self.events.append({"type": "taint", "session": s.id, "path": path, "ts": time.time()})

    def _observe_read(self, s: Session, a: ActionEvent, target: Path, content: str) -> None:
        """A read of a SecretIndex file (symlink/case/./ safe) or of content holding secret material taints the ledger."""
        ids = []
        for e in s.secrets.entries.values():
            try:
                if os.path.samefile(target, s.workspace / e.path):
                    ids.append(e.id)
            except OSError:
                continue
        ids = sorted(set(ids) | set(s.secrets.scan(content)))
        if ids:
            self._taint(s, a, str(target.relative_to(s.workspace)), ids)

    def _observe_shell(self, s: Session, a: ActionEvent) -> None:
        """Conservative until strace/diff observation lands: a command naming a secret file taints."""
        cmd = a.args.get("command", "").casefold()
        names = {Path(e.path).name.casefold() for e in s.secrets.entries.values()}
        hit = [n for n in names if n in cmd]
        if hit:
            self._taint(s, a, hit[0], [])
