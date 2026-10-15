"""Docker sandbox: one container per session, default-deny network, one exit through the egress proxy.

Layout (all created by ensure_infra):
  clearance-internal   Docker network with --internal: no route out, no external DNS, no host access
  clearance-proxy      mitmproxy on clearance-internal + bridge; every request is held for a gateway decision
  clearance-sbx-<sid>  agent workspace container on clearance-internal only, workspace bind-mounted at /workspace

The sandbox holds workspace secrets (the repo's .env) but never control secrets: no git token, no API keys,
no mitmproxy CA private key (only the public cert is mounted).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

NETWORK = "clearance-internal"
PROXY = "clearance-proxy"
IMAGE = "clearance-sandbox"
PROXY_IMAGE = "mitmproxy/mitmproxy:latest"
DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker"
STATE_DIR = Path(os.environ.get("CLEARANCE_STATE_DIR", Path.home() / ".clearance"))
MAX_OUTPUT = 20_000
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv"}
_PKG_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-\[\]=<>~!,@/:+]*$")


class SandboxError(RuntimeError):
    pass


def docker(*args: str, check: bool = True, timeout: float = 120, env: dict | None = None,
           input: str | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, **(env or {})}, input=input)
    if check and p.returncode != 0:
        raise SandboxError(f"docker {args[0]} failed: {p.stderr.strip()[:400]}")
    return p


def docker_available() -> bool:
    try:
        return docker("info", "--format", "{{.ServerVersion}}", check=False, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _truncate(s: str) -> str:
    return s if len(s) <= MAX_OUTPUT else s[:MAX_OUTPUT] + f"\n... [{len(s) - MAX_OUTPUT} bytes truncated]"


def _tree_state(root: Path) -> dict[str, tuple[float, int]]:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            p = Path(dirpath) / f
            try:
                st = p.lstat()
            except OSError:
                continue
            out[str(p.relative_to(root))] = (st.st_mtime, st.st_size)
    return out


@dataclass
class ExecResult:
    exit_code: int
    output: str
    files_written: list[str]
    snapshot_id: str | None = None


class DockerExecutor:
    """Runs shell, git and package actions inside the session container. fs actions stay host-side on the
    bind-mounted workspace (the gateway already confines and resolves those paths)."""

    def __init__(self, gateway_url: str, proxy_token: str, git_token: str | None = None):
        self.gateway_url = gateway_url  # as seen from the proxy container
        self.proxy_token = proxy_token
        self.git_token = git_token if git_token is not None else os.environ.get("CLEARANCE_GIT_TOKEN")
        self.proxy_ip: str | None = None
        self.ca_cert = STATE_DIR / "mitm" / "mitmproxy-ca-cert.pem"
        self.containers: dict[str, str] = {}  # session id -> container name
        self._snap_n = 0

    # ---------- infrastructure ----------
    def ensure_infra(self) -> None:
        if docker("network", "inspect", NETWORK, check=False).returncode != 0:
            docker("network", "create", "--internal", NETWORK)
        if docker("image", "inspect", IMAGE, check=False).returncode != 0:
            docker("build", "-q", "-t", IMAGE, "-f", str(DOCKER_DIR / "sandbox.Dockerfile"), str(DOCKER_DIR),
                   timeout=900)
        (STATE_DIR / "mitm").mkdir(parents=True, exist_ok=True)
        docker("rm", "-f", PROXY, check=False)  # recreate so the proxy holds this gateway's token
        docker("run", "-d", "--name", PROXY, "--network", "bridge",
               "--add-host", "host.docker.internal:host-gateway",
               "-v", f"{STATE_DIR / 'mitm'}:/home/mitmproxy/.mitmproxy",
               "-v", f"{DOCKER_DIR}:/addon:ro",
               "-e", f"CLEARANCE_PROXY_TOKEN={self.proxy_token}",
               "-e", f"CLEARANCE_GATEWAY={self.gateway_url}",
               PROXY_IMAGE, "mitmdump", "--listen-port", "8080", "-s", "/addon/proxy_addon.py",
               "--set", "connection_strategy=lazy", "--set", "block_global=false", "--quiet")
        docker("network", "connect", NETWORK, PROXY)
        self.proxy_ip = docker("inspect", "-f", f'{{{{(index .NetworkSettings.Networks "{NETWORK}").IPAddress}}}}',
                               PROXY).stdout.strip()
        for _ in range(100):
            if self.ca_cert.exists() and self._proxy_listening():
                return
            time.sleep(0.2)
        raise SandboxError("egress proxy did not come up (no CA certificate or listener)")

    def _proxy_listening(self) -> bool:
        logs = docker("logs", PROXY, check=False, timeout=10)
        running = docker("inspect", "-f", "{{.State.Running}}", PROXY, check=False, timeout=10).stdout.strip()
        if running != "true":
            raise SandboxError(f"egress proxy exited: {(logs.stdout + logs.stderr)[-400:]}")
        return True

    def teardown(self) -> None:
        for name in list(self.containers.values()):
            docker("rm", "-f", name, check=False)
        self.containers.clear()
        docker("rm", "-f", PROXY, check=False)

    # ---------- sessions ----------
    def start(self, session_id: str, workspace: Path) -> str:
        """Start the session container; returns its IP on the internal network (used to attribute egress)."""
        if self.proxy_ip is None:
            raise SandboxError("ensure_infra() must run before sessions start")
        name = f"clearance-sbx-{session_id}"
        proxy = f"http://{self.proxy_ip}:8080"
        ca = "/etc/clearance/ca.pem"
        env = {"HTTP_PROXY": proxy, "HTTPS_PROXY": proxy, "http_proxy": proxy, "https_proxy": proxy,
               "NO_PROXY": "localhost,127.0.0.1", "no_proxy": "localhost,127.0.0.1",
               "SSL_CERT_FILE": ca, "REQUESTS_CA_BUNDLE": ca, "CURL_CA_BUNDLE": ca, "PIP_CERT": ca,
               "NODE_EXTRA_CA_CERTS": ca, "GIT_SSL_CAINFO": ca, "npm_config_cafile": ca,
               "npm_config_proxy": proxy, "npm_config_https_proxy": proxy}
        args = ["run", "-d", "--name", name, "--network", NETWORK,
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "512", "--memory", "2g",
                "-v", f"{workspace}:/workspace", "-v", f"{self.ca_cert}:{ca}:ro", "-w", "/workspace"]
        for k, v in env.items():
            args += ["-e", f"{k}={v}"]
        docker(*args, IMAGE)
        self.containers[session_id] = name
        return docker("inspect", "-f", f'{{{{(index .NetworkSettings.Networks "{NETWORK}").IPAddress}}}}',
                      name).stdout.strip()

    def stop(self, session_id: str) -> None:
        name = self.containers.pop(session_id, None)
        if name:
            docker("rm", "-f", name, check=False)
        snaps = docker("images", "-q", f"clearance-snap-{session_id}", check=False).stdout.split()
        if snaps:
            docker("rmi", "-f", *snaps, check=False)

    def _exec(self, session_id: str, script: str, env: dict | None = None, timeout: float = 300) -> tuple[int, str]:
        args = ["exec", "-w", "/workspace"]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        try:
            p = docker(*args, self.containers[session_id], "sh", "-c", script, check=False, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {int(timeout)}s"
        return p.returncode, _truncate(p.stdout + p.stderr)

    def snapshot(self, session_id: str) -> str | None:
        """Restore point for workspace files: a stash-style commit on a shadow ref (worktree untouched)."""
        code, out = self._exec(session_id, 'git rev-parse --git-dir >/dev/null 2>&1 || exit 3; '
                                           's=$(git stash create 2>/dev/null); [ -n "$s" ] || s=$(git rev-parse HEAD 2>/dev/null); '
                                           '[ -n "$s" ] || exit 3; git update-ref "refs/clearance/$N" "$s" && echo "$s"',
                               env={"N": f"snap-{self._next_snap()}"}, timeout=30)
        return out.strip().splitlines()[-1] if code == 0 and out.strip() else None

    def _next_snap(self) -> int:
        self._snap_n += 1
        return self._snap_n

    # ---------- surfaces ----------
    def shell(self, session_id: str, workspace: Path, command: str) -> ExecResult:
        snap = self.snapshot(session_id)
        before = _tree_state(workspace)
        code, out = self._exec(session_id, command, timeout=300)
        after = _tree_state(workspace)
        written = sorted(p for p, st in after.items() if before.get(p) != st)
        written += sorted(f"(deleted) {p}" for p in before if p not in after)
        return ExecResult(code, out, written, snap)

    def git(self, session_id: str, workspace: Path, op: str, args: dict) -> ExecResult:
        if op == "status":
            code, out = self._exec(session_id, "git status --short --branch")
        elif op == "diff":
            code, out = self._exec(session_id, "git --no-pager diff --no-color")
        elif op == "commit":
            snap = self.snapshot(session_id)
            before = _tree_state(workspace)
            code, out = self._exec(session_id, 'git add -A && git commit -q -m "$MSG" && git log --oneline -1',
                                   env={"MSG": str(args.get("message", ""))})
            return ExecResult(code, out, sorted(p for p, st in _tree_state(workspace).items()
                                                if before.get(p) != st), snap)
        else:
            raise SandboxError(f"unsupported git op: {op}")
        return ExecResult(code, out, [])

    def pkg(self, session_id: str, package: str, registry: str | None, manager: str = "pip") -> ExecResult:
        if not _PKG_NAME.match(package):
            raise SandboxError("invalid package specifier")
        snap = docker("commit", self.containers[session_id], f"clearance-snap-{session_id}:{self._next_snap()}",
                      check=False, timeout=120).stdout.strip() or None
        if manager == "npm":
            script = 'npm install --no-fund --no-audit "$PKG"' + (' --registry "$REG"' if registry else "")
        else:
            script = 'pip install -q --disable-pip-version-check "$PKG"' + (' --index-url "$REG"' if registry else "")
        code, out = self._exec(session_id, script, env={"PKG": package, "REG": registry or ""}, timeout=600)
        return ExecResult(code, out, [], snap)

    # ---------- push (gateway-side, with the custodied token) ----------
    def prepare_push(self, workspace: Path, branch: str, remote: str) -> tuple[str, str]:
        """Clone the workspace bare (repo config and hooks are not copied), then compute the exact outgoing
        commits as text for the secret scan. The push later runs from this same clone, so what was scanned
        is what is pushed. Returns (clone_dir, outgoing_text)."""
        tmp = tempfile.mkdtemp(prefix="clearance-push-")
        clone = Path(tmp) / "repo.git"
        safe = ["-c", "core.hooksPath=/dev/null", "-c", "credential.helper=", "-c", "protocol.file.allow=always"]
        p = subprocess.run(["git", *safe, "clone", "-q", "--bare", "--no-local", str(workspace), str(clone)],
                           capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            shutil.rmtree(tmp, ignore_errors=True)
            raise SandboxError(f"could not clone workspace for push: {p.stderr.strip()[:200]}")
        url = self.remote_url(remote)
        ls = subprocess.run(["git", *safe, "ls-remote", url, f"refs/heads/{branch}"], capture_output=True,
                            text=True, timeout=60)
        base = ls.stdout.split()[0] if ls.returncode == 0 and ls.stdout.strip() else None
        have_base = base and subprocess.run(["git", "-C", str(clone), "cat-file", "-e", base],
                                            capture_output=True).returncode == 0
        rng = [f"{base}..refs/heads/{branch}"] if have_base else [f"refs/heads/{branch}"]
        log = subprocess.run(["git", *safe, "-C", str(clone), "log", "-p", "--no-color", "--format=%H%n%an%n%B", *rng],
                             capture_output=True, text=True, timeout=120)
        refs = subprocess.run(["git", "-C", str(clone), "for-each-ref", "--format=%(refname)"],
                              capture_output=True, text=True).stdout
        return str(clone), (log.stdout[:5_000_000] + "\n" + refs)

    def remote_url(self, remote: str) -> str:
        if remote.startswith(("/", "file://")):
            return remote
        host_path = remote.removeprefix("https://").removesuffix(".git")
        if self.git_token:
            return f"https://x-access-token:{self.git_token}@{host_path}.git"
        return f"https://{host_path}.git"

    def push(self, clone_dir: str, branch: str, remote: str) -> ExecResult:
        if not remote.startswith(("/", "file://")) and not self.git_token:
            raise SandboxError("no git credentials configured on the gateway")
        try:
            p = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "credential.helper=",
                                "-c", "protocol.file.allow=always", "-C", clone_dir, "push", "-q",
                                self.remote_url(remote), f"refs/heads/{branch}:refs/heads/{branch}"],
                               capture_output=True, text=True, timeout=180)
            out = (p.stdout + p.stderr).replace(self.git_token or "\0", "***")
            return ExecResult(p.returncode, out or "pushed", [])
        finally:
            shutil.rmtree(Path(clone_dir).parent, ignore_errors=True)
