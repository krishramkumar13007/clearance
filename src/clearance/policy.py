"""Control layer: hard policies and trusted destinations. Rules only escalate."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

POLICY_VERSION = "p1"

_DESTRUCTIVE_SHELL = [
    (r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+|--recursive\s+|--force\s+)+(/|~|\$HOME|\*|\.)(/?)(\s|$)", "recursive delete of root/home/cwd"),
    (r"\bfind\s+(/|~|\.)\S*\s.*-delete\b", "bulk find -delete"),
    (r"\bgit\s+(clean\s+-\S*[fdx]|reset\s+--hard|push\s+.*--delete)", "destructive git operation"),
    (r"\brm\s+(-[a-zA-Z]*\s+)*\S*\.git/?(\s|$)", "delete of .git directory"),
    (r"\b(mkfs|fdisk|wipefs)\b", "filesystem destruction"),
    (r"\bdd\s+.*of=/dev/", "raw device write"),
    (r":\(\)\s*\{\s*:\|:&\s*\};:", "fork bomb"),
    (r"\b(shutdown|reboot|halt)\b", "system power command"),
    (r"\bchmod\s+-R\s+0?777\s+/(\s|$)", "recursive permission change on root"),
]


@dataclass
class Policy:
    trusted_destinations: set[str] = field(
        default_factory=lambda: {
            "registry.npmjs.org",
            "pypi.org",
            "files.pythonhosted.org",
            "api.tokenfactory.nebius.com",
        }
    )
    git_remote: str | None = None
    protected_branches: set[str] = field(default_factory=lambda: {"main", "master"})
    push_to_protected_authorized: bool = False

    def __post_init__(self) -> None:
        if self.git_remote:
            self.trusted_destinations.add(self.git_remote)

    def is_trusted(self, destination: str | None) -> bool:
        return destination is not None and destination in self.trusted_destinations


def path_in_workspace(path: str) -> bool:
    """Paths are workspace-relative; reject absolutes, home refs and traversal."""
    if path.startswith(("/", "~")):
        return False
    return ".." not in PurePosixPath(path).parts


def hard_policy_violation(policy: Policy, surface: str, op: str, args: dict) -> str | None:
    """Return a short reason string if the action violates a hard policy, else None."""
    if surface == "fs":
        path = args.get("path", "")
        if args.get("_escapes_workspace") or not path_in_workspace(path):
            return "path outside workspace"
        if op == "delete" and PurePosixPath(path).parts[:1] == (".git",):
            return "delete of .git directory"
    if surface == "shell":
        cmd = args.get("command", "")
        for pattern, reason in _DESTRUCTIVE_SHELL:
            if re.search(pattern, cmd):
                return reason
        if args.get("cwd") and not path_in_workspace(args["cwd"]):
            return "shell outside workspace"
    if surface == "git" and op == "push":
        if not policy.git_remote:
            return "no git remote configured"
        if args.get("remote", policy.git_remote) != policy.git_remote:
            return "push only to the configured remote"
        branch = normalize_branch(args.get("branch", ""))
        if branch in policy.protected_branches and not policy.push_to_protected_authorized:
            return "unauthorized push to protected branch"
        if args.get("force"):
            return "force push"
    return None


def normalize_branch(ref: str) -> str:
    """Strip refspec noise (+, HEAD:, refs/heads/); an empty ref pushes the current branch, assume protected."""
    ref = ref.lstrip("+").split(":")[-1]
    ref = ref.removeprefix("refs/heads/")
    return ref or "main"
