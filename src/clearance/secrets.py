"""SecretIndex: workspace secret values held in gateway memory only.

Raw values are never persisted or logged; everything else references secret ids.
"""
from __future__ import annotations

import base64
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

SECRET_FILE_PATTERNS = (".env", ".env.*", "*.pem", "*.key", "id_rsa", "id_ed25519", "credentials*", "*.p12")
EXAMPLE_SUFFIXES = (".example", ".sample", ".template")
_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
# "client_secret": "...", token: ... in JSON/YAML credential files
_KEYED = re.compile(r"""["']?([A-Za-z0-9_.-]*(?:secret|token|key|password|passwd|credential)[A-Za-z0-9_.-]*)["']?\s*[:=]\s*["']([^"']+)["']""", re.I)
MIN_SECRET_LEN = 8
COMMON_VALUES = {"production", "development", "localhost", "postgres", "password", "changeme", "undefined"}


def _offset_b64(b: bytes) -> set[str]:
    """Base64 substrings that survive a different byte alignment inside a larger payload."""
    out = set()
    for k, drop in ((1, 2), (2, 3)):
        enc = base64.b64encode(b"\x00" * k + b).decode().rstrip("=")
        enc = enc[drop:]
        if len(enc) > 4:
            out.add(enc[:-2])  # trailing chars depend on what follows the value
    enc = base64.b64encode(b).decode().rstrip("=")
    if len(enc) > 4:
        out.add(enc[:-2])
    return out


def _variants(raw: str) -> list[str]:
    b = raw.encode()
    out = _offset_b64(b) | {
        raw,
        base64.b64encode(b).decode(),
        base64.urlsafe_b64encode(b).decode(),
        base64.b64encode(b).decode().rstrip("="),
        b.hex(),
        b.hex().upper(),
        urllib.parse.quote(raw, safe=""),
        urllib.parse.quote_plus(raw),
        raw[::-1],
    }
    return [v for v in out if v]


def _clean_value(v: str) -> str:
    v = v.strip()
    if v[:1] in "'\"":
        q = v[0]
        end = v.find(q, 1)
        return v[1:end] if end > 0 else v[1:]
    return re.split(r"\s+#", v, maxsplit=1)[0].strip()  # drop inline comment


@dataclass
class SecretEntry:
    id: str
    path: str  # workspace-relative
    name: str
    _raw: str = field(repr=False)
    _variants: list[str] = field(repr=False, default_factory=list)

    def matches(self, haystack: str) -> bool:
        return any(v in haystack for v in self._variants)


@dataclass
class SecretIndex:
    entries: dict[str, SecretEntry] = field(default_factory=dict)

    @property
    def paths(self) -> set[str]:
        return {e.path for e in self.entries.values()}

    def ids_for_path(self, path: str) -> list[str]:
        return [e.id for e in self.entries.values() if e.path == path]

    def add(self, path: str, name: str, raw: str) -> str | None:
        if len(raw) < MIN_SECRET_LEN or raw.lower() in COMMON_VALUES:
            return None
        sid = f"s{len(self.entries) + 1}"
        self.entries[sid] = SecretEntry(sid, path, name, raw, _variants(raw))
        return sid

    def scan_workspace(self, root: Path) -> None:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or ".git" in p.parts or "node_modules" in p.parts:
                continue
            if p.name.endswith(EXAMPLE_SUFFIXES):
                continue
            if not any(p.match(g) for g in SECRET_FILE_PATTERNS):
                continue
            rel = str(p.relative_to(root))
            text = p.read_text(errors="ignore")
            if p.suffix in (".pem", ".key") or p.name.startswith("id_"):
                self.add(rel, p.name, text.strip())
                continue
            if p.suffix == ".pem" or "PRIVATE KEY" in text:
                for line in text.splitlines():
                    if line and not line.startswith("-----"):
                        self.add(rel, p.name, line.strip())
            for m in _KEYED.finditer(text):
                self.add(rel, m.group(1), m.group(2).strip())
            for line in text.splitlines():
                m = _ASSIGN.match(line)
                if m and not line.lstrip().startswith("#"):
                    self.add(rel, m.group(1), _clean_value(m.group(2)))

    def scan(self, payload: str | bytes | None) -> list[str]:
        """Return ids of secrets whose raw value or encoded variant appears in payload."""
        if not payload:
            return []
        text = payload.decode(errors="ignore") if isinstance(payload, bytes) else payload
        return [e.id for e in self.entries.values() if e.matches(text)]
