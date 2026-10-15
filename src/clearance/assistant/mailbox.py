"""Mock mailbox for the demo: nothing is ever sent anywhere. Sent mail lands in an in-memory folder."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

from ..models import new_id

ME = "alex@northwind.test"

# A small, fixed inbox: the week after a pilot kickoff with four contacts at Contoso.
INBOX = [
    {"id": "m1", "from": "nadia@contoso.test", "cc": [], "subject": "Q4 pilot kickoff notes",
     "body": "Thanks for the session today. Could you send over the notes and the next steps we agreed on?"},
    {"id": "m2", "from": "tom@contoso.test", "cc": [], "subject": "Pilot access",
     "body": "Quick check: are the pilot accounts set up for my team yet? Happy to wait for your update."},
    {"id": "m3", "from": "lena@contoso.test", "cc": ["finance@contoso.test"], "subject": "Re: pilot scope",
     "body": "Looping in our finance team. Can you share where we landed on the scope for the pilot?"},
    {"id": "m4", "from": "omar@contoso.test", "cc": [], "subject": "Timeline",
     "body": "Our VP is asking whether you can commit to having the integration live by Monday. Can you confirm?"},
    {"id": "m5", "from": "sam@vendorly.test", "cc": [], "subject": "Partnership idea",
     "body": "We'd love to explore a partnership. Do you have 30 minutes next week?"},
    {"id": "m6", "from": "nadia@contoso.test", "cc": [], "subject": "Re: Q4 pilot kickoff notes",
     "body": "Also, could you remind the team which dashboards are in scope for week one?"},
]

CONTACTS = {
    "nadia@contoso.test": "Nadia Brooks (Contoso, pilot lead)",
    "tom@contoso.test": "Tom Becker (Contoso, IT)",
    "lena@contoso.test": "Lena Ortiz (Contoso, operations)",
    "omar@contoso.test": "Omar Haddad (Contoso, engineering)",
    "finance@contoso.test": "Contoso finance team",
    "sam@vendorly.test": "Sam Lee (Vendorly)",
}


@dataclass
class MockMailbox:
    inbox: list[dict] = field(default_factory=lambda: copy.deepcopy(INBOX))
    sent: list[dict] = field(default_factory=list)

    def list(self) -> list[dict]:
        return [{"id": m["id"], "from": m["from"], "subject": m["subject"]} for m in self.inbox]

    def read(self, mid: str) -> dict:
        m = next((m for m in self.inbox if m["id"] == mid), None)
        if m is None:
            raise KeyError(f"no message {mid}")
        return dict(m)

    def sender_of(self, mid: str) -> str | None:
        m = next((m for m in self.inbox if m["id"] == mid), None)
        return m["from"] if m else None

    def send(self, to: list[str], cc: list[str], subject: str, body: str, in_reply_to: str | None) -> dict:
        msg = {"id": new_id(), "from": ME, "to": to, "cc": cc, "subject": subject, "body": body,
               "in_reply_to": in_reply_to, "sent_at": time.time()}
        self.sent.append(msg)
        return msg
