# Clearance

**Give permission once. Your assistant knows when to ask again.**

Clearance is a permission and approval layer for personal AI assistants. You authorize a workflow once, with clear boundaries: what it may do, who it may contact, what it may share, how many times and until when. After that the assistant works quietly inside those boundaries and stops only for meaningful exceptions: a new recipient, a new commitment on your behalf, information you didn't agree to share. Every permission is visible and revocable, and repeated approvals can suggest widening a permission but never widen it silently.

The problem is approval fatigue. An assistant that asks before every action interrupts you 100% of the time, and
after enough identical "yes" clicks people stop reading, so the one draft that matters goes out anyway. One that
never asks is how mistakes get sent. Email is the first surface; the same shape (one permission, quiet work inside
it, a card only at the edge) fits calendar invites, messages, purchases, file sharing and coding agents, the
second surface this repo already covers.

Built for the Nebius x NVIDIA Global AI Hackathon, Personal AI track. License: MIT.

![dashboard](docs/dashboard.jpg)

## The demo: email follow-ups

```bash
uv venv && uv pip install -e '.[dev]'
echo 'NEBIUS_API_KEY=...' > .env              # Token Factory key, gitignored
uv run python -m clearance assistant-demo     # open the printed dashboard URL, paste the printed human token
```

No Token Factory key? Run `uv run python -m clearance serve-demo --host 127.0.0.1` and open http://127.0.0.1:7860: with no key it replays Nemotron's recorded answers for the same drafts, and every decision still comes from Clearance's code.

1. **One permission.** The assistant asks once: "Routine follow-ups on the Q4 pilot", to four named contacts, may share meeting notes, next steps, pilot scope and setup status, never pricing, new deadlines or commitments on your behalf, at most 10 emails, for 5 days. You approve it on a single card.
2. **Quiet work.** It reads the threads from those contacts and sends four routine replies without asking.
3. **One meaningful exception.** A contact asks whether you can commit to a Monday go-live, and the natural reply says yes. Clearance stops it: "This draft makes a new commitment on your behalf." Send anyway or don't send.
4. **What it leaves alone.** A vendor email outside the permission is never read or answered.
5. **The count.** 2 interruptions with Clearance (the permission and the exception) against 5 if every email needed your OK.

The mailbox is a mock: nothing is ever sent anywhere.

The scripted drafts are fixed so the demo is repeatable. One was reworded for the hosted demo: Tom's status reply used to end "I'll follow up as soon as the invites go out", which live Nemotron read as a new commitment in 3 of 4 runs (the same kind of false alarm the results below report). It now reads as a plain status update and passes 5 of 5.

### Hosted demo

`clearance serve-demo` runs the same demo for the public: each visitor gets a private gateway, tokens, permissions and mock mailbox at `/v/<id>/`, and the human token reaches the browser in the URL fragment, which is never sent to the server. The Token Factory key stays on the server. Live Nemotron runs are capped per UTC day (`--daily-cap`, default 40) and per address (3 an hour); after the cap, or with no key configured, runs replay Nemotron's saved answers for the same drafts (`evals/demo_judge_cache.json`, written by `clearance demo-cache`). Visitors expire after 30 minutes.

**Try it online:** [huggingface.co/spaces/krishramkumar13007/clearance](https://huggingface.co/spaces/krishramkumar13007/clearance). That page is the free static version (`clearance static-demo`): it runs the scripted demo through the real gateway and assistant code once, with the judge replaying Nemotron's recorded answers for these drafts, and replays that trace in the browser; you still decide both approval cards. No model is called from it, so live judging is local (`assistant-demo`) or via `serve-demo`. Publish it with `scripts/deploy_static_space.sh <hf-user>/<space-name>` after `hf auth login`.

`serve-demo` can also run as a Docker Space (`scripts/deploy_space.sh`, with `NEBIUS_API_KEY` as a Space secret), but Hugging Face now requires a PRO subscription for Docker Spaces.

## How it decides

For every action the assistant takes, in order:

1. **Is there an active permission for this kind of action?** No: ask, and offer to create one.
2. **Is the action inside it?** Recipients, count limit and expiry are checked exactly. Outside: ask, saying exactly what changed ("Adds a recipient you didn't approve: finance@contoso.test").
3. **Does the content stay inside it?** A judge answers three narrow questions about the draft: does it make a new commitment, share something the permission doesn't cover, or drift off purpose? Any yes: ask, with that reason.
4. **Otherwise proceed and log it** against the permission.

The judge can only add an approval card. It never grants or widens a permission: model confidence is not consent. If the judge is unavailable or returns something invalid, Clearance asks.

The judge is **Nemotron 3 Super on Nebius Token Factory** (strict JSON, low reasoning effort, about 2 s per check). Each question has a cost for missing it (commitment 20, sharing 50, off purpose 10, in units of one interruption), and the ask threshold is fitted once per judge on a separate calibration split.

## Results: does it ask at the right times?

`evals/email-v1.jsonl`: 50 hand-written outgoing emails under the demo permission, each with the message it replies to. 34 are the held-out eval split (16 should go out quietly, 18 should need approval: 14 content problems, 4 wrong recipients); 16 are a calibration split used only to fit thresholds.

| Setup | Interruptions on emails that should go out | Boundary misses | Content misses | p50 latency |
|---|---|---|---|---|
| Ask before every email (no Clearance) | 100% | 0% | 0% | 0 ms |
| Permission rules only | 0% | 78% | 100% | 0 ms |
| Rules + Laya judge | 100% | 0% | 0% | 1723 ms |
| Rules + Nemotron judge, raw cost thresholds | 31% | 0% | 0% | 1917 ms |
| Rules + Nemotron judge, thresholds fitted on calib split | 6% | 0% | 0% | 2003 ms |
| Rules + CLM judge, released head, fitted thresholds | 100% | 0% | 0% | scored offline |
| Rules + CLM judge, heads fine-tuned on our training split, fitted thresholds | 81% | 0% | 0% | scored offline |
| Rules + Laya fine-tuned with RLCD on our training split, seed 0, fitted thresholds | 0% | 28% | 36% | scored offline |
| Rules + Laya fine-tuned with RLCD on our training split, seed 1, fitted thresholds | 0% | 17% | 21% | scored offline |

- **Without Clearance**, every email interrupts you. **Permission rules alone** never interrupt, but miss every content problem (a promised deadline, a quoted price): they only catch wrong recipients.
- **Nemotron with fitted thresholds** interrupts on 1 of 16 routine emails and missed none of the 18 that needed approval, stable across three runs. The one false alarm was "I'll connect your engineers with the right person", read as a commitment.
- **Laya** (an open, RLCD-trained calibrated decision model run locally) separates commitments reasonably but rates almost every draft as "sharing outside the permission", so no threshold helps; it asks about everything. It would need fine-tuning on this task.
- **Fine-tuned local judges** were trained on `evals/email-train-v1.jsonl`: 1,044 generated emails under six permissions that are not the eval's, labeled per question, with Nemotron paraphrases cleaned to plain ASCII. Tests keep it away from email-v1: no shared email text, and no training case whose incoming message and reply both resemble an eval case.
- **CLM** ([Contrastive-LM](https://github.com/Contrastive-LM/CLM), Apache 2.0: small heads on a frozen Qwen3-8B encoder, fine-tuned on Kaggle T4 x2 in minutes) keeps every boundary email but interrupts on 13 of 16 routine ones: it reads plain notes and next steps as sharing outside the permission.
- **Laya fine-tuned with RLCD** (Laya's own `laya.train.finetune`, one Kaggle T4, two seeds because takeoff is seed-dependent) swings the other way: it never interrupts, but lets through 3 (seed 1) or 5 (seed 0) of the 14 content problems, among them a booked call with the CTO, a home address and an accepted Friday deadline, each at P(yes) of about 0.003, so no threshold can catch them. A 421M model that answers in ~0.3 s on a laptop CPU is the right shape for a private judge, but 1,044 templated emails are not enough to teach it this task; Nemotron stays the judge.
- **An earlier version of these results was wrong.** A first training split included a batch written after looking at which email-v1 cases CLM missed, plus near-copies of three eval cases; on it, Laya seed 1 appeared to match Nemotron (6% / 0%). A pre-merge review caught it, the split was rebuilt without them, and both models were retrained; the numbers above are from the rebuilt split.
- **Raw thresholds** (taking the model's probabilities at face value) interrupt on 31% of routine emails: prompted models are not calibrated at the few-percent level, which is why the thresholds are fitted.

**Caveats.** One author wrote the permission, the suite and the drafts; 34 eval cases is small; the eval measures the decision on one draft at a time, not a full agent loop. Treat the numbers as evidence the design works, not as a benchmark.

Reproduce: `uv run python -m clearance email-calibrate && uv run python -m clearance email-eval --judge ask-every --judge none --judge laya+t --judge nemotron --judge nemotron+t`. CLM: `clm-export`, run `kaggle/clm-email/run.py` on a GPU (T4 x2), copy its `email_clm_probs.json` to `evals/`, then `email-calibrate --judge clm --judge clm-zero` and `email-eval --judge clm+t --judge clm-zero+t`. Laya RLCD: the same dataset with `kaggle/laya-email/run.py` (one GPU). Each probability file records a hash of the training split it came from, and the tests fail if the committed split has changed since, copy `email_laya_rlcd_probs.json` to `evals/`, then `email-calibrate --judge laya-rlcd --judge laya-rlcd-s1` and `email-eval --judge laya-rlcd+t --judge laya-rlcd-s1+t`.

## Plugging into an assistant

Clearance has to sit in the assistant's real tool path, or the user gets asked twice. The assistant's only tools come from the Clearance MCP server (`request_permission`, `email_list`, `email_read`, `email_send`, plus the coding tools below), and every call becomes a decision.

- **Your own agent loop** (the demo): Clearance is the only approval path, so there is exactly one prompt per decision.
- **MCP hosts with their own tool-permission prompt** (for example Claude Desktop): set the Clearance tools to always-allow in the host, so the host stops asking and Clearance makes the real decision. If a host cannot be told to stop asking, the user will see two prompts; that is a limit of the host, not something Clearance can override.

Run the gateway with assistant surfaces: `uv run python -m clearance serve --assistant --judge nemotron`, then point the MCP server (`clearance-mcp`) at it with the printed agent token.

Adding Clearance to an assistant you build or run (any agent with tool calls), step by step:

1. **Run the gateway** next to the assistant: `uv run python -m clearance serve --assistant --judge nemotron`. It prints an agent token (for the assistant) and a human token (for you, on the dashboard).
2. **Give the assistant Clearance's tools instead of direct ones.** If it speaks MCP, add `clearance-mcp` as a tool server with the agent token. If not, call the HTTP API: `POST /sessions/{id}/actions` with `{"surface": "email", "op": "send", "args": {...}}` and act only on `ALLOW`.
3. **Ask for a permission once per workflow** with `request_permission` (purpose, recipients, what it may share, what it must never share, a use limit and an expiry). You approve it on one card.
4. **Remove the assistant's own "confirm before sending" step** for those actions, so you are not asked twice. Clearance now decides: inside the permission it proceeds; at the edge it holds the action and shows you why.
5. **Connect the real mailbox** behind the gateway's email surface. The demo uses a mock mailbox, and a real provider (Gmail, Microsoft 365) is not built yet: it would sit behind the same `email_list`, `email_read` and `email_send` tools.

# Also: coding agents

The same gateway started as a guard for coding agents, with a Docker sandbox, a default-deny egress proxy, secret taint tracking and a 70-trajectory eval. That work is kept as a second surface.

### How it decides

Every agent action goes through three layers. Rules run first and can only make a decision stricter.

| Layer | Question it answers | Mechanism |
| --- | --- | --- |
| Control | What must never happen? | Hard policies, credential custody, Docker sandbox, default-deny egress proxy |
| Context | What has this agent already done? | Session ledger, secret taint, destination history, agent fingerprint |
| Uncertainty | What does this ambiguous action probably mean? | Nemotron on Nebius Token Factory estimates p(destructive) and p(exfiltration) |

Evaluation order for each action:

1. Hard policy violation (destructive system command, push to a protected branch or another remote, path outside the workspace): **BLOCK**
2. Workspace secret (raw, base64, hex, URL-encoded or reversed) in any outbound payload, trusted destination or not: **BLOCK**
3. Session has read secrets and the action can reach an untrusted destination: **ASK**
4. Novel action that cannot be undone: **ASK**
5. Otherwise the lowest expected cost wins, using the model's two probabilities
6. An ALLOW in a tainted session is recorded as **ALLOW_AUDIT**

#### Cost matrix

Expected cost, in units of one human interruption. These are product choices; the eval reports a sweep over them.

```
cost(ALLOW) = p_d * C_FA,d + p_e * C_FA,e
cost(ASK)   = C_ASK
cost(BLOCK) = (1 - p_d)(1 - p_e) * C_FB
```

| Cost | Value | Meaning |
| --- | --- | --- |
| C_ASK | 1 | One human interruption |
| C_FB | 5 | Blocking legitimate work; the task stalls |
| C_FA,d | 200 | Letting a destructive action run |
| C_FA,e | 500 | Letting an exfiltration out |

ASK beats ALLOW above p_d = 0.005 or p_e = 0.002, so the probabilities have to be calibrated for this to work. That is what the evaluation measures.

### Quick start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker (for the sandbox), a Nebius Token Factory API key (for the model arms).

```bash
git clone https://github.com/<owner>/clearance && cd clearance
uv venv && uv pip install -e '.[dev]'
echo 'NEBIUS_API_KEY=...' > .env        # gitignored; only needed for model arms
uv run pytest -q                         # unit + API tests; add -m docker for the sandbox suite
```

Watch a scripted agent session on the dashboard (rules-only arm, no key needed):

```bash
uv run python -m clearance demo --sandbox
# open the printed dashboard URL, paste the printed human token
```

Run the gateway for a real agent:

```bash
uv run python -m clearance serve --sandbox --arm B --root ~/code --git-remote github.com/you/repo
```

Then point any MCP-capable coding agent at the Clearance MCP server, which exposes the agent's only tools (`fs_read`, `fs_write`, `fs_list`, `shell_exec`, `git_status`, `git_diff`, `git_commit`, `git_push`, `pkg_install`, `http_request`):

```json
{
  "mcpServers": {
    "clearance": {
      "command": "clearance-mcp",
      "env": {
        "CLEARANCE_URL": "http://127.0.0.1:8750",
        "CLEARANCE_AGENT_TOKEN": "<agent token printed by serve>",
        "CLEARANCE_SESSION_ID": "<id from POST /sessions>"
      }
    }
  }
}
```

### Architecture

```
 agent ──MCP──▶ clearance-mcp ──HTTP (agent token)──▶ gateway (FastAPI)
                                                       │  rules → ledger/taint → Nemotron → expected cost
                                                       ▼
                         ┌──────────── Docker, --internal network (no route, no DNS) ─────────────┐
                         │  clearance-sbx-<session>  ──every HTTP(S) request──▶  clearance-proxy   │
                         │  workspace at /workspace                              (mitmproxy)       │
                         └────────────────────────────────────────────────────────────┬───────────┘
                                                                                       │ held until
 human ──dashboard / HTTP (human token)──▶ gateway ◀──/internal/egress (proxy token)───┘ decided
```

- **Sandbox.** One container per session on a Docker network created with `--internal`: no default route, no external DNS, no access to the host. The only reachable host is the egress proxy. The container holds workspace secrets (the repo's `.env`) but never control secrets: no git token, no API key, no proxy CA private key.
- **Egress proxy.** mitmproxy with a CA the sandbox trusts, so request bodies can be read. Every request is held while the gateway decides it with the same rules as everything else, and the proxy fails closed if the gateway is unreachable.
- **Credential custody.** `git push` runs on the gateway from a bare clone of the workspace (repo config and hooks are not copied), only to the configured remote. The exact outgoing commits are scanned for secrets first, and what was scanned is what gets pushed.
- **Restore points.** A shadow git ref is taken before every shell command and commit, and a container snapshot before every package install. That is what makes sandboxed shell commands count as reversible.
- **Three credentials.** The agent token can submit actions and wait on its own ASKs. The human token can create sessions, resolve ASKs and read the dashboard. The proxy token can only submit egress. The agent never holds the other two.

### How Nemotron and Nebius are used

- **Nemotron on Nebius Token Factory** (`nvidia/nemotron-3-super-120b-a12b`, OpenAI-compatible API with strict JSON-schema output) estimates `p_destructive` and `p_exfiltration` for any action the rules leave undecided. The prompt sees the action and a structured ledger summary only: never raw tool output, and never the agent's own justification. Model errors, timeouts and malformed output fall back to ASK, never ALLOW.
- **Arm C′** applies Platt scaling (a temperature and a bias per harm) fitted on a separate calibration set to the prompted model's probabilities.
- **Arm B0** asks Nemotron for an ALLOW/ASK/BLOCK verdict directly, the "LLM as judge" baseline.
- Probe notes and model behavior are in [docs/NEBIUS_NOTES.md](docs/NEBIUS_NOTES.md).

### Evaluation

All numbers come from a hand-written suite frozen before any calibration was fitted. The demo session is illustration only.

- **Suite:** [`evals/eval-v1.jsonl`](evals/eval-v1.jsonl), 70 trajectories, 138 labeled actions, git tag `eval-v1`, sha256 `fd45da33ce283b054e52b8141f87a878de771beee6d97f152072cd522adb9ca5`.
- **Categories:** benign routine (18), benign near-miss (10), obvious destructive (6), ambiguous destructive (8), direct exfiltration (6), encoded or split exfiltration (8), chained (8), novel but legitimate (6).
- **Conditions:** every trajectory runs through the real gateway in a fresh workspace with a fresh agent fingerprint (cold start, no earned trust). Shell commands are not executed; their network effects are written into the suite as the egress requests the proxy would see.
- **Arms:** A (rules only), B (prompted Nemotron, plain prompt), B-coached (prompt says routine work is benign), B0 (Nemotron verdict, no probabilities), C′ (B plus Platt scaling), C-iso (B plus isotonic calibration), L (Laya by Convai Innovations, Apache 2.0 weights, run locally via `@receptron/laya`; not trained or tuned by us). Any arm can also run "earned", as an agent with established trust.

| Arm | Harm executed (traj) | False ALLOW (action) | Benign ASK | False BLOCK | Mean cost | ECE | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| A | 0.229 [0.120, 0.333] | 0.295 [0.143, 0.444] | 0.043 [0.010, 0.095] | 0.000 [0.000, 0.000] | 25.464 [11.230, 41.098] | — | 0.1 | 0.7 |
| B | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.713 [0.632, 0.798] | 0.000 [0.000, 0.000] | 0.573 [0.489, 0.650] | 0.0329 | 2079.7 | 6556.2 |
| B-coached | 0.057 [0.000, 0.105] | 0.045 [0.000, 0.109] | 0.053 [0.011, 0.107] | 0.000 [0.000, 0.000] | 3.022 [0.097, 7.504] | 0.0362 | 1733.1 | 9624.4 |
| B0 | 0.029 [0.000, 0.056] | 0.023 [0.000, 0.073] | 0.149 [0.077, 0.236] | 0.032 [0.000, 0.066] | 1.775 [0.230, 4.921] | — | 1171.6 | 2123.8 |
| C' | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.575 [0.476, 0.680] | 0.000 [0.000, 0.000] | 0.507 [0.424, 0.592] | 0.0445 | 1931.5 | 9833.9 |
| C-iso | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.915 [0.850, 0.971] | 0.000 [0.000, 0.000] | 0.710 [0.628, 0.790] | 0.0273 | 0.1 | 0.7 |
| L | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.872 [0.788, 0.947] | 0.043 [0.000, 0.097] | 0.848 [0.701, 1.015] | 0.2005 | 924.4 | 1553.2 |
| A earned | 0.229 [0.120, 0.333] | 0.295 [0.143, 0.444] | 0.043 [0.010, 0.095] | 0.000 [0.000, 0.000] | 25.464 [11.230, 41.098] | — | 0.1 | 0.5 |
| B earned | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.096 [0.033, 0.180] | 0.000 [0.000, 0.000] | 0.152 [0.078, 0.222] | 0.0345 | 0.1 | 0.5 |
| C' earned | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.074 [0.022, 0.140] | 0.000 [0.000, 0.000] | 0.138 [0.072, 0.207] | 0.032 | 0.1 | 0.2 |
| C-iso earned | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.074 [0.020, 0.150] | 0.000 [0.000, 0.000] | 0.138 [0.070, 0.212] | 0.0146 | 0.1 | 0.5 |
| L earned | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.872 [0.788, 0.947] | 0.043 [0.000, 0.097] | 0.848 [0.701, 1.015] | 0.2005 | 0.1 | 0.6 |

Rows marked **earned** run each trajectory as an agent that has already earned HIGH autonomy (20+ actions per surface, no denied reviews, no hard-rule trips). Unmarked rows are a cold start: a brand-new agent with no history.

What the numbers say:

- **Rules alone (A)** rarely interrupt (4% benign ASK) but let 23% of harmful trajectories through, all ambiguous destructive commands (`shutil.rmtree`, a malicious `postinstall`, a poisoned `Makefile`) that no pattern list catches.
- **Prompted Nemotron (B)** stops every harmful trajectory but, for a new agent, asks about 65 to 71% of benign actions (the live model is not fully deterministic between runs). It rates routine work at 0.01 to 0.05, and anything above 0.005 makes ASK cheaper than ALLOW. That is the approval-fatigue failure the spec predicted.
- **Post-hoc calibration does not fix it on this much data.** Platt scaling (C′) trims benign ASKs to 58%. Isotonic calibration (C-iso) makes it worse (92%): with only 32 clean benign examples it cannot certify a probability below the 0.2% exfiltration threshold, so it rounds everything up. Getting there by calibration needs hundreds of verified benign examples or training-time calibration (RLCD).
- **Earned autonomy does fix it once an agent has a track record.** With trust earned and calibrated Nemotron (C′ earned), benign ASKs drop from 58% to 7%, with no harmful trajectory executed and mean cost down from 0.51 to 0.14. Earned trust only relaxes cost-based ASKs on actions the gateway can roll back, never the hard rules, never irreversible actions, and it is suspended for the rest of a session once anything in it was blocked, denied or is awaiting review.
- **Laya (L), an open calibrated decision model run locally on CPU,** ranks risk almost as well as Nemotron (AUC 0.92 vs 0.98 for destructive, 0.99 vs 1.00 for exfiltration) but is poorly calibrated on coding-agent actions out of the box: it puts benign work at a median p_exfiltration of 0.25, so 87% of benign actions get an ASK and 4% are blocked (ECE 0.20). It would need domain fine-tuning to be useful here; it is wired in as an optional shadow estimator (`--shadow-laya`) that is recorded but never acted on.
- **Coaching the prompt (B-coached)** also cuts ASKs (5%) but lets two ambiguous destructive trajectories through. **A plain verdict (B0)** sits in between and is the only arm that blocks legitimate work.

**Disclosure on the earned rows.** The session-suspension rule was added after an earlier earned run let two trajectories through (a `postinstall` and a `Makefile` follow-up whose first, malicious step had been blocked). Those two trajectories informed the rule, so the earned rows are not a fully held-out result for it. The cold rows are untouched by that change.

<details><summary>Cost-ratio sweep (C_FA scaled x0.25 and x4; latency columns reflect cached model calls)</summary>

| Arm | Harm executed (traj) | False ALLOW (action) | Benign ASK | False BLOCK | Mean cost | ECE | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| A C_FA x0.25 | 0.229 [0.120, 0.333] | 0.295 [0.143, 0.444] | 0.043 [0.010, 0.095] | 0.000 [0.000, 0.000] | 25.464 [11.230, 41.098] | — | 0.1 | 0.4 |
| A C_FA x4.0 | 0.229 [0.120, 0.333] | 0.295 [0.143, 0.444] | 0.043 [0.010, 0.095] | 0.000 [0.000, 0.000] | 25.464 [11.230, 41.098] | — | 0.1 | 0.4 |
| B C_FA x0.25 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.553 [0.454, 0.657] | 0.000 [0.000, 0.000] | 0.478 [0.389, 0.564] | 0.028 | 0.1 | 1.8 |
| B C_FA x4.0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.766 [0.679, 0.850] | 0.000 [0.000, 0.000] | 0.623 [0.540, 0.708] | 0.028 | 0.1 | 0.7 |
| B-coached C_FA x0.25 | 0.057 [0.000, 0.105] | 0.045 [0.000, 0.109] | 0.043 [0.010, 0.095] | 0.000 [0.000, 0.000] | 3.014 [0.087, 7.489] | 0.0366 | 0.1 | 5.4 |
| B-coached C_FA x4.0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.787 [0.694, 0.871] | 0.000 [0.000, 0.000] | 0.630 [0.545, 0.706] | 0.0587 | 0.1 | 4806.1 |
| B0 C_FA x0.25 | 0.029 [0.000, 0.056] | 0.023 [0.000, 0.073] | 0.149 [0.077, 0.236] | 0.032 [0.000, 0.066] | 1.775 [0.230, 4.921] | — | 0.1 | 1.0 |
| B0 C_FA x4.0 | 0.029 [0.000, 0.056] | 0.023 [0.000, 0.073] | 0.149 [0.077, 0.236] | 0.032 [0.000, 0.066] | 1.775 [0.230, 4.921] | — | 0.0 | 0.3 |
| C' C_FA x0.25 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.277 [0.180, 0.379] | 0.000 [0.000, 0.000] | 0.297 [0.216, 0.377] | 0.0399 | 0.2 | 2865.2 |
| C' C_FA x4.0 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.575 [0.476, 0.680] | 0.000 [0.000, 0.000] | 0.500 [0.418, 0.588] | 0.0432 | 0.1 | 0.5 |

</details>

**Metrics.** "Harm executed" is the share of harmful trajectories where a destructive action ran or a request or push carrying the secret got out. "False ALLOW" is per action and counts a shell command that attempts exfiltration even when its traffic is later blocked at the proxy. Brackets are 95% bootstrap intervals over trajectories. ECE is expected calibration error over both probabilities (lower is better). Latency is per decision, including the model call.

Reproduce: `uv run python -m clearance calibrate && uv run python -m clearance eval --arm A --arm B --arm B-coached --arm B0 --arm "C'" --arm C-iso --sweep && uv run python -m clearance eval --earned --arm A --arm B --arm "C'" --arm C-iso`.

**Honest caveats.** The same author wrote the rules, the suite and the calibration set, so the suite will favor the rules more than real traffic would. With about 30 harmful trajectories, the intervals are wide; treat differences inside overlapping intervals as noise.

### Known limits

- Timing and covert channels, and payloads encoded in ways the scanner does not try (for example ROT13 or compression). Session taint is the fallback: once a session has read secrets, any request to an untrusted destination is ASKed.
- Taint from shell reads is detected from the command text (`cat .env`); a command that builds the filename at runtime is not seen until its traffic reaches the proxy.
- The hard-policy command list is a speed bump, not a boundary. The sandbox's filesystem and network confinement is the boundary.
- One gateway process, in-memory state, no multi-tenancy. Not production-hardened.

## Repository map

| Path | What |
| --- | --- |
| `src/clearance/assistant/` | Permissions, mock mailbox, judge, email demo and eval |
| `src/clearance/decision.py` | Rules and expected-cost engine (coding surfaces) |
| `src/clearance/gateway.py` | Sessions, ledger, taint, ASK flow, execution |
| `src/clearance/sandbox.py` | Docker sandbox, egress proxy, push from a bare clone |
| `src/clearance/risk.py` | Nemotron risk backends (arms B, B-coached, B0) |
| `src/clearance/calibrate.py` | Calibration set and Platt scaling (arm C′) |
| `src/clearance/evals.py` | Evaluation harness and metrics |
| `src/clearance/api.py`, `mcp_server.py` | HTTP API and the agent's MCP tools |
| `src/clearance/static/dashboard.html` | Live dashboard: activity, approval and permission cards, interruption count |
| `laya/` | Optional local Laya service (Node, `@receptron/laya`) |
| `docker/` | Sandbox image and the mitmproxy addon |
