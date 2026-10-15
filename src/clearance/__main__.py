"""python -m clearance serve [--root DIR] [--port N] [--arm A|B|B-coached]"""
from __future__ import annotations

import argparse
import atexit
import os
import secrets
from pathlib import Path

import uvicorn

from .api import create_app
from .gateway import Gateway
from .policy import Policy
from .risk import NemotronRisk


def main() -> None:
    p = argparse.ArgumentParser(prog="clearance")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--root", default=".", help="directory sessions' workspaces must live under")
    s.add_argument("--port", type=int, default=8750)
    s.add_argument("--arm", default="B", choices=["A", "B", "B-coached"])
    s.add_argument("--git-remote", default=None)
    s.add_argument("--sandbox", action="store_true", help="run shell/git/pkg in Docker behind the egress proxy")
    s.add_argument("--shadow-laya", action="store_true", help="also score every action with the local Laya service "
                   "(recorded only, never acted on)")
    s.add_argument("--assistant", action="store_true", help="enable permission and (mock) email surfaces")
    s.add_argument("--judge", default="nemotron", choices=["none", "nemotron", "laya", "laya-calibrated"])
    d = sub.add_parser("demo", help="start the gateway and run a scripted agent session")
    d.add_argument("--port", type=int, default=8750)
    d.add_argument("--arm", default="A", choices=["A", "B", "B-coached"])
    d.add_argument("--sandbox", action="store_true")
    d.add_argument("--shadow-laya", action="store_true")
    d.add_argument("--pause", type=float, default=2.5)
    c = sub.add_parser("calibrate", help="fit arm C' (Platt) and C-iso (isotonic) on the separate calibration set")
    c.add_argument("--method", choices=["platt", "isotonic", "both"], default="both")
    ad = sub.add_parser("assistant-demo", help="email follow-up demo: one permission, quiet sends, one exception")
    ad.add_argument("--port", type=int, default=8750)
    ad.add_argument("--judge", default="nemotron", choices=["none", "nemotron", "laya", "laya-calibrated"])
    ad.add_argument("--pause", type=float, default=2.0)
    ad.add_argument("--llm-drafts", action="store_true", help="let Nemotron write the replies (less repeatable)")
    ee = sub.add_parser("email-eval", help="score judges on the email permission suite")
    ee.add_argument("--judge", action="append", choices=["none", "nemotron", "laya", "laya-calibrated", "ask-every",
                                                         "nemotron+t", "laya+t", "laya-calibrated+t",
                                                         "clm", "clm-zero", "clm+t", "clm-zero+t",
                                                         "laya-rlcd", "laya-rlcd+t", "laya-rlcd-s1", "laya-rlcd-s1+t"])
    ec = sub.add_parser("email-calibrate", help="fit Laya's calibration map and per-judge thresholds on the email "
                        "calib split")
    ec.add_argument("--judge", action="append", choices=["laya", "laya-calibrated", "nemotron", "clm", "clm-zero",
                                                         "laya-rlcd", "laya-rlcd-s1"],
                    help="only fit these judges' thresholds, keeping the others (default: refit Laya and Nemotron)")
    ce = sub.add_parser("clm-export", help="write the Kaggle dataset and metadata for the CLM email fine-tune")
    ce.add_argument("--kaggle-user", default=None)
    sub.add_parser("email-train-data", help="write the email fine-tuning split (disjoint from email-v1)")
    sd = sub.add_parser("serve-demo", help="public demo: a private gateway per visitor, live judge runs capped per day")
    sd.add_argument("--host", default="0.0.0.0")
    sd.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    sd.add_argument("--daily-cap", type=int, default=int(os.environ.get("CLEARANCE_DEMO_DAILY_CAP", "40")),
                    help="live Nemotron runs per UTC day; later runs replay saved answers")
    sd.add_argument("--per-ip-hour", type=int, default=3)
    sd.add_argument("--pause", type=float, default=2.0)
    st = sub.add_parser("static-demo", help="write the demo as one static page (replays saved judge answers)")
    st.add_argument("--out", default="dist/static-demo")
    sub.add_parser("demo-cache", help="save Nemotron's answers for the scripted demo replies (Token Factory)")
    sub.add_parser("email-paraphrase", help="cache Nemotron paraphrases of the training templates (Token Factory)")
    e = sub.add_parser("eval", help="run evaluation arms over the frozen suite")
    e.add_argument("--earned", action="store_true", help="each trajectory runs as an agent that has already earned "
                   "HIGH autonomy (clean reviewed history)")
    e.add_argument("--arm", action="append", choices=["A", "B", "B-coached", "B0", "C'", "C-iso", "L"],
                   help="repeatable; default A")
    e.add_argument("--sweep", action="store_true", help="also report the cost-ratio sweep (C_FA x0.25, x4)")
    a = p.parse_args()
    if a.cmd == "eval":
        return run_eval(a)
    if a.cmd == "demo":
        return run_demo(a)
    if a.cmd == "assistant-demo":
        return run_assistant_demo(a)
    if a.cmd == "email-eval":
        return run_email_eval(a)
    if a.cmd == "email-calibrate":
        return run_email_calibrate(a.judge)
    if a.cmd == "clm-export":
        import json

        from .assistant.clm_judge import export
        return print(json.dumps(export(a.kaggle_user)))
    if a.cmd == "serve-demo":
        from .hosted import Limits, create_hosted_app
        app = create_hosted_app(a.port, Limits(daily_cap=a.daily_cap, per_ip_hour=a.per_ip_hour), pause=a.pause)
        return uvicorn.run(app, host=a.host, port=a.port, log_level="warning", proxy_headers=True,
                           forwarded_allow_ips="*")
    if a.cmd == "static-demo":
        import json

        from .static_demo import write
        return print(json.dumps(write(Path(a.out))))
    if a.cmd == "demo-cache":
        import json

        from .hosted import write_cache
        return print(json.dumps(write_cache()))
    if a.cmd == "email-paraphrase":
        import json

        from .assistant.train_data import paraphrase_all
        return print(json.dumps(paraphrase_all()))
    if a.cmd == "email-train-data":
        import json

        from .assistant.train_data import write
        return print(json.dumps(write()))
    if a.cmd == "calibrate":
        return run_calibrate(a.method)

    risk = None if a.arm == "A" else NemotronRisk(variant="coached" if a.arm == "B-coached" else "plain")
    agent_tok, human_tok, proxy_tok = (secrets.token_urlsafe(24) for _ in range(3))
    executor = None
    if a.sandbox:
        from .sandbox import DockerExecutor
        executor = DockerExecutor(f"http://host.docker.internal:{a.port}", proxy_tok)
        executor.ensure_infra()
        atexit.register(executor.teardown)
    gw = Gateway(Policy(git_remote=a.git_remote), risk_fn=risk, executor=executor, shadow_fn=_shadow(a),
                 assistant=_assistant(a))
    app = create_app(gw, agent_tok, human_tok, Path(a.root), proxy_token=proxy_tok)
    print(f"arm={a.arm} sandbox={a.sandbox}\nagent token: {agent_tok}\nhuman token: {human_tok}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port)


def _assistant(a):
    if not getattr(a, "assistant", False):
        return None
    import json
    from dataclasses import replace

    from .assistant.core import Assistant
    from .assistant.judge import JudgeCosts
    from .evals import ROOT_EVALS
    costs, th = JudgeCosts(), ROOT_EVALS / "email_thresholds.json"
    if th.exists() and a.judge in json.loads(th.read_text()):
        costs = replace(costs, scale=json.loads(th.read_text())[a.judge]["scale"])
    return Assistant(judge=make_judge(a.judge), costs=costs)


def _shadow(a):
    if not getattr(a, "shadow_laya", False):
        return None
    from .risk import LayaRisk
    return LayaRisk()


def make_judge(name: str):
    from .assistant.clm_judge import LAYA_RLCD_PROBS, ClmJudge
    from .assistant.judge import LayaJudge, NemotronJudge
    from .evals import ROOT_EVALS
    return {"none": lambda: None, "nemotron": NemotronJudge, "laya": LayaJudge,
            "laya-calibrated": lambda: LayaJudge.calibrated(ROOT_EVALS / "email_laya_temperatures.json"),
            "clm": ClmJudge, "clm-zero": lambda: ClmJudge(head="zero", name="clm-zero"),
            # Laya fine-tuned with RLCD on Kaggle (kaggle/laya-email/run.py), replayed the same way as CLM
            "laya-rlcd": lambda: ClmJudge(head="seed0", path=LAYA_RLCD_PROBS, name="laya-rlcd"),
            "laya-rlcd-s1": lambda: ClmJudge(head="seed1", path=LAYA_RLCD_PROBS, name="laya-rlcd-s1")}[name]()


def run_email_calibrate(judges: list[str] | None = None) -> None:
    import json

    from .assistant.evals import fit_calibration, fit_scale
    from .assistant.judge import LayaJudge
    from .evals import ROOT_EVALS
    if judges is None or "laya-calibrated" in judges:
        params = fit_calibration(LayaJudge())
        (ROOT_EVALS / "email_laya_temperatures.json").write_text(json.dumps(params, indent=1) + "\n")
        print("laya map", json.dumps(params))
    path = ROOT_EVALS / "email_thresholds.json"
    scales = json.loads(path.read_text()) if path.exists() else {}  # keep judges not refitted (e.g. Kaggle ones)
    scales.update({name: fit_scale(make_judge(name)) for name in (judges or ("laya", "laya-calibrated", "nemotron"))})
    path.write_text(json.dumps(scales, indent=1) + "\n")
    print("thresholds", json.dumps(scales))


def run_email_eval(a) -> None:
    import json

    from .assistant.evals import run
    from .evals import RESULTS, ROOT_EVALS
    out = {}
    for name in a.judge or ["none"]:
        if name == "ask-every":
            from .assistant.evals import load, score
            rows = [{"id": c["id"], "expected": c["expected"], "why": c["why"], "decision": "ASK", "by": "baseline",
                     "ms": 0, "probs": None} for c in load("eval")]
            r = {"rows": rows, "metrics": score(rows)}
        elif name.endswith("+t"):
            from dataclasses import replace

            from .assistant.judge import JudgeCosts
            scale = json.loads((ROOT_EVALS / "email_thresholds.json").read_text())[name[:-2]]["scale"]
            r = run(make_judge(name[:-2]), costs=replace(JudgeCosts(), scale=scale))
        else:
            r = run(make_judge(name))
        out[name] = r["metrics"]
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"email_{name}.json").write_text(json.dumps(r, indent=1))
        m = r["metrics"]
        print(f"{name:<16} interruptions {m['interruption_rate']:.0%}  boundary misses {m['boundary_miss_rate']:.0%} "
              f"(content {m['content_miss_rate']:.0%})  judge errors {m['judge_errors']}  p50 {m['latency_ms_p50']} ms",
              flush=True)
    prev = RESULTS / "email_summary.json"
    merged = {**(json.loads(prev.read_text()) if prev.exists() else {}), **out}
    prev.write_text(json.dumps(merged, indent=1))


def run_assistant_demo(a) -> None:
    import tempfile
    import threading
    import time

    from .assistant.core import Assistant
    from .assistant.demo import run
    from .models import AgentFingerprint

    agent_tok, human_tok = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    import json
    from dataclasses import replace

    from .assistant.judge import JudgeCosts
    from .evals import ROOT_EVALS
    costs = JudgeCosts()
    th = ROOT_EVALS / "email_thresholds.json"
    if th.exists() and a.judge in json.loads(th.read_text()):
        costs = replace(costs, scale=json.loads(th.read_text())[a.judge]["scale"])  # fitted on the calib split
    gw = Gateway(Policy(), assistant=Assistant(judge=make_judge(a.judge), costs=costs))
    root = Path(tempfile.mkdtemp(prefix="clearance-assistant-"))
    app = create_app(gw, agent_tok, human_tok, root)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=a.port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    s = gw.start_session(root, AgentFingerprint("nvidia/nemotron-3-super-120b-a12b", "mcp-assistant",
                                                "mock-mailbox", "p1"))
    print(f"dashboard: http://127.0.0.1:{a.port}/dashboard\nhuman token: {human_tok}\n"
          f"session {s.id}  judge={a.judge}\n", flush=True)
    run(f"http://127.0.0.1:{a.port}", agent_tok, s.id, pause=a.pause, llm_drafts=a.llm_drafts)
    from .assistant.core import interruptions
    i = interruptions(gw.records[s.id])
    print(f"\nInterruptions: {i['with_clearance']} with Clearance vs {i['without_clearance']} if every email "
          f"needed your OK. Sent: {len(gw.assistant.mailbox.sent)}.\nDashboard stays up (Ctrl-C to quit)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def run_demo(a) -> None:
    import threading
    import time

    from .demo import make_workspace, run_agent
    from .models import AgentFingerprint
    from .sandbox import STATE_DIR

    root = STATE_DIR / "demo"
    ws = make_workspace(root)
    risk = None if a.arm == "A" else NemotronRisk(variant="coached" if a.arm == "B-coached" else "plain")
    agent_tok, human_tok, proxy_tok = (secrets.token_urlsafe(24) for _ in range(3))
    executor = None
    if a.sandbox:
        from .sandbox import DockerExecutor
        executor = DockerExecutor(f"http://host.docker.internal:{a.port}", proxy_tok)
        executor.ensure_infra()
        atexit.register(executor.teardown)
    gw = Gateway(Policy(git_remote=None), risk_fn=risk, executor=executor, shadow_fn=_shadow(a))
    app = create_app(gw, agent_tok, human_tok, root, proxy_token=proxy_tok)
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0" if a.sandbox else "127.0.0.1", port=a.port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    s = gw.start_session(ws, AgentFingerprint("nvidia/nemotron-3-super-120b-a12b", "mcp-v1",
                                              "docker" if a.sandbox else "local", "p1"))
    print(f"dashboard: http://127.0.0.1:{a.port}/dashboard\nhuman token: {human_tok}\n"
          f"session {s.id}  arm={a.arm}  sandbox={a.sandbox}\n")
    run_agent(f"http://127.0.0.1:{a.port}", agent_tok, s.id, pause=a.pause)
    print("\nscripted session finished; dashboard stays up (Ctrl-C to quit)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def run_calibrate(method: str = "both") -> None:
    import json

    from .calibrate import PARAMS, PARAMS_ISO, calibration_set, collect_estimates, fit, fit_iso

    items = calibration_set()
    est = collect_estimates(NemotronRisk(variant="plain"), items,
                            progress=lambda i, n: print(f"\rcalibration set: {i}/{n}", end="", flush=True))
    print()
    if method in ("platt", "both"):
        params = fit(est)
        PARAMS.write_text(json.dumps(params, indent=1) + "\n")
        print(json.dumps(params, indent=1))
    if method in ("isotonic", "both"):
        iso = fit_iso(est)
        PARAMS_ISO.write_text(json.dumps(iso, indent=1) + "\n")
        print(json.dumps(iso, indent=1))


def run_eval(a) -> None:
    import json
    from dataclasses import replace

    from .decision import CostMatrix
    from .evals import RESULTS, make_risk_fn, run_arm, save, suite_hash, summary_table

    print(f"suite eval-v1 sha256={suite_hash()}")
    results, sweep = {}, {}
    for arm in a.arm or ["A"]:
        risk = make_risk_fn(arm)
        name = f"{arm} earned" if a.earned else arm
        r = run_arm(name, risk, earned=a.earned, progress=lambda i, n: print(f"\r{name}: {i}/{n}", end="", flush=True))
        print()
        save(r)
        results[name] = r.metrics
        if a.earned:
            continue
        if a.sweep:
            base = CostMatrix()
            for mult in (0.25, 4.0):
                cm = replace(base, false_allow_destructive=base.false_allow_destructive * mult,
                             false_allow_exfil=base.false_allow_exfil * mult)
                m = run_arm(arm, risk, costs=cm).metrics  # model arms reuse cached estimates
                sweep[f"{arm} C_FA x{mult}"] = m
    prev = RESULTS / "summary.json"
    if prev.exists():  # merge with earlier runs so the table always covers every arm evaluated
        old = json.loads(prev.read_text())
        results = {**old.get("arms", {}), **results}
        sweep = {**old.get("sweep", {}), **sweep}
    order = ["A", "B", "B-coached", "B0", "C'", "C-iso", "L", "A earned", "B earned", "C' earned", "C-iso earned",
             "L earned"]
    results = dict(sorted(results.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))
    table = summary_table(results)
    print(table)
    if sweep:
        print("\nCost-ratio sweep:\n" + summary_table(sweep))
    (RESULTS / "summary.md").write_text(table + ("\n\nCost-ratio sweep:\n" + summary_table(sweep) if sweep else "") + "\n")
    (RESULTS / "summary.json").write_text(json.dumps({"arms": results, "sweep": sweep}, indent=1))


if __name__ == "__main__":
    main()
