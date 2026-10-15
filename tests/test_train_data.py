import json

import pytest

from clearance.assistant.evals import EVAL_PERMISSION as PERMISSION
from clearance.assistant.evals import load
from clearance.assistant.judge import QUESTIONS
from clearance.assistant.train_data import OUT, _words, build, max_overlap


def test_training_split_is_well_formed_and_deterministic():
    cases = build()
    assert cases == build() and len(cases) >= 300
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert set(c["labels"]) == set(QUESTIONS) and set(c["labels"].values()) <= {0, 1}
        assert c["expected"] == ("ask" if any(c["labels"].values()) else "proceed")
        assert c["email"]["to"] == [c["replying_to"]["from"]] and not c["email"]["cc"]  # rules cases stay out
    for q in QUESTIONS:
        assert sum(c["labels"][q] for c in cases) >= 50


def test_training_split_is_disjoint_from_email_v1():
    cases = build()
    assert all(c["permission"]["purpose"] != PERMISSION["purpose"] for c in cases)  # held-out permission
    ev = load()
    ev_bodies = {c["email"]["body"] for c in ev}
    assert not any(c["email"]["body"] in ev_bodies for c in cases)
    assert max_overlap(cases, ev) < 0.4
    # a near-copy is one whose incoming message AND reply both resemble an email-v1 case (the hiring/vacation
    # templates removed in review scored 0.3+ on both)
    J = lambda a, b: len(_words(a) & _words(b)) / len(_words(a) | _words(b))
    assert max(min(J(c["replying_to"]["body"], e["replying_to"]["body"]), J(c["email"]["body"], e["email"]["body"]))
               for c in cases for e in ev) < 0.3


def test_training_bodies_carry_no_style_cue():
    for c in build():
        assert c["email"]["body"].isascii(), c["id"]  # paraphrase punctuation would otherwise mark one class
        assert len(c["email"]["body"].split()) >= 6, c["id"]  # no degenerate paraphrases


def test_committed_file_matches_generator():
    assert [json.loads(l) for l in OUT.read_text().splitlines()] == build()


def test_clm_cases_match_the_states_email_eval_sends(tmp_path):
    from clearance.assistant import clm_judge
    from clearance.assistant.evals import run

    cs = clm_judge.cases()
    assert sum(c["split"] == "train" for c in cs) == len(build())
    assert not any(c["id"].startswith("r") for c in cs)  # rules decide recipient cases
    probs = tmp_path / "p.json"
    probs.write_text(json.dumps({"finetuned": {c["key"]: {q: 0.0 for q in QUESTIONS} for c in cs}}))
    rows = run(clm_judge.ClmJudge(path=probs))["rows"]
    assert not any(r["by"] == "judge.unavailable" for r in rows)  # every eval state found by key
    assert all(r["decision"] == "ASK" for r in rows if r["id"].startswith("r"))


@pytest.mark.parametrize("fname,heads", [("email_clm_probs.json", ("zero", "finetuned")),
                                         ("email_laya_rlcd_probs.json", ("zero", "seed0", "seed1"))])
def test_committed_probs_cover_current_eval_states(fname, heads):
    from clearance.assistant import clm_judge

    d = json.loads((clm_judge.ROOT / "evals" / fname).read_text())
    keys = {c["key"] for c in clm_judge.cases() if c["split"] != "train"}
    if "train_sha" in d["meta"]:  # results from before the stamp was added cannot be checked this way
        assert d["meta"]["train_sha"] == clm_judge.train_sha([c for c in clm_judge.cases() if c["split"] == "train"]), \
            f"{fname} was trained on a different training split; rerun the Kaggle script"
    for h in heads:
        assert keys <= set(d[h]), f"{fname}:{h} is stale; rerun the Kaggle script"
        assert all(set(QUESTIONS) <= set(d[h][k]) for k in keys)


def test_unknown_state_becomes_ask(tmp_path):
    from clearance.assistant import clm_judge
    from clearance.assistant.evals import run

    p = tmp_path / "p.json"
    p.write_text(json.dumps({"finetuned": {}}))
    judged = [r for r in run(clm_judge.ClmJudge(path=p))["rows"] if not r["id"].startswith("r")]
    assert judged and all(r["decision"] == "ASK" and r["by"] == "judge.unavailable" for r in judged)


@pytest.mark.parametrize("judges,expected", [(["clm"], {"offline": 0.1, "clm": 0.5}),
                                             (None, {"offline": 0.1, "laya": 0.5, "laya-calibrated": 0.5,
                                                     "nemotron": 0.5})])
def test_calibrate_keeps_judges_it_does_not_refit(tmp_path, monkeypatch, judges, expected):
    import clearance.__main__ as m
    import clearance.assistant.evals as ae
    import clearance.evals as ce

    monkeypatch.setattr(ce, "ROOT_EVALS", tmp_path)
    (tmp_path / "email_thresholds.json").write_text(json.dumps({"offline": {"scale": 0.1}}))
    monkeypatch.setattr(ae, "fit_scale", lambda j: {"scale": 0.5})
    monkeypatch.setattr(ae, "fit_calibration", lambda j: {"a": 1.0, "b": 0.0})
    monkeypatch.setattr(m, "make_judge", lambda name: object())
    m.run_email_calibrate(judges)
    out = json.loads((tmp_path / "email_thresholds.json").read_text())
    assert {k: v["scale"] for k, v in out.items()} == expected
    assert (tmp_path / "email_laya_temperatures.json").exists() == (judges is None)


def test_paraphrase_filter_drops_changed_numbers_and_copies(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import clearance.assistant.train_data as td
    import clearance.risk as risk

    monkeypatch.setattr(td, "PARAPHRASES", tmp_path / "p.json")
    monkeypatch.setattr(td, "templates", lambda: [("ask", "premium is $14 per user.")])
    monkeypatch.setattr(risk, "load_api_key", lambda: "test")
    alts = ["premium costs $14 per seat.", "premium is $12 per user.", "premium is $14 per user."]
    resp = SimpleNamespace(raise_for_status=lambda: None,
                           json=lambda: {"choices": [{"message": {"content": json.dumps({"paraphrases": alts})}}]})
    td.paraphrase_all(client=SimpleNamespace(post=lambda *a, **k: resp))
    assert json.loads((tmp_path / "p.json").read_text()) == {"premium is $14 per user.": ["premium costs $14 per seat."]}
