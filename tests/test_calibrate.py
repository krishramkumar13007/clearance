import json

from clearance.calibrate import CalibratedRisk, _sigmoid, calibration_set, fit, fit_platt
from clearance.evals import FIXTURE_SECRETS, load_suite


def test_calibration_set_is_disjoint_from_eval():
    calib = json.dumps(calibration_set())
    for s in FIXTURE_SECRETS.values():
        assert s not in calib
    eval_cmds = {a["args"].get("command") for t in load_suite() for a in t["actions"] if a["surface"] == "shell"}
    calib_cmds = {a["args"].get("command") for a, _, _ in calibration_set() if a["surface"] == "shell"}
    assert not (eval_cmds & calib_cmds)


def test_platt_pulls_overconfident_benign_down():
    # model says 0.7 for 20 benign actions and 0.9 for 5 harmful ones
    ps = [0.7] * 20 + [0.9] * 5
    ys = [0] * 20 + [1] * 5
    a, b = fit_platt(ps, ys)
    assert _sigmoid(a * 0.8473 + b) < 0.3   # logit(0.7)
    assert _sigmoid(a * 2.1972 + b) > 0.5   # logit(0.9)


def test_calibrated_risk_applies_params():
    est = [{"p_d": 0.7, "p_e": 0.0, "label": "benign"}] * 10 + [{"p_d": 0.95, "p_e": 0.0, "label": "destructive"}] * 5
    params = fit(est)
    cr = CalibratedRisk(params)
    pd, _ = cr.apply(0.7, 0.0)
    assert pd < 0.5


def test_isotonic_maps_benign_low_end_down_and_keeps_order():
    from clearance.calibrate import apply_isotonic, fit_isotonic
    ps = [0.0] * 10 + [0.01] * 30 + [0.05] * 10 + [0.9] * 8 + [0.95] * 4
    ys = [0] * 10 + [0] * 30 + [0] * 10 + [1] * 8 + [1] * 4
    curve = fit_isotonic(ps, ys)
    low, high = apply_isotonic(curve, 0.01), apply_isotonic(curve, 0.9)
    assert low < 0.05 and high > 0.5
    assert 0 < low  # never exactly zero
    vals = [apply_isotonic(curve, p) for p in (0.0, 0.01, 0.05, 0.5, 0.9, 1.0)]
    assert vals == sorted(vals)
