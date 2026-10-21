"""flare.review: the supervised review priority and its gate on the number of confirmed families."""
import json

import numpy as np

from flare import review as RV


def _table(names, families):
    rng = np.random.default_rng(0)
    return dict(scorers=names, mean=[0.0] * len(names), scale=[1.0] * len(names), coef=[1.0] + [0.0] * (len(names) - 1),
                intercept=0.0, reference_known=sorted(rng.normal(0, 1, 400).tolist()), families=families, fitted="test")


def test_score_and_p_value():
    rc = RV.ReviewCombiner(_table(["energy", "msp"], {"nova": 5, "GRB": 1, "FBOT": 1}))
    hi, lo = rc.score({"energy": 3.0, "msp": 0.0}), rc.score({"energy": -3.0, "msp": 0.0})
    assert hi["review_score"] == 3.0 and hi["review_p"] < 0.01 < lo["review_p"]
    assert hi["n_confirmations"] == 7 and rc.score({"energy": 1.0}) is None       # a missing scorer gives no score


def test_one_family_is_not_enough(tmp_path, monkeypatch):
    path = tmp_path / "rc.json"
    path.write_text(json.dumps(_table(["energy"], {"nova": 30})))
    monkeypatch.setenv("FLARE_REVIEW_COMBINER", str(path))
    assert RV.load(tmp_path) is None                                              # learns the family, not unfamiliarity
    path.write_text(json.dumps(_table(["energy"], {"nova": 30, "GRB": 2, "FBOT": 1})))
    assert RV.load(tmp_path) is not None


def test_service_reports_review_priority(tmp_path, monkeypatch):
    """With a usable combiner in FLARE_REVIEW_COMBINER, the full-curve output carries the review block and the
    flat annotation; the alarm stays the blind consensus."""
    from flare.skyportal import analyze, _classifier
    names = list(_classifier(None).consensus.pipeline.active_methods)
    path = tmp_path / "rc.json"; path.write_text(json.dumps(_table(names, {"nova": 3, "GRB": 1, "FBOT": 1})))
    monkeypatch.setenv("FLARE_REVIEW_COMBINER", str(path))
    rows = [(61300.0 + 2.0 * k, 1 + k % 2, 18.0 + 0.04 * abs(k - 12), 0.05) for k in range(30)]
    r = analyze(rows, None, dict(context="none", plot=False, forecast=False, lifetime=False), resource_id="rv", work_dir=str(tmp_path))
    a = r["results"]["classification"]["anomaly"]
    assert r["results"]["classification"]["model"] == "bts6"
    assert a["review"] is not None and 0 < a["review"]["review_p"] <= 1 and a["alarm_score"] == "consensus"
    assert r["annotations_flat"]["flare_review_p"] == a["review"]["review_p"]
