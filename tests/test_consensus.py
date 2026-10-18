"""flare.consensus: the shipped blind-consensus alarm (needs luna-ood >= 0.3)."""
import json
import shutil
from pathlib import Path

import pytest

luna = pytest.importorskip("luna")
if tuple(int(x) for x in luna.__version__.split(".")[:2]) < (0, 3):
    pytest.skip("luna-ood >= 0.3 required", allow_module_level=True)

from flare import consensus as C  # noqa: E402
from flare import skyportal as S  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models" / "bts6"
DIRS = [MODELS] + [MODELS / "early" / f"h{h}" for h in (5, 10, 20)]


def test_every_model_ships_a_consensus():
    for d in DIRS:
        c = C.load(d)
        assert c is not None and c.fars == (0.01, 0.05, 0.10), d
        assert len(c.pipeline.active_methods) == 14


def test_manifest_mismatch_is_refused(tmp_path):
    for f in ("consensus.pkl.gz", "manifest.json"):
        shutil.copy(MODELS / "early" / "h5" / f, tmp_path / f)
    man = json.loads((tmp_path / "manifest.json").read_text())
    man["consensus.pkl.gz"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError):
        C.load(tmp_path)


def test_skyportal_reports_consensus_alarms():
    rows = [(61312.0 + 0.5 * k, 1 + k % 2, 19.0 - 0.05 * k, 0.05) for k in range(9)]
    r = S.analyze(rows, None, dict(context="none", plot=False, forecast=False), resource_id="cons", work_dir=".")
    a = r["results"]["classification"]["anomaly"]
    assert a["alarm_score"] == "consensus" and a["calibration"] == "h5"
    assert set(a["alarms"]) == {"1%", "5%", "10%"}
    assert 0 < a["alarm_p"] <= 1
    # nested rates: flagged at 1% implies flagged at 5% and 10%
    assert a["alarms"]["1%"] <= a["alarms"]["5%"] <= a["alarms"]["10%"]
    assert a["alarm_far"] == min((float(k[:-1]) / 100 for k, v in a["alarms"].items() if v), default=None)


def test_consensus_alarm_drives_the_verdict():
    base = {"predicted": "SN_Ia", "probabilities": {"SN_Ia": 0.9, "SN_CC": 0.1}, "prediction_set": ["SN_Ia"],
            "alpha": 0.1, "p_values": {}, "credibility": 0.5, "argmax_excluded_from_set": False,
            "anomaly": {"energy": -1.0, "energy_percentile": 40.0, "novelty_p": 0.6, "likelihood_ratio": 0.5,
                        "p_anomaly": 0.01, "base_rate": 0.0326, "threshold_1pct_far": 1.47,
                        "above_1pct_far": False, "calibrated": True,
                        "alarms": {"1%": True, "5%": True, "10%": True}}}
    assert S.rule_verdict(base, {}, 20)["verdict"] == "anomaly_review"
    base["anomaly"]["alarms"] = {"1%": False, "5%": True, "10%": True}
    assert S.rule_verdict(base, {}, 20)["verdict"] == "ordinary"
