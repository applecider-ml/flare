"""flare.skyportal: the verdict ladder and the annotation payload (no network, no model)."""
from flare import skyportal as S


def _cls(**over):
    base = {"predicted": "SN_Ia", "probabilities": {"SN_Ia": 0.9, "SN_CC": 0.1}, "prediction_set": ["SN_Ia"], "alpha": 0.1,
            "p_values": {}, "credibility": 0.5, "argmax_excluded_from_set": False,
            "anomaly": {"energy": -1.0, "energy_percentile": 40.0, "novelty_p": 0.6, "likelihood_ratio": 0.5, "p_anomaly": 0.01,
                        "base_rate": 0.0326, "threshold_1pct_far": 1.47, "above_1pct_far": False}}
    base.update(over)
    return base


def test_rule_verdict_ladder():
    assert S.rule_verdict(_cls(), {}, 20)["verdict"] == "ordinary"
    assert S.rule_verdict(_cls(), {}, 5)["verdict"] == "insufficient_data"
    assert S.rule_verdict(_cls(), {"pm_over_error": 8.0}, 20)["verdict"] == "likely_stellar_or_agn"
    assert S.rule_verdict(_cls(argmax_excluded_from_set=True), {}, 20)["verdict"] == "anomaly_review"
    assert S.rule_verdict(_cls(predicted="TDE", prediction_set=["TDE"]), {"host_sep": 0.1}, 20)["verdict"] == "needs_spectrum"


def test_annotations_are_flat_and_finite():
    ann = S.annotations_for(_cls(), {"verdict": "ordinary", "priority": 0})
    assert ann["flare_class"] == "SN_Ia" and ann["flare_set"] == "SN_Ia" and ann["flare_priority"] == 0
    assert all(not isinstance(v, (dict, list)) for v in ann.values())


def test_resolve_coords_from_params():
    assert S.resolve_coords({"ra": "10.5", "dec": "-3.25"}, "obj") == (10.5, -3.25)
