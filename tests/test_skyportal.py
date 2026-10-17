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
    assert S.rule_verdict(_cls(), {}, 2)["verdict"] == "insufficient_data"
    assert S.rule_verdict(_cls(prediction_set=["SN_Ia", "SN_CC"]), {}, 20)["verdict"] == "ambiguous"
    assert S.rule_verdict(_cls(), {"pm_over_error": 8.0}, 20)["verdict"] == "likely_stellar_or_agn"
    assert S.rule_verdict(_cls(argmax_excluded_from_set=True), {}, 20)["verdict"] == "anomaly_review"
    assert S.rule_verdict(_cls(predicted="TDE", prediction_set=["TDE"]), {"host_sep": 0.1}, 20)["verdict"] == "needs_spectrum"


def test_annotations_are_flat_and_finite():
    ann = S.annotations_for(_cls(), {"verdict": "ordinary", "priority": 0})
    assert ann["flare_class"] == "SN_Ia" and ann["flare_set"] == "SN_Ia" and ann["flare_priority"] == 0
    assert all(not isinstance(v, (dict, list)) for v in ann.values())


def test_skyportal_shapes():
    ann = S.skyportal_annotations(_cls(), {"verdict": "ordinary", "priority": 0})
    assert isinstance(ann, list) and ann[0]["origin"] == "FLARE" and ann[0]["data"]["flare_p_SN_Ia"] == 0.9
    cl = S.skyportal_classifications(_cls())
    assert cl == [{"taxonomy": "Sitewide Taxonomy", "classification": "Ia", "probability": 0.9, "ml": True, "origin": "FLARE"}]
    assert all(S.FLARE_TO_TAXONOMY[c] for c in ("SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"))


def test_resolve_coords_from_params():
    assert S.resolve_coords({"ra": "10.5", "dec": "-3.25"}, "obj") == (10.5, -3.25)


def test_forecast_overlay_shape():
    """The analysis returns a SkyPortal model_lightcurve overlay: {filter: [[mjd, mag, lo68, hi68], ...]}."""
    import math
    from flare.skyportal import analyze
    t0 = 60500.0; rows = []
    for k, dt in enumerate([0.0, 0.9, 2.1, 3.0, 4.2, 5.1, 6.0, 7.2, 8.1, 9.0, 10.2, 11.0]):
        for fid, off in ((1, 0.0), (2, 0.15)):
            mag = 19.5 - 1.2 * (1 - math.exp(-dt / 4.0)) + off
            rows.append((t0 + dt + 0.01 * fid, fid, mag, 0.05))
    r = analyze(sorted(rows), None, dict(context="none", plot=False), resource_id="synthetic", work_dir=".")
    fc = r["results"]["forecast"]
    assert fc["status"] == "ok" and fc["forecast_horizon_days"] == 15
    ml = r["model_lightcurve"]
    assert set(ml) == {"ztfg", "ztfr"}
    for curve in ml.values():
        assert all(len(p) == 4 for p in curve)
        mjd = [p[0] for p in curve]; assert mjd == sorted(mjd) and mjd[0] > t0
        assert all(p[2] <= p[1] <= p[3] for p in curve)


def test_forecast_skipped_on_sparse_light_curve():
    from flare.skyportal import analyze
    rows = [(61312.15, 1, 15.86, 0.03), (61312.21, 2, 15.89, 0.03)]
    r = analyze(rows, None, dict(context="none", plot=False), resource_id="sparse", work_dir=".")
    assert r["results"]["forecast"]["status"] == "skipped" and "model_lightcurve" not in r


def test_noise_points_dropped_and_no_class_on_insufficient_data():
    """ZTF26abxwsuv-like: two real detections plus forced-photometry noise (SNR < 3).
    The noise is dropped by the detection floor, triage is insufficient_data, and no headline
    class is returned (predicted = None, no SkyPortal classification), probabilities kept."""
    from flare.skyportal import analyze
    real = [(61312.15, 1, 15.86, 0.03), (61312.21, 2, 15.89, 0.03)]
    noise = [(61312.3 + 0.7 * k, 1 + k % 2, 19.5 - 0.1 * k, 0.45 + 0.02 * k) for k in range(16)]
    r = analyze(real + noise, None, dict(context="none", plot=False), resource_id="noisy", work_dir=".")
    res = r["results"]
    assert res["n_rows_input"] == 18 and res["n_detections"] == 2 and res["model"] == "bts6_h5"
    assert res["triage"]["verdict"] == "insufficient_data"
    assert res["classification"]["predicted"] is None and res["classification"]["predicted_if_forced"]
    assert abs(sum(res["classification"]["probabilities"].values()) - 1) < 1e-3   # probabilities are rounded to 4 decimals
    assert r["classifications"] == [] and "insufficient data" in r["message"]


def test_host_mismatch_withholds_the_cv_label():
    """ZTF26abydsbe-like: a CV-like fast light curve 3 arcsec from an extended source with nothing underneath."""
    cv = _cls(predicted="CV", probabilities={"CV": 0.8, "SN_CC": 0.15, "SN_Ia": 0.05}, prediction_set=["CV"], credibility=0.26)
    galaxy = {"host_sep": 3.0, "host_ext": 0.9, "near_sep": 3.0, "near_ext": 0.9}
    assert S.host_mismatch(cv, galaxy)
    assert S.rule_verdict(cv, galaxy, 10, 4, 4) == {"verdict": "host_mismatch", "priority": 2, "source": "rules"}
    # a point source underneath (a quiescent counterpart) keeps the CV label
    assert not S.host_mismatch(cv, {**galaxy, "near_sep": 0.3, "near_ext": 0.0})
    assert S.rule_verdict(cv, {**galaxy, "near_sep": 0.3, "near_ext": 0.0}, 10, 4, 4)["verdict"] == "ordinary"
    # no host information, or a non-CV label: nothing changes
    assert not S.host_mismatch(cv, {}) and not S.host_mismatch(_cls(), galaxy)
    assert S.rule_verdict(cv, {}, 10, 4, 4)["verdict"] == "ordinary"


def test_phase_routing_and_uncalibrated_alarm():
    assert [S.phase_horizon(d) for d in (0.0, 3.0, 7.4, 7.5, 14.9, 15.0, 34.9, 35.0, 200.0)] == [5, 5, 5, 10, 10, 20, 20, None, None]
    # the full-curve energy alarm is ignored under an early model and below 8 detections
    hot = _cls(anomaly={**_cls()["anomaly"], "above_1pct_far": True, "calibrated": False})
    assert S.rule_verdict(hot, {}, 20)["verdict"] == "ordinary"
    hot = _cls(anomaly={**_cls()["anomaly"], "above_1pct_far": True, "calibrated": True})
    assert S.rule_verdict(hot, {}, 20)["verdict"] == "anomaly_review"
    assert S.rule_verdict(hot, {}, 5)["verdict"] == "ordinary"


def test_early_model_is_used_and_set_withholds_the_label():
    """A 4-day light curve goes to the day-5 model; a non-singleton set posts no classification."""
    from flare.skyportal import analyze
    rows = [(61312.0 + 0.5 * k, 1 + k % 2, 19.0 - 0.05 * k, 0.05) for k in range(9)]
    r = analyze(rows, None, dict(context="none", plot=False, forecast=False), resource_id="early", work_dir=".")
    res = r["results"]
    assert res["model"] == "bts6_h5" and res["classification"]["model"] == "bts6_h5"
    assert res["classification"]["anomaly"]["calibrated"] is True and res["classification"]["anomaly"]["calibration"] == "h5"
    if len(res["classification"]["prediction_set"]) == 1:
        assert res["triage"]["verdict"] in ("ordinary", "needs_spectrum", "anomaly_review") or res["classification"]["predicted"] is None
    else:
        # a non-singleton set posts nothing, or the coarse "Supernova" when every member is a supernova type
        pred = res["classification"]["predicted"]
        assert pred in (None, "SN")
        assert r["classifications"] == [] if pred is None else r["classifications"][0]["classification"] == "Supernova"


def test_coarse_supernova_label():
    assert S.coarse_label(_cls(prediction_set=["SN_Ia", "SN_CC", "SLSN"])) == "SN"
    assert S.coarse_label(_cls(prediction_set=["SN_Ia", "TDE"])) is None
    assert S.coarse_label(_cls(prediction_set=["SN_Ia"])) is None
    c = _cls(predicted="SN", prediction_set=["SN_Ia", "SN_CC"], probabilities_coarse={"SN": 0.97})
    post = S.skyportal_classifications(c)
    assert post[0]["classification"] == "Supernova" and post[0]["probability"] == 0.97
    # the nuclear-TDE spectrum rule needs a set of at most two
    wide = _cls(prediction_set=["SN_Ia", "SN_CC", "SLSN", "TDE"])
    assert S.rule_verdict(wide, {"host_sep": 0.1}, 20)["verdict"] == "ambiguous"
    assert S.rule_verdict(_cls(prediction_set=["SN_Ia", "TDE"]), {"host_sep": 0.1}, 20)["verdict"] == "needs_spectrum"


def test_four_conformal_numbers_in_output():
    from flare.skyportal import analyze
    rows = [(61312.0 + 0.5 * k, 1 + k % 2, 19.0 - 0.05 * k, 0.05) for k in range(9)]
    r = analyze(rows, None, dict(context="none", plot=False, forecast=False), resource_id="four", work_dir=".")
    c = r["results"]["classification"]
    assert c["nominal_coverage"] == 0.9 and 0 < c["confidence"] <= 1 and 0 < c["p_value_floor"] < 0.02
    assert c["coverage_of_predicted"] is not None and 0.85 <= c["coverage_of_predicted"] <= 0.95
    flat = r["annotations_flat"]
    assert "flare_confidence" in flat and "flare_coverage" in flat


def test_no_specific_label_unless_the_set_is_that_singleton():
    """anomaly_review with the argmax excluded by its own set must not post the argmax (2026-10-06 review)."""
    from flare.skyportal import skyportal_classifications
    c = _cls(predicted="SN_Ia", prediction_set=["SN_CC", "SLSN"], argmax_excluded_from_set=True, credibility=0.05)
    assert S.rule_verdict(c, {}, 20)["verdict"] == "anomaly_review"
    # the adapter withholds the label: emulate the analyze() rule on the dict
    singleton_ok = len(c["prediction_set"]) == 1 and c["predicted"] in c["prediction_set"]
    assert not singleton_ok
    withheld = {**c, "predicted_if_forced": c["predicted"], "predicted": S.coarse_label(c)}
    assert withheld["predicted"] == "SN"          # all-supernova set -> coarse label, not "Ia"
    withheld["probabilities_coarse"] = {"SN": 1.0}
    assert skyportal_classifications(withheld)[0]["classification"] == "Supernova"
    c2 = _cls(predicted="SN_Ia", prediction_set=["SN_Ia", "TDE"])
    assert S.coarse_label(c2) is None and skyportal_classifications({**c2, "predicted": None}) == []
