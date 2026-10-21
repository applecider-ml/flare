"""Nightly refit of the review combiner (flare.review) from operator confirmations on Fritz.

1. Harvest: the sources saved to the service's groups that received a human classification with a
   spectrum since the last run; the Sitewide-Taxonomy label maps to a trained class (label 0) or to an
   out-of-taxonomy family (label 1, with its family). Conflicting human labels are skipped.
2. Score: for each new confirmation, its ZTF photometry on Fritz (duplicate epochs collapsed to the
   smallest error, as SkyPortal's analysis payload does) goes through flare.skyportal's own steps, and the
   fourteen LUNA scores of the full-curve consensus are stored with the label (--store, a CSV).
3. Fit: a class-balanced logistic regression over the standardised scores of every stored confirmation;
   the reference for the review p-value is the out-of-fold combiner value of the stored known-class
   objects (five folds), so that the p-value of a new object compares it with known objects that did not
   fit the weights. The JSON is written to --out (point FLARE_REVIEW_COMBINER at it, or ship it in the
   model directory) only when the confirmations span at least flare.review.MIN_FAMILIES families.

Read-only on Fritz (FRITZ_TOKEN). Run, e.g. nightly:
  python scripts/review_nightly.py --groups 1840 1955 1876 1937 --store review_store.csv --out review_combiner.json
A first store can be seeded from a scored table with --seed <csv: obj_id,label,family,<scorer columns>>.
"""
from __future__ import annotations

import argparse, datetime as dt, json, os, sys, time, urllib.parse, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

TRAINED = {"SN_Ia": ["Ia", "Ia-pec", "Ia-norm", "Ia-91T", "Ia-91bg", "Ia-99aa", "Ia-00cx", "Ia-02cx", "Ia-CSM", "Ia-03fg", "Ia-18byg"],
           "SN_CC": ["Ib", "Ib-pec", "Ib-norm", "Ibn", "Ic", "Ic-BL", "Ic-pec", "Ic-norm", "Icn", "Ib/c", "Type II", "IIn",
                     "II-norm", "IIP", "IIL", "IIb", "IIb-pec", "II-pec"],
           "SLSN": ["Ic-SLSN", "Ic.5-SLSN", "II-SLSN"], "TDE": ["Tidal Disruption Event"],
           "AGN": ["AGN", "Seyfert", "QSO", "Blazar", "OVV", "BL Lac"],
           "CV": ["Cataclysmic", "Nova-like", "SW Sex", "VY Scl", "UX UMa", "AM CVn", "Polars", "AM Her", "DQ Her", "U Gem",
                  "SS Cyg", "Z Cam", "SU UMaj", "ER U Maj", "WZ Sge"]}
FAMILIES = {"Novae": "nova", "Classical Nova": "nova", "Recurrent Nova": "nova", "Luminous Red Nova": "LRN", "ILRT": "ILRT",
            "Ca-rich": "Ca-rich", "FBOT": "FBOT", "S Doradus": "LBV", "GRB": "GRB", "short GRB": "GRB", "long GRB": "GRB",
            "X-ray Flash": "GRB", "afterglow": "GRB", "kilonova": "kilonova"}
LABEL = {c: (0, k) for k, cs in TRAINED.items() for c in cs} | {c: (1, f) for c, f in FAMILIES.items()}
FILTERS = {"ztfg": 1, "ztfr": 2, "ztfi": 3}


def get(path, **q):
    url = "https://fritz.science" + path + ("?" + urllib.parse.urlencode(q, doseq=True) if q else "")
    req = urllib.request.Request(url, headers={"Authorization": f"token {os.environ['FRITZ_TOKEN']}"})
    for k in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=300))["data"]
        except Exception as e:  # noqa: BLE001
            err = e; time.sleep(5 * (k + 1))
    raise err


def harvest(groups, since):
    """(obj_id, label, family, ra, dec, classified_at) for human classifications with a spectrum since `since`."""
    rows, page = [], 1
    while True:
        d = get("/api/sources", group_ids=",".join(map(str, groups)), hasSpectrum="true", includeClassifications="true",
                numPerPage=500, pageNumber=page)
        for s in d.get("sources", []):
            hum = [c for c in s.get("classifications", []) if not c.get("ml", False) and c.get("classification") in LABEL]
            new = [c for c in hum if (c.get("created_at") or "") >= since]
            if not new:
                continue
            labs = {LABEL[c["classification"]] for c in hum}
            if len({lab for lab, _ in labs}) != 1 or len(labs) != 1:
                continue                                 # conflicting human labels: skipped
            lab, fam = labs.pop()
            rows.append(dict(obj_id=s["id"], label=lab, family=fam, ra=s.get("ra"), dec=s.get("dec"),
                             classified_at=max(c["created_at"] for c in new)))
        if len(d.get("sources", [])) < 500:
            break
        page += 1
    return pd.DataFrame(rows)


def ztf_rows(oid):
    best = {}
    for p in get(f"/api/sources/{oid}/photometry", format="mag"):
        f = FILTERS.get(p.get("filter")); ins = str(p.get("instrument_name") or "ZTF").upper()
        if f is None or ins != "ZTF" or p.get("mag") is None or p.get("magerr") is None:
            continue
        key = (round(p["mjd"], 5), f)
        if key not in best or p["magerr"] < best[key][3]:
            best[key] = (p["mjd"], f, p["mag"], p["magerr"])
    return sorted(best.values())


def scores_for(oid, ra, dec):
    """The fourteen LUNA scores of the full-curve consensus, through flare.skyportal's own steps."""
    from flare.context import context_features
    from flare.fetch import detections, to_events
    from flare.skyportal import DEFAULTS, feature_frame, _classifier
    rows = sorted(detections(ztf_rows(oid), DEFAULTS["min_snr"]))
    ev = to_events(rows, horizon_days=DEFAULTS["horizon_days"])
    if ev is None or len(ev) < 3:
        return None
    X, _, _ = feature_frame(ev, context_features(ra, dec, peak_mag=None), None)
    clf = _classifier(None)
    if getattr(clf, "consensus", None) is None:
        return None
    return clf.consensus.score(X.reindex(columns=clf.feature_names), clf.ad_space)[0]["scores"]


def fit(store: pd.DataFrame, names: list[str]) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    M = store[names].to_numpy(float); y = store.label.to_numpy(int)
    mean, scale = M.mean(0), M.std(0) + 1e-9
    Z = np.nan_to_num((M - mean) / scale)
    lr = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Z, y)
    oof = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, y):
        oof[te] = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Z[tr], y[tr]).decision_function(Z[te])
    fam = store[store.label == 1].family.value_counts().to_dict()
    return dict(scorers=names, mean=mean.tolist(), scale=scale.tolist(), coef=lr.coef_[0].tolist(),
                intercept=float(lr.intercept_[0]), reference_known=sorted(oof[y == 0].tolist()),
                families={k: int(v) for k, v in fam.items()}, n_known=int((y == 0).sum()), n_anomalous=int((y == 1).sum()),
                fitted=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
                note="class-balanced logistic regression over the standardised LUNA scores of the full-curve consensus, "
                     "fitted on operator confirmations; reference = out-of-fold values of the known-class confirmations")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", type=int, nargs="+", required=True)
    ap.add_argument("--store", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--since", default=None, help="ISO date; default: the latest classified_at in the store")
    ap.add_argument("--seed", type=Path, default=None)
    a = ap.parse_args()
    from flare import review as RV
    store = pd.read_csv(a.store) if a.store.exists() else (pd.read_csv(a.seed) if a.seed else pd.DataFrame())
    since = a.since or (store.classified_at.dropna().max() if "classified_at" in store and store.classified_at.notna().any() else "2026-01-01")
    new = harvest(a.groups, since)
    new = new[~new.obj_id.isin(store.obj_id)] if len(store) else new
    rows = []
    for r in new.itertuples():
        try:
            s = scores_for(r.obj_id, r.ra, r.dec)
        except Exception as e:  # noqa: BLE001 - one source failing must not stop the night
            print(f"  {r.obj_id}: skipped ({type(e).__name__})"); continue
        if s:
            rows.append(dict(obj_id=r.obj_id, label=r.label, family=r.family, classified_at=r.classified_at, **s))
    if rows:
        store = pd.concat([store, pd.DataFrame(rows)], ignore_index=True)
    store.to_csv(a.store, index=False)
    names = [c for c in store.columns if c not in ("obj_id", "label", "family", "classified_at", "source")]
    fam = store[store.label == 1].family.nunique() if len(store) else 0
    print(f"since {since}: {len(new)} new confirmations, {len(rows)} scored; store {len(store)} "
          f"({int((store.label == 0).sum()) if len(store) else 0} known, {int((store.label == 1).sum()) if len(store) else 0} anomalous in {fam} families)")
    if fam < RV.MIN_FAMILIES:
        print(f"fewer than {RV.MIN_FAMILIES} families confirmed: combiner not written"); return
    table = fit(store, names)
    a.out.write_text(json.dumps(table))
    print(f"wrote {a.out}: families {table['families']}")


if __name__ == "__main__":
    main()
