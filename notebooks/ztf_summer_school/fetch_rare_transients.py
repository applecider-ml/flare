"""Fetch real out-of-catalog transients from the BOOM broker (or ALeRCE).

These classes do NOT exist in the training taxonomy — they are genuinely new to
every model in this repository. The categories mirror the out-of-taxonomy
validation of ASTRANet (Sasli et al. 2026, arXiv:2607.08044), photometric
edition. The script downloads ZTF difference photometry, converts it to the
training event format, and saves one .npz per object into rare_transients/.

Objects (API-verified, pass the survey quality filter: >=8 detections,
>=2 per g/r band, within 100 days of first detection):

    FBOT      : AT2018cow, AT2021csp
    Ca-rich   : SN2019ehk, SN2021gno, SN2022oqm
    gap/ILRT  : AT2019abn
    SLSN-I    : SN2020ank
    SN Iax    : SN2020kyg

GRB afterglows (e.g. AT2021any) and the jetted TDE AT2022cmc fade too fast to
accumulate 8 public ZTF detections — an honest cadence limitation worth
discussing in class.

Backends
--------
BOOM (default when configured): the team's alert broker. Set either
    BOOM_URL + BOOM_TOKEN                  (pre-issued token), or
    BOOM_URL + BOOM_USERNAME + BOOM_PASSWORD   (token via POST /auth)
Photometry is read Kowalski-style with POST /queries/find on the
ZTF_alerts / ZTF_alerts_aux catalogs.

ALeRCE (public fallback): no credentials needed; used automatically when
BOOM_URL is not set, so the notebook reproduces anywhere.

Run:  python fetch_rare_transients.py [--backend auto|boom|alerce]
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

OBJECTS = {  # display name -> (ZTF oid, class label)
    "AT2018cow": ("ZTF18abcfcoo", "FBOT"),
    "AT2021csp": ("ZTF21aakilyd", "FBOT"),
    "SN2019ehk": ("ZTF19aatesgp", "Ca-rich"),
    "SN2021gno": ("ZTF21aaqhhfu", "Ca-rich"),
    "SN2022oqm": ("ZTF22aasxgjp", "Ca-rich"),
    "AT2019abn": ("ZTF19aadyppr", "ILRT"),
    "SN2020ank": ("ZTF20aahbfmf", "SLSN-I"),
    "SN2020kyg": ("ZTF20abjbgjj", "SN Iax"),
}
OUT = Path(__file__).resolve().parent / "rare_transients"
HORIZON = 100.0
ZP = 23.9
LOG_CONST = 1.0 / np.log(10)
COLOR_WINDOW = 1.5


# ---------------------------------------------------------------- backends --
def _http_json(url, data=None, headers=None, form=False, retries=3):
    for i in range(retries):
        try:
            body = None
            hdrs = dict(headers or {})
            if data is not None:
                if form:
                    body = urllib.parse.urlencode(data).encode()
                    hdrs["Content-Type"] = "application/x-www-form-urlencoded"
                else:
                    body = json.dumps(data).encode()
                    hdrs["Content-Type"] = "application/json"
            req = urllib.request.Request(url, data=body, headers=hdrs)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"request failed: {url}")


def fetch_alerce(oid):
    """-> list of (mjd, fid, magpsf, sigmapsf) from the public ALeRCE API."""
    lc = _http_json(f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve")
    return [(d["mjd"], d["fid"], d["magpsf"], d["sigmapsf"])
            for d in lc["detections"]
            if d.get("magpsf") is not None and d.get("fid") in (1, 2, 3)]


class Boom:
    """Minimal BOOM API client (POST /auth, POST /queries/find)."""

    def __init__(self):
        self.url = os.environ["BOOM_URL"].rstrip("/")
        token = os.environ.get("BOOM_TOKEN")
        if not token:
            resp = _http_json(f"{self.url}/auth",
                              data={"username": os.environ["BOOM_USERNAME"],
                                    "password": os.environ["BOOM_PASSWORD"]},
                              form=True)
            token = resp["access_token"]
        self.headers = {"Authorization": f"Bearer {token}"}

    def find(self, catalog, filt, projection):
        resp = _http_json(f"{self.url}/queries/find",
                          data={"catalog_name": catalog, "filter": filt,
                                "projection": projection},
                          headers=self.headers)
        return resp.get("data", resp) or []

    def fetch(self, oid):
        """-> list of (mjd, fid, magpsf, sigmapsf) from alerts + prv history."""
        dets = {}
        # triggered alerts
        for doc in self.find("ZTF_alerts", {"objectId": oid},
                             {"candidate.jd": 1, "candidate.fid": 1,
                              "candidate.magpsf": 1, "candidate.sigmapsf": 1}):
            c = doc.get("candidate", {})
            if c.get("magpsf") is not None:
                dets[round(c["jd"], 5)] = (c["jd"] - 2400000.5, c["fid"],
                                           c["magpsf"], c["sigmapsf"])
        # previous-candidate history (aux collection, keyed by object id)
        for doc in self.find("ZTF_alerts_aux", {"_id": oid},
                             {"prv_candidates": 1}):
            for c in doc.get("prv_candidates", []) or []:
                if c.get("magpsf") is not None and c.get("fid") in (1, 2, 3):
                    dets.setdefault(round(c["jd"], 5),
                                    (c["jd"] - 2400000.5, c["fid"],
                                     c["magpsf"], c["sigmapsf"]))
        return sorted(dets.values())


# ------------------------------------------------------------- conversion --
def to_events(rows):
    """(mjd, fid, magpsf, sigmapsf) rows -> (n,15) training-format array."""
    rows = sorted((m, int(f) - 1, mag, err) for m, f, mag, err in rows)
    mjd = np.array([r[0] for r in rows])
    keep = mjd <= mjd[0] + HORIZON
    mjd = mjd[keep]
    band = np.array([r[1] for r in rows])[keep]
    mag = np.array([r[2] for r in rows])[keep]
    err = np.array([r[3] for r in rows])[keep]

    flux = 10.0 ** (-0.4 * (mag - ZP))
    flux_err = err * flux / (2.5 * LOG_CONST)
    logf = np.log10(np.clip(flux, 1e-6, None))
    logf_err = flux_err * LOG_CONST / flux

    dt = mjd - mjd[0]
    dt_prev = np.diff(np.r_[mjd[0], mjd])
    oh = np.eye(3)[np.clip(band, 0, 2)]

    def colour(i, b1, b2):
        m = band == (b2 if band[i] == b1 else b1)
        if band[i] not in (b1, b2) or not m.any():
            return np.nan, np.nan
        j = np.argmin(np.abs(mjd[m] - mjd[i]))
        if abs(mjd[m][j] - mjd[i]) > COLOR_WINDOW:
            return np.nan, np.nan
        lf1 = logf[i] if band[i] == b1 else logf[m][j]
        lf2 = logf[m][j] if band[i] == b1 else logf[i]
        return -2.5 * (lf1 - lf2), 2.5 * np.hypot(logf_err[i], logf_err[m][j])

    n = len(mjd)
    g_r = np.full(n, np.nan); g_r_e = np.full(n, np.nan)
    r_i = np.full(n, np.nan); r_i_e = np.full(n, np.nan)
    for i in range(n):
        g_r[i], g_r_e[i] = colour(i, 0, 1)
        r_i[i], r_i_e[i] = colour(i, 1, 2)
    has_gr = np.isfinite(g_r).astype(float)
    has_ri = np.isfinite(r_i).astype(float)

    return np.column_stack([dt, dt_prev, band, logf, logf_err,
                            oh[:, 0], oh[:, 1], oh[:, 2],
                            np.nan_to_num(g_r), np.nan_to_num(g_r_e),
                            np.nan_to_num(r_i), np.nan_to_num(r_i_e),
                            has_gr, has_ri,
                            np.full(n, -1.0)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "boom", "alerce"])
    args = ap.parse_args()
    use_boom = (args.backend == "boom"
                or (args.backend == "auto" and os.environ.get("BOOM_URL")))
    boom = Boom() if use_boom else None
    print(f"backend: {'BOOM (' + boom.url + ')' if boom else 'ALeRCE (public)'}")

    OUT.mkdir(exist_ok=True)
    meta = {}
    for name, (oid, cls) in OBJECTS.items():
        rows = boom.fetch(oid) if boom else fetch_alerce(oid)
        data = to_events(rows)
        band = data[:, 2].astype(int)
        ng, nr = (band == 0).sum(), (band == 1).sum()
        ok = len(data) >= 8 and ng >= 2 and nr >= 2
        np.savez(OUT / f"{oid}.npz", data=data,
                 columns=np.array(["dt", "dt_prev", "band_id", "logflux",
                                   "logflux_err", "band_ztfg", "band_ztfr",
                                   "band_ztfi", "g_r", "g_r_err", "r_i",
                                   "r_i_err", "has_g_r", "has_r_i", "label"]),
                 label=np.int64(-1))
        meta[oid] = {"name": name, "class": cls, "n_events": int(len(data)),
                     "n_g": int(ng), "n_r": int(nr), "passes_quality": bool(ok)}
        print(f"{name:10s} ({cls:7s}) {oid}: {len(data):3d} events "
              f"(g={ng}, r={nr})  quality={'OK' if ok else 'FAIL'}")
        time.sleep(0.5)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved -> {OUT}/")


if __name__ == "__main__":
    main()
