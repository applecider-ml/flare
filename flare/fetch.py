"""Photometry retrieval: BOOM first, public ALeRCE as the fallback.

BOOM is the broker this group operates; it is preferred whenever credentials
are present because it serves both triggered alerts and the previous-candidate
history in one query. Configuration is by environment:

    BOOM_URL        base URL of the BOOM API           (enables the backend)
    BOOM_TOKEN      bearer token, or
    BOOM_USERNAME / BOOM_PASSWORD for password auth

Without BOOM_URL the public ALeRCE API is used, so everything reproduces
anywhere with no credentials. Both backends return the same tuples
(mjd, fid, magpsf, sigmapsf), which `to_events` converts into the (N, 15)
event-array format the rest of the package consumes.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

import numpy as np

ZP = 23.9
LOG_CONST = 1.0 / np.log(10)
COLOR_WINDOW = 1.5
DEFAULT_HORIZON = 100.0

Row = Tuple[float, int, float, float]


def _http_json(url, data=None, headers=None, form=False, retries=3):
    for i in range(retries):
        try:
            body, hdrs = None, dict(headers or {})
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

    def fetch(self, oid: str) -> List[Row]:
        """Alerts plus previous-candidate history, deduplicated by epoch."""
        dets = {}
        for doc in self.find("ZTF_alerts", {"objectId": oid},
                             {"candidate.jd": 1, "candidate.fid": 1,
                              "candidate.magpsf": 1, "candidate.sigmapsf": 1}):
            c = doc.get("candidate", {})
            if c.get("magpsf") is not None and c.get("fid") in (1, 2, 3):
                dets[round(c["jd"], 5)] = (c["jd"] - 2400000.5, c["fid"],
                                           c["magpsf"], c["sigmapsf"])
        for doc in self.find("ZTF_alerts_aux", {"_id": oid},
                             {"prv_candidates": 1}):
            for c in doc.get("prv_candidates", []) or []:
                if c.get("magpsf") is not None and c.get("fid") in (1, 2, 3):
                    dets.setdefault(round(c["jd"], 5),
                                    (c["jd"] - 2400000.5, c["fid"],
                                     c["magpsf"], c["sigmapsf"]))
        return sorted(dets.values())


def fetch_alerce(oid: str) -> List[Row]:
    lc = _http_json(f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve")
    return [(d["mjd"], d["fid"], d["magpsf"], d["sigmapsf"])
            for d in lc.get("detections", [])
            if d.get("magpsf") is not None and d.get("fid") in (1, 2, 3)]


def coords(oid: str) -> Optional[tuple]:
    """(ra, dec) in degrees for a ZTF object id, or None.

    The photometry rows carry no position, and every external context block is
    a positional crossmatch, so this is fetched separately.
    """
    try:
        d = _http_json(f"https://api.alerce.online/ztf/v1/objects/{oid}")
        return float(d["meanra"]), float(d["meandec"])
    except Exception:
        return None


def fetch_photometry(oid: str, backend: str = "auto") -> List[Row]:
    """(mjd, fid, magpsf, sigmapsf) rows for one ZTF object id."""
    use_boom = (backend == "boom"
                or (backend == "auto" and os.environ.get("BOOM_URL")))
    if use_boom:
        return Boom().fetch(oid)
    return fetch_alerce(oid)


def to_events(rows: List[Row],
              horizon_days: float = DEFAULT_HORIZON) -> Optional[np.ndarray]:
    """(mjd, fid, mag, magerr) -> (N, 15) event array; None if empty."""
    rows = sorted((m, int(f) - 1, mag, err) for m, f, mag, err in rows)
    if not rows:
        return None
    mjd = np.array([r[0] for r in rows])
    keep = mjd <= mjd[0] + horizon_days
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
    gr = np.full(n, np.nan); gre = np.full(n, np.nan)
    ri = np.full(n, np.nan); rie = np.full(n, np.nan)
    for i in range(n):
        gr[i], gre[i] = colour(i, 0, 1)
        ri[i], rie[i] = colour(i, 1, 2)
    has_gr = np.isfinite(gr).astype(float)
    has_ri = np.isfinite(ri).astype(float)
    return np.column_stack([
        dt, dt_prev, band, logf, logf_err, oh[:, 0], oh[:, 1], oh[:, 2],
        np.nan_to_num(gr), np.nan_to_num(gre), np.nan_to_num(ri),
        np.nan_to_num(rie), has_gr, has_ri,
        np.full(n, -1.0)]).astype(np.float32)


def fetch_events(oid: str, backend: str = "auto",
                 horizon_days: float = DEFAULT_HORIZON) -> Optional[np.ndarray]:
    """One call: photometry query -> event array."""
    return to_events(fetch_photometry(oid, backend), horizon_days)
