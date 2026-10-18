"""PS1 point source AT the transient position: the quiescent counterpart.

The Gaia block established presence-of-a-point-source as a discriminant but
saturates at G~20.5, which is why it saw 2/37 nova counterparts and only 18%
of CVs. Pan-STARRS DR2 reaches ~23 mag and was observed 2010--2014, before
every BTS transient, so a detection inside 1.5 arcsec is a clean pre-outburst
counterpart -- the quiescent CV or nova system, an LBV/LRN progenitor, or the
galaxy nucleus -- with zero contamination from the transient itself.

Columns (NaN when nothing is inside 1.5"):
    pos_sep      arcsec to the nearest PS1 source
    pos_r/i      its PSF magnitudes
    pos_gr/ri    PSF colours
    pos_star     iPSF - iKron < 0.05  (point-source morphology)
    pos_ndet     nDetections (variability-ish proxy for persistence)

Login node (network). Resumable.

Run:  python scripts/fetch_ps1_position.py [--workers 6] [--limit N]
"""
import os
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", REPO / "paper_data"))  # python scripts/setup_data.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_hosts import COLS, HEADERS, URL, sep_arcsec   # noqa: E402
from fetch_gaia_wise import alerce_coords                # noqa: E402

OUT = ROOT / "data" / "ps1_position.csv"
SCHOOL = REPO / "notebooks" / "ztf_summer_school_v2" / "rare_transients"
RADIUS_DEG = 1.5 / 3600.0
FIELDS = ["obj_id", "pos_sep", "pos_g", "pos_r", "pos_i", "pos_gr", "pos_ri",
          "pos_star", "pos_ndet", "status"]
QCOLS = COLS + ["gMeanPSFMag", "iMeanKronMag"]


def query(ra, dec, tries=4):
    q = urllib.parse.urlencode({
        "ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "radius": f"{RADIUS_DEG:.6f}",
        "nDetections.gte": 2, "columns": ",".join(QCOLS), "pagesize": 50})
    for a in range(tries):
        try:
            req = urllib.request.Request(f"{URL}?{q}", headers=HEADERS)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode())
        except Exception:
            if a == tries - 1:
                return None
            time.sleep(1.5 * (a + 1))


def one(row):
    oid, ra, dec = row
    j = query(ra, dec)
    if j is None:
        return dict(obj_id=oid, status="error")
    recs = j.get("data", [])
    if not recs:
        return dict(obj_id=oid, status="empty")
    if isinstance(recs[0], list):
        d = pd.DataFrame(recs, columns=[c["name"] for c in j["info"]])
    else:
        d = pd.DataFrame(recs)
    for c in d.columns:
        if c not in ("objID",):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace(-999.0, np.nan)
    d["sep"] = sep_arcsec(ra, dec, d.raMean.values, d.decMean.values)
    h = d.sort_values("sep").iloc[0]

    def g(v):
        return float(v) if np.isfinite(v) else np.nan
    star = (h.iMeanPSFMag - h.iMeanKronMag
            if np.isfinite(h.iMeanPSFMag) and np.isfinite(h.iMeanKronMag)
            else np.nan)
    return dict(obj_id=oid, status="ok", pos_sep=float(h.sep),
                pos_g=g(h.gMeanPSFMag), pos_r=g(h.rMeanPSFMag),
                pos_i=g(h.iMeanPSFMag),
                pos_gr=g(h.gMeanPSFMag - h.rMeanPSFMag),
                pos_ri=g(h.rMeanPSFMag - h.iMeanPSFMag),
                pos_star=float(star < 0.05) if np.isfinite(star) else np.nan,
                pos_ndet=g(h.nDetections))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wanted", default="", help="CSV with an obj_id column; default = the benchmark splits")
    ap.add_argument("--out", default="", help="output CSV; default = the benchmark file")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)
    cat = pd.read_csv(ROOT / "data" / "bts_catalog.csv").rename(
        columns={"ZTFID": "obj_id"})

    def deg(v, is_ra):
        p = [float(x) for x in str(v).split(":")]
        s = -1 if str(v).strip().startswith("-") else 1
        return s * (abs(p[0]) + p[1] / 60 + (p[2] if len(p) > 2 else 0) / 3600) \
            * (15 if is_ra else 1)

    wanted = pd.read_csv(a.wanted)[["obj_id"]] if a.wanted else pd.concat(
        [pd.read_csv(ROOT / "data" / "splits" / f"{s}.csv")[["obj_id"]]
         for s in ["train", "val", "test", "anomaly"]], ignore_index=True).drop_duplicates("obj_id")
    todo = wanted.merge(cat[["obj_id", "RA", "Dec"]], on="obj_id", how="left")
    rows = [(r.obj_id, deg(r.RA, True), deg(r.Dec, False))
            for r in todo.dropna(subset=["RA", "Dec"]).itertuples()]
    school = {} if a.wanted else json.loads((SCHOOL / "meta.json").read_text())
    have = {r[0] for r in rows}
    for k in school:
        if k not in have:
            c = alerce_coords(k)
            if c:
                rows.append((k, c[0], c[1]))
    done = set(pd.read_csv(OUT).obj_id) if OUT.exists() else set()
    rows = [r for r in rows if r[0] not in done]
    if a.limit:
        rows = rows[:a.limit]
    print(f"{len(done)} done, {len(rows)} to go", flush=True)
    if not rows:
        return
    if not OUT.exists():
        pd.DataFrame(columns=FIELDS).to_csv(OUT, index=False)
    t0, n, buf = time.time(), 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(one, rows):
            buf.append(res)
            n += 1
            if len(buf) >= 50 or n == len(rows):
                pd.DataFrame(buf).reindex(columns=FIELDS).to_csv(
                    OUT, mode="a", header=False, index=False)
                buf = []
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n}/{len(rows)}  {rate:.1f}/s  "
                      f"eta {(len(rows) - n) / max(rate, 1e-9) / 60:.0f} min",
                      flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
