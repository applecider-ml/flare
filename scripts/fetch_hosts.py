"""Host-galaxy photometry from Pan-STARRS DR2, without a redshift.

The distance oracle showed SLSN F1 going 0.41 -> 0.75 once absolute magnitude is
known, but a redshift is not available for a general alert. Host *photometry*
is: whenever Pan-STARRS resolves a galaxy at the transient's position, its
colour, brightness and the transient's offset from it are measurable at alert
time and need no distance at all. The physics is well established -- superluminous
SNe sit in blue, low-mass dwarfs, Ca-rich transients sit far out in the halos of
early-type galaxies -- and both are things our classifier and our anomaly layer
currently get wrong.

For each position this queries the PS1 DR2 mean-object catalogue within 30",
keeps sources that look extended (PSF minus Kron magnitude above 0.05, the
standard PS1 separator), and records the nearest one plus the field around it:

    host_sep         arcsec to the nearest extended source
    host_r, host_i   Kron magnitudes
    host_gr, host_ri, host_iz    colours
    host_ext         PSF - Kron in i, how extended the match is
    n_ext_30         extended neighbours within 30", a crowding proxy
    near_sep         arcsec to the nearest source of any kind
    near_ext         whether that nearest source is extended

WARNING for whoever analyses this. Host presence is *not* independent of class.
AGN live in the nucleus of their host, so they always have one at zero offset;
CVs are Galactic stars with no host at all. That is the same missingness trap
the redshift experiment measured at twelve points of macro F1, in a new costume.
Report the detection rate per class before reporting any performance number, and
train with the host block randomly dropped so the model cannot lean on its
absence.

Runs on the login node: the compute nodes have no outbound network. Resumable --
finished positions are appended to the output and skipped on restart.

Run:  python scripts/fetch_hosts.py [--limit N]
"""
import os
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", REPO / "paper_data"))  # python scripts/setup_data.py
OUT = ROOT / "data" / "hosts.csv"
URL = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.json"
RADIUS_DEG = 30.0 / 3600.0
COLS = ["objID", "raMean", "decMean", "nDetections",
        "gMeanKronMag", "rMeanKronMag", "iMeanKronMag", "zMeanKronMag",
        "iMeanPSFMag", "rMeanPSFMag"]
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120 Safari/537.36"}
FIELDS = ["obj_id", "host_sep", "host_r", "host_i", "host_gr", "host_ri",
          "host_iz", "host_ext", "n_ext_30", "near_sep", "near_ext", "status"]


def query(ra, dec, tries=4):
    q = urllib.parse.urlencode({
        "ra": f"{ra:.6f}", "dec": f"{dec:.6f}", "radius": f"{RADIUS_DEG:.6f}",
        "nDetections.gte": 3, "columns": ",".join(COLS), "pagesize": 200})
    for a in range(tries):
        try:
            req = urllib.request.Request(f"{URL}?{q}", headers=HEADERS)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            if a == tries - 1:
                return None
            time.sleep(1.5 * (a + 1))
    return None


def sep_arcsec(ra0, dec0, ra, dec):
    d = np.radians(dec0)
    return 3600.0 * np.hypot((ra - ra0) * np.cos(d), dec - dec0)


def one(row):
    """Query one position and reduce it to the host feature block."""
    oid, ra, dec = row
    j = query(ra, dec)
    if j is None:
        return dict(obj_id=oid, status="error")
    recs = j.get("data", [])
    if not recs:
        return dict(obj_id=oid, status="empty")
    # the service ignores the `columns` parameter and returns everything, so the
    # column names have to come from the info block rather than be assumed
    if isinstance(recs[0], list):
        d = pd.DataFrame(recs, columns=[c["name"] for c in j["info"]])
    else:
        d = pd.DataFrame(recs)
    for c in COLS[1:]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce").replace(-999.0, np.nan)
    if "raMean" not in d or d.raMean.isna().all():
        return dict(obj_id=oid, status="empty")
    d["sep"] = sep_arcsec(ra, dec, d.raMean.values, d.decMean.values)
    d = d.sort_values("sep")

    out = dict(obj_id=oid, status="ok", near_sep=float(d.sep.iloc[0]))
    ext = d.iMeanPSFMag - d.iMeanKronMag
    d = d.assign(ext=ext)
    out["near_ext"] = float(ext.iloc[0]) if np.isfinite(ext.iloc[0]) else np.nan
    gal = d[(d.ext > 0.05) & d.rMeanKronMag.notna()]
    out["n_ext_30"] = int(len(gal))
    if len(gal):
        h = gal.iloc[0]
        out.update(host_sep=float(h.sep), host_r=float(h.rMeanKronMag),
                   host_i=float(h.iMeanKronMag) if np.isfinite(h.iMeanKronMag) else np.nan,
                   host_gr=float(h.gMeanKronMag - h.rMeanKronMag),
                   host_ri=float(h.rMeanKronMag - h.iMeanKronMag),
                   host_iz=float(h.iMeanKronMag - h.zMeanKronMag),
                   host_ext=float(h.ext))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wanted", default="", help="CSV with an obj_id column; default = the benchmark splits")
    ap.add_argument("--out", default="", help="output CSV; default = the benchmark file")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)

    cat = pd.read_csv(ROOT / "data" / "bts_catalog.csv").rename(columns={"ZTFID": "obj_id"})
    cat = cat[["obj_id", "RA", "Dec"]]
    wanted = pd.read_csv(a.wanted)[["obj_id"]] if a.wanted else pd.concat(
        [pd.read_csv(ROOT / "data" / "splits" / f"{s}.csv")[["obj_id"]]
         for s in ["train", "val", "test", "anomaly"]], ignore_index=True)
    todo = wanted.merge(cat, on="obj_id", how="left").dropna(subset=["RA", "Dec"])
    todo = todo.drop_duplicates("obj_id")

    done = set()
    if OUT.exists():
        done = set(pd.read_csv(OUT).obj_id)
    todo = todo[~todo.obj_id.isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"{len(done)} already fetched, {len(todo)} to go", flush=True)
    if todo.empty:
        return

    # RA/Dec in the BTS catalogue are sexagesimal strings
    def deg(v, is_ra):
        if isinstance(v, str) and ":" in v:
            p = [float(x) for x in v.split(":")]
            s = -1 if p[0] < 0 or v.strip().startswith("-") else 1
            val = abs(p[0]) + p[1] / 60 + (p[2] if len(p) > 2 else 0) / 3600
            return s * val * (15 if is_ra else 1)
        return float(v)

    rows = [(r.obj_id, deg(r.RA, True), deg(r.Dec, False))
            for r in todo.itertuples()]

    if not OUT.exists():
        pd.DataFrame(columns=FIELDS).to_csv(OUT, index=False)
    t0, n = time.time(), 0
    buf = []
    with ThreadPoolExecutor(a.workers) as ex:
        for res in ex.map(one, rows):
            buf.append(res)
            n += 1
            if len(buf) >= 100 or n == len(rows):
                pd.DataFrame(buf).reindex(columns=FIELDS).to_csv(
                    OUT, mode="a", header=False, index=False)
                buf = []
                rate = n / max(time.time() - t0, 1e-9)
                print(f"  {n}/{len(rows)}  {rate:.1f}/s  "
                      f"eta {(len(rows)-n)/max(rate,1e-9)/60:.0f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
