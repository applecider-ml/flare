"""Point-source context from Gaia DR3 and AllWISE, via Data Lab TAP.

The two context blocks the model does not yet see, each targeting a confusion
the light curve cannot resolve:

    Gaia DR3   a parallax or proper motion at the transient position is the
               galactic/extragalactic discriminant -- CVs (and the classical
               novae hiding in the anomaly stream) are stars; supernovae are
               not.  gaia_sep, parallax, parallax_over_error, pm, pm_over_error,
               gaia_g, ruwe.
    AllWISE    W1-W2 is the classical AGN mid-IR discriminant (Stern+ 2012),
               attacking the AGN/TDE confusion.  wise_sep, w1, w1w2, w2w3.

Match radii: 2" for Gaia (sub-arcsec astrometry), 6" for WISE (its PSF).
A missing match is information (no star, no mid-IR source) but coverage varies,
so the training guard is the same 30% dropout as the host block.

Runs on the login node (compute nodes have no internet). Resumable.

Run:  python scripts/fetch_gaia_wise.py [--limit N] [--workers N]
"""
import os
import argparse
import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", REPO / "paper_data"))  # python scripts/setup_data.py
OUT = ROOT / "data" / "context_gaia_wise.csv"
SCHOOL = REPO / "notebooks" / "ztf_summer_school_v2" / "rare_transients"
TAP = "https://datalab.noirlab.edu/tap/sync"
GAIA_BOX = 2.0 / 3600.0
WISE_BOX = 6.0 / 3600.0
FIELDS = ["obj_id", "gaia_sep", "parallax", "parallax_over_error", "pm",
          "pm_over_error", "gaia_g", "ruwe", "wise_sep", "w1", "w1w2", "w2w3",
          "status"]

QG = ("SELECT ra, dec, parallax, parallax_error, pmra, pmdec, "
      "pmra_error, pmdec_error, phot_g_mean_mag, ruwe "
      "FROM gaia_dr3.gaia_source "
      "WHERE ra BETWEEN {ralo:.6f} AND {rahi:.6f} "
      "AND dec BETWEEN {declo:.6f} AND {dechi:.6f}")
QW = ("SELECT ra, dec, w1mpro, w2mpro, w3mpro "
      "FROM allwise.source "
      "WHERE ra BETWEEN {ralo:.6f} AND {rahi:.6f} "
      "AND dec BETWEEN {declo:.6f} AND {dechi:.6f}")


def tap(q, tries=4):
    url = TAP + "?" + urllib.parse.urlencode(
        dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q))
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode()
            if "," in text.split("\n", 1)[0]:
                return pd.read_csv(io.StringIO(text))
            return pd.DataFrame()
        except Exception:
            if a == tries - 1:
                return None
            time.sleep(2.0 * (a + 1))


def box(q, ra, dec, half):
    dra = half / max(np.cos(np.radians(dec)), 0.05)
    return q.format(ralo=ra - dra, rahi=ra + dra,
                    declo=dec - half, dechi=dec + half)


def nearest(d, ra, dec):
    cosd = np.cos(np.radians(dec))
    d = d.copy()
    d["sep"] = 3600.0 * np.hypot((d.ra - ra) * cosd, d.dec - dec)
    return d.sort_values("sep").iloc[0]


def one(row):
    oid, ra, dec = row
    out = dict(obj_id=oid)
    status = []
    g = tap(box(QG, ra, dec, GAIA_BOX))
    if g is None:
        status.append("gaia_err")
    elif len(g):
        h = nearest(g, ra, dec)
        pm = np.hypot(h.pmra, h.pmdec)
        pme = np.hypot(h.pmra_error, h.pmdec_error)
        out.update(gaia_sep=float(h.sep),
                   parallax=float(h.parallax) if np.isfinite(h.parallax) else np.nan,
                   parallax_over_error=float(h.parallax / h.parallax_error)
                   if np.isfinite(h.parallax) and h.parallax_error > 0 else np.nan,
                   pm=float(pm) if np.isfinite(pm) else np.nan,
                   pm_over_error=float(pm / pme)
                   if np.isfinite(pm) and pme > 0 else np.nan,
                   gaia_g=float(h.phot_g_mean_mag)
                   if np.isfinite(h.phot_g_mean_mag) else np.nan,
                   ruwe=float(h.ruwe) if np.isfinite(h.ruwe) else np.nan)
        status.append("gaia_ok")
    else:
        status.append("gaia_none")
    w = tap(box(QW, ra, dec, WISE_BOX))
    if w is None:
        status.append("wise_err")
    elif len(w):
        h = nearest(w, ra, dec)
        out.update(wise_sep=float(h.sep),
                   w1=float(h.w1mpro) if np.isfinite(h.w1mpro) else np.nan,
                   w1w2=float(h.w1mpro - h.w2mpro)
                   if np.isfinite(h.w1mpro) and np.isfinite(h.w2mpro) else np.nan,
                   w2w3=float(h.w2mpro - h.w3mpro)
                   if np.isfinite(h.w2mpro) and np.isfinite(h.w3mpro) else np.nan)
        status.append("wise_ok")
    else:
        status.append("wise_none")
    out["status"] = "+".join(status)
    return out


def alerce_coords(oid):
    try:
        with urllib.request.urlopen(
                f"https://api.alerce.online/ztf/v1/objects/{oid}", timeout=30) as r:
            d = json.load(r)
        return float(d["meanra"]), float(d["meandec"])
    except Exception:
        return None


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

    # school extras: coordinates from ALeRCE
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
