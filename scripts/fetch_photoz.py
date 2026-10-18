"""Host photo-z from Legacy Survey DR10, via the Data Lab TAP service.

The oracle measured what distance is worth (SLSN F1 0.41 -> 0.75); host
photometry recovered only part of it because apparent magnitude is a noisy
distance proxy. DR10 supplies the deployable middle ground: a photometric
redshift for the host galaxy, available at alert time, no spectrum anywhere.
With z_phot the peak apparent magnitude becomes a pseudo-absolute magnitude.

For each position: a ~30 arcsec box on ls_dr10.tractor joined to
ls_dr10.photo_z, nearest extended source (type != 'PSF') taken as the host:

    pz_sep      arcsec to that host candidate
    pz_type     tractor morphological type (REX/EXP/DEV/SER)
    pz_mag_r/z  model magnitudes
    z_phot      z_phot_median        z_phot_err   z_phot_std
    z_spec      spectroscopic z if DR10 carries one (do NOT feed to the model
                blindly -- same follow-up bias as the BTS redshift)
    pz_n_gal    extended sources in the box (crowding)

The same missingness rules as the host block apply: DR10 does not cover the
whole ZTF footprint and depth varies, so train with dropout and report
with/without subsets separately.

Runs on the login node (network-bound; compute nodes have no internet).
Resumable via the output CSV.

Run:  python scripts/fetch_photoz.py [--limit N] [--workers N]
"""
import os
import argparse
import csv
import io
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", REPO / "paper_data"))  # python scripts/setup_data.py
OUT = ROOT / "data" / "hosts_photoz.csv"
TAP = "https://datalab.noirlab.edu/tap/sync"
BOX = 30.0 / 3600.0
FIELDS = ["obj_id", "pz_sep", "pz_type", "pz_mag_r", "pz_mag_z",
          "z_phot", "z_phot_std", "z_spec", "pz_n_gal", "status"]
# DR10 is DECam-based and stops near dec +32; DR9 carries BASS/MzLS in the
# north. Query DR10 first, fall back to DR9 -- ZTF is a northern survey.
Q = ("SELECT t.ra, t.dec, t.type, t.mag_r, t.mag_z, "
     "p.z_phot_median, p.z_phot_std, p.z_spec "
     "FROM {sur}.tractor t JOIN {sur}.photo_z p ON t.ls_id=p.ls_id "
     "WHERE t.ra BETWEEN {ralo:.6f} AND {rahi:.6f} "
     "AND t.dec BETWEEN {declo:.6f} AND {dechi:.6f}")


def query(ra, dec, survey="ls_dr10", tries=4):
    dra = BOX / max(np.cos(np.radians(dec)), 0.05)
    q = Q.format(sur=survey, ralo=ra - dra, rahi=ra + dra,
                 declo=dec - BOX, dechi=dec + BOX)
    url = TAP + "?" + urllib.parse.urlencode(
        dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q))
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                text = r.read().decode()
            if text.startswith("ls_id") or "," in text.split("\n", 1)[0]:
                return pd.read_csv(io.StringIO(text))
            return None
        except Exception:
            if a == tries - 1:
                return None
            time.sleep(2.0 * (a + 1))
    return None


def one(row):
    oid, ra, dec = row
    survey = "dr10"
    d = query(ra, dec, "ls_dr10")
    if d is None or not len(d):
        survey = "dr9"
        d = query(ra, dec, "ls_dr9")
    if d is None:
        return dict(obj_id=oid, status="error")
    if not len(d):
        return dict(obj_id=oid, status="empty")
    d = d.replace(-99.0, np.nan)
    d = d[d.type != "PSF"].copy()
    if not len(d):
        return dict(obj_id=oid, status="no_gal")
    cosd = np.cos(np.radians(dec))
    d["sep"] = 3600.0 * np.hypot((d.ra - ra) * cosd, d.dec - dec)
    d = d.sort_values("sep")
    h = d.iloc[0]
    return dict(obj_id=oid, status=f"ok_{survey}", pz_sep=float(h.sep), pz_type=str(h.type),
                pz_mag_r=float(h.mag_r) if np.isfinite(h.mag_r) else np.nan,
                pz_mag_z=float(h.mag_z) if np.isfinite(h.mag_z) else np.nan,
                z_phot=float(h.z_phot_median) if np.isfinite(h.z_phot_median) else np.nan,
                z_phot_std=float(h.z_phot_std) if np.isfinite(h.z_phot_std) else np.nan,
                z_spec=float(h.z_spec) if np.isfinite(h.z_spec) else np.nan,
                pz_n_gal=int(len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wanted", default="", help="CSV with an obj_id column; default = the benchmark splits")
    ap.add_argument("--out", default="", help="output CSV; default = the benchmark file")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out)

    cat = pd.read_csv(ROOT / "data" / "bts_catalog.csv").rename(columns={"ZTFID": "obj_id"})

    def deg(v, is_ra):
        p = [float(x) for x in str(v).split(":")]
        s = -1 if str(v).strip().startswith("-") else 1
        return s * (abs(p[0]) + p[1] / 60 + (p[2] if len(p) > 2 else 0) / 3600) \
            * (15 if is_ra else 1)

    wanted = pd.read_csv(a.wanted)[["obj_id"]] if a.wanted else pd.concat(
        [pd.read_csv(ROOT / "data" / "splits" / f"{s}.csv")[["obj_id"]]
         for s in ["train", "val", "test", "anomaly"]], ignore_index=True)
    todo = wanted.merge(cat[["obj_id", "RA", "Dec"]], on="obj_id", how="left") \
                 .dropna(subset=["RA", "Dec"]).drop_duplicates("obj_id")
    done = set(pd.read_csv(OUT).obj_id) if OUT.exists() else set()
    todo = todo[~todo.obj_id.isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    if todo.empty:
        return
    rows = [(r.obj_id, deg(r.RA, True), deg(r.Dec, False)) for r in todo.itertuples()]

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
                      f"eta {(len(rows) - n) / max(rate, 1e-9) / 60:.0f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
