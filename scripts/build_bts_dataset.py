"""Build a spectroscopically verified ZTF transient dataset from the BTS sample.

Source of labels: the ZTF Bright Transient Survey sample explorer
(https://sites.astro.caltech.edu/ztf/bts/explorer.php), which publishes
~12.9k spectroscopically classified transients with a fine-grained taxonomy.
Photometry: ZTF alert difference photometry, fetched per object (ALeRCE public
API by default; BOOM when BOOM_URL is configured) and converted to the same
event format used by the AppleCiDEr light-curve module.

Why this dataset. The existing AppleCiDEr photometric set carries ten
spectroscopic subclasses collapsed to five broad classes, and therefore has no
representatives of the rare families that matter for open-world discovery. BTS
supplies them with verified spectroscopic types: SLSN-I, SLSN-II, novae,
SN Ibn/Icn, SN Iax, SN Ia-CSM/SC, Ca-rich, ILRT, LBV, LRN and TDE subtypes.

Output layout (data/):
    bts_catalog.csv                  the raw BTS table
    manifest.csv                     obj_id, filepath, bts_type, broad, group,
                                     n_events, n_g, n_r, passes_quality
    events/<ZTFID>.npz               per-object event array (n,15) + columns

Groups: "core" (tier A, trained classes) | "peculiar" (tier B, unusual
members of a trained class) | "novel" (tier C, no parent class) | "ambiguous"
(novae, which sit between CV and a class of their own).

Run:
    python scripts/build_bts_dataset.py --subset novel     # tier C
    python scripts/build_bts_dataset.py --subset peculiar  # tier B
    python scripts/build_bts_dataset.py --subset all       # everything, ~5 h
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", REPO / "paper_data"))  # python scripts/setup_data.py
DATA = ROOT / "data"
EVENTS = DATA / "events"
BTS_URL = "https://sites.astro.caltech.edu/ztf/bts/explorer.php?f=s&format=csv"

HORIZON = 100.0
ZP = 23.9
LOG_CONST = 1.0 / np.log(10)
COLOR_WINDOW = 1.5
MIN_OBS, MIN_G, MIN_R, MIN_BANDS = 8, 2, 2, 2

# ---------------------------------------------------------------- taxonomy --
# Three tiers of novelty relative to the five-class scheme {SNI,SNII,CV,AGN,TDE}.
#
# TIER A ("core"): the classes the model is trained on, including the ordinary
#   spectroscopic subtypes that clearly belong to them.
# TIER B ("peculiar"): unusual members of a class that *does* exist in the
#   scheme. A correct classifier should still return the parent class; a high
#   anomaly score here means "atypical member", not "new class". The mapping is
#   not our invention: for 45 of these objects our own labelled set already
#   assigns the parent (e.g. BTS "SN Ibn" -> label "SN Ib", "SN Iax" -> "SN Ia",
#   "SLSN-II" -> "SN II"), which is the empirical justification for tier B.
# TIER C ("novel"): no natural parent among the five classes.
#
# The ambiguous case is the classical nova: physically an outburst in a
# cataclysmic-variable system, so arguably tier B under CV, but BTS lists it
# separately from "CV" and it is a distinct observational phenomenon. It is
# kept in its own group ("ambiguous") and excluded from the headline tier-C
# count, then reported separately.

CORE = {
    # thermonuclear
    "SN Ia": "SNI", "SN Ia-91T": "SNI", "SN Ia-91bg": "SNI", "SN Ia-pec": "SNI",
    # stripped-envelope
    "SN Ib": "SNI", "SN Ic": "SNI", "SN Ib/c": "SNI", "SN Ic-BL": "SNI",
    "SN Ib-pec": "SNI", "SN Ic-pec": "SNI",
    # hydrogen-rich
    "SN II": "SNII", "SN IIP": "SNII", "SN IIn": "SNII", "SN IIb": "SNII",
    "SN IIL": "SNII",
    # non-SN
    "CV": "CV", "AGN": "AGN", "TDE": "TDE",
}

PECULIAR = {          # tier B: parent class exists
    "SN Iax": "SNI", "SN Ia-CSM": "SNI", "SN Ia-SC": "SNI",
    "SN Ibn": "SNI", "SN Icn": "SNI",
    "Ca-rich": "SNI", "SN Ca-rich-Ca": "SNI",
    "SLSN-II": "SNII", "SN II-pec": "SNII", "SN IIn-pec": "SNII",
    "TDE-H-He": "TDE", "TDE-He": "TDE", "TDE-featureless": "TDE",
}

NOVEL = ["SLSN-I", "LBV", "LRN", "ILRT"]   # tier C: no parent class
AMBIGUOUS = ["nova"]                        # reported separately (vs CV)
# excluded entirely: uncertain types ("CV?", "AGN?") and catch-alls ("Other")


# ------------------------------------------------------------- photometry --
UA = {"User-Agent": "Mozilla/5.0 (compatible; flare-dataset-builder/1.0)"}


def http_json(url, retries=3, timeout=30):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def fetch_alerce(oid):
    lc = http_json(f"https://api.alerce.online/ztf/v1/objects/{oid}/lightcurve")
    return [(d["mjd"], d["fid"], d["magpsf"], d["sigmapsf"])
            for d in lc.get("detections", [])
            if d.get("magpsf") is not None and d.get("fid") in (1, 2, 3)]


def to_events(rows):
    """(mjd, fid, mag, magerr) -> (n,15) array in the AppleCiDEr event format."""
    rows = sorted((m, int(f) - 1, mag, err) for m, f, mag, err in rows)
    mjd = np.array([r[0] for r in rows])
    keep = mjd <= mjd[0] + HORIZON
    mjd, band = mjd[keep], np.array([r[1] for r in rows])[keep]
    mag, err = np.array([r[2] for r in rows])[keep], np.array([r[3] for r in rows])[keep]

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
    has_gr, has_ri = np.isfinite(gr).astype(float), np.isfinite(ri).astype(float)

    return np.column_stack([
        dt, dt_prev, band, logf, logf_err, oh[:, 0], oh[:, 1], oh[:, 2],
        np.nan_to_num(gr), np.nan_to_num(gre), np.nan_to_num(ri),
        np.nan_to_num(rie), has_gr, has_ri, np.full(n, -1.0)]).astype(np.float32)


COLUMNS = np.array(["dt", "dt_prev", "band_id", "logflux", "logflux_err",
                    "band_ztfg", "band_ztfr", "band_ztfi", "g_r", "g_r_err",
                    "r_i", "r_i_err", "has_g_r", "has_r_i", "label"])


# ------------------------------------------------------------------ build --
def load_catalog(refresh=False):
    path = DATA / "bts_catalog.csv"
    if refresh or not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(BTS_URL, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    lines = [l for l in open(path, newline="") if l.strip()]
    rows = list(csv.reader(lines))
    hdr, data = rows[0], rows[1:]
    ti, zi = hdr.index("type"), hdr.index("ZTFID")
    out = []
    for r in data:
        if len(r) <= ti:
            continue
        t = r[ti].strip()
        if t in CORE:
            out.append((r[zi], t, CORE[t], "core"))
        elif t in PECULIAR:
            out.append((r[zi], t, PECULIAR[t], "peculiar"))
        elif t in NOVEL:
            out.append((r[zi], t, "NOVEL", "novel"))
        elif t in AMBIGUOUS:
            out.append((r[zi], t, "CV?", "ambiguous"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="novel",
                    choices=["core", "peculiar", "novel", "ambiguous", "all"])
    ap.add_argument("--refresh-catalog", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cat = load_catalog(args.refresh_catalog)
    if args.subset != "all":
        cat = [c for c in cat if c[3] == args.subset]
    if args.limit:
        cat = cat[:args.limit]
    EVENTS.mkdir(parents=True, exist_ok=True)
    print(f"catalog: {len(cat)} objects (subset={args.subset})", flush=True)

    mpath = DATA / f"manifest_{args.subset}.csv"
    done = set()
    if mpath.exists():                       # resumable
        done = set(pd.read_csv(mpath).obj_id)
        print(f"resuming: {len(done)} already fetched", flush=True)

    new, t0 = [], time.time()
    for k, (oid, btype, broad, group) in enumerate(cat, 1):
        if oid in done:
            continue
        try:
            rows = fetch_alerce(oid)
            if not rows:
                raise ValueError("no detections")
            ev = to_events(rows)
            b = ev[:, 2].astype(int)
            ng, nr, ni = (b == 0).sum(), (b == 1).sum(), (b == 2).sum()
            nb = int((np.array([ng, nr, ni]) > 0).sum())
            ok = len(ev) >= MIN_OBS and ng >= MIN_G and nr >= MIN_R and nb >= MIN_BANDS
            np.savez(EVENTS / f"{oid}.npz", data=ev, columns=COLUMNS,
                     label=np.int64(-1))
            new.append(dict(obj_id=oid, filepath=f"events/{oid}.npz",
                            bts_type=btype, broad=broad, group=group,
                            n_events=int(len(ev)), n_g=int(ng), n_r=int(nr),
                            passes_quality=bool(ok)))
        except Exception as e:
            new.append(dict(obj_id=oid, filepath="", bts_type=btype,
                            broad=broad, group=group, n_events=0, n_g=0, n_r=0,
                            passes_quality=False))
            print(f"  [{k}] {oid} {btype}: {type(e).__name__}", flush=True)
        if k % 100 == 0:
            el = time.time() - t0
            print(f"  {k}/{len(cat)}  {el/60:.1f} min  "
                  f"(eta {(len(cat)-k)*el/max(k,1)/60:.0f} min)", flush=True)
            _flush(mpath, new)
    _flush(mpath, new)

    df = pd.read_csv(mpath)
    good = df[df.passes_quality]
    print(f"\nfetched {len(df)} | pass quality {len(good)} ({len(good)/max(len(df),1):.0%})")
    print(good.groupby(["group", "bts_type"]).size().to_string())


def _flush(mpath, new):
    if not new:
        return
    df = pd.DataFrame(new)
    if mpath.exists():
        df = pd.concat([pd.read_csv(mpath), df], ignore_index=True)
        df = df.drop_duplicates("obj_id", keep="last")
    df.to_csv(mpath, index=False)
    new.clear()


if __name__ == "__main__":
    main()
