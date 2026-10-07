"""External context at alert time: host, photo-z, and point source; no spectrum.

Three blocks, all optional and all fetched from public archives given only a
sky position:

  host photometry (Pan-STARRS DR2 via the MAST catalog API): angular offset to
  the nearest extended source, its Kron magnitudes and colours, and field
  crowding. The extended/point separation is the standard PS1 cut
  iPSF - iKron > 0.05.

  photometric redshift (Legacy Surveys DR10, falling back to DR9 for the
  northern sky DR10 does not cover): z_phot of the nearest extended source,
  from which the peak apparent magnitude becomes a pseudo-absolute magnitude
  M_pseudo = m_peak - 5 log10(d_L(z_phot)/10 pc), flat LCDM (H0=70, Om=0.3).

  point-source context (Gaia DR3 and AllWISE via the Data Lab TAP service):
  Gaia astrometry within 2 arcsec -- parallax and proper motion with their
  significances, G, RUWE, and the separation itself; the *presence* of a
  counterpart is the galactic/nuclear discriminant (98% of AGN and 84% of TDEs
  have one, offset supernovae rarely do). AllWISE within 6 arcsec supplies the
  mid-infrared colour W1-W2, the classical AGN discriminant.

Missingness discipline (measured in the paper, Sections on the missingness
trap): the bts6 models are trained with each block independently blanked on
30% of training rows, so an object with no host or no redshift degrades
gracefully instead of being misread. Never impute these columns; pass NaN.
"""
from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
from typing import Optional

import numpy as np
import pandas as pd

MAST = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.json"
TAP = "https://datalab.noirlab.edu/tap/sync"
BOX = 30.0 / 3600.0
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) flare/1.0"}
PS1_COLS = ["objID", "raMean", "decMean", "nDetections",
            "gMeanKronMag", "rMeanKronMag", "iMeanKronMag", "zMeanKronMag",
            "iMeanPSFMag", "rMeanPSFMag"]
GAIA_BOX = 2.0 / 3600.0
WISE_BOX = 6.0 / 3600.0
GAIA_Q = ("SELECT ra, dec, parallax, parallax_error, pmra, pmdec, "
          "pmra_error, pmdec_error, phot_g_mean_mag, ruwe "
          "FROM gaia_dr3.gaia_source "
          "WHERE ra BETWEEN {ralo:.6f} AND {rahi:.6f} "
          "AND dec BETWEEN {declo:.6f} AND {dechi:.6f}")
WISE_Q = ("SELECT ra, dec, w1mpro, w2mpro, w3mpro "
          "FROM allwise.source "
          "WHERE ra BETWEEN {ralo:.6f} AND {rahi:.6f} "
          "AND dec BETWEEN {declo:.6f} AND {dechi:.6f}")
LS_Q = ("SELECT t.ra, t.dec, t.type, t.mag_r, t.mag_z, "
        "p.z_phot_median, p.z_phot_std, p.z_spec "
        "FROM {sur}.tractor t JOIN {sur}.photo_z p ON t.ls_id=p.ls_id "
        "WHERE t.ra BETWEEN {ralo:.6f} AND {rahi:.6f} "
        "AND t.dec BETWEEN {declo:.6f} AND {dechi:.6f}")


def _get(url: str, timeout: int = 60, tries: int = 4) -> Optional[bytes]:
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if a == tries - 1:
                return None
            time.sleep(1.5 * (a + 1))
    return None


def _sep_arcsec(ra0, dec0, ra, dec):
    return 3600.0 * np.hypot((ra - ra0) * np.cos(np.radians(dec0)), dec - dec0)


def _num(row, key):
    """A finite float from a dict/Series row (native catalogue column), else NaN.
    Lets the point-source blocks share one code path for a fetched row and a
    caller-provided (pre-staged) row."""
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        try:
            v = getattr(row, key)
        except AttributeError:
            return np.nan
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _provided_sep(row, ra, dec):
    """Separation for a provided row: its own ``sep`` if present, else from ra/dec."""
    sep = _num(row, "sep")
    if np.isfinite(sep):
        return sep
    rra, rdec = _num(row, "ra"), _num(row, "dec")
    if np.isfinite(rra) and np.isfinite(rdec):
        return float(_sep_arcsec(ra, dec, rra, rdec))
    return np.nan


def host_features(ra: float, dec: float) -> dict:
    """PS1 host block for one position; NaN-filled dict on any failure."""
    out = {k: np.nan for k in ["host_sep", "host_r", "host_i", "host_gr",
                               "host_ri", "host_iz", "host_ext", "n_ext_30",
                               "near_sep", "near_ext"]}
    q = urllib.parse.urlencode(dict(
        ra=f"{ra:.6f}", dec=f"{dec:.6f}", radius=f"{BOX:.6f}",
        **{"nDetections.gte": 3}, pagesize=200))
    raw = _get(f"{MAST}?{q}")
    if raw is None:
        return out
    try:
        j = json.loads(raw.decode())
        recs = j.get("data", [])
        if not recs:
            return out
        # the service ignores any column selection; names come from the info block
        d = (pd.DataFrame(recs, columns=[c["name"] for c in j["info"]])
             if isinstance(recs[0], list) else pd.DataFrame(recs))
    except Exception:
        return out
    for c in PS1_COLS[1:]:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce").replace(-999.0, np.nan)
    if "raMean" not in d or d.raMean.isna().all():
        return out
    d["sep"] = _sep_arcsec(ra, dec, d.raMean.values, d.decMean.values)
    d = d.sort_values("sep")
    d["ext"] = d.iMeanPSFMag - d.iMeanKronMag
    out["near_sep"] = float(d.sep.iloc[0])
    out["near_ext"] = float(d.ext.iloc[0]) if np.isfinite(d.ext.iloc[0]) else np.nan
    gal = d[(d.ext > 0.05) & d.rMeanKronMag.notna()]
    out["n_ext_30"] = float(len(gal))
    if len(gal):
        h = gal.iloc[0]
        out.update(host_sep=float(h.sep), host_r=float(h.rMeanKronMag),
                   host_i=float(h.iMeanKronMag) if np.isfinite(h.iMeanKronMag) else np.nan,
                   host_gr=float(h.gMeanKronMag - h.rMeanKronMag),
                   host_ri=float(h.rMeanKronMag - h.iMeanKronMag),
                   host_iz=float(h.iMeanKronMag - h.zMeanKronMag),
                   host_ext=float(h.ext))
    return out


def lumdist_mpc(z):
    """Flat LCDM comoving-distance integral, H0=70, Om=0.3. Vectorised."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z) & (z > 0) & (z < 5)
    if ok.any():
        c_h0 = 299792.458 / 70.0
        zi = z[ok]
        grid = np.linspace(0, zi.max(), 2048)
        einv = 1.0 / np.sqrt(0.3 * (1 + grid) ** 3 + 0.7)
        cum = np.concatenate([[0], np.cumsum((einv[1:] + einv[:-1]) / 2
                                             * np.diff(grid))])
        out[ok] = np.interp(zi, grid, cum) * c_h0 * (1 + zi)
    return out


def photoz(ra: float, dec: float) -> dict:
    """Nearest-extended-source photo-z from LS DR10, DR9 fallback."""
    out = dict(z_phot=np.nan, z_phot_std=np.nan, pz_sep=np.nan)
    dra = BOX / max(np.cos(np.radians(dec)), 0.05)
    for survey in ("ls_dr10", "ls_dr9"):
        q = LS_Q.format(sur=survey, ralo=ra - dra, rahi=ra + dra,
                        declo=dec - BOX, dechi=dec + BOX)
        raw = _get(TAP + "?" + urllib.parse.urlencode(
            dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q)))
        if raw is None:
            continue
        try:
            d = pd.read_csv(io.StringIO(raw.decode())).replace(-99.0, np.nan)
        except Exception:
            continue
        if not len(d):
            continue
        d = d[d.type != "PSF"]
        if not len(d):
            continue
        d = d.assign(sep=_sep_arcsec(ra, dec, d.ra.values, d.dec.values)) \
             .sort_values("sep")
        h = d.iloc[0]
        if np.isfinite(h.z_phot_median):
            out.update(z_phot=float(h.z_phot_median),
                       z_phot_std=float(h.z_phot_std)
                       if np.isfinite(h.z_phot_std) else np.nan,
                       pz_sep=float(h.sep))
            return out
    return out


def m_pseudo(peak_mag: float, z_phot: float) -> float:
    """Pseudo-absolute peak magnitude from a photometric redshift."""
    d = lumdist_mpc(z_phot)[0]
    if not np.isfinite(d) or not np.isfinite(peak_mag):
        return float("nan")
    return float(peak_mag - 5 * np.log10(d * 1e6 / 10.0))


def _tap_box(q_template: str, ra: float, dec: float, half: float):
    dra = half / max(np.cos(np.radians(dec)), 0.05)
    q = q_template.format(ralo=ra - dra, rahi=ra + dra,
                          declo=dec - half, dechi=dec + half)
    raw = _get(TAP + "?" + urllib.parse.urlencode(
        dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=q)))
    if raw is None:
        return None
    try:
        d = pd.read_csv(io.StringIO(raw.decode()))
    except Exception:
        return None
    if not len(d):
        return d
    return d.assign(sep=_sep_arcsec(ra, dec, d.ra.values, d.dec.values))             .sort_values("sep")


def gaia_features(ra: float, dec: float, row=None) -> dict:
    """Nearest Gaia DR3 source within 2 arcsec; NaN block if none. ``row`` (a
    pre-staged nearest-source dict with Gaia's native columns) skips the TAP query
    for an offline run."""
    out = dict(gaia_sep=np.nan, parallax=np.nan, parallax_over_error=np.nan,
               pm=np.nan, pm_over_error=np.nan, gaia_g=np.nan, ruwe=np.nan)
    if row is None:
        d = _tap_box(GAIA_Q, ra, dec, GAIA_BOX)
        if d is None or not len(d):
            return out
        h = d.iloc[0]
        sep = float(h.sep)
    else:
        h, sep = row, _provided_sep(row, ra, dec)
    plx, plxe = _num(h, "parallax"), _num(h, "parallax_error")
    pm = np.hypot(_num(h, "pmra"), _num(h, "pmdec"))
    pme = np.hypot(_num(h, "pmra_error"), _num(h, "pmdec_error"))
    out.update(
        gaia_sep=sep,
        parallax=plx,
        parallax_over_error=float(plx / plxe) if np.isfinite(plx) and plxe > 0 else np.nan,
        pm=float(pm) if np.isfinite(pm) else np.nan,
        pm_over_error=float(pm / pme) if np.isfinite(pm) and pme > 0 else np.nan,
        gaia_g=_num(h, "phot_g_mean_mag"),
        ruwe=_num(h, "ruwe"))
    return out


def wise_features(ra: float, dec: float, row=None) -> dict:
    """Nearest AllWISE source within 6 arcsec; NaN block if none. ``row`` (a
    pre-staged nearest-source dict with w1mpro/w2mpro/w3mpro) skips the TAP query
    for an offline run."""
    out = dict(wise_sep=np.nan, w1=np.nan, w1w2=np.nan, w2w3=np.nan)
    if row is None:
        d = _tap_box(WISE_Q, ra, dec, WISE_BOX)
        if d is None or not len(d):
            return out
        h = d.iloc[0]
        sep = float(h.sep)
    else:
        h, sep = row, _provided_sep(row, ra, dec)
    w1, w2, w3 = _num(h, "w1mpro"), _num(h, "w2mpro"), _num(h, "w3mpro")
    out.update(
        wise_sep=sep,
        w1=w1,
        w1w2=float(w1 - w2) if np.isfinite(w1) and np.isfinite(w2) else np.nan,
        w2w3=float(w2 - w3) if np.isfinite(w2) and np.isfinite(w3) else np.nan)
    return out


def pos_features(ra: float, dec: float, row=None) -> dict:
    """PS1 pre-outburst counterpart at the position (nearest source in 1.5"). ``row``
    (a pre-staged PS1 nearest-source dict with the *Mean mags) skips the MAST query
    for an offline run."""
    out = dict(pos_sep=np.nan, pos_g=np.nan, pos_r=np.nan, pos_i=np.nan,
               pos_gr=np.nan, pos_ri=np.nan, pos_star=np.nan, pos_ndet=np.nan)
    if row is None:
        q = urllib.parse.urlencode({
            "ra": f"{ra:.6f}", "dec": f"{dec:.6f}",
            "radius": f"{1.5 / 3600.0:.7f}", "nDetections.gte": 2,
            "pagesize": 50})
        raw = _get(MAST + "?" + q)
        if raw is None:
            return out
        try:
            j = json.loads(raw.decode())
        except Exception:
            return out
        recs = j.get("data", [])
        if not recs:
            return out
        if isinstance(recs[0], list):
            d = pd.DataFrame(recs, columns=[c["name"] for c in j["info"]])
        else:
            d = pd.DataFrame(recs)
        for c in d.columns:
            if c != "objID":
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.replace(-999.0, np.nan)
        d = d.assign(sep=_sep_arcsec(ra, dec, d.raMean.values, d.decMean.values)).sort_values("sep")
        h = d.iloc[0]
        sep = float(h.sep)
    else:
        h, sep = row, _provided_sep(row, ra, dec)
    gm, rm, im = _num(h, "gMeanPSFMag"), _num(h, "rMeanPSFMag"), _num(h, "iMeanPSFMag")
    ik = _num(h, "iMeanKronMag")
    star = im - ik if np.isfinite(im) and np.isfinite(ik) else np.nan
    out.update(pos_sep=sep, pos_g=gm, pos_r=rm, pos_i=im,
               pos_gr=float(gm - rm) if np.isfinite(gm) and np.isfinite(rm) else np.nan,
               pos_ri=float(rm - im) if np.isfinite(rm) and np.isfinite(im) else np.nan,
               pos_star=float(star < 0.05) if np.isfinite(star) else np.nan,
               pos_ndet=_num(h, "nDetections"))
    return out


def context_features(ra: float, dec: float,
                     peak_mag: Optional[float] = None,
                     provided: Optional[dict] = None) -> dict:
    """The full external block for one alert: host + M_pseudo + point source.

    Live by default (queries MAST/Data Lab). Pass ``provided`` to run fully
    OFFLINE from pre-staged cross-matches (e.g. BOOM), making no network calls -- a
    dict with any of these keys:
      - ``gaia`` / ``wise`` / ``pos``: the nearest-source row (native catalogue
        columns: parallax/pmra/phot_g_mean_mag/ruwe; w1mpro/w2mpro/w3mpro;
        gMeanPSFMag/.../iMeanKronMag/nDetections) -> computed into features here.
      - ``host`` / ``photoz``: the aggregated blocks precomputed by the caller
        (the 30" PS1 host and LS photo-z can't be rebuilt from a single row), as a
        dict of the ``host_*`` / ``z_phot*`` feature values.
      - ``z_phot``: host photo-z for M_pseudo when ``photoz`` isn't supplied.
    A missing key yields that block's NaN values -- never a network call."""
    if provided is None:  # live (unchanged default)
        out = host_features(ra, dec)
        pz = photoz(ra, dec)
        out["M_pseudo"] = m_pseudo(peak_mag, pz["z_phot"]) if peak_mag is not None else float("nan")
        out.update(gaia_features(ra, dec))
        out.update(wise_features(ra, dec))
        out.update(pos_features(ra, dec))
        return out

    # Offline: compute each block from the staged rows, no network.
    out = {k: np.nan for k in ["host_sep", "host_r", "host_i", "host_gr", "host_ri",
                               "host_iz", "host_ext", "n_ext_30", "near_sep", "near_ext"]}
    out.update(provided.get("host") or {})
    pz = provided.get("photoz") or {}
    out.setdefault("z_phot", pz.get("z_phot", provided.get("z_phot", np.nan)))
    for k in ("z_phot_std", "pz_sep"):
        if k in pz:
            out[k] = pz[k]
    out["M_pseudo"] = m_pseudo(peak_mag, out.get("z_phot")) if peak_mag is not None else float("nan")
    out.update(gaia_features(ra, dec, row=provided.get("gaia")))
    out.update(wise_features(ra, dec, row=provided.get("wise")))
    out.update(pos_features(ra, dec, row=provided.get("pos")))
    return out
