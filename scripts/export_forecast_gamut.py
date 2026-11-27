"""Train, calibrate, export, and verify the GAMUT forecast bundle on ZTF BTS.

Follows FLARE's cross-conformal export convention (scripts/train_early_cc.py,
scripts/export_venn_abers.py, scripts/deployed_check.py):
  * Training and calibration use only benchmark/splits/{train,val}.csv (8,922
    objects); benchmark/splits/test.csv (1,575 objects) is untouched until the
    post-export deployed check.
  * Five stratified folds (seed 42) over train+val produce out-of-fold phase
    classifier probabilities and hypothesis-conditional quantile forecasts
    (mu, q16, q84 with one-hot class inputs hyp_0..hyp_5).
  * Calibration exports both modes per horizon h in {5, 10, 15, 20, 30}:
      - 'crossfit' (default): leave-one-object-out smoothed standardized
        residual densities, temperature scaling, and dual fixed-point weights
        with object-level conformal risk-control thresholds over train+val OOF;
      - 'split': nuisances and dual weights fitted on stratified half A, with
        per-class object-level thresholds certified on disjoint half B.
  * Final shipped C1 boosters are fitted on train+val (85/15 fit/early-stop)
    and verified with SHA-256 manifests and an end-to-end check on test.csv.

Run:
    python scripts/export_forecast_gamut.py --out models/bts6/forecast_gamut --threads 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold, train_test_split

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "scripts"))

from train_early_cc import hierarchy_nocal  # noqa: E402

from flare.analysis import FlareAnalysis, feature_frame, observed_events  # noqa: E402
from flare.config import GW_COLS, HOST_COLS, POS_COLS  # noqa: E402
from flare.data import COL, load_events  # noqa: E402
from flare.forecast_gamut import (  # noqa: E402
    CLASSES,
    EGRID,
    ESTEP,
    FLARE_GRID,
    LEAD_EDGES,
    SCALE_FLOOR,
    SUPPORTED_HORIZONS,
    GamutForecaster,
    apply_temperature,
)

CLF_HORIZON = {5: 5, 10: 10, 15: 20, 20: 20, 30: 20}
FORECAST_PARAMS = dict(
    objective="regression_l1",
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=100,
    feature_fraction=0.6,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=0.0,
    verbosity=-1,
    seed=42,
)
QUANTILES = {"q16": 0.16, "q84": 0.84}
T_MAX = 90.0
_G = {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dumps_compact(obj: dict) -> str:
    raw = json.dumps(obj, indent=2, allow_nan=False)
    compact = re.sub(
        r"\[\s+((?:-?\d+(?:\.\d+)?(?:e[+-]?\d+)?,\s*)*-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s+\]",
        lambda m: "[" + re.sub(r"\s+", " ", m.group(1)).strip() + "]",
        raw,
    )
    return compact + "\n"


def context_table(bench_dir: Path, ids) -> pd.DataFrame:
    out = pd.DataFrame({"obj_id": ids})
    for filename, cols in [
        ("hosts.csv", HOST_COLS),
        ("context_gaia_wise.csv", GW_COLS),
        ("ps1_position.csv", POS_COLS),
        ("hosts_photoz.csv", ["z_phot"]),
    ]:
        src = pd.read_csv(bench_dir / filename).drop_duplicates("obj_id")
        out = out.merge(src[["obj_id"] + cols], how="left", on="obj_id", validate="one_to_one")
    return out.set_index("obj_id")


def _init_worker(bench_str: str, ctx: pd.DataFrame):
    _G.update(bench=Path(bench_str), ctx=ctx)


def _extract_object(job):
    i, fp, oid = job
    arr = load_events(str(_G["bench"] / fp))
    t, band, lf = arr[:, COL["dt"]], arr[:, COL["band_id"]], arr[:, COL["logflux"]]
    mag = 23.9 - 2.5 * lf
    ctx = {k: (np.nan if pd.isna(v) else float(v)) for k, v in _G["ctx"].loc[oid].items()}
    out = {}
    for h in SUPPORTED_HORIZONS:
        try:
            pre = observed_events(arr, float(h))
        except ValueError:
            out[h] = None
            continue
        f, peak = feature_frame(pre, ctx)
        fut = (t > h) & (t <= T_MAX) & np.isin(band, (0, 1)) & np.isfinite(mag)
        out[h] = (
            f.iloc[0].to_dict(),
            float(peak),
            np.column_stack([np.full(fut.sum(), i), t[fut], band[fut], mag[fut] - peak]),
        )
    return i, out


def ensure_cache(bench_dir: Path, cache_dir: Path, workers: int) -> None:
    if (cache_dir / "meta.parquet").exists() and all(
        (cache_dir / f"X_h{h}.parquet").exists() and (cache_dir / f"F_h{h}.parquet").exists()
        for h in SUPPORTED_HORIZONS
    ):
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    pool = pd.concat(
        [pd.read_csv(bench_dir / "splits" / f"{s}.csv").assign(split=s) for s in ("train", "val", "test")],
        ignore_index=True,
    )
    pool = pool[pool.cls.isin(CLASSES)].reset_index(drop=True)
    y = np.array([CLASSES.index(c) for c in pool.cls])
    ctx = context_table(bench_dir, pool.obj_id.values)
    jobs = [(i, r.filepath, r.obj_id) for i, r in enumerate(pool.itertuples())]
    with Pool(workers, initializer=_init_worker, initargs=(str(bench_dir), ctx)) as p:
        res = dict(p.imap_unordered(_extract_object, jobs, chunksize=32))
    meta = pd.DataFrame(
        dict(
            obj_id=pool.obj_id,
            filepath=pool.filepath,
            cls=pool.cls,
            y=y,
            split=pool.split,
            bts_type=pool.bts_type,
        )
    )
    meta.to_parquet(cache_dir / "meta.parquet")
    for h in SUPPORTED_HORIZONS:
        rows, peaks, f_list, keep = [], [], [], np.zeros(len(pool), bool)
        for i in range(len(pool)):
            r = res[i][h]
            if r is None:
                rows.append({})
                peaks.append(np.nan)
                continue
            rows.append(r[0])
            peaks.append(r[1])
            f_list.append(r[2])
            keep[i] = True
        x_df = pd.DataFrame(rows)
        x_df.insert(0, "peak_mag", peaks)
        x_df.insert(0, "has_prefix", keep)
        x_df.to_parquet(cache_dir / f"X_h{h}.parquet")
        f_df = pd.DataFrame(np.concatenate(f_list), columns=["obj", "t", "band", "dmag"]).astype(
            dict(obj=int, band=int)
        )
        f_df["lead"] = f_df.t - h
        f_df.to_parquet(cache_dir / f"F_h{h}.parquet")


def cap_rows(rows: np.ndarray, f_obj: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if cap <= 0:
        return rows
    rng = np.random.default_rng(seed)
    keep = []
    obj = f_obj[rows]
    order = np.argsort(obj, kind="stable")
    rows, obj = rows[order], obj[order]
    starts = np.r_[0, np.flatnonzero(np.diff(obj)) + 1, len(obj)]
    for s0, s1 in zip(starts[:-1], starts[1:], strict=False):
        r = rows[s0:s1]
        keep.append(r if len(r) <= cap else np.sort(rng.choice(r, cap, replace=False)))
    return np.concatenate(keep)


def long_frame(x_df: pd.DataFrame, feat_cols: list[str], f_df: pd.DataFrame, idx_rows: np.ndarray) -> pd.DataFrame:
    sub = f_df.iloc[idx_rows]
    d = x_df.iloc[sub.obj.to_numpy()][feat_cols].reset_index(drop=True)
    d["fc_day"] = sub.t.to_numpy()
    d["fc_band"] = sub.band.to_numpy().astype(float)
    d["fc_lead"] = sub.lead.to_numpy()
    return d


def with_class(d_df: pd.DataFrame, cls_idx: np.ndarray, k_cls: int = 6) -> pd.DataFrame:
    out = d_df.copy()
    for c in range(k_cls):
        out[f"hyp_{c}"] = (cls_idx == c).astype(float)
    return out


def fit_lgb(params: dict, x_tr: pd.DataFrame, y_tr: np.ndarray, x_es: pd.DataFrame, y_es: np.ndarray, threads: int):
    p = dict(params, num_threads=threads)
    d = lgb.Dataset(x_tr, y_tr, free_raw_data=True)
    e = lgb.Dataset(x_es, y_es, reference=d)
    return lgb.train(p, d, 4000, valid_sets=[e], callbacks=[lgb.early_stopping(150, verbose=False)])


def fit_temperature(probs: np.ndarray, z: np.ndarray) -> float:
    lp = np.log(np.maximum(probs, 1e-8))

    def nll(log_t):
        s = lp / np.exp(log_t)
        s = s - s.max(axis=1, keepdims=True)
        return float(-(s[np.arange(len(z)), z] - np.log(np.exp(s).sum(axis=1))).mean())

    return float(np.exp(minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded").x))


def residual_density_parts(e: np.ndarray, bw_mult: float = 2.0):
    e = np.clip(e[np.isfinite(e)], EGRID[0], EGRID[-1])
    if len(e) < 5:
        e_pad = np.r_[e, np.random.default_rng(0).normal(0.0, 1.0, 50)]
        d, _, _, _ = residual_density_parts(e_pad, bw_mult=bw_mult)
        return d, None, None, len(e)
    hist = np.bincount(
        np.clip(np.round((e - EGRID[0]) / ESTEP).astype(int), 0, len(EGRID) - 1),
        minlength=len(EGRID),
    ).astype(float)
    iqr_sd = (np.quantile(e, 0.75) - np.quantile(e, 0.25)) / 1.349
    sd = min(float(np.std(e)), float(iqr_sd)) or float(np.std(e)) or 1.0
    bw = max(bw_mult * 1.06 * sd * (len(e) ** -0.2), 2.0 * ESTEP)
    k = np.arange(-int(4.0 * bw / ESTEP), int(4.0 * bw / ESTEP) + 1) * ESTEP
    ker = np.exp(-0.5 * (k / bw) ** 2)
    ker /= ker.sum()
    conv = np.convolve(hist, ker, mode="same")
    norm = float(conv.sum() * ESTEP)
    return np.maximum(conv / norm, 1e-9), ker, norm, len(e)


def class_thresholds(scores: np.ndarray, z: np.ndarray, obj: np.ndarray, k_cls: int, alpha: float) -> np.ndarray:
    t = np.full(k_cls, np.nan)
    for c in range(k_cls):
        m = z == c
        if not m.any():
            continue
        sc = scores[m]
        oc = obj[m]
        uo, inv, cnt = np.unique(oc, return_inverse=True, return_counts=True)
        n = len(uo)
        w = 1.0 / cnt[inv]
        budget = alpha * (n + 1) - 1.0
        order = np.argsort(sc)
        cw = np.cumsum(w[order])
        k = int(np.searchsorted(cw, budget, side="right"))
        t[c] = float(sc[order[k]]) if k < len(sc) else np.inf
        if budget < 0:
            t[c] = -np.inf
    return t


def horizon_cells(lead_bin_cal: np.ndarray, obj_cal: np.ndarray, z_cal: np.ndarray, k_cls: int, nb: int, n_target: int):
    groups = [[b] for b in range(nb)]
    tot = np.array([len(np.unique(obj_cal[z_cal == c])) for c in range(k_cls)])
    big = np.flatnonzero(tot >= n_target)

    def counts(g):
        m = np.isin(lead_bin_cal, g)
        return np.array([len(np.unique(obj_cal[m & (z_cal == c)])) for c in big])

    while len(groups) > 1:
        mins = [int(counts(g).min()) if len(big) else n_target for g in groups]
        i = int(np.argmin(mins))
        if mins[i] >= n_target:
            break
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if mins[i - 1] <= mins[i + 1] else i + 1
        a_idx, b_idx = sorted((i, j))
        groups[a_idx] = groups[a_idx] + groups[b_idx]
        del groups[b_idx]
    gid = np.zeros(nb, dtype=int)
    for k, g in enumerate(groups):
        gid[g] = k
    return gid


def pack_densities(dens: np.ndarray, floor: float = 1e-9) -> list[dict]:
    packed = []
    k_cls, n_bins, _ = dens.shape
    for c in range(k_cls):
        for b in range(n_bins):
            d = dens[c, b]
            active = d > floor * 1.0001
            padded = np.r_[False, active, False]
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            segments = [
                [int(s), np.round(d[s:e], 7).tolist()]
                for s, e in zip(edges[0::2], edges[1::2], strict=False)
            ]
            packed.append(dict(class_index=c, lead_bin=b, floor=floor, segments=segments))
    return packed


def fit_gamut_mode(
    probs_obj: np.ndarray,
    y_obj: np.ndarray,
    obj_rows: np.ndarray,
    yt_rows: np.ndarray,
    lead_bin_rows: np.ndarray,
    hmu_rows: np.ndarray,
    hsig_rows: np.ndarray,
    cal_rows: np.ndarray,
    nb: int = 10,
    crc: bool = False,
    bw_mult: float = 2.0,
    min_bin: int = 150,
    n_iter: int = 100,
    seed: int = 0,
) -> dict:
    k_cls = len(CLASSES)
    cal_objs = np.unique(obj_rows[cal_rows])
    if crc:
        strat = y_obj[cal_objs] if np.bincount(y_obj[cal_objs], minlength=k_cls).min() >= 2 else None
        a_objs, b_objs = train_test_split(cal_objs, test_size=0.5, stratify=strat, random_state=seed)
        a_objs, b_objs = np.sort(a_objs), np.sort(b_objs)
        in_a = np.isin(obj_rows[cal_rows], a_objs)
        cal_a, cal_b = cal_rows[in_a], cal_rows[~in_a]
    else:
        a_objs = cal_objs
        cal_a, cal_b = cal_rows, cal_rows

    pi = np.bincount(y_obj[a_objs], minlength=k_cls) / len(a_objs)
    temp = fit_temperature(probs_obj[a_objs], y_obj[a_objs])
    post_all = apply_temperature(probs_obj, temp)

    z_a = y_obj[obj_rows[cal_a]]
    b_a_raw = lead_bin_rows[cal_a]
    dens = np.zeros((k_cls, nb, len(EGRID)), dtype=float)
    src = {}
    e_a = (yt_rows[cal_a][:, None] - hmu_rows[cal_a]) / hsig_rows[cal_a]
    for c in range(k_cls):
        mc = z_a == c
        d_all, k_all, n_all, c_all = residual_density_parts(e_a[mc, c], bw_mult=bw_mult)
        for b in range(nb):
            mb = mc & (b_a_raw == b)
            if mb.sum() >= min_bin:
                d_b, k_b, n_b, c_b = residual_density_parts(e_a[mb, c], bw_mult=bw_mult)
                dens[c, b] = d_b
                src[(c, b)] = (mb, k_b, n_b, c_b)
            else:
                dens[c, b] = d_all
                src[(c, b)] = (mc, k_all, n_all, c_all)

    # Leave-one-object-out own-class residual density on cal_a
    idx_own = np.clip(
        np.round((np.clip(e_a[np.arange(len(cal_a)), z_a], EGRID[0], EGRID[-1]) - EGRID[0]) / ESTEP).astype(int),
        0,
        len(EGRID) - 1,
    )
    loo_own = dens[z_a, b_a_raw, idx_own].copy()
    obj_a = obj_rows[cal_a]
    for (c, b), (sample, ker, norm, n_pts) in src.items():
        if ker is None:
            continue
        ev = np.flatnonzero((z_a == c) & (b_a_raw == b))
        if not len(ev):
            continue
        half = len(ker) // 2
        smp = np.flatnonzero(sample)
        so = obj_a[smp]
        order = np.argsort(so, kind="stable")
        smp, so = smp[order], so[order]
        starts = np.searchsorted(so, obj_a[ev], "left")
        ends = np.searchsorted(so, obj_a[ev], "right")
        for i_ev, r_ev in enumerate(ev):
            js = smp[starts[i_ev] : ends[i_ev]]
            dd = idx_own[r_ev] - idx_own[js] + half
            own = ker[dd[(dd >= 0) & (dd < len(ker))]].sum()
            m_cnt = len(js)
            if n_pts - m_cnt > 0:
                loo_own[r_ev] = max((dens[c, b, idx_own[r_ev]] * norm - own) / (norm * (n_pts - m_cnt) / n_pts), 1e-9)
    loo_own[(e_a[np.arange(len(cal_a)), z_a] < EGRID[0]) | (e_a[np.arange(len(cal_a)), z_a] > EGRID[-1])] = 1e-12

    def row_terms(rows):
        pr = post_all[obj_rows[rows]]
        e = (yt_rows[rows][:, None] - hmu_rows[rows]) / hsig_rows[rows]
        idx = np.clip(np.round((e - EGRID[0]) / ESTEP).astype(int), 0, len(EGRID) - 1)
        fv = dens[np.arange(k_cls)[None, :], lead_bin_rows[rows][:, None], idx] / hsig_rows[rows]
        fv[(e < EGRID[0]) | (e > EGRID[-1])] = 1e-12
        sb = (pr * hsig_rows[rows]).sum(axis=1)
        return pr * fv * sb[:, None]

    terms_a = row_terms(cal_a)
    pr_a = post_all[obj_rows[cal_a]]
    sb_a = (pr_a * hsig_rows[cal_a]).sum(axis=1)
    terms_a[np.arange(len(cal_a)), z_a] = pr_a[np.arange(len(cal_a)), z_a] * loo_own / hsig_rows[cal_a, z_a] * sb_a
    terms_b = terms_a if cal_b is cal_a else row_terms(cal_b)

    levels_out = {}
    for level_str, alpha in (("0.68", 0.32), ("0.95", 0.05)):
        n_big = int(np.ceil(2.0 / alpha))
        n_min = int(np.ceil(1.0 / alpha - 1.0 - 1e-9))
        gid = horizon_cells(lead_bin_rows[cal_a], obj_rows[cal_a], z_a, k_cls, nb, n_big)
        cell_a = gid[lead_bin_rows[cal_a]]
        cell_b = gid[lead_bin_rows[cal_b]]
        n_cells = int(gid.max()) + 1
        v_mat = np.ones((n_cells, k_cls), dtype=float)
        t_vec = np.full(n_cells, np.nan, dtype=float)
        excluded = []
        for cell_id in range(n_cells):
            ma = cell_a == cell_id
            mb = cell_b == cell_id
            if not mb.any():
                continue
            zb_a = y_obj[obj_rows[cal_a[ma]]]
            nobj_a = np.array([len(np.unique(obj_rows[cal_a[ma]][zb_a == c])) for c in range(k_cls)])
            keep = nobj_a >= n_min
            v = 1.0 / np.maximum(pi, 1e-6)
            for _ in range(n_iter):
                tau = class_thresholds(terms_a[ma] @ v, zb_a, obj_rows[cal_a[ma]], k_cls, alpha)
                ok = keep & np.isfinite(tau) & (tau > 0)
                if not ok.any():
                    break
                ref = float(np.exp(np.mean(np.log(tau[ok]))))
                upd = np.where(ok, (ref / np.where(ok, tau, ref)) ** 0.5, 1.0)
                v = v * np.clip(upd, 0.5, 2.0)
                v = v / float(np.exp(np.mean(np.log(v))))
                if np.max(np.abs(np.log(tau[ok] / ref))) < 0.005:
                    break
            zb_b = y_obj[obj_rows[cal_b[mb]]]
            t_b = class_thresholds(terms_b[mb] @ v, zb_b, obj_rows[cal_b[mb]], k_cls, alpha)
            nobj_b = np.array([len(np.unique(obj_rows[cal_b[mb]][zb_b == c])) for c in range(k_cls)])
            ok_b = (nobj_b >= n_min) & np.isfinite(t_b)
            for c in range(k_cls):
                if 0 < nobj_b[c] < n_min:
                    excluded.append(dict(cell=int(cell_id), cls=CLASSES[c], n_objects=int(nobj_b[c])))
            t_val = float(np.min(t_b[ok_b])) if ok_b.any() else float(np.nanmin(t_b))
            v_mat[cell_id] = v
            t_vec[cell_id] = t_val
        if np.isnan(t_vec).any():
            good = np.flatnonzero(~np.isnan(t_vec))
            for bad in np.flatnonzero(np.isnan(t_vec)):
                j = int(good[np.argmin(np.abs(good - bad))])
                v_mat[bad] = v_mat[j]
                t_vec[bad] = t_vec[j]
        levels_out[level_str] = dict(
            alpha=alpha,
            cell_map=gid.tolist(),
            dual_weights=np.round(v_mat, 7).tolist(),
            thresholds=np.round(t_vec, 7).tolist(),
            min_objects_for_threshold=n_min,
            excluded_class_cells=excluded,
        )

    note = (
        "GAMUT dual-weighted posterior-mixture level sets calibrated on five-fold out-of-fold predictions "
        f"over BTS train+val ({len(cal_objs):,} objects; test split untouched). "
        + (
            "Split mode fits nuisances and dual weights on half A and per-class object-level thresholds on half B; "
            if crc
            else "Crossfit mode fits leave-one-object-out residual laws, dual weights and thresholds on all OOF objects; "
        )
        + "validity of the shipped bundle is checked empirically out of fold and on the untouched test split."
    )
    return dict(
        temperature=round(temp, 6),
        n_calibration_objects=int(len(cal_objs)),
        residual_densities=pack_densities(dens),
        levels=levels_out,
        note=note,
    )


def evaluate_lofo_oof(
    probs_oof: np.ndarray,
    y_pool: np.ndarray,
    f_pool: pd.DataFrame,
    hmu_oof: np.ndarray,
    hsig_oof: np.ndarray,
    folds: list,
) -> dict:
    fobj = f_pool.obj.to_numpy(dtype=int)
    yt = f_pool.dmag.to_numpy(dtype=float)
    lb = np.searchsorted(LEAD_EDGES, f_pool.t.to_numpy(dtype=float), side="right")
    valid = np.isfinite(yt) & np.all(np.isfinite(hmu_oof), axis=1) & np.all(np.isfinite(hsig_oof), axis=1)
    out = {}
    for mode_name, crc_flag in (("gamut_crossfit", False), ("gamut_split", True)):
        cov = {"0.68": np.zeros(len(f_pool), dtype=bool), "0.95": np.zeros(len(f_pool), dtype=bool)}
        for tr_o, te_o in folds:
            cal_r = np.flatnonzero(valid & np.isin(fobj, tr_o))
            te_r = np.flatnonzero(valid & np.isin(fobj, te_o))
            cfg = fit_gamut_mode(probs_oof, y_pool, fobj, yt, lb, hmu_oof, hsig_oof, cal_r, nb=10, crc=crc_flag)
            from flare.forecast_gamut import unpack_densities

            dens = unpack_densities(cfg["residual_densities"], len(CLASSES), 10)
            pr = apply_temperature(probs_oof[fobj[te_r]], float(cfg["temperature"]))
            e = (yt[te_r][:, None] - hmu_oof[te_r]) / hsig_oof[te_r]
            idx = np.clip(np.round((e - EGRID[0]) / ESTEP).astype(int), 0, len(EGRID) - 1)
            fv = dens[np.arange(len(CLASSES))[None, :], lb[te_r][:, None], idx] / hsig_oof[te_r]
            fv[(e < EGRID[0]) | (e > EGRID[-1])] = 1e-12
            sb = (pr * hsig_oof[te_r]).sum(axis=1)
            terms_te = pr * fv * sb[:, None]
            for lvl_key in ("0.68", "0.95"):
                lvl = cfg["levels"][lvl_key]
                cm = np.asarray(lvl["cell_map"], dtype=int)
                wv = np.asarray(lvl["dual_weights"], dtype=float)
                thr = np.asarray(lvl["thresholds"], dtype=float)
                cells = cm[lb[te_r]]
                cov[lvl_key][te_r] = (terms_te * wv[cells]).sum(axis=1) >= thr[cells]
        ev = np.flatnonzero(valid)
        yc = y_pool[fobj[ev]]
        uo, inv = np.unique(fobj[ev], return_inverse=True)
        yc_obj = y_pool[uo]
        mode_res = dict(
            n_objects=int(len(uo)),
            n_points=int(len(ev)),
            n_objects_per_class={CLASSES[c]: int((yc_obj == c).sum()) for c in range(len(CLASSES))},
        )
        for lvl_key, tag in (("0.68", "68"), ("0.95", "95")):
            cv = cov[lvl_key][ev]
            c_obj = np.bincount(inv, cv.astype(float)) / np.bincount(inv)
            per_pt = {CLASSES[c]: round(float(cv[yc == c].mean()), 3) for c in range(len(CLASSES))}
            per_ob = {CLASSES[c]: round(float(c_obj[yc_obj == c].mean()), 3) for c in range(len(CLASSES))}
            mode_res[f"marginal_{tag}"] = round(float(cv.mean()), 4)
            mode_res[f"worst_class_{tag}"] = round(min(per_pt.values()), 3)
            mode_res[f"worst_class_object_{tag}"] = round(min(per_ob.values()), 3)
            mode_res[f"per_class_{tag}"] = per_pt
            mode_res[f"per_class_object_{tag}"] = per_ob
        out[mode_name] = mode_res
    return out


def evaluate_test_split(
    engine: FlareAnalysis,
    gamut_fc: GamutForecaster,
    bench_dir: Path,
    meta_df: pd.DataFrame,
    x_df: pd.DataFrame,
    f_df: pd.DataFrame,
    ctx_df: pd.DataFrame,
    horizon: int,
) -> dict:
    from flare.forecast_gamut import adaptive_grid, extract_components

    te_idx = np.flatnonzero(meta_df.split.to_numpy() == "test")
    pos_of_obj = {int(o): idx for idx, o in enumerate(te_idx)}
    clf = engine._gamut_classifier(gamut_fc.classifier_horizon(horizon))
    meta_grid = engine._metadata("forecast", horizon)
    grid_days = np.asarray(meta_grid["grid_days"], dtype=float)
    mond = meta_grid["conformal"].get("mondrian")
    mult_map = meta_grid["conformal"]["multipliers"]
    anchor_hw = meta_grid["conformal"]["anchored_halfwidth_at_horizon"]

    x_te_all = x_df.iloc[te_idx]
    probs_te_all = clf.predict_proba(x_te_all.reindex(columns=clf.feature_names).astype(float))
    grid_clf = engine._classifier_at(horizon)
    probs_grid_clf = grid_clf.predict_proba(x_te_all.reindex(columns=grid_clf.feature_names).astype(float))

    # End-to-end FlareAnalysis parity check on 12 test objects (including rare classes)
    gamut_single = GamutForecaster(gamut_fc.directory, num_threads=1)
    sample_objs = []
    for c in range(len(CLASSES)):
        cand = te_idx[meta_df.y.to_numpy()[te_idx] == c]
        if len(cand):
            sample_objs.extend(cand[:2].tolist())
    for i in sample_objs:
        row = meta_df.iloc[i]
        arr = load_events(str(bench_dir / row.filepath))
        try:
            pre = observed_events(arr, float(horizon))
        except ValueError:
            continue
        ctx = {k: (np.nan if pd.isna(v) else float(v)) for k, v in ctx_df.loc[row.obj_id].items()}
        e2e = engine.forecast(arr, horizon, context=ctx, recipe="gamut")
        direct = gamut_single.predict(
            pre, x_df.iloc[[i]], probs_te_all[pos_of_obj[int(i)]], float(x_df["peak_mag"].iloc[i]), horizon
        )
        for b_name in ("g", "r"):
            if e2e["bands"][b_name]["status"] == "ok":
                np.testing.assert_allclose(
                    e2e["bands"][b_name]["lower68"], direct["bands"][b_name]["lower68"], rtol=0, atol=1e-4
                )

    # Filter test future points to anchored (object, band) pairs (identical eligibility across methods)
    last_mag_te = np.full((len(te_idx), 2), np.nan)
    for pos_i, i in enumerate(te_idx):
        arr = load_events(str(bench_dir / meta_df.filepath.iloc[i]))
        try:
            pre = observed_events(arr, float(horizon))
        except ValueError:
            continue
        for b_id in (0, 1):
            in_b = pre[pre[:, COL["band_id"]] == b_id]
            if len(in_b):
                last_mag_te[pos_i, b_id] = float(23.9 - 2.5 * in_b[-1, COL["logflux"]])

    f_te = f_df[np.isin(f_df.obj.to_numpy(), te_idx)].reset_index(drop=True)
    te_pos_rows = np.array([pos_of_obj[int(o)] for o in f_te.obj.to_numpy()], dtype=int)
    anchored_mask = np.isfinite(last_mag_te[te_pos_rows, f_te.band.to_numpy(dtype=int)])
    f_te = f_te.iloc[anchored_mask].reset_index(drop=True)
    te_pos_rows = te_pos_rows[anchored_mask]

    # 1) Batched shipped grid forecast across all test objects
    grid_frames = [
        x_te_all.reindex(columns=meta_grid["feature_names"]).astype(float).assign(fc_day=float(d), fc_band=float(b))
        for b in (0, 1)
        for d in grid_days
    ]
    xf_grid_all = pd.concat(grid_frames, ignore_index=True)
    p_grid_all = {
        k: engine._predict("forecast", meta_grid["models"][k], xf_grid_all)
        .reshape(2, len(grid_days), len(te_idx))
        .transpose(2, 0, 1)
        for k in ("pooled", "q16", "q84")
    }
    sc_grid_all = np.maximum(0.5 * np.abs(p_grid_all["q84"] - p_grid_all["q16"]), SCALE_FLOOR)

    records = {m: [] for m in ("grid", "gamut_crossfit", "gamut_split")}
    peaks_te = x_te_all["peak_mag"].to_numpy(dtype=float)
    y_cls_te = meta_df.y.to_numpy()[te_idx]

    for pos_i in range(len(te_idx)):
        pred_cls = CLASSES[int(probs_grid_clf[pos_i].argmax())]
        if mond and pred_cls in mond["classes"]:
            m68 = np.asarray(mond["classes"][pred_cls]["m68"], dtype=float)
            m95 = np.asarray(mond["classes"][pred_cls]["m95"], dtype=float)
        else:
            m68 = np.full((2, len(grid_days)), mult_map["pooled"]["m68"])
            m95 = np.full((2, len(grid_days)), mult_map["pooled"]["m95"])
        rows_i = np.flatnonzero(te_pos_rows == pos_i)
        if not len(rows_i):
            continue
        for b_id in (0, 1):
            rb = rows_i[f_te.band.to_numpy()[rows_i] == b_id]
            if not len(rb):
                continue
            last_m = last_mag_te[pos_i, b_id]
            t_pts = f_te.t.to_numpy(dtype=float)[rb]
            y_true = f_te.dmag.to_numpy(dtype=float)[rb] + peaks_te[pos_i]
            g_days = np.r_[float(horizon), grid_days]
            g_mean = np.r_[last_m, p_grid_all["pooled"][pos_i, b_id] + peaks_te[pos_i]]
            g_hw68 = np.r_[anchor_hw["hw68"], m68[b_id] * sc_grid_all[pos_i, b_id]]
            g_hw95 = np.r_[anchor_hw["hw95"], m95[b_id] * sc_grid_all[pos_i, b_id]]
            g_mag = np.interp(t_pts, g_days, g_mean)
            g_lo68 = np.interp(t_pts, g_days, g_mean - g_hw68)
            g_hi68 = np.interp(t_pts, g_days, g_mean + g_hw68)
            g_lo95 = np.interp(t_pts, g_days, g_mean - g_hw95)
            g_hi95 = np.interp(t_pts, g_days, g_mean + g_hw95)
            for k_pt in range(len(t_pts)):
                records["grid"].append(
                    (
                        int(y_cls_te[pos_i]),
                        int(te_idx[pos_i]),
                        abs(y_true[k_pt] - g_mag[k_pt]),
                        bool(g_lo68[k_pt] <= y_true[k_pt] <= g_hi68[k_pt]),
                        float(g_hi68[k_pt] - g_lo68[k_pt]),
                        bool(g_lo95[k_pt] <= y_true[k_pt] <= g_hi95[k_pt]),
                        float(g_hi95[k_pt] - g_lo95[k_pt]),
                    )
                )

    # 2) Batched GAMUT C1 predictions across all test future points
    meta_g = gamut_fc.metadata(horizon)
    feat_names = meta_g["feature_names"]
    boosters_g = {name: gamut_fc._booster(cfg) for name, cfg in meta_g["models"].items()}
    base_te = long_frame(x_df, feat_names, f_te, np.arange(len(f_te))).astype(float)
    stacked_te = pd.concat(
        [with_class(base_te, np.full(len(f_te), c)) for c in range(len(CLASSES))],
        ignore_index=True,
    )
    preds_g = {
        name: b.predict(stacked_te, num_threads=engine.num_threads).reshape(len(CLASSES), len(f_te)).T
        for name, b in boosters_g.items()
    }
    hmu_te = preds_g["mu"]
    hsig_te = np.maximum(0.5 * np.abs(preds_g["q84"] - preds_g["q16"]), SCALE_FLOOR)
    lead_bins_te = np.searchsorted(LEAD_EDGES, f_te.t.to_numpy(dtype=float), side="right")
    yt_dmag = f_te.dmag.to_numpy(dtype=float)
    obj_ids_rows = f_te.obj.to_numpy(dtype=int)
    y_cls_rows = meta_df.y.to_numpy()[obj_ids_rows]

    for mode_name, rec_key in (("crossfit", "gamut_crossfit"), ("split", "gamut_split")):
        cal_b = meta_g["calibration_modes"][mode_name]
        dens = cal_b["_dens"]
        phat_obj = apply_temperature(probs_te_all, float(cal_b["temperature"]))
        phat_rows = phat_obj[te_pos_rows]
        act_rows = phat_rows >= float(meta_g.get("prune_threshold", 1e-3))
        act_rows[np.arange(len(f_te)), phat_rows.argmax(axis=1)] = True

        def _eval_score_chunk(lvl_cfg, dens_arr, pr_c, hm_c, hs_c, lb_c, y_g):
            cell_map = np.asarray(lvl_cfg["cell_map"], dtype=int)
            wv = np.asarray(lvl_cfg["dual_weights"], dtype=float)
            thr = np.asarray(lvl_cfg["thresholds"], dtype=float)
            cells = cell_map[lb_c]
            sbar = (hs_c * pr_c).sum(axis=1)
            sc = np.zeros_like(y_g)
            for c in range(len(CLASSES)):
                e = (y_g - hm_c[:, c][:, None]) / hs_c[:, c][:, None]
                idx = np.clip(np.round((e - EGRID[0]) / ESTEP).astype(int), 0, len(EGRID) - 1)
                fc = dens_arr[c][lb_c[:, None], idx] / hs_c[:, c][:, None]
                fc[(e < EGRID[0]) | (e > EGRID[-1])] = 0.0
                sc += wv[cells, c][:, None] * pr_c[:, c][:, None] * fc
            sc *= sbar[:, None]
            ins = sc >= thr[cells][:, None]
            emp = ~ins.any(axis=1)
            if emp.any():
                ins[np.flatnonzero(emp), sc[emp].argmax(axis=1)] = True
            return sc, ins

        for s0 in range(0, len(f_te), 2000):
            s1 = min(s0 + 2000, len(f_te))
            sl = slice(s0, s1)
            pr_chunk = phat_rows[sl]
            hm_chunk = hmu_te[sl]
            hs_chunk = hsig_te[sl]
            lb_chunk = lead_bins_te[sl]
            lo_g = np.where(act_rows[sl], hm_chunk - 8.0 * hs_chunk, np.inf).min(axis=1)
            hi_g = np.where(act_rows[sl], hm_chunk + 8.0 * hs_chunk, -np.inf).max(axis=1)
            y_grid, _ = adaptive_grid(lo_g, hi_g, hm_chunk, hs_chunk)

            sc68, in68 = _eval_score_chunk(
                cal_b["levels"]["0.68"], dens, pr_chunk, hm_chunk, hs_chunk, lb_chunk, y_grid
            )
            _, in95 = _eval_score_chunk(
                cal_b["levels"]["0.95"], dens, pr_chunk, hm_chunk, hs_chunk, lb_chunk, y_grid
            )
            in95 = in95 | in68
            lo68, hi68, w68, comp68 = extract_components(y_grid, in68, offset=0.0)
            _, _, w95, comp95 = extract_components(y_grid, in95, offset=0.0)
            post_m = (hm_chunk * pr_chunk).sum(axis=1)
            mode_m = y_grid[np.arange(len(y_grid)), sc68.argmax(axis=1)]
            pred_dmag = np.where((post_m < lo68) | (post_m > hi68), mode_m, post_m)

            for idx_local in range(s1 - s0):
                yt = float(yt_dmag[s0 + idx_local])
                c68 = any(lo <= yt <= hi for lo, hi in comp68[idx_local])
                c95 = any(lo <= yt <= hi for lo, hi in comp95[idx_local])
                records[rec_key].append(
                    (
                        int(y_cls_rows[s0 + idx_local]),
                        int(obj_ids_rows[s0 + idx_local]),
                        abs(yt - float(pred_dmag[idx_local])),
                        c68,
                        float(w68[idx_local]),
                        c95,
                        float(w95[idx_local]),
                    )
                )

    summary = {}
    for key, rows in records.items():
        df_r = pd.DataFrame(rows, columns=["y", "obj", "err", "cov68", "w68", "cov95", "w95"])
        obj_grp = df_r.groupby("obj")
        obj_mae = float(obj_grp["err"].mean().median())
        obj_df = obj_grp.agg(y=("y", "first"), cov68=("cov68", "mean"), cov95=("cov95", "mean"))
        n_obj_cls = {
            CLASSES[c]: int((obj_df.y == c).sum())
            for c in range(len(CLASSES))
            if (obj_df.y == c).any()
        }
        per_cls_68 = {
            CLASSES[c]: round(float(df_r.loc[df_r.y == c, "cov68"].mean()), 3)
            for c in range(len(CLASSES))
            if (df_r.y == c).any()
        }
        per_obj_68 = {
            CLASSES[c]: round(float(obj_df.loc[obj_df.y == c, "cov68"].mean()), 3)
            for c in range(len(CLASSES))
            if (obj_df.y == c).any()
        }
        per_cls_95 = {
            CLASSES[c]: round(float(df_r.loc[df_r.y == c, "cov95"].mean()), 3)
            for c in range(len(CLASSES))
            if (df_r.y == c).any()
        }
        per_obj_95 = {
            CLASSES[c]: round(float(obj_df.loc[obj_df.y == c, "cov95"].mean()), 3)
            for c in range(len(CLASSES))
            if (obj_df.y == c).any()
        }
        summary[key] = dict(
            n_objects=int(df_r.obj.nunique()),
            n_points=int(len(df_r)),
            n_objects_per_class=n_obj_cls,
            median_object_mae=round(obj_mae, 4),
            marginal_68=round(float(df_r.cov68.mean()), 4),
            worst_class_68=round(min(per_cls_68.values()), 3),
            worst_class_object_68=round(min(per_obj_68.values()), 3),
            mean_width_68=round(float(df_r.w68.mean()), 4),
            per_class_68=per_cls_68,
            per_class_object_68=per_obj_68,
            marginal_95=round(float(df_r.cov95.mean()), 4),
            worst_class_95=round(min(per_cls_95.values()), 3),
            worst_class_object_95=round(min(per_obj_95.values()), 3),
            mean_width_95=round(float(df_r.w95.mean()), 4),
            per_class_95=per_cls_95,
            per_class_object_95=per_obj_95,
        )
    summary["grid_note"] = (
        "Shipped models/bts6/forecast boosters were fitted on train+val+test (10,497 objects), "
        "so grid test-split residuals are in-sample; see models/bts6/forecast/h*.json for grid's "
        "out-of-fold per-class coverage and oof_lofo_check for GAMUT's 5-fold out-of-fold coverage."
    )
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, default=PKG / "benchmark")
    ap.add_argument("--cache", type=Path, default=PKG / "paper_data" / "gamut_cache")
    ap.add_argument("--out", type=Path, default=PKG / "models" / "bts6" / "forecast_gamut")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--cap", type=int, default=24)
    a = ap.parse_args()

    t_start = time.time()
    ensure_cache(a.benchmark, a.cache, workers=min(a.threads, 12))
    meta_df = pd.read_parquet(a.cache / "meta.parquet")
    pool_mask = meta_df.split.isin(["train", "val"]).to_numpy()
    pool_idx = np.flatnonzero(pool_mask)
    assert np.array_equal(pool_idx, np.arange(len(pool_idx))), "train+val must precede test in meta"
    n_pool = len(pool_idx)
    y_pool = meta_df.y.to_numpy()[:n_pool]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n_pool), y_pool))

    x_tables = {h: pd.read_parquet(a.cache / f"X_h{h}.parquet") for h in SUPPORTED_HORIZONS}
    feat_cols = json.loads((PKG / "models" / "bts6" / "forecast" / "h5.json").read_text())["feature_names"]

    # Step 1: 5-fold out-of-fold phase classifier probabilities on train+val
    oof_probs_path = a.cache / "oof_phase_probs.npz"
    if oof_probs_path.exists():
        z_p = np.load(oof_probs_path)
        p_oof_by_h = {h: z_p[f"P_h{h}"] for h in SUPPORTED_HORIZONS}
    else:
        p_oof_by_h = {h: np.full((n_pool, len(CLASSES)), np.nan) for h in SUPPORTED_HORIZONS}
        clf_args = SimpleNamespace(threads=a.threads, rounds=2000)
        for ch in sorted(set(CLF_HORIZON.values())):
            x_ch = x_tables[ch].iloc[:n_pool][feat_cols].reset_index(drop=True)
            targets_h = [h for h, mapped in CLF_HORIZON.items() if mapped == ch]
            for k, (tr0, te) in enumerate(folds):
                tr, es = train_test_split(tr0, test_size=0.15, stratify=y_pool[tr0], random_state=k)
                clf_m = hierarchy_nocal(x_ch, y_pool, tr, es, clf_args)
                for h in targets_h:
                    xh_te = x_tables[h].iloc[te][clf_m.feature_names]
                    p_oof_by_h[h][te] = clf_m.predict_proba(xh_te)
                print(
                    f"phase clf h={ch} fold {k}: acc {np.mean(p_oof_by_h[ch][te].argmax(1) == y_pool[te]):.3f}",
                    flush=True,
                )
        np.savez_compressed(oof_probs_path, **{f"P_h{h}": v for h, v in p_oof_by_h.items()})

    a.out.mkdir(parents=True, exist_ok=True)
    ctx_df = context_table(a.benchmark, meta_df.obj_id.values)

    # Step 2: per-horizon OOF C1 forecasts, GAMUT calibration, and final C1 booster export
    for h in SUPPORTED_HORIZONS:
        t_h = time.time()
        f_df = pd.read_parquet(a.cache / f"F_h{h}.parquet")
        f_pool = f_df[f_df.obj < n_pool].reset_index(drop=True)
        f_obj = f_pool.obj.to_numpy()
        yt_pool = f_pool.dmag.to_numpy()
        lead_bins = np.searchsorted(LEAD_EDGES, f_pool.t.to_numpy(), side="right")

        oof_c1_path = a.cache / f"oof_c1_h{h}.npz"
        model_specs = {
            "mu": FORECAST_PARAMS,
            **{q: dict(FORECAST_PARAMS, objective="quantile", alpha=v) for q, v in QUANTILES.items()},
        }
        if oof_c1_path.exists():
            zc = np.load(oof_c1_path)
            hmu_oof, hq16_oof, hq84_oof = zc["mu"], zc["q16"], zc["q84"]
        else:
            hmu_oof = np.full((len(f_pool), len(CLASSES)), np.nan)
            hq16_oof = np.full((len(f_pool), len(CLASSES)), np.nan)
            hq84_oof = np.full((len(f_pool), len(CLASSES)), np.nan)
            targets_arr = {"mu": hmu_oof, "q16": hq16_oof, "q84": hq84_oof}
            for k, (tr_obj, te_obj) in enumerate(folds):
                fit_obj, es_obj = train_test_split(tr_obj, test_size=0.15, stratify=y_pool[tr_obj], random_state=k)
                r_fit = cap_rows(np.flatnonzero(np.isin(f_obj, fit_obj)), f_obj, a.cap, 1000 + k)
                r_es = cap_rows(np.flatnonzero(np.isin(f_obj, es_obj)), f_obj, a.cap, 2000 + k)
                r_te = np.flatnonzero(np.isin(f_obj, te_obj))
                x_fit = with_class(long_frame(x_tables[h], feat_cols, f_pool, r_fit), y_pool[f_obj[r_fit]])
                x_es = with_class(long_frame(x_tables[h], feat_cols, f_pool, r_es), y_pool[f_obj[r_es]])
                x_te_base = long_frame(x_tables[h], feat_cols, f_pool, r_te)
                y_fit, y_es = yt_pool[r_fit], yt_pool[r_es]
                for name, prm in model_specs.items():
                    m = fit_lgb(prm, x_fit, y_fit, x_es, y_es, a.threads)
                    stacked_te = pd.concat(
                        [with_class(x_te_base, np.full(len(r_te), c)) for c in range(len(CLASSES))],
                        ignore_index=True,
                    )
                    preds = m.predict(stacked_te, num_threads=a.threads).reshape(len(CLASSES), len(r_te)).T
                    targets_arr[name][r_te] = preds
                print(f"h={h} OOF C1 fold {k}: {len(r_te)} pts ({time.time() - t_h:.0f}s)", flush=True)
            np.savez_compressed(oof_c1_path, mu=hmu_oof, q16=hq16_oof, q84=hq84_oof)

        hsig_oof = np.maximum(0.5 * np.abs(hq84_oof - hq16_oof), SCALE_FLOOR)
        ok_rows = np.flatnonzero(
            np.all(np.isfinite(hmu_oof), axis=1) & np.all(np.isfinite(hsig_oof), axis=1) & np.isfinite(yt_pool)
        )

        cal_crossfit = fit_gamut_mode(
            p_oof_by_h[h], y_pool, f_obj, yt_pool, lead_bins, hmu_oof, hsig_oof, ok_rows, nb=10, crc=False
        )
        cal_split = fit_gamut_mode(
            p_oof_by_h[h], y_pool, f_obj, yt_pool, lead_bins, hmu_oof, hsig_oof, ok_rows, nb=10, crc=True
        )
        lofo_oof = evaluate_lofo_oof(p_oof_by_h[h], y_pool, f_pool, hmu_oof, hsig_oof, folds)

        # Fit final shipped C1 boosters on all of train+val (85/15 fit/early-stop, seed 42)
        c1_features = list(feat_cols) + ["fc_day", "fc_band", "fc_lead"] + [f"hyp_{c}" for c in range(len(CLASSES))]
        need_train = any(not (a.out / f"h{h}_{name}.txt").exists() for name in model_specs)
        if need_train:
            fit_all, es_all = train_test_split(np.arange(n_pool), test_size=0.15, stratify=y_pool, random_state=42)
            rf_all = cap_rows(np.flatnonzero(np.isin(f_obj, fit_all)), f_obj, a.cap, 3000 + h)
            re_all = cap_rows(np.flatnonzero(np.isin(f_obj, es_all)), f_obj, a.cap, 4000 + h)
            xf_all = with_class(long_frame(x_tables[h], feat_cols, f_pool, rf_all), y_pool[f_obj[rf_all]])
            xe_all = with_class(long_frame(x_tables[h], feat_cols, f_pool, re_all), y_pool[f_obj[re_all]])
        models_meta = {}
        for name, prm in model_specs.items():
            fname = f"h{h}_{name}.txt"
            fpath = a.out / fname
            if fpath.exists():
                booster = lgb.Booster(model_file=str(fpath))
                n_trees = int(booster.num_trees())
                models_meta[name] = dict(file=fname, rounds=n_trees, features=c1_features)
                print(f"h={h} existing {name}: {n_trees} rounds -> {fname}", flush=True)
            else:
                booster = fit_lgb(prm, xf_all, yt_pool[rf_all], xe_all, yt_pool[re_all], a.threads)
                booster.save_model(str(fpath), num_iteration=booster.best_iteration)
                models_meta[name] = dict(
                    file=fname,
                    rounds=int(booster.best_iteration),
                    features=c1_features,
                )
                print(f"h={h} final {name}: {booster.best_iteration} rounds -> {fname}", flush=True)

        grid_days = [int(d) for d in FLARE_GRID if d > h]
        meta_json = dict(
            horizon_days=int(h),
            classifier_horizon_days=CLF_HORIZON[h],
            grid_days=grid_days,
            n_lead_bins=10,
            prune_threshold=1e-3,
            feature_names=feat_cols,
            models=models_meta,
            training=dict(
                objects=int(n_pool),
                cap_points_per_object=int(a.cap),
                population="BTS train+val splits (8,922 objects); test split (1,575 objects) untouched",
                params=FORECAST_PARAMS,
            ),
            calibration_modes=dict(crossfit=cal_crossfit, split=cal_split),
            oof_lofo_check=lofo_oof,
        )
        (a.out / f"h{h}.json").write_text(dumps_compact(meta_json))

    # Write initial manifest so GamutForecaster can load and run the untouched test-split check
    files_hash = {p.name: sha256_file(p) for p in sorted(a.out.iterdir()) if p.suffix in (".json", ".txt") and p.name != "manifest.json"}
    (a.out / "manifest.json").write_text(
        dumps_compact(dict(algorithm="sha256", files=files_hash, export_script_sha256=sha256_file(Path(__file__))))
    )

    # Step 3: Untouched test-split verification (1,575 objects)
    engine = FlareAnalysis(num_threads=a.threads)
    gamut_fc = GamutForecaster(a.out, num_threads=a.threads)
    for h in SUPPORTED_HORIZONS:
        f_df_h = pd.read_parquet(a.cache / f"F_h{h}.parquet")
        test_summary = evaluate_test_split(
            engine, gamut_fc, a.benchmark, meta_df, x_tables[h], f_df_h, ctx_df, h
        )
        meta_path = a.out / f"h{h}.json"
        meta_json = json.loads(meta_path.read_text())
        meta_json["deployed_test_check"] = test_summary
        meta_path.write_text(dumps_compact(meta_json))
        lofo_c = meta_json["oof_lofo_check"]["gamut_crossfit"]
        c_s = test_summary["gamut_crossfit"]
        s_s = test_summary["gamut_split"]
        print(
            f"h={h:2d} | LOFO OOF 68% worst={lofo_c['worst_class_68']:.3f} (obj {lofo_c['worst_class_object_68']:.3f}), "
            f"95% worst={lofo_c['worst_class_95']:.3f} | TEST crossfit 68% marg={c_s['marginal_68']:.3f} w={c_s['mean_width_68']:.3f} "
            f"| TEST split 68% worst={s_s['worst_class_68']:.3f} w={s_s['mean_width_68']:.3f}",
            flush=True,
        )

    files_hash = {p.name: sha256_file(p) for p in sorted(a.out.iterdir()) if p.suffix in (".json", ".txt") and p.name != "manifest.json"}
    (a.out / "manifest.json").write_text(
        dumps_compact(dict(algorithm="sha256", files=files_hash, export_script_sha256=sha256_file(Path(__file__))))
    )
    # Re-verify manifest integrity
    verified_fc = GamutForecaster(a.out, num_threads=2)
    for name in verified_fc.manifest:
        verified_fc.checked_path(name)
    print(f"Export and untouched test verification complete in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
