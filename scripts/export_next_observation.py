"""Train, calibrate, export, and verify next-observation closure models on ZTF BTS.

Follows the pre-registered split protocol of the next-observation closure study:
  * ``benchmark/splits/train.csv`` (7,347 objects):
      - five-fold out-of-fold hypothesis-conditional LightGBM quantile models
        (mu, q16, q84) collect per-class standardized next-point residuals on
        real detections within (h, h + 2] days;
      - final shipped C1 boosters (h5, h10) are fitted on ``train`` only
        (85/15 fit/early-stop).
  * ``benchmark/splits/val.csv`` (1,575 objects):
      - fits temperature T on phase-classifier probabilities;
      - fits inductive Venn-Abers step tables (``flare.venn_abers.build_table``)
        for both the shipped Gaussian score (``grid``) and the latent-class
        next-point mixture (``gamut``) against realised singleton outcomes.
  * ``benchmark/splits/test.csv`` (1,575 objects, untouched):
      - evaluates ECE, Brier score, log-loss, ROC AUC, and the five-bin +-0.10
        reliability criterion, recording ``deployed_test_check`` in ``h{h}.json``.

Run:
    python scripts/export_next_observation.py --out models/bts6/next_observation --threads 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from flare import load_analysis  # noqa: E402
from flare.analysis import (  # noqa: E402
    NEXT_OBS_DT,
    NEXT_OBS_ERR,
    NEXT_OBS_SCATTER,
    append_detection,
    feature_frame,
    observed_events,
)
from flare.config import GW_COLS, HOST_COLS, POS_COLS  # noqa: E402
from flare.data import COL, load_events, recompute_colors  # noqa: E402
from flare.next_observation import (  # noqa: E402
    CLASSES,
    SCALE_FLOOR,
    SUPPORTED_HORIZONS,
    NextObservationPredictor,
    apply_temperature,
    venn_abers_binary,
)
from flare.venn_abers import build_table  # noqa: E402

WINDOW = 2.0
N_DRAW_CLASS = 8
P_MIN = 0.01
T_MAX = 90.0
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


def _init_extract(bench_str: str, ctx: pd.DataFrame):
    _G.update(bench=Path(bench_str), ctx=ctx)


def _extract_obj(job):
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
    with mp.Pool(workers, initializer=_init_extract, initargs=(str(bench_dir), ctx)) as p:
        res = dict(p.imap_unordered(_extract_obj, jobs, chunksize=32))
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


def _init_sim(bench_str: str, h: int, q_map: dict, ctx: pd.DataFrame):
    os.environ["OMP_NUM_THREADS"] = "1"
    eng = load_analysis(num_threads=1)
    _G.update(bench=Path(bench_str), eng=eng, clf=eng._classifier_at(h), h=h, q_map=q_map, ctx=ctx)


def _sim_object(job):
    """Evaluate real singleton outcome, shipped grid score, and GAMUT raw mixture score for one object."""
    oid, fp, split, y_cls, phat, hmu_next, hsig_next, real_nexts = job
    eng = _G["eng"]
    clf = _G["clf"]
    h = _G["h"]
    q_map = _G["q_map"]
    ctx = {k: (np.nan if pd.isna(v) else float(v)) for k, v in _G["ctx"].loc[oid].items()}
    full = load_events(str(_G["bench"] / fp))
    try:
        pre = observed_events(full, float(h))
    except ValueError:
        return []
    pre_clean = recompute_colors(pre)
    x_pre, peak = feature_frame(pre_clean, ctx)
    set0 = clf.conformal_report(x_pre.reindex(columns=clf.feature_names).astype(float))[0]["set"]
    fc_grid = eng.forecast(pre, h, ctx, recipe="grid")
    t_sim = float(h) + NEXT_OBS_DT
    rng = np.random.default_rng(0)

    active = phat >= P_MIN
    active[int(phat.argmax())] = True
    keep = np.flatnonzero(active)
    weights = phat[keep] / phat[keep].sum()

    # Collect all feature extractions for this object into one classifier batch
    batch_frames = []
    slices = []
    # Note: rng in next_observation_value advances across ('g', 'r') in order when status == 'ok'
    grid_draws_by_band = {}
    for b_id, b_name in enumerate(("g", "r")):
        curve = fc_grid["bands"].get(b_name, {})
        if curve.get("status") == "ok":
            (d0, d1), (m0, m1) = curve["days"][:2], curve["magnitude"][:2]
            mean_m = m0 + (m1 - m0) * NEXT_OBS_DT / (d1 - d0)
            grid_draws_by_band[b_id] = rng.normal(mean_m, NEXT_OBS_SCATTER[h][b_id], 16)

    for b_id, t_real, mag_real, err_real in real_nexts:
        start_idx = len(batch_frames)
        # 1) Real next detection
        aug_real = recompute_colors(append_detection(pre_clean, t_real, b_id, mag_real, err_real))
        fx_real, _ = feature_frame(aug_real, ctx)
        batch_frames.append(fx_real)
        # 2) Shipped grid draws (if defined in this band)
        has_grid = b_id in grid_draws_by_band
        if has_grid:
            for m_val in grid_draws_by_band[b_id]:
                aug_g = recompute_colors(append_detection(pre_clean, t_sim, b_id, float(m_val), NEXT_OBS_ERR))
                fx_g, _ = feature_frame(aug_g, ctx)
                batch_frames.append(fx_g)
        # 3) GAMUT hypothesis quantile draws at t_sim = h + 1.0
        for c in keep:
            draws_c = peak + hmu_next[b_id, c] + hsig_next[b_id, c] * q_map[int(c)]
            for m_val in draws_c:
                aug_c = recompute_colors(append_detection(pre_clean, t_sim, b_id, float(m_val), NEXT_OBS_ERR))
                fx_c, _ = feature_frame(aug_c, ctx)
                batch_frames.append(fx_c)
        slices.append((b_id, t_real, start_idx, has_grid))

    batch_df = pd.concat(batch_frames, ignore_index=True).reindex(columns=clf.feature_names).astype(float)
    reports = clf.conformal_report(batch_df)
    sizes = np.array([len(r["set"]) for r in reports], dtype=int)

    out_rows = []
    n_q = len(next(iter(q_map.values())))
    for b_id, t_real, s_idx, has_grid in slices:
        cur = s_idx
        size_after = int(sizes[cur])
        cur += 1
        if has_grid:
            g_sizes = sizes[cur : cur + 16]
            s_grid = float((g_sizes == 1).mean())
            cur += 16
        else:
            s_grid = float("nan")
        c_sizes = sizes[cur : cur + len(keep) * n_q].reshape(len(keep), n_q)
        pclose_c = (c_sizes == 1).mean(axis=1)
        s_gamut_raw = float(np.sum(weights * pclose_c))
        out_rows.append(
            dict(
                obj_id=oid,
                split=split,
                y=int(y_cls),
                band=int(b_id),
                t_next_real=float(t_real),
                set_before=len(set0),
                outcome=int(size_after == 1),
                s_grid=s_grid,
                s_gamut_raw=s_gamut_raw,
            )
        )
    return out_rows


def ece_15(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, bins)
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in parts if len(b)))


def five_bin_check(p: np.ndarray, y: np.ndarray, tol: float = 0.10) -> dict:
    edges = np.linspace(0.0, 1.0, 6)
    devs = []
    bins_info = []
    for i in range(5):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < 4 else (p <= edges[i + 1]))
        if not m.any():
            continue
        dev = float(y[m].mean() - p[m].mean())
        devs.append(abs(dev))
        bins_info.append(
            dict(
                bin_lo=round(float(edges[i]), 2),
                n=int(m.sum()),
                mean_pred=round(float(p[m].mean()), 4),
                realised_rate=round(float(y[m].mean()), 4),
                deviation=round(dev, 4),
            )
        )
    return dict(
        bins_within_0p10=int(sum(d <= tol for d in devs)),
        n_populated_bins=len(devs),
        max_abs_deviation=round(max(devs) if devs else 0.0, 4),
        bins=bins_info,
    )


def score_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p_clip = np.clip(np.asarray(p, dtype=float), 1e-4, 1.0 - 1e-4)
    y_f = np.asarray(y, dtype=float)
    return dict(
        n=int(len(y_f)),
        ece=round(ece_15(p_clip, y_f), 4),
        brier=round(float(np.mean((p_clip - y_f) ** 2)), 4),
        logloss=round(float(-np.mean(y_f * np.log(p_clip) + (1.0 - y_f) * np.log(1.0 - p_clip))), 4),
        auc=round(float(roc_auc_score(y_f, p_clip)), 4) if 0.0 < y_f.mean() < 1.0 else None,
        five_bin_reliability=five_bin_check(p_clip, y_f),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", type=Path, default=PKG / "benchmark")
    ap.add_argument("--cache", type=Path, default=PKG / "paper_data" / "gamut_cache")
    ap.add_argument("--out", type=Path, default=PKG / "models" / "bts6" / "next_observation")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cap", type=int, default=24)
    a = ap.parse_args()

    t_start = time.time()
    ensure_cache(a.benchmark, a.cache, workers=min(a.workers, 12))
    meta_df = pd.read_parquet(a.cache / "meta.parquet")
    y_all = meta_df.y.to_numpy()
    split_all = meta_df.split.to_numpy()
    tr_idx = np.flatnonzero(split_all == "train")
    val_idx = np.flatnonzero(split_all == "val")
    te_idx = np.flatnonzero(split_all == "test")
    ctx_df = context_table(a.benchmark, meta_df.obj_id.values)
    feat_cols = json.loads((PKG / "models" / "bts6" / "forecast" / "h5.json").read_text())["feature_names"]

    a.out.mkdir(parents=True, exist_ok=True)
    eng_main = load_analysis(num_threads=a.threads)

    for h in SUPPORTED_HORIZONS:
        t_h = time.time()
        x_df = pd.read_parquet(a.cache / f"X_h{h}.parquet")
        f_df = pd.read_parquet(a.cache / f"F_h{h}.parquet")
        f_obj = f_df.obj.to_numpy()
        f_band = f_df.band.to_numpy()
        f_t = f_df.t.to_numpy()
        f_y = f_df.dmag.to_numpy()

        model_specs = {
            "mu": FORECAST_PARAMS,
            **{q: dict(FORECAST_PARAMS, objective="quantile", alpha=v) for q, v in QUANTILES.items()},
        }

        # 1) 5-fold OOF C1 on train split ONLY -> class-specific next-point residual quantiles Q_c
        oof_tr_path = a.cache / f"nextobs_c1oof_tr_h{h}.npz"
        if oof_tr_path.exists():
            zo = np.load(oof_tr_path)
            oof_rows, hmu_oof, hq16_oof, hq84_oof = zo["rows"], zo["mu"], zo["q16"], zo["q84"]
        else:
            rows_list = []
            preds_oof = {n: [] for n in model_specs}
            for k, (tr_sub, te_sub) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(tr_idx, y_all[tr_idx])):
                f_sub, e_sub = train_test_split(tr_idx[tr_sub], test_size=0.15, stratify=y_all[tr_idx[tr_sub]], random_state=k)
                rf = cap_rows(np.flatnonzero(np.isin(f_obj, f_sub)), f_obj, a.cap, 100 + k)
                re = cap_rows(np.flatnonzero(np.isin(f_obj, e_sub)), f_obj, a.cap, 200 + k)
                rp = np.flatnonzero(np.isin(f_obj, tr_idx[te_sub]))
                xf = with_class(long_frame(x_df, feat_cols, f_df, rf), y_all[f_obj[rf]])
                xe = with_class(long_frame(x_df, feat_cols, f_df, re), y_all[f_obj[re]])
                xp_base = long_frame(x_df, feat_cols, f_df, rp)
                for name, prm in model_specs.items():
                    m = fit_lgb(prm, xf, f_y[rf], xe, f_y[re], a.threads)
                    stacked = pd.concat([with_class(xp_base, np.full(len(rp), c)) for c in range(len(CLASSES))], ignore_index=True)
                    preds_oof[name].append(m.predict(stacked, num_threads=a.threads).reshape(len(CLASSES), len(rp)).T)
                rows_list.append(rp)
                print(f"nextobs h={h} train-OOF fold {k}: {len(rp)} rows", flush=True)
            oof_rows = np.concatenate(rows_list)
            hmu_oof = np.concatenate(preds_oof["mu"])
            hq16_oof = np.concatenate(preds_oof["q16"])
            hq84_oof = np.concatenate(preds_oof["q84"])
            np.savez_compressed(oof_tr_path, rows=oof_rows, mu=hmu_oof, q16=hq16_oof, q84=hq84_oof)

        hsig_oof = np.maximum(0.5 * np.abs(hq84_oof - hq16_oof), SCALE_FLOOR)
        pos_oof = {int(r): j for j, r in enumerate(oof_rows)}
        resid_by_cls = {c: [] for c in range(len(CLASSES))}
        for i_tr in tr_idx:
            for b_id in (0, 1):
                m_pts = np.flatnonzero((f_obj == i_tr) & (f_band == b_id) & (f_t > h) & (f_t <= h + WINDOW))
                if not len(m_pts):
                    continue
                r0 = int(m_pts[np.argmin(f_t[m_pts])])
                if r0 not in pos_oof:
                    continue
                j = pos_oof[r0]
                c_true = int(y_all[i_tr])
                val_e = (f_y[r0] - hmu_oof[j, c_true]) / hsig_oof[j, c_true]
                if np.isfinite(val_e):
                    resid_by_cls[c_true].append(float(val_e))
        pooled_resid = np.concatenate([np.asarray(v, dtype=float) for v in resid_by_cls.values() if len(v)])
        qs = (np.arange(N_DRAW_CLASS) + 0.5) / N_DRAW_CLASS
        q_map = {
            c: np.quantile(np.asarray(resid_by_cls[c], dtype=float) if len(resid_by_cls[c]) >= 20 else pooled_resid, qs)
            for c in range(len(CLASSES))
        }
        print(f"h={h} train OOF next-point residual counts: {[len(resid_by_cls[c]) for c in range(len(CLASSES))]}", flush=True)

        # 2) Fit final shipped next-observation C1 boosters on train split ONLY (85/15 fit/stop, seed 0)
        fit_tr, es_tr = train_test_split(tr_idx, test_size=0.15, stratify=y_all[tr_idx], random_state=0)
        rf_tr = cap_rows(np.flatnonzero(np.isin(f_obj, fit_tr)), f_obj, a.cap, 11)
        re_tr = cap_rows(np.flatnonzero(np.isin(f_obj, es_tr)), f_obj, a.cap, 12)
        xf_tr = with_class(long_frame(x_df, feat_cols, f_df, rf_tr), y_all[f_obj[rf_tr]])
        xe_tr = with_class(long_frame(x_df, feat_cols, f_df, re_tr), y_all[f_obj[re_tr]])
        boosters = {}
        models_meta = {}
        for name, prm in model_specs.items():
            booster = fit_lgb(prm, xf_tr, f_y[rf_tr], xe_tr, f_y[re_tr], a.threads)
            fname = f"h{h}_{name}.txt"
            booster.save_model(str(a.out / fname), num_iteration=booster.best_iteration)
            boosters[name] = booster
            models_meta[name] = dict(file=fname, rounds=int(booster.best_iteration), features=list(xf_tr.columns))
            print(f"h={h} final train {name}: {booster.best_iteration} rounds -> {fname}", flush=True)

        # 3) Fit temperature T on val split
        clf = eng_main._classifier_at(h)
        val_has = x_df.iloc[val_idx]["has_prefix"].to_numpy(dtype=bool)
        val_valid_idx = val_idx[val_has]
        p_val_raw = clf.predict_proba(x_df.iloc[val_valid_idx].reindex(columns=clf.feature_names).astype(float))
        temp = fit_temperature(p_val_raw, y_all[val_valid_idx])
        print(f"h={h} val temperature T={temp:.4f}", flush=True)

        # 4) Precompute C1 next-night (t = h + 1.0) forecasts for all val + test objects
        eval_idx = np.concatenate([val_idx, te_idx])
        p_eval_raw = clf.predict_proba(x_df.iloc[eval_idx].reindex(columns=clf.feature_names).astype(float))
        phat_eval = apply_temperature(p_eval_raw, temp)

        t_sim = float(h) + NEXT_OBS_DT
        eval_long_g = x_df.iloc[eval_idx][feat_cols].astype(float).reset_index(drop=True).assign(fc_day=t_sim, fc_band=0.0, fc_lead=NEXT_OBS_DT)
        eval_long_r = x_df.iloc[eval_idx][feat_cols].astype(float).reset_index(drop=True).assign(fc_day=t_sim, fc_band=1.0, fc_lead=NEXT_OBS_DT)
        eval_both = pd.concat([eval_long_g, eval_long_r], ignore_index=True)
        stacked_eval = pd.concat([with_class(eval_both, np.full(len(eval_both), c)) for c in range(len(CLASSES))], ignore_index=True)
        preds_eval = {
            name: booster.predict(stacked_eval, num_threads=a.threads).reshape(len(CLASSES), 2, len(eval_idx)).transpose(2, 1, 0)
            for name, booster in boosters.items()
        }
        hmu_eval = preds_eval["mu"]  # (len(eval_idx), 2, 6)
        hsig_eval = np.maximum(0.5 * np.abs(preds_eval["q84"] - preds_eval["q16"]), SCALE_FLOOR)

        # Build simulation jobs for val + test objects that have a real next detection within (h, h + 2]
        sim_cache_path = a.cache / f"nextobs_sim_h{h}.parquet"
        if sim_cache_path.exists():
            sim_df = pd.read_parquet(sim_cache_path)
        else:
            jobs = []
            for pos_e, obj_i in enumerate(eval_idx):
                row_m = meta_df.iloc[obj_i]
                full = load_events(str(a.benchmark / row_m.filepath))
                real_nexts = []
                for b_id in (0, 1):
                    m_n = np.flatnonzero(
                        (full[:, COL["dt"]] > h) & (full[:, COL["dt"]] <= h + WINDOW) & (full[:, COL["band_id"]] == b_id)
                    )
                    if not len(m_n):
                        continue
                    j0 = int(m_n[np.argmin(full[m_n, COL["dt"]])])
                    t_r = float(full[j0, COL["dt"]])
                    mag_r = 23.9 - 2.5 * float(full[j0, COL["logflux"]])
                    err_r = 2.5 * float(full[j0, COL["logflux_err"]])
                    real_nexts.append((b_id, t_r, mag_r, err_r))
                if real_nexts:
                    jobs.append(
                        (
                            row_m.obj_id,
                            row_m.filepath,
                            row_m.split,
                            int(row_m.y),
                            phat_eval[pos_e],
                            hmu_eval[pos_e],
                            hsig_eval[pos_e],
                            real_nexts,
                        )
                    )
            print(f"h={h}: simulating {len(jobs)} val+test objects across {a.workers} workers...", flush=True)
            with mp.get_context("spawn").Pool(
                a.workers, initializer=_init_sim, initargs=(str(a.benchmark), h, q_map, ctx_df)
            ) as pool_mp:
                sim_rows = [r for sub in pool_mp.imap_unordered(_sim_object, jobs, chunksize=4) for r in sub]
            sim_df = pd.DataFrame(sim_rows)
            sim_df.to_parquet(sim_cache_path, index=False)

        # 5) Fit Venn-Abers step tables on val split ONLY
        val_sim = sim_df[sim_df.split == "val"].copy()
        val_grid_ok = val_sim[np.isfinite(val_sim.s_grid)]
        grid_va_table = build_table(val_grid_ok.s_grid.to_numpy(), val_grid_ok.outcome.to_numpy())
        gamut_va_table = build_table(val_sim.s_gamut_raw.to_numpy(), val_sim.outcome.to_numpy())

        # 6) Evaluate on untouched test split
        te_sim = sim_df[sim_df.split == "test"].copy()
        te_common = te_sim[np.isfinite(te_sim.s_grid)].copy()
        y_te = te_common.outcome.to_numpy(dtype=float)
        p_grid_raw = te_common.s_grid.to_numpy(dtype=float)
        p_grid_va = np.array([venn_abers_binary(v, grid_va_table)[0] for v in p_grid_raw])
        p_gamut_raw = te_common.s_gamut_raw.to_numpy(dtype=float)
        p_gamut_va = np.array([venn_abers_binary(v, gamut_va_table)[0] for v in p_gamut_raw])

        te_undef = te_sim[~np.isfinite(te_sim.s_grid)]
        test_check = dict(
            n_test_pairs_common=int(len(te_common)),
            n_test_objects_common=int(te_common.obj_id.nunique()),
            n_test_pairs_single_band_only=int(len(te_undef)),
            outcome_rate=round(float(y_te.mean()), 4),
            grid_uncalibrated=score_metrics(p_grid_raw, y_te),
            grid_venn_abers=score_metrics(p_grid_va, y_te),
            gamut_uncalibrated=score_metrics(p_gamut_raw, y_te),
            gamut_venn_abers=score_metrics(p_gamut_va, y_te),
        )
        if len(te_undef) >= 10:
            p_undef_va = np.array([venn_abers_binary(v, gamut_va_table)[0] for v in te_undef.s_gamut_raw.to_numpy()])
            test_check["gamut_on_single_band_prefixes"] = score_metrics(p_undef_va, te_undef.outcome.to_numpy())

        meta_json = dict(
            horizon_days=int(h),
            p_min=P_MIN,
            temperature=round(temp, 6),
            feature_names=feat_cols,
            models=models_meta,
            residual_quantiles={str(c): np.round(q_map[c], 6).tolist() for c in range(len(CLASSES))},
            residual_train_oof_counts={CLASSES[c]: len(resid_by_cls[c]) for c in range(len(CLASSES))},
            grid_venn_abers=grid_va_table,
            gamut_venn_abers=gamut_va_table,
            training=dict(
                c1_train_objects=int(len(tr_idx)),
                val_calibration_pairs=int(len(val_sim)),
                population="C1 boosters and OOF residual quantiles fitted on BTS train only; temperature and Venn-Abers fitted on BTS val; BTS test untouched",
            ),
            calibration_note=(
                f"Latent-class next-night mixture calibrated with inductive Venn-Abers on BTS validation detections "
                f"(day {h}: test Brier {test_check['gamut_venn_abers']['brier']:.3f} vs "
                f"{test_check['grid_uncalibrated']['brier']:.3f} uncalibrated grid, "
                f"AUC {test_check['gamut_venn_abers']['auc']:.3f} vs {test_check['grid_uncalibrated']['auc']:.3f}, "
                f"ECE {test_check['gamut_venn_abers']['ece']:.3f})."
            ),
            deployed_test_check=test_check,
        )
        (a.out / f"h{h}.json").write_text(dumps_compact(meta_json))
        print(
            f"TEST next_obs h={h:2d} ({time.time() - t_h:.0f}s) | "
            f"grid raw: Brier={test_check['grid_uncalibrated']['brier']:.3f} ECE={test_check['grid_uncalibrated']['ece']:.3f} "
            f"AUC={test_check['grid_uncalibrated']['auc']:.3f} bins={test_check['grid_uncalibrated']['five_bin_reliability']['bins_within_0p10']}/5 | "
            f"grid+VA: Brier={test_check['grid_venn_abers']['brier']:.3f} ECE={test_check['grid_venn_abers']['ece']:.3f} "
            f"bins={test_check['grid_venn_abers']['five_bin_reliability']['bins_within_0p10']}/5 | "
            f"GAMUT+VA: Brier={test_check['gamut_venn_abers']['brier']:.3f} ECE={test_check['gamut_venn_abers']['ece']:.3f} "
            f"AUC={test_check['gamut_venn_abers']['auc']:.3f} bins={test_check['gamut_venn_abers']['five_bin_reliability']['bins_within_0p10']}/5",
            flush=True,
        )

    files_hash = {
        p.name: sha256_file(p)
        for p in sorted(a.out.iterdir())
        if p.suffix in (".json", ".txt") and p.name != "manifest.json"
    }
    (a.out / "manifest.json").write_text(
        dumps_compact(dict(algorithm="sha256", files=files_hash, export_script_sha256=sha256_file(Path(__file__))))
    )

    # End-to-end runtime check on first 5 test objects
    predictor = NextObservationPredictor(a.out, num_threads=2)
    for name in predictor.manifest:
        predictor.checked_path(name)
    for idx_check in te_idx[:5]:
        r_m = meta_df.iloc[idx_check]
        ev_c = load_events(str(a.benchmark / r_m.filepath))
        ctx_c = {k: (np.nan if pd.isna(v) else float(v)) for k, v in ctx_df.loc[r_m.obj_id].items()}
        for h in SUPPORTED_HORIZONS:
            res_g = eng_main.next_observation_value(ev_c, h, context=ctx_c, recipe="gamut")
            json.dumps(res_g, allow_nan=False)
    print(f"Next-observation export and verification complete in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
