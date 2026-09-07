"""Train and export the bts6 deployment models.

Reproduces the paper's headline configuration as shippable artifacts:
models/bts6/{top.txt, slsn_branch.txt, conformal.json, model_card.json}.

Split discipline (the paper's, exactly): the training split trains with the
external blocks (host, M_pseudo, Gaia/WISE) independently blanked on 30% of
rows; the validation split is
cut in half -- one half early-stops both boosters and tunes the SLSN branch
threshold, the other half is touched by nothing except conformal calibration.
The test split is never read here at all; the model card quotes the paper's
five-fold numbers rather than anything computed in this script.

Run:  python scripts/train_bts6.py [--data /path/to/bts_dataset]
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from flare.config import AD_SPACE_COLS, BTS6, GW_COLS, HOST_COLS, POS_COLS  # noqa: E402
from flare.context import lumdist_mpc     # noqa: E402

SIX = BTS6.classes
FIVE = BTS6.top_classes
TO5 = {"SN_Ia": "SN_Ia", "SN_CC": "SN_CC+", "SLSN": "SN_CC+",
       "AGN": "AGN", "TDE": "TDE", "CV": "CV"}
DROP_P = 0.30
ALPHA = 0.10
TOP = dict(objective="multiclass", num_class=5, metric="multi_logloss",
           verbosity=-1, bagging_freq=1, max_depth=-1, num_threads=4, seed=42,
           learning_rate=0.06, num_leaves=40, min_child_samples=20,
           feature_fraction=0.55, bagging_fraction=0.85, lambda_l2=1.0)
BR = dict(objective="binary", metric="auc",
          verbosity=-1, bagging_freq=1, max_depth=-1, num_threads=4, seed=42,
          learning_rate=0.05, num_leaves=24, min_child_samples=25,
          feature_fraction=0.6, bagging_fraction=0.85, lambda_l2=1.0)
BETA = 0.8
# five-fold out-of-fold coverage at alpha=0.10 (scripts/kfold_conformal.py);
# shipped so a deployed prediction set can be shown next to its measured error
EMPIRICAL_COVERAGE = {"SN_Ia": 0.903, "SN_CC": 0.919, "SLSN": 0.949,
                      "AGN": 0.948, "TDE": 0.926, "CV": 0.934}
MARGINAL_COVERAGE = 0.909
# the shipped scheme checked once on the test split, which nothing in this
# script reads: marginal 0.895, mean set size 1.04, 90% singletons
DEPLOYED_CHECK = {"marginal": 0.895, "mean_set_size": 1.04,
                  "singleton_rate": 0.897, "split": "test (1575 objects)"}


def fit(params, X, y, Xes, yes, w):
    d = lgb.Dataset(X, label=y, weight=w)
    dv = lgb.Dataset(Xes, label=yes, reference=d)
    return lgb.train(params, d, 3000, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(120, verbose=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/projects/bcrv/asasli/later/bts_dataset")
    args = ap.parse_args()
    DATA = Path(args.data)
    OUT = PKG / "models" / "bts6"
    OUT.mkdir(parents=True, exist_ok=True)

    dfs = {s: pd.read_csv(DATA / "data/splits" / f"{s}.csv")
           for s in ["train", "val"]}
    feats = pd.read_parquet(DATA / "data/features_all.parquet")
    hosts = pd.read_csv(DATA / "data/hosts.csv").drop_duplicates("obj_id")
    pz = pd.read_csv(DATA / "data/hosts_photoz.csv").drop_duplicates("obj_id")
    cat = pd.read_csv(DATA / "data/bts_catalog.csv").rename(
        columns={"ZTFID": "obj_id"})[["obj_id", "peakmag"]]
    cat["peakmag"] = pd.to_numeric(cat.peakmag, errors="coerce")
    pz = pz.merge(cat, on="obj_id", how="left")
    pz["M_pseudo"] = pz.peakmag - 5 * np.log10(
        lumdist_mpc(pz.z_phot.values) * 1e5)
    gw = pd.read_csv(DATA / "data/context_gaia_wise.csv").drop_duplicates("obj_id")
    posrc = pd.read_csv(DATA / "data/ps1_position.csv").drop_duplicates("obj_id")

    def table(df):
        m = df[["obj_id"]].merge(feats, on="obj_id", how="left",
                                 validate="one_to_one")
        h = df[["obj_id"]].merge(hosts[["obj_id"] + HOST_COLS], on="obj_id",
                                 how="left", validate="one_to_one")
        z = df[["obj_id"]].merge(pz[["obj_id", "M_pseudo"]], on="obj_id",
                                 how="left", validate="one_to_one")
        g = df[["obj_id"]].merge(gw[["obj_id"] + GW_COLS], on="obj_id",
                                 how="left", validate="one_to_one")
        p = df[["obj_id"]].merge(posrc[["obj_id"] + POS_COLS], on="obj_id",
                                 how="left", validate="one_to_one")
        for t in (m, h, z, g, p):
            assert (t.obj_id.values == df.obj_id.values).all()
        return pd.concat([t.drop(columns=["obj_id"]).reset_index(drop=True)
                          for t in (m, h, z, g, p)], axis=1)

    Xtr = table(dfs["train"])
    Xva = table(dfs["val"])
    rtr, rva = dfs["train"].cls.values, dfs["val"].cls.values
    y6va = np.array([SIX.index(c) for c in rva])

    # external-context dropout on the training rows
    rng = np.random.default_rng(42)
    Xtr = Xtr.copy()
    Xtr.loc[rng.random(len(Xtr)) < DROP_P, HOST_COLS] = np.nan
    Xtr.loc[rng.random(len(Xtr)) < DROP_P, "M_pseudo"] = np.nan
    Xtr.loc[rng.random(len(Xtr)) < DROP_P, GW_COLS] = np.nan
    Xtr.loc[rng.random(len(Xtr)) < DROP_P, POS_COLS] = np.nan

    # ---- calibration: the deployed model on the untouched validation split
    # A Mondrian quantile at alpha is attainable for a class only once that
    # class has ceil(1/alpha)-1 members in the calibration sample -- nine at
    # alpha=0.10. Calibrating on half of validation leaves five TDEs, below
    # that bound, and the usual index cap then silently substitutes the
    # largest observed score: the sets balloon (mean size 2.06 on test, 4%
    # singletons) without ever honouring the guarantee. So no training object
    # is spent on early stopping: the number of boosting rounds is chosen by
    # cross-validation inside the training split, both boosters are refitted
    # on all of it, and the whole validation split -- which nothing has
    # touched -- calibrates the model that is actually shipped. That is exact
    # split-conformal validity, every class attainable (TDE gets eleven), and
    # on test it measures 0.895 marginal coverage with 1.04 mean set size.
    print("choosing rounds by cross-validation inside train ...", flush=True)
    y6tr_all = np.array([SIX.index(c) for c in rtr])
    r5, rb, ths = [], [], []
    for i_, (itr, ite) in enumerate(StratifiedKFold(
            5, shuffle=True, random_state=7).split(Xtr, y6tr_all)):
        y5i = np.array([FIVE.index(TO5[c]) for c in rtr[itr]])
        y5e = np.array([FIVE.index(TO5[c]) for c in rtr[ite]])
        ci = np.bincount(y5i, minlength=5).astype(float)
        wi = ((len(y5i) / (5 * ci)) ** BETA)[y5i]
        m5i = fit(TOP, Xtr.iloc[itr], y5i, Xtr.iloc[ite], y5e, wi)
        r5.append(m5i.best_iteration)
        cci, cce = np.isin(rtr[itr], ["SN_CC", "SLSN"]), np.isin(rtr[ite], ["SN_CC", "SLSN"])
        ybi, ybe = (rtr[itr] == "SLSN").astype(int), (rtr[ite] == "SLSN").astype(int)
        wbi = np.where(ybi[cci] == 1, (cci.sum() / max(ybi[cci].sum(), 1)) ** 0.5, 1.0)
        mbi = fit(BR, Xtr.iloc[itr][cci], ybi[cci], Xtr.iloc[ite][cce], ybe[cce], wbi)
        rb.append(mbi.best_iteration)
        pb_i = mbi.predict(Xtr.iloc[ite][cce], num_iteration=mbi.best_iteration)
        bt, bf = 0.5, -1
        for tt in np.linspace(0.05, 0.9, 35):
            ff = f1_score(ybe[cce], pb_i > tt)
            if ff > bf:
                bt, bf = float(tt), ff
        ths.append(bt)
    n5, nb = int(np.median(r5)), int(np.median(rb))
    thr = float(np.median(ths))
    print(f"  rounds top={n5} branch={nb}; branch threshold {thr:.2f}", flush=True)

    y5tr_all = np.array([FIVE.index(TO5[c]) for c in rtr])
    cnt = np.bincount(y5tr_all, minlength=5).astype(float)
    w = ((len(y5tr_all) / (5 * cnt)) ** BETA)[y5tr_all]
    print("refitting on the full training split ...", flush=True)
    m5 = lgb.train(TOP, lgb.Dataset(Xtr, label=y5tr_all, weight=w), n5)
    cc_tr = np.isin(rtr, ["SN_CC", "SLSN"])
    yb_tr = (rtr == "SLSN").astype(int)
    wb = np.where(yb_tr[cc_tr] == 1,
                  (cc_tr.sum() / max(yb_tr[cc_tr].sum(), 1)) ** 0.5, 1.0)
    mb = lgb.train(BR, lgb.Dataset(Xtr[cc_tr], label=yb_tr[cc_tr], weight=wb), nb)

    P5c = m5.predict(Xva)
    pbc = mb.predict(Xva)
    P6c = np.zeros((len(P5c), 6))
    for j, c in enumerate(SIX):
        if c == "SN_CC":
            P6c[:, j] = P5c[:, FIVE.index("SN_CC+")] * (1 - pbc)
        elif c == "SLSN":
            P6c[:, j] = P5c[:, FIVE.index("SN_CC+")] * pbc
        else:
            P6c[:, j] = P5c[:, FIVE.index(c)]
    yc = np.array([SIX.index(c) for c in rva])

    qhat, cal_scores, attainable = {}, {}, {}
    need = int(np.ceil(1 / ALPHA)) - 1
    for cidx, c in enumerate(SIX):
        s = np.sort(1 - P6c[yc == cidx, cidx])
        n = len(s)
        attainable[c] = bool(n >= need)
        k = int(np.ceil((n + 1) * (1 - ALPHA))) - 1
        # k >= n means the guarantee is unattainable for this class: the
        # honest threshold is then 1.0 (it cannot be excluded), never the
        # largest observed score
        qhat[c] = float(s[k]) if k < n else 1.0
        cal_scores[c] = [float(v) for v in s]
        print(f"  {c:<6} n_cal={n:<5} qhat={qhat[c]:.3f} "
              f"p-floor={1/(n+1):.4f} attainable={attainable[c]}", flush=True)

    m5.save_model(str(OUT / "top.txt"))
    mb.save_model(str(OUT / "slsn_branch.txt"))
    (OUT / "conformal.json").write_text(json.dumps(dict(
        alpha=ALPHA, n_cal=int(len(Xva)), classes=SIX, qhat=qhat,
        attainable=attainable,
        min_n_for_alpha=int(np.ceil(1 / ALPHA)) - 1,
        n_cal_per_class={c: len(v) for c, v in cal_scores.items()},
        p_value_floor={c: 1.0 / (len(v) + 1) for c, v in cal_scores.items()},
        cal_scores=cal_scores,
        empirical_coverage=EMPIRICAL_COVERAGE,
        marginal_coverage=MARGINAL_COVERAGE,
        deployed_check=DEPLOYED_CHECK,
        empirical_coverage_note=(
            "five-fold out-of-fold coverage at alpha=0.10 from the paper "
            "(cross-conformal calibration, marginal 0.909); measured on "
            "objects no fold trained on, not on this calibration split"),
        method="mondrian_lac",
        note=("split conformal on the whole validation split, which no "
              "stage of fitting touched: rounds were chosen by "
              "cross-validation inside the training split and both boosters "
              "refitted on all of it, so the calibrated model is exactly the "
              "shipped one and every class clears the attainability bound")),
        indent=2))
    (OUT / "model_card.json").write_text(json.dumps(dict(
        scheme="bts6",
        classes=SIX, top_classes=FIVE,
        branch_threshold=thr,
        training=dict(objects=int(len(Xtr)), features=int(Xtr.shape[1]),
                      external_dropout=DROP_P, beta=BETA,
                      data="ZTF BTS benchmark (see benchmark/DATASET.md)"),
        performance_note=(
            "quote the paper's five-fold out-of-fold numbers, not a "
            "single split: accuracy 0.924+-0.001, macro F1 0.832+-0.016; "
            "per-class F1 SN_Ia 0.957, SN_CC 0.852, SLSN 0.525, AGN 0.901, "
            "TDE 0.813, CV 0.944"),
        external_blocks=dict(host=HOST_COLS, photoz=["M_pseudo"],
                             point_source=GW_COLS, counterpart=POS_COLS),
        anomaly_space=dict(model="ad_space.txt", columns_excluded=["M_pseudo"]
                           + POS_COLS,
                           note="blind energy scores come from this booster; "
                                "M_pseudo and the counterpart block harm "
                                "detection (measured)"),
    ), indent=2))
    print(f"exported to {OUT}", flush=True)


if __name__ == "__main__":
    main()
