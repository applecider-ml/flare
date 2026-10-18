"""Assemble the 7-class BTS dataset and write stratified train/val/test splits.

Photometric taxonomy (six classes). Five of them map 1:1 onto ASTRANet's
FINE_10; only SN_CC fans out, and that is where spectroscopy takes over:

    SN_Ia -> SN_Ia          SLSN -> SLSN        AGN -> AGN
    TDE   -> TDE            CV   -> CV
    SN_CC -> SN_II | SN_IIn | SN_IIb | SN_Ibc | SN_Ibn     (spectra needed)

The II/Ibc distinction is deliberately *not* attempted photometrically: both
are H-poor explosions with similar light-curve shapes, and forcing the split
costs 7 pp of macro F1 (0.731 -> 0.659) while SN_Ibc itself only reaches
F1 0.32 (granularity study of the FLARE paper, in preparation).

Tier-B "peculiar" objects are folded into their parent class for training --
they are unusual members of a class the model knows, not new classes -- and are
flagged so their behaviour can be analysed separately. Tier-C ("novel") and the
ambiguous novae are held out entirely: they are the anomaly-detection benchmark
and must never be seen in training.

Outputs: data/splits/{train,val,test}.csv  and  data/splits/anomaly.csv
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("FLARE_PAPER_DATA", REPO / "paper_data"))  # python scripts/setup_data.py
DATA, OUT = ROOT / "data", ROOT / "data" / "splits"

PHOT6 = {
    "SN_Ia":  ["SN Ia", "SN Ia-91T", "SN Ia-91bg", "SN Ia-pec",
               "SN Iax", "SN Ia-SC"],
    "SN_CC":  ["SN II", "SN IIP", "SN IIn", "SN IIb", "SN IIL",
               "SN II-pec", "SN IIn-pec", "SN Ia-CSM",
               "SN Ib", "SN Ic", "SN Ib/c", "SN Ic-BL", "SN Ib-pec",
               "SN Ic-pec", "SN Ibn", "SN Icn"],
    "SLSN":   ["SLSN-I", "SLSN-II"],
    "AGN":    ["AGN"],
    "TDE":    ["TDE", "TDE-H-He", "TDE-He", "TDE-featureless"],
    "CV":     ["CV"],
}
# the ASTRANet FINE_10 classes each photometric class hands over to
HANDOVER = {
    "SN_Ia": ["SN_Ia"], "SLSN": ["SLSN"], "AGN": ["AGN"],
    "TDE": ["TDE"], "CV": ["CV"],
    "SN_CC": ["SN_II", "SN_IIn", "SN_IIb", "SN_Ibc", "SN_Ibn"],
}
TYPE2CLS = {t: c for c, ts in PHOT6.items() for t in ts}
CLASSES = list(PHOT6)
SEED = 42


def main():
    frames = []
    for g in ["core", "peculiar", "novel", "ambiguous"]:
        p = DATA / f"manifest_{g}.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True).drop_duplicates("obj_id")
    df = df[df.passes_quality].copy()
    print(f"quality-passing objects: {len(df)}")

    df["cls"] = df.bts_type.map(TYPE2CLS)
    held = df[df.cls.isna()].copy()          # tier C + novae: never trained on
    lab = df[df.cls.notna()].copy()

    OUT.mkdir(parents=True, exist_ok=True)
    held["role"] = np.where(held.group == "ambiguous", "ambiguous", "novel")
    held.to_csv(OUT / "anomaly.csv", index=False)
    print(f"\nheld out for anomaly detection: {len(held)}")
    print(held.groupby(["role", "bts_type"]).size().to_string())

    # stratified 70/15/15 on the 7 classes
    tr, tmp = train_test_split(lab, test_size=0.30, stratify=lab.cls,
                               random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.cls,
                              random_state=SEED)
    for name, d in [("train", tr), ("val", va), ("test", te)]:
        d.to_csv(OUT / f"{name}.csv", index=False)

    print(f"\n=== 6-class splits (train {len(tr)} / val {len(va)} / test {len(te)}) ===")
    tab = pd.DataFrame({s: d.cls.value_counts() for s, d in
                        [("train", tr), ("val", va), ("test", te)]}).fillna(0).astype(int)
    tab["total"] = tab.sum(1)
    print(tab.loc[[c for c in CLASSES if c in tab.index]].to_string())
    npec = int(lab.group.eq("peculiar").sum())
    print(f"\n(of which {npec} are tier-B peculiar members, folded into their parent class)")


if __name__ == "__main__":
    main()
