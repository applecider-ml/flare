"""Minimal FLARE usage: classify test objects of the shipped benchmark with uncertainty."""
import flare
from flare import benchmark

benchmark.prepare()                                   # unpacks benchmark/light_curves.tar.gz once
clf = flare.load_classifier()                         # six classes, trained on train+val
X, y, table = benchmark.load_split("test")
X = X.reindex(columns=clf.feature_names).head(10)
proba = clf.predict_proba(X)
sets = clf.prediction_sets(X)                          # 90% class-conditional conformal sets
for oid, p, s in zip(table.obj_id, proba, sets):
    print(f"{oid:14s} -> {benchmark.SIX[p.argmax()]:6s}  p={p.max():.2f}  set={s}")
