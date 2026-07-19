"""Minimal FLARE usage — classify a few objects with uncertainty sets."""
import glob
from flare import FlareClassifier

clf = FlareClassifier.from_pretrained()
files = sorted(glob.glob("/path/to/photo_events/test/*.npz"))[:10]
proba = clf.predict_proba_from_files(files)
labels = clf.predict_from_files(files)
sets = clf.prediction_sets_from_files(files)      # 90% class-conditional coverage
for f, l, p, s in zip(files, labels, proba, sets):
    print(f"{f.split('/')[-1]:24s} -> {l:5s}  p={p.max():.2f}  set={s}")
