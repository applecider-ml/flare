# FLARE Hackathon — find what you were never taught to find

One day, three tracks, one theme: **anomaly detection on real ZTF light
curves**. Pick the track that matches your experience — every track produces a
concrete result by the end of the day, and every track starts from something
you saw in the demo session.

Teams of 2-3 work well; mixing experience levels inside a team is encouraged.

---

## Setup (do this first, ~10 minutes)

```bash
git clone https://github.com/applecider-ml/flare.git
cd flare
pip install -e .
export FLARE_DATA=/path/to/photo_events     # ask for the data location
```

Sanity check — this should print a class name in a few seconds:

```python
from flare import FlareClassifier
clf = FlareClassifier.from_pretrained()
```

Everything below uses pretrained models bundled in `models/` — no GPU, no
long training runs. The relevant notebooks are already executed, so you can
read every expected output before you run anything.

| You will lean on | What it is |
|---|---|
| `tutorial.ipynb` | the full anomaly pipeline, executed end to end |
| `build_your_own_flare.ipynb` | training fundamentals + the exercises Track A extends |
| `inside_the_network.ipynb` | the mini network + the exercises Track C extends |
| `train_no_tde.py` | rebuilds the 4-class "no-TDE" model (~1 minute) |
| `fusion_net.py` | loader for FusionNet, the neural companion (Track C) |
| `models/flare_lgbm_no_tde.txt` | the classifier that has never seen a TDE |
| `models/flare_lgbm_day10.txt` | the day-10 early-warning variant |

---

## Track A — First detection

*For you if: you are newer to machine learning and want a guided path with
real astronomy at the end.*

**Goal.** Understand what a classifier does when it meets a class it was never
trained on — and see it with your own histograms.

**Steps.**

1. Warm up with the three exercises in `build_your_own_flare.ipynb`
   (class-weight strength, feature importance, early stopping).
2. Do the final challenge: evaluate the **day-10 model**. How much
   classification is possible from only ten days of data? Which classes become
   identifiable first, and does that ordering make physical sense?
3. Now the anomaly part. Load the **no-TDE model** and push the test set
   through it (`tutorial.ipynb` Act 2 shows the pattern):
   - Where do the real TDEs land? Which class absorbs them, and why does that
     make sense physically?
   - Plot the histogram of maximum predicted probability for known-class
     objects vs the hidden TDEs. What do you see?
4. Build the simplest possible anomaly detector from that: flag everything
   whose maximum probability falls below a threshold. Choose the threshold
   yourself and report how many TDEs you catch vs how many ordinary objects
   you drag along.

**Deliverable.** The two plots from steps 3-4 plus three sentences: what the
classifier does with an unknown class, and what your simple detector's
trade-off is.

**Stretch.** Compare your threshold detector against the entropy of the full
probability vector instead of the maximum. Which is better here, and why?

---

## Track B — Beat the 0.91

*For you if: you are comfortable training models and want a measurable target
with a leaderboard.*

**Goal.** The demo showed the reference pipeline: a blind rank-average
consensus of anomaly scorers reaches AUC 0.72 on a pool of 2,026 objects
hiding 56 TDEs; adding just 9 labelled examples to the combiner lifts it to
**AUC 0.905**, with 20 of the 56 hidden TDEs inside 49 alerts at a 1%
false-alarm rate. Your job: reproduce it, then beat it.

**Rules.** The 56 hidden TDEs and the evaluation pool are fixed — never fit
anything on them. Only the 9 validation TDEs may be used as labels, exactly as
in the reference. Metric: AUC on the pool, plus TDEs recovered at 1%
false-alarm rate. Report both.

**Steps.**

1. Reproduce the reference numbers from `tutorial.ipynb` Act 2. If your AUC
   does not match, stop and find out why — that is half the learning.
2. Now improve any stage you like:
   - the **scorers**: drop weak ones, add your own (a new feature-space
     distance, a per-class calibrated score, an isolation forest on a feature
     subset you select with SHAP);
   - the **combiner**: different supervised models on top of the scores,
     different ways to spend the 9 labels;
   - the **features**: is there a physics feature that separates TDEs which
     the current set underuses? (Colour evolution is the known signal — what
     else?)
3. Log every attempt: AUC, TDEs at 1% FAR, one line on what changed.

**Deliverable.** Your best pipeline, its two numbers, and the attempt log.
Whiteboard leaderboard, updated whenever a team improves.

**Stretch.** `beyond_the_catalog.ipynb` scores 8 genuine out-of-taxonomy
transients (FBOT, Ca-rich, ILRT, SLSN-I, SN Iax); half rank above the 89th
percentile, but the SLSN-I hides. Does *your* improved pipeline see it?

---

## Track C — The unmeasured

*For you if: you want an open question nobody knows the answer to, or you came
here for the Hyrax integration.*

Three genuinely open problems — pick one.

**C1. Anomaly detection with the network.** FusionNet
(`fusion_net.py` + `models/fusion_net/`) has never been used for anomaly
detection — this measurement does not exist anywhere. Ideas: use the entropy
or maximum of its temperature-calibrated probabilities as scorers; extract the
pooled representation before the classifier head and run distance-based
scores in that embedding space; ensemble disagreement between the three seeds.
Evaluate on the Track B pool with the Track B rules so the numbers are
directly comparable. Does the sequence model see anomalies the feature model
misses — or the reverse?

**C2. Architecture questions with teeth.** The three exercises in
`inside_the_network.ipynb`, done properly: FiLM conditioning vs plain
band-concatenation; attention pooling vs masked mean; and the day-10
early-warning cap (`CAP = 30`). Each is a controlled experiment on the mini
network — small enough to run on CPU, real enough that the answer is not
obvious. Report effect sizes, not just "better".

**C3. Hyrax-ify FLARE.** Wrap FLARE (and, if you get there, FusionNet) as a
model in Hyrax, following the patterns from the Hyrax session. Target: the
classifier callable through the Hyrax interface, with the conformal
prediction sets exposed. This connects the light-curve work to the same
infrastructure as the rest of AppleCiDEr.

**Deliverable.** Whatever you measured, honestly reported — including null
results. A clean negative ("embedding distances do not beat the feature
scorers, here is the comparison") is a real contribution.

A note worth knowing: C1 and C2 are questions the FLARE project itself has
not answered yet. If your measurement holds up, it feeds directly into the
ongoing work — with credit. Talk to Argyro before the end of the day.

---

## End of day

Each team gets three minutes: one slide or one figure, the numbers, and the
one thing you would try next. Honest nulls are as welcome as wins — the goal
is a measurement, not a victory lap.
