# ZTF Summer School: session materials (v2)

Two notebooks, one session, one hackathon. Both read the BTS benchmark that ships inside this
repository (`benchmark/`, loaded with `flare.benchmark`) and use its six classes: SN_Ia, SN_CC,
SLSN, AGN, TDE and CV. No external data download is needed.

| Notebook | What it covers | Time |
|---|---|---|
| `deep_architecture.ipynb` | build a deep sequence classifier (`LightCurveGRU`, a GRU with FiLM band conditioning and attention pooling) on **raw** light curves from scratch: architecture, training, full evaluation report. CPU only, every line visible. | ~30 min |
| `flare_pipeline.ipynb` | use the **pretrained FLARE** bts6 classifier: classification, conformal prediction sets with abstention, Venn-Abers intervals and an early (day-10) model; then a five-class model that has never seen a TDE, blind anomaly detection of the hidden TDEs with LUNA, and an active-learning step with a few labelled TDEs. | ~30 min |

## Setup

```bash
git clone https://github.com/applecider-ml/flare.git
cd flare
pip install -e .
pip install torch                                   # for deep_architecture.ipynb
pip install git+https://github.com/asasli/luna.git  # LUNA, the anomaly-detection package
```

The light curves are unpacked from `benchmark/light_curves.tar.gz` on first use
(`flare.benchmark.prepare()`).

## Hackathon

- **Anomaly-detection track** starts from `flare_pipeline.ipynb`, section 7: score the 53
  out-of-taxonomy objects of the benchmark's anomaly split (`flare.benchmark.load_split("anomaly")`,
  37 novae and 16 rarer transients) and the peculiar transients in `rare_transients/`, find which
  ones hide, improve the detector, and map the value-of-labels curve. Tasks are listed at the end
  of the notebook. Note that six of the eight objects in `rare_transients/` carry an in-taxonomy
  BTS label (SN_CC or SLSN) and sit in the benchmark's train or validation split; only AT2018cow
  and AT2019abn are out of taxonomy for the six-class models.
- **Hyrax track** starts from `deep_architecture.ipynb`: wrap the trained `LightCurveGRU` as a
  Hyrax model, following the patterns from the Hyrax session.

`train_no_tde.py` trains the five-class no-TDE model on the benchmark's train split (early
stopping on the validation split) in under a minute; the notebook calls it directly, and
`python train_no_tde.py --out flare_no_tde.txt` saves the model if we want to change what counts
as "known". `fetch_rare_transients.py` re-downloads the peculiar transients from public brokers.
