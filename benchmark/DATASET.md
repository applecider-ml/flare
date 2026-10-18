# The BTS benchmark

Spectroscopically confirmed ZTF transients for photometric classification and
anomaly detection, as used in the FLARE paper. 10,549 objects pass the quality
cut (at least 8 detections, with at least 2 in each of *g* and *r*, within 100
days of the first detection): 10,497 belong to the six trained classes and 52 to
families outside the taxonomy. AT 2018cow (ZTF18abcfcoo), listed as "other" in
the BTS catalogue, is added by hand, which brings the out-of-taxonomy hold-out
to 53 objects and the benchmark to 10,550.

| class | N | | held out | N |
|---|---|---|---|---|
| SN Ia | 7159 | | classical novae | 37 |
| SN CC | 2559 | | LBV | 6 |
| CV | 304 | | LRN | 4 |
| AGN | 251 | | ILRT | 3 |
| SLSN | 156 | | Ca-rich | 2 |
| TDE | 68 | | FBOT | 1 |

## What is in this directory

| file | content |
|---|---|
| `light_curves.tar.gz` | the light curves of all 10,550 objects (`events/<obj_id>.npz`, 8 MB) |
| `splits/{train,val,test}.csv` | the splits of the paper (7347 / 1575 / 1575), stratified |
| `splits/anomaly.csv` | the out-of-taxonomy hold-out (53 objects) |
| `features_all.parquet` | the 160 light-curve features, keyed by `obj_id` |
| `hosts.csv` | Pan-STARRS DR2 host photometry per position |
| `hosts_photoz.csv` | Legacy Surveys DR10 (DR9 fallback) host photometric redshifts |
| `context_gaia_wise.csv` | Gaia DR3 (2") and AllWISE (6") point-source context |
| `ps1_position.csv` | Pan-STARRS DR2 pre-outburst counterpart at the position (1.5") |
| `bts_catalog.csv` | snapshot of the BTS sample-explorer catalogue |
| `manifest_*.csv` | the graded-novelty tiers (core / peculiar / novel / ambiguous) |

Join the tables by `obj_id`, never by row position.

## Light curves

```
tar -xzf benchmark/light_curves.tar.gz -C benchmark/
```

gives `benchmark/events/<obj_id>.npz`, the path in the `filepath` column of the
split files. Each file holds an `(N, 15)` float32 array of detections (time since
the first detection, time since the previous one, band, log-flux and its error,
band one-hots and the paired colours) and its column names.

The light curves can also be rebuilt from public photometry with
`python paper/build_bts_dataset.py`: set `BOOM_URL` (and `BOOM_TOKEN`, or
`BOOM_USERNAME` / `BOOM_PASSWORD`) to query through the BOOM broker; without
credentials the public ALeRCE API is used.

## Provenance

Labels: ZTF Bright Transient Survey sample explorer (spectroscopic types).
Photometry: ZTF alerts via BOOM/ALeRCE. Host photometry: Pan-STARRS DR2 via the
MAST catalog API. Photometric redshifts: Legacy Surveys DR10/DR9 via the Astro
Data Lab TAP service. The quality cuts, the taxonomy mapping and the novelty
tiers are implemented in `paper/build_bts_dataset.py` and `paper/make_splits.py`.
