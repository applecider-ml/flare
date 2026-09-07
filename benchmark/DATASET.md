# The BTS benchmark

Spectroscopically confirmed ZTF transients for photometric classification and
anomaly detection. 10,549 objects pass the quality cut (≥8 detections, ≥2 in
each of *g* and *r*, within 100 days of first detection); 10,497 belong to the
six trained classes and 52 to out-of-taxonomy families, with five curated
objects bringing the anomaly hold-out to 57.

| class | N | | held out | N |
|---|---|---|---|---|
| SN Ia | 7159 | | novae | 37 |
| SN CC | 2559 | | LBV | 6 |
| CV | 304 | | Ca-rich | 5 |
| AGN | 251 | | LRN | 4 |
| SLSN | 156 | | ILRT | 3 |
| TDE | 68 | | FBOT | 2 |

## What is in this directory

| file | content |
|---|---|
| `splits/{train,val,test}.csv` | the paper's splits (7347 / 1575 / 1575), stratified |
| `splits/anomaly.csv` | the out-of-taxonomy hold-out |
| `features_all.parquet` | the 160 light-curve features, keyed by `obj_id` |
| `hosts.csv` | Pan-STARRS DR2 host photometry per position |
| `hosts_photoz.csv` | Legacy Surveys DR10 (DR9 fallback) host photo-z |
| `context_gaia_wise.csv` | Gaia DR3 astrometry (2") + AllWISE mid-IR (6") at each position |
| `ps1_position.csv` | PS1 DR2 pre-outburst counterpart at the position (1.5") |
| `bts_catalog.csv` | the BTS sample-explorer catalogue snapshot |
| `manifest_*.csv` | the graded-novelty tiers (core / peculiar / novel / ambiguous) |

Always join by `obj_id`, never by row position (the splits are re-derived
whenever the taxonomy changes; positional joins misalign silently).

## Light curves

The per-object event arrays (`(N, 15)` float32 per object, ~59 MB total) are
not stored in git.

- **Zenodo**: the archived copy lives at DOI TO-BE-MINTED (upload pending).
  Unpack into `benchmark/events/` and the split manifests' `filepath` column
  resolves against it.
- **Re-fetch**: `python paper/build_bts_dataset.py` rebuilds them from public
  photometry. Set `BOOM_URL` (and `BOOM_TOKEN`, or `BOOM_USERNAME` /
  `BOOM_PASSWORD`) to query through the BOOM broker; without credentials the
  public ALeRCE API is used. The script is resumable via its manifest. The
  same backend selection is available programmatically as
  `flare.fetch.fetch_events(oid)`.

## Provenance

Labels: ZTF Bright Transient Survey sample explorer (spectroscopic types).
Photometry: ZTF public alerts via BOOM/ALeRCE. Host photometry: Pan-STARRS DR2
via the MAST catalog API. Photometric redshifts: Legacy Surveys DR10/DR9 via
the Astro Data Lab TAP service. Quality cuts, taxonomy mapping and the graded
novelty tiers are documented in the paper and implemented in
`paper/build_bts_dataset.py` and `paper/make_splits.py`.
