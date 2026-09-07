# docs — the project page and how the module reports

This folder is served as the project's GitHub Pages site at
https://applecider-ml.github.io/flare/ (Settings → Pages → branch `main`,
folder `/docs`). `index.html` is the landing page; `.nojekyll` keeps GitHub
from running Jekyll over the static files.

Two renderings of the same per-alert output, both built from genuine model
outputs on real ZTF objects (no invented values anywhere).

| file | what it is |
|---|---|
| `flare_console.html` | **the production proposal**: a triage queue plus the full per-object case. Our own design; this is how we would want a deployment to report. Built by `make_production_console.py` from `prod_demo.json`. |
| `boom_panel.html` | the same content rendered inside **BOOM's** own design system, as a concrete integration proposal for the broker team. Built by `make_boom_panel.py` from `boom_demo.json`; every token, tile rule and band colour is read from boom-astro/boom's frontend source. |

Regenerate the data with `paper/conformal_examples.py` (per-object conformal
and anomaly outputs) — the JSON files here are snapshots of exactly that.

Both pages state the two things a score alone cannot: the error the prediction
set admits (90% ± 10% at α = 0.10, plus the coverage measured out of fold) and
the probability that the object is out of taxonomy, with the base rate that
probability assumes.
