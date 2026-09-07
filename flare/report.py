"""Render per-alert FLARE output as a self-contained HTML console.

One page: a triage queue and the full case for each object. Used by
`flare report` and by docs/make_production_console.py. Every value shown is
model output; nothing here invents a number.

A record is the dict produced by `flare.report.build_record`, or anything with
the same keys: conformal (from HierarchicalFlare.conformal_report), anom (from
anomaly_probability), proba, mag, ctx and the identity fields.
"""
import json


CLS = ["SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"]
NICE = {"SN_Ia": "SN Ia", "SN_CC": "SN CC", "SLSN": "SLSN",
        "AGN": "AGN", "TDE": "TDE", "CV": "CV"}
BAND = {"g": "#4ea72e", "r": "#e0483c", "i": "#e0a03c"}


def status(v):
    """Three states, decided by the two calibrated quantities, not by taste."""
    a, c = v["anom"], v["conformal"]
    if a["novelty_p"] <= 0.01:
        return ("anomaly", "Anomaly",
                "Energy clears the 1% false-alarm budget"
                + (" and the prediction set excludes the argmax class"
                   if c["predicted"] not in c["set"] else ""))
    if c["predicted"] in ("TDE", "SLSN") and len(c["set"]) == 1:
        return ("follow", "Follow-up",
                "Singleton set for a class whose defining property only "
                "spectroscopy resolves")
    if len(c["set"]) > 1:
        return ("ambig", "Ambiguous",
                f"{len(c['set'])} classes survive at 10% error — hand over as "
                "a candidate set, not a label")
    return ("routine", "Routine",
            "Singleton set, energy in the bulk of the known-class stream")


def evidence(ctx, pred):
    """The context that actually moved the decision, in physical language."""
    out = []
    if "gaia_sep" in ctx:
        out.append(("Gaia point source", f'{ctx["gaia_sep"]:.2f}&Prime; away',
                    "a persistent source sits at the position — nuclear or stellar"))
    else:
        out.append(("Gaia point source", "none",
                    "nothing persistent at the position above the Gaia limit"))
    if "w1w2" in ctx:
        agn = ctx["w1w2"] > 0.8
        out.append(("WISE W1&minus;W2", f'{ctx["w1w2"]:.2f}',
                    "mid-infrared colour of an accreting nucleus" if agn
                    else "no mid-infrared AGN excess"))
    else:
        out.append(("WISE W1&minus;W2", "no match", "below the AllWISE limit"))
    if "pos_sep" in ctx:
        out.append(("Pre-outburst counterpart",
                    f'{ctx["pos_sep"]:.2f}&Prime; · r = {ctx.get("pos_r", float("nan")):.1f}',
                    "Pan-STARRS saw something here years before the outburst"))
    else:
        out.append(("Pre-outburst counterpart", "none",
                    "nothing at this position in Pan-STARRS before the transient"))
    if "host_sep" in ctx:
        nuc = ctx["host_sep"] < 0.5
        out.append(("Host offset", f'{ctx["host_sep"]:.2f}&Prime;',
                    "on the nucleus" if nuc else "displaced from its host"))
    else:
        out.append(("Host offset", "no host", "no extended source within 30&Prime;"))
    if "z_phot" in ctx:
        out.append(("Photometric redshift", f'{ctx["z_phot"]:.3f}',
                    f'gives M<sub>pseudo</sub> = {ctx.get("M_pseudo", float("nan")):.1f}'))
    else:
        out.append(("Photometric redshift", "none", "no distance proxy available"))
    return out


def lc_svg(mag, w=620, h=200):
    pl, pr, pt, pb = 46, 14, 16, 32
    pts = [(t, m) for b in mag for t, m in mag[b]]
    if not pts:
        return ""
    xs = [p[0] for p in pts]; ms = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    m0, m1 = min(ms) - .3, max(ms) + .3
    X = lambda v: pl + (v - x0) / max(x1 - x0, 1e-9) * (w - pl - pr)
    Y = lambda v: pt + (v - m0) / max(m1 - m0, 1e-9) * (h - pt - pb)
    s = [f'<svg viewBox="0 0 {w} {h}" class="lc" role="img" '
         f'aria-label="light curve, magnitude against days from first detection">']
    for k in range(5):
        yy = pt + k * (h - pt - pb) / 4
        s.append(f'<line x1="{pl}" y1="{yy:.1f}" x2="{w-pr}" y2="{yy:.1f}" class="gl"/>')
        s.append(f'<text x="{pl-8}" y="{yy+3.5:.1f}" class="tk" text-anchor="end">'
                 f'{m0 + k*(m1-m0)/4:.1f}</text>')
    for k in range(5):
        xx = pl + k * (w - pl - pr) / 4
        s.append(f'<text x="{xx:.0f}" y="{h-12}" class="tk" text-anchor="middle">'
                 f'{x0 + k*(x1-x0)/4:.0f}</text>')
    s.append(f'<text x="{(pl+w-pr)/2:.0f}" y="{h-1}" class="al" text-anchor="middle">'
             f'days from first detection</text>')
    s.append(f'<text x="12" y="{(pt+h-pb)/2:.0f}" class="al" text-anchor="middle" '
             f'transform="rotate(-90 12 {(pt+h-pb)/2:.0f})">magnitude</text>')
    for b in ("g", "r", "i"):
        if b not in mag:
            continue
        for t, m in sorted(mag[b]):
            s.append(f'<circle cx="{X(t):.1f}" cy="{Y(m):.1f}" r="3.2" '
                     f'fill="{BAND[b]}" opacity=".9"/>')
    for i, b in enumerate([b for b in ("g", "r", "i") if b in mag]):
        s.append(f'<circle cx="{w-pr-52}" cy="{pt+8+i*15}" r="3.2" fill="{BAND[b]}"/>'
                 f'<text x="{w-pr-42}" y="{pt+11.5+i*15}" class="tk">ZTF {b}</text>')
    s.append('</svg>')
    return "".join(s)


def detail(oid, v):
    c, a = v["conformal"], v["anom"]
    st, label, why = status(v)
    setstr = ", ".join(NICE[x] for x in c["set"]) if c["set"] else "empty"
    probs = "".join(
        f'<div class="pr{" in" if k in c["set"] else ""}{" top" if k == c["predicted"] else ""}">'
        f'<span class="pk">{NICE[k]}</span>'
        f'<span class="pbar"><i style="width:{v["proba"][k]*100:.1f}%"></i></span>'
        f'<span class="pn">{v["proba"][k]*100:.0f}%</span>'
        f'<span class="pp">p&nbsp;{c["p_values"][k]:.3f}</span></div>' for k in CLS)
    ev = "".join(f'<tr><td class="ek">{k}</td><td class="ev">{val}</td>'
                 f'<td class="ew">{note}</td></tr>' for k, val, note in evidence(v["ctx"], c["predicted"]))
    excluded = c["predicted"] not in c["set"]
    return f'''<article class="detail" id="d-{oid}" hidden>
  <header class="dhead">
    <div>
      <h2 class="mono">{oid}</h2>
      <p class="sub">{v["iau"]} &middot; {v["ra"]} {v["dec"]} &middot; {v["n_det"]} detections
        over {v["span"]:.0f} d &middot; peak {v["peakmag"]:.2f}</p>
    </div>
    <span class="chip {st}">{label}</span>
  </header>
  <div class="action {st}"><b>{label}.</b> {why}.</div>

  <section class="panel">
    <h3>Light curve</h3>
    {lc_svg(v["mag"])}
  </section>

  <div class="cols">
    <section class="panel">
      <h3>Classification <span class="tag">6 classes · 190 features</span></h3>
      {probs}
      <div class="setline">
        <div class="setbox mono">{{{setstr}}}</div>
        <div class="guarantee">
          <span class="gbig">90% <span class="pm">&plusmn; 10%</span></span>
          <span class="gtxt">coverage the set is calibrated for &mdash; it may
            miss the true class one time in ten, by construction</span>
        </div>
        <table class="kv">
          <tr><td>measured out of fold, {NICE[c["predicted"]]}</td>
            <td class="n">{c["empirical_coverage"]*100:.1f}% &plusmn; {v["cov_err"]*100:.1f}%
              <span class="dim">n&nbsp;=&nbsp;{v["cov_n"]}</span></td></tr>
          <tr><td>credibility <span class="dim">does it match any class at all</span></td>
            <td class="n">{c["credibility"]:.3f}</td></tr>
          <tr><td>confidence <span class="dim">is the runner-up excluded</span></td>
            <td class="n">{c["confidence"]:.3f}</td></tr>
        </table>
        {'<p class="flagline">The most probable class is <b>not in the set</b>: its p-value is below the 10% error level, so the classifier is contradicting itself.</p>' if excluded else ''}
      </div>
    </section>

    <section class="panel">
      <h3>Anomaly <span class="tag">own 181-column space</span></h3>
      <div class="pbig {st}">
        <span class="pnum">{a["p"]*100:.1f}%</span>
        <span class="pci">({a["lo"]*100:.0f}&ndash;{a["hi"]*100:.0f}%)</span>
      </div>
      <p class="psub">probability this is outside the six-class taxonomy, at the
        benchmark's {a["base"]*100:.1f}% base rate</p>
      <table class="kv">
        <tr><td>likelihood ratio <span class="dim">carries no prior — rescale
          for your stream</span></td><td class="n">{a["lr"]:.1f}&times;</td></tr>
        <tr><td>novelty p-value <span class="dim">1 known-class object in
          {1/a["novelty_p"]:.0f} is this extreme</span></td>
          <td class="n">{a["novelty_p"]:.4f}</td></tr>
        <tr><td>energy</td><td class="n">{v["energy"]:+.2f}</td></tr>
      </table>
      <div class="gauge">
        <div class="gtrack"><div class="gfill" style="width:{a["pct"]:.1f}%"></div>
          <div class="gthr" title="1% false-alarm budget"></div></div>
        <div class="gcap"><span>percentile against the known-class stream</span>
          <b>{a["pct"]:.1f}</b></div>
      </div>
    </section>
  </div>

  <section class="panel">
    <h3>Evidence <span class="tag">context at alert time, no spectrum</span></h3>
    <table class="eve">{ev}</table>
  </section>

  <footer class="prov">
    <span><b>truth (held out):</b> {v["truth"]}</span>
    <span>bts6 · conformal &alpha;=0.10, calibrated on 1575 held-out objects ·
      anomaly calibrated on 1575 known + 52 out-of-taxonomy</span>
  </footer>
</article>'''


def queue_row(oid, v, i):
    c, a = v["conformal"], v["anom"]
    st, label, _ = status(v)
    return f'''<button class="qrow{' sel' if i == 0 else ''}" data-id="{oid}"
      aria-controls="d-{oid}">
  <span class="qdot {st}" aria-hidden="true"></span>
  <span class="qmain">
    <span class="qid mono">{oid}</span>
    <span class="qcls">{NICE[c["predicted"]]}{'' if len(c["set"]) == 1 else f' · {len(c["set"])} in set'}</span>
  </span>
  <span class="qp">{a["p"]*100:.0f}%</span>
</button>'''


CSS = """
:root{
  --ground:#14101a; --panel:#1c1622; --raised:#241d2c; --line:#332a3d;
  --ink:#f3edf4; --dim:#a99cb3; --faint:#7d7189;
  --rose:#e0679a; --plum:#a06cd5;
  --ok:#4ea72e; --warn:#e0a03c; --bad:#e0483c; --amber:#c88a2e;
}
:root[data-theme="light"]{
  --ground:#faf7fb; --panel:#ffffff; --raised:#f4eef6; --line:#e6dced;
  --ink:#231c2a; --dim:#6b5f75; --faint:#8d8296;
  --rose:#c2477a; --plum:#6d3fa0;
  --ok:#3d7d2a; --warn:#a8721f; --bad:#b3373a;
}
@media (prefers-color-scheme:light){
  :root:not([data-theme="dark"]){
    --ground:#faf7fb; --panel:#ffffff; --raised:#f4eef6; --line:#e6dced;
    --ink:#231c2a; --dim:#6b5f75; --faint:#8d8296;
    --rose:#c2477a; --plum:#6d3fa0;
    --ok:#3d7d2a; --warn:#a8721f; --bad:#b3373a;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  font-size:14px;line-height:1.55;-webkit-font-smoothing:antialiased}
.mono{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace}
.dim{color:var(--dim);font-weight:400}
h1,h2,h3{font-family:Fraunces,Georgia,serif;margin:0;text-wrap:balance}
/* ---- masthead ---- */
.top{display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap;
  padding:1.1rem 1.5rem;border-bottom:1px solid var(--line);background:var(--panel)}
.brand{font-size:1.35rem;font-weight:600;letter-spacing:-.01em}
.brand span{color:var(--rose)}
.tagline{color:var(--dim);font-size:.85rem;max-width:60ch}
.build{margin-left:auto;font-size:.72rem;color:var(--faint);
  font-family:"IBM Plex Mono",monospace;text-align:right}
.proposal{margin:0;padding:.5rem 1.5rem;background:var(--raised);
  border-bottom:1px solid var(--line);font-size:.78rem;color:var(--dim)}
.proposal b{color:var(--ink)}
/* ---- layout ---- */
.shell{display:grid;grid-template-columns:19rem minmax(0,1fr);gap:0;
  min-height:calc(100vh - 8rem)}
@media (max-width:900px){.shell{grid-template-columns:1fr}}
.queue{border-right:1px solid var(--line);background:var(--panel);
  padding:1rem .75rem;display:flex;flex-direction:column;gap:.25rem}
.qhead{display:flex;justify-content:space-between;align-items:baseline;
  padding:0 .5rem .5rem;font-size:.72rem;letter-spacing:.09em;
  text-transform:uppercase;color:var(--faint)}
.qrow{display:flex;align-items:center;gap:.6rem;width:100%;text-align:left;
  background:none;border:1px solid transparent;border-radius:.5rem;
  padding:.5rem .6rem;cursor:pointer;color:inherit;font:inherit}
.qrow:hover{background:var(--raised)}
.qrow.sel{background:var(--raised);border-color:var(--line)}
.qrow:focus-visible{outline:2px solid var(--rose);outline-offset:1px}
.qdot{width:.5rem;height:.5rem;border-radius:99px;flex:0 0 auto}
.qdot.routine{background:var(--ok)}.qdot.follow{background:var(--plum)}
.qdot.ambig{background:var(--warn)}.qdot.anomaly{background:var(--bad)}
.qmain{display:flex;flex-direction:column;min-width:0;flex:1}
.qid{font-size:.8rem}
.qcls{font-size:.72rem;color:var(--dim)}
.qp{font-size:.8rem;font-variant-numeric:tabular-nums;color:var(--dim)}
.stage{padding:1.25rem 1.5rem 2.5rem;min-width:0}
/* ---- detail ---- */
.dhead{display:flex;align-items:flex-start;justify-content:space-between;
  gap:1rem;flex-wrap:wrap;margin-bottom:.8rem}
.dhead h2{font-size:1.5rem}
.sub{margin:.15rem 0 0;color:var(--dim);font-size:.82rem}
.chip{font-size:.72rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
  padding:.2rem .7rem;border-radius:99px;border:1px solid}
.chip.routine{color:var(--ok);border-color:var(--ok)}
.chip.follow{color:var(--plum);border-color:var(--plum)}
.chip.ambig{color:var(--warn);border-color:var(--warn)}
.chip.anomaly{color:var(--bad);border-color:var(--bad)}
.action{padding:.6rem .9rem;border-radius:.5rem;background:var(--panel);
  border-left:3px solid var(--line);margin-bottom:1rem;font-size:.88rem}
.action.routine{border-left-color:var(--ok)}
.action.follow{border-left-color:var(--plum)}
.action.ambig{border-left-color:var(--warn)}
.action.anomaly{border-left-color:var(--bad)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:.75rem;
  padding:1rem 1.1rem;margin-bottom:1rem}
.panel h3{font-size:1.05rem;margin-bottom:.7rem;display:flex;
  align-items:baseline;gap:.6rem;flex-wrap:wrap}
.tag{font-family:"IBM Plex Sans",sans-serif;font-size:.7rem;font-weight:400;
  color:var(--faint);letter-spacing:.02em}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media (max-width:780px){.cols{grid-template-columns:1fr}}
.lc{width:100%;height:auto;display:block}
.gl{stroke:var(--line)}
.tk{font-size:9.5px;fill:var(--faint);font-family:"IBM Plex Mono",monospace}
.al{font-size:9.5px;fill:var(--faint)}
/* probabilities */
.pr{display:grid;grid-template-columns:3.6rem 1fr 2.4rem 3.9rem;align-items:center;
  gap:.5rem;padding:.1rem 0;font-size:.8rem}
.pk{color:var(--dim)}
.pr.top .pk{color:var(--ink);font-weight:600}
.pbar{height:.5rem;background:var(--raised);border-radius:3px;overflow:hidden}
.pbar i{display:block;height:100%;background:var(--line)}
.pr.in .pbar i{background:var(--rose)}
.pr.top .pbar i{background:var(--plum)}
.pn{text-align:right;font-variant-numeric:tabular-nums}
.pp{text-align:right;font-size:.72rem;color:var(--faint);
  font-family:"IBM Plex Mono",monospace}
.setline{margin-top:.8rem;border-top:1px solid var(--line);padding-top:.8rem}
.setbox{background:var(--raised);border-radius:.5rem;padding:.4rem .7rem;
  font-size:.95rem;margin-bottom:.6rem}
.guarantee{display:flex;align-items:baseline;gap:.7rem;flex-wrap:wrap;
  margin-bottom:.6rem}
.gbig{font-family:Fraunces,serif;font-size:1.6rem;font-weight:600;
  font-variant-numeric:tabular-nums;line-height:1}
.pm{color:var(--dim);font-size:1.1rem}
.gtxt{color:var(--dim);font-size:.75rem;flex:1;min-width:12rem}
.kv{width:100%;border-collapse:collapse}
.kv td{padding:.2rem 0;font-size:.78rem;color:var(--dim);vertical-align:top}
.kv td.n{text-align:right;color:var(--ink);white-space:nowrap;
  font-variant-numeric:tabular-nums;font-family:"IBM Plex Mono",monospace}
.flagline{margin:.6rem 0 0;padding:.5rem .7rem;border-radius:.4rem;
  background:var(--raised);font-size:.78rem;border-left:3px solid var(--bad)}
.pbig{display:flex;align-items:baseline;gap:.5rem}
.pnum{font-family:Fraunces,serif;font-size:2.1rem;font-weight:600;line-height:1;
  font-variant-numeric:tabular-nums}
.pbig.anomaly .pnum{color:var(--bad)}
.pci{color:var(--dim);font-size:.9rem;font-variant-numeric:tabular-nums}
.psub{margin:.15rem 0 .6rem;color:var(--dim);font-size:.75rem}
.gauge{margin-top:.7rem}
.gtrack{position:relative;height:.5rem;background:var(--raised);border-radius:3px}
.gfill{height:100%;background:linear-gradient(90deg,var(--plum),var(--rose));
  border-radius:3px}
.gthr{position:absolute;top:-4px;bottom:-4px;left:99%;width:2px;background:var(--bad)}
.gcap{display:flex;justify-content:space-between;font-size:.72rem;
  color:var(--faint);margin-top:.3rem}
.gcap b{color:var(--ink);font-variant-numeric:tabular-nums}
.eve{width:100%;border-collapse:collapse}
.eve td{padding:.35rem 0;border-bottom:1px solid var(--line);font-size:.8rem;
  vertical-align:top}
.eve tr:last-child td{border-bottom:0}
.ek{color:var(--dim);width:11rem}
.ev{font-family:"IBM Plex Mono",monospace;width:9rem;white-space:nowrap}
.ew{color:var(--faint);font-size:.76rem}
.prov{display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;
  font-size:.74rem;color:var(--faint);padding-top:.3rem}
.prov b{color:var(--dim);font-weight:600}
.legend{padding:0 1.5rem 2rem;color:var(--faint);font-size:.78rem;max-width:78ch}
.legend b{color:var(--dim)}
"""

JS = """
const rows = document.querySelectorAll('.qrow');
const shows = id => {
  document.querySelectorAll('.detail').forEach(d => d.hidden = d.id !== 'd-' + id);
  rows.forEach(r => r.classList.toggle('sel', r.dataset.id === id));
};
rows.forEach(r => r.addEventListener('click', () => shows(r.dataset.id)));
shows(rows[0].dataset.id);
"""


def page(demo):
    order = sorted(demo, key=lambda o: -demo[o]["anom"]["p"])
    q = "".join(queue_row(o, demo[o], i) for i, o in enumerate(order))
    d = "".join(detail(o, demo[o]) for o in order)
    n_anom = sum(1 for o in order if status(demo[o])[0] == "anomaly")
    return f'''<title>FLARE Alert Console</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<header class="top">
  <div>
    <div class="brand">FL<span>A</span>RE</div>
    <p class="tagline">Photometric triage for optical transients: six classes, a
      calibrated prediction set, and a probability that the object belongs to
      none of them.</p>
  </div>
  <div class="build">bts6 · 190 features<br>&alpha; = 0.10 · 1% false-alarm budget</div>
</header>
<p class="proposal"><b>Production proposal.</b> Every number on this page is a
  genuine output of the shipped models on real ZTF objects — including one the
  detector misses. What a live deployment would add is throughput, not different
  numbers.</p>
<div class="shell">
  <nav class="queue" aria-label="alert queue">
    <div class="qhead"><span>Queue · {len(order)}</span><span>{n_anom} flagged</span></div>
    {q}
  </nav>
  <main class="stage">{d}</main>
</div>
<p class="legend"><b>How to read it.</b> The prediction set is the operational
  output, not the argmax: it is calibrated so that it contains the true class
  90% of the time, and the 10% it admits is stated rather than hidden. A set
  with two classes is the model declining to choose, which is the correct
  hand-off to spectroscopy. The anomaly probability is separate machinery —
  scored in its own representation, calibrated against the benchmark's
  out-of-taxonomy objects — and it is quoted with the base rate it assumes,
  because the same likelihood ratio gives a very different probability in a raw
  alert stream than in a spectroscopically confirmed sample.</p>
<script>{JS}</script>'''



def render(records: dict) -> str:
    """The whole page as a string."""
    return page(records)


# ---------------------------------------------------------------------------
# building a record from an object id, end to end
# ---------------------------------------------------------------------------
def build_record(obj_id, clf=None, ra=None, dec=None, peak_mag=None,
                 events=None, context=None, horizon_days=100.0):
    """Everything the console needs for one object, from an id or an array.

    Fetches photometry (BOOM if configured, else ALeRCE) and the four external
    context blocks unless they are passed in, runs the classifier, the
    conformal layer and the anomaly layer, and returns a record. Any block
    that cannot be fetched is left missing rather than imputed -- the models
    are trained for that.
    """
    import numpy as np
    import pandas as pd

    from . import data as D
    from .config import BTS6
    from .features import extract_from_array

    if clf is None:
        from . import load_classifier
        clf = load_classifier("bts6")

    if events is None:
        from .fetch import coords, fetch_photometry, to_events
        rows = fetch_photometry(obj_id)
        events = to_events(rows, horizon_days=horizon_days)
        if events is None:
            raise RuntimeError(f"no photometry for {obj_id}")
        if ra is None or dec is None:
            c = coords(obj_id)
            if c:
                ra, dec = c
    events = events[events[:, 0] <= horizon_days]

    if context is None and ra is not None and dec is not None:
        from .context import context_features
        bands0 = D.reconstruct_bands(events)
        if peak_mag is None and bands0:
            peak_mag = float(min(23.9 - 2.5 * b.logflux.max() for b in bands0.values()))
        context = context_features(ra, dec, peak_mag=peak_mag)
    context = {k: float(v) for k, v in (context or {}).items()
               if isinstance(v, (int, float)) and np.isfinite(v)}

    X = pd.DataFrame([extract_from_array(events)])
    if context:
        X = pd.concat([X, pd.DataFrame([context])], axis=1)
    for col in clf.feature_names:
        if col not in X.columns:
            X[col] = np.nan
    X = X[clf.feature_names]

    conf = clf.conformal_report(X)[0]
    anom = clf.anomaly_probability(X)[0]
    bands = D.reconstruct_bands(events)
    mag = {b: [[float(t), float(23.9 - 2.5 * f)]
               for t, f in zip(bands[b].t, bands[b].logflux)] for b in bands}
    cov = conf["empirical_coverage"] or float("nan")
    n = dict(SN_Ia=7159, SN_CC=2559, SLSN=156, AGN=251, TDE=68,
             CV=304).get(conf["predicted"], 0)
    return dict(
        iau="", ra="" if ra is None else f"{ra:.5f}",
        dec="" if dec is None else f"{dec:+.5f}",
        truth="not labelled", n_det=int(len(events)),
        span=float(events[:, 0].max()) if len(events) else 0.0,
        peakmag=peak_mag if peak_mag is not None else float("nan"),
        conformal=conf,
        cov_err=float(np.sqrt(cov * (1 - cov) / n)) if n else float("nan"),
        cov_n=n,
        proba={c: round(float(p), 4)
               for c, p in zip(BTS6.classes, clf.predict_proba(X)[0])},
        anom=dict(p=anom["p_anomaly"], lo=float("nan"), hi=float("nan"),
                  lr=anom["likelihood_ratio"], base=anom["base_rate"],
                  novelty_p=anom.get("novelty_p", float("nan")),
                  pct=anom.get("energy_percentile", float("nan"))),
        energy=round(anom["energy"], 2), ctx=context, mag=mag)
