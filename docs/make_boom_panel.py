"""Render the FLARE alert panel as it would appear inside BOOM's own webapp.

Not a screenshot and not an invention: the chrome, the card grid, the "ML
Scores" tiles, the band colours and every design token below are read out of
boom-astro/boom's frontend source (Tailwind v4 tokens in src/index.css,
ClassificationsV2.tsx, Lightcurve.tsx, CrossmatchCard.tsx, section-cards.tsx),
and the FLARE numbers are genuine outputs of the shipped bts6 models.

Usage:  python make_boom_panel.py boom_demo.json out.html
"""
import json
import sys

# ---- BOOM's own tokens, verbatim from frontend/src/index.css ---------------
TOK_DARK = dict(bg="oklch(0.141 0.005 285.823)", card="oklch(0.21 0.006 285.885)",
                fg="oklch(0.985 0 0)", muted="oklch(0.274 0.006 286.033)",
                mutedfg="oklch(0.705 0.015 286.067)", border="oklch(1 0 0 / 10%)",
                sidebar="oklch(0.21 0.006 285.885)", accent="oklch(0.274 0.006 286.033)")
TOK_LIGHT = dict(bg="oklch(1 0 0)", card="oklch(1 0 0)",
                 fg="oklch(0.141 0.005 285.823)", muted="oklch(0.967 0.001 286.375)",
                 mutedfg="oklch(0.552 0.016 285.938)", border="oklch(0.92 0.004 286.32)",
                 sidebar="oklch(0.985 0 0)", accent="oklch(0.967 0.001 286.375)")
ZTF = "oklch(0.6 0.142 253.497)"          # --ztf
BAND = {"g": "#38b000ea", "r": "#ef233be7", "i": "#fcc049e3"}   # Lightcurve.tsx
GREEN, AMBER, RED = "#10b981", "#f59e0b", "#ef4444"             # ClassificationsV2
NICE = {"SN_Ia": "SN Ia", "SN_CC": "SN CC", "SLSN": "SLSN",
        "AGN": "AGN", "TDE": "TDE", "CV": "CV"}
CLS = ["SN_Ia", "SN_CC", "SLSN", "AGN", "TDE", "CV"]


def tile(name, score, sub=""):
    """CompactHeatmap tile: three-bucket traffic light at 0.7 / 0.4."""
    bg = GREEN if score > 0.7 else AMBER if score > 0.4 else RED
    return (f'<div class="tile" style="background:{bg}">'
            f'<div class="trow"><span class="tname">{name}</span></div>'
            f'<div class="trow2"><span class="tpct">{score*100:.0f}%</span>'
            f'<span class="tsub">{sub}</span></div></div>')


def lightcurve(mag, mjd0, w=560, h=210):
    pad_l, pad_r, pad_t, pad_b = 44, 12, 14, 30
    pts = [(mjd0 + t, m, b) for b in mag for t, m, _ in mag[b]]
    if not pts:
        return ""
    xs = [p[0] for p in pts]; ms = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    m0, m1 = min(ms) - 0.25, max(ms) + 0.25          # inverted: bright at top
    X = lambda v: pad_l + (v - x0) / max(x1 - x0, 1e-9) * (w - pad_l - pad_r)
    Y = lambda v: pad_t + (v - m0) / max(m1 - m0, 1e-9) * (h - pad_t - pad_b)
    s = [f'<svg viewBox="0 0 {w} {h}" class="lc">']
    for k in range(5):
        yy = pad_t + k * (h - pad_t - pad_b) / 4
        mv = m0 + k * (m1 - m0) / 4
        s.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{w-pad_r}" y2="{yy:.1f}" class="gl"/>')
        s.append(f'<text x="{pad_l-6}" y="{yy+3:.1f}" class="tick" text-anchor="end">{mv:.1f}</text>')
    for k in range(4):
        xx = pad_l + k * (w - pad_l - pad_r) / 3
        xv = x0 + k * (x1 - x0) / 3
        s.append(f'<text x="{xx:.0f}" y="{h-10}" class="tick" text-anchor="middle">{xv:.0f}</text>')
    s.append(f'<text x="{w/2:.0f}" y="{h-0.5}" class="axl" text-anchor="middle">MJD</text>')
    s.append(f'<text x="11" y="{h/2:.0f}" class="axl" text-anchor="middle" '
             f'transform="rotate(-90 11 {h/2:.0f})">magnitude</text>')
    for b in ("g", "r", "i"):
        if b not in mag:
            continue
        for t, m, e in mag[b]:
            x, y = X(mjd0 + t), Y(m)
            s.append(f'<line x1="{x:.1f}" y1="{Y(m-e):.1f}" x2="{x:.1f}" '
                     f'y2="{Y(m+e):.1f}" stroke="{BAND[b]}" stroke-width="1" opacity=".55"/>')
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{BAND[b]}"/>')
    lx = w - pad_r - 60
    for i, b in enumerate([b for b in ("g", "r", "i") if b in mag]):
        s.append(f'<circle cx="{lx}" cy="{pad_t+8+i*14}" r="3.5" fill="{BAND[b]}"/>'
                 f'<text x="{lx+9}" y="{pad_t+11+i*14}" class="tick">ZTF {b}</text>')
    s.append('</svg>')
    return "".join(s)


def xmatch(ctx):
    """Cross-matches card: BOOM lists catalogues with matches first, then the
    empty ones at opacity 40 with a '0 matches' badge and a disabled trigger."""
    cats = []
    if "host_sep" in ctx:
        cats.append(("PS1_DR2 (host)", [("sep [arcsec]", f'{ctx["host_sep"]:.2f}'),
                                        ("rKronMag", f'{ctx.get("host_r", float("nan")):.2f}'),
                                        ("iKronMag", f'{ctx.get("host_i", float("nan")):.2f}'),
                                        ("g-r", f'{ctx.get("host_gr", float("nan")):.2f}')]))
    if "pos_sep" in ctx:
        cats.append(("PS1_DR2 (at position)", [("sep [arcsec]", f'{ctx["pos_sep"]:.2f}'),
                                               ("rPSFMag", f'{ctx.get("pos_r", float("nan")):.2f}'),
                                               ("point source", "yes" if ctx.get("pos_star") == 1 else "no"),
                                               ("nDetections", f'{ctx.get("pos_ndet", 0):.0f}')]))
    if "z_phot" in ctx:
        cats.append(("LS_DR10_PHOTOZ", [("z_phot", f'{ctx["z_phot"]:.4f}'),
                                        ("M_pseudo", f'{ctx.get("M_pseudo", float("nan")):.2f}')]))
    if "gaia_sep" in ctx:
        cats.append(("Gaia_DR3", [("sep [arcsec]", f'{ctx["gaia_sep"]:.2f}'),
                                  ("phot_g_mean_mag", f'{ctx.get("gaia_g", float("nan")):.2f}'),
                                  ("parallax_over_error", f'{ctx.get("parallax_over_error", 0):.1f}')]))
    if "w1w2" in ctx:
        cats.append(("AllWISE", [("sep [arcsec]", f'{ctx.get("wise_sep", float("nan")):.2f}'),
                                 ("W1", f'{ctx.get("w1", float("nan")):.2f}'),
                                 ("W1-W2", f'{ctx["w1w2"]:.2f}')]))
    empties = [n for n, k in [("Gaia_DR3", "gaia_sep"), ("AllWISE", "w1w2"),
                              ("LS_DR10_PHOTOZ", "z_phot"),
                              ("PS1_DR2 (at position)", "pos_sep")] if k not in ctx]
    out = []
    for i, (name, rows) in enumerate(cats):
        body = "".join(f'<tr><td class="k">{k}</td><td class="v">{v}</td></tr>'
                       for k, v in rows)
        open_ = " open" if i == 0 else ""
        out.append(f'<details class="acc"{open_}><summary><span>{name}</span>'
                   f'<span class="badge">1 match</span></summary>'
                   f'<table class="xt">{body}</table></details>')
    for name in empties:
        out.append(f'<div class="acc empty"><summary><span>{name}</span>'
                   f'<span class="badge">0 matches</span></summary></div>')
    return "".join(out)


def flare_card(v):
    c = v["conformal"]
    a = v["anom"]
    setstr = ", ".join(NICE[x] for x in c["set"]) if c["set"] else "empty"
    pv = c["p_values"]
    rows = "".join(
        f'<div class="pv{" hit" if k in c["set"] else ""}">'
        f'<span class="pk">{NICE[k]}</span>'
        f'<span class="pb"><i style="width:{min(pv[k],1)*100:.1f}%"></i></span>'
        f'<span class="pn">{pv[k]:.3f}</span></div>' for k in CLS)
    nom = (1 - c["alpha"]) * 100
    err = c["alpha"] * 100
    cov = c["empirical_coverage"] * 100
    cerr = c["coverage_err"] * 100
    p = a["p_anomaly"] * 100
    plo, phi = a["p_lo"] * 100, a["p_hi"] * 100
    e_pct = a["energy_percentile"]
    over = a["novelty_p"] <= 0.01
    per = 1 / a["novelty_p"]
    return f'''
<section class="card span2">
  <div class="chead"><h3>FLARE <span class="dim">· photometric classification</span></h3>
    <span class="pill">bts6</span></div>
  <div class="cbody">
    <div class="two">
      <div>
        <div class="lab">prediction set</div>
        <div class="setbox">{{{setstr}}}</div>
        <div class="guar">
          <div class="gnum">{nom:.0f}% <span class="pm">&plusmn; {err:.0f}%</span></div>
          <div class="gsub">target coverage &plusmn; the error the set admits
            (&alpha; = {c["alpha"]:.2f}, class-conditional)</div>
        </div>
        <table class="err">
          <tr><td>measured out of fold, {NICE[c["predicted"]]}
              <span class="dim">n = {c["coverage_n"]}</span></td>
              <td class="n">{cov:.1f}% &plusmn; {cerr:.1f}%</td></tr>
          <tr><td>credibility <span class="dim">max p — matches any class at all</span></td>
              <td class="n">{c["credibility"]:.3f}</td></tr>
          <tr><td>confidence <span class="dim">1 &minus; second p</span></td>
              <td class="n">{c["confidence"]:.3f}</td></tr>
          <tr><td>p-value floor <span class="dim">1/(n<sub>cal</sub>+1)</span></td>
              <td class="n">{c["p_value_floor"]:.3f}</td></tr>
        </table>
      </div>
      <div>
        <div class="lab">conformal p-values <span class="dim">set = {{p &gt; &alpha;}}</span></div>
        {rows}
        <div class="lab" style="margin-top:.8rem">is it an anomaly?</div>
        <div class="anom {'alert' if over else 'ok'}">
          <div class="anum">{p:.1f}%<span class="pm"> ({plo:.0f}&ndash;{phi:.0f}%)</span></div>
          <div class="asub">probability it is out of taxonomy, at the benchmark&rsquo;s
            own {a["base_rate"]*100:.1f}% base rate</div>
          <table class="err" style="margin-top:.4rem">
            <tr><td>likelihood ratio <span class="dim">prior-free</span></td>
                <td class="n">{a["likelihood_ratio"]:.1f}&times;</td></tr>
            <tr><td>novelty p-value <span class="dim">1 in {per:.0f} known-class objects
                is this extreme</span></td><td class="n">{a["novelty_p"]:.4f}</td></tr>
          </table>
        </div>
      </div>
    </div>
    <div class="lab" style="margin-top:.9rem">anomaly energy
      <span class="dim">· own 181-column space, red mark = 1% false-alarm budget</span></div>
    <div class="gauge"><div class="gtrack"><div class="gfill" style="width:{e_pct:.1f}%"></div>
      <div class="gthr"></div></div>
      <div class="gcap"><span>percentile against the known-class stream</span>
        <b class="{'hot' if over else ''}">{e_pct:.1f}</b></div></div>
    <div class="verdict {'alert' if over else 'ok'}">{v["verdict"]}</div>
  </div>
</section>'''


ICONS = {  # lucide/tabler paths as used by the real sidebar
 "search": "M10 4a6 6 0 1 0 0 12 6 6 0 0 0 0-12M20 20l-4.5-4.5",
 "chart": "M4 19h16M7 16V9M12 16V5M17 16v-5",
 "tree": "M9 5h6v4H9zM3 15h6v4H3zM15 15h6v4h-6zM12 9v3M6 15v-3h12v3",
 "db": "M4 6c0-1.1 3.6-2 8-2s8 .9 8 2-3.6 2-8 2-8-.9-8-2M4 6v12c0 1.1 3.6 2 8 2s8-.9 8-2V6",
 "book": "M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z",
 "help": "M12 19h.01M9.1 9a3 3 0 1 1 4 2.8c-.8.4-1.1 1.2-1.1 2.2",
}


def page(demo):
    order = list(demo)
    main_id = order[-1]              # the nova: the interesting one
    v = demo[main_id]
    c = v["conformal"]
    rail = "".join(
        f'<div class="ic" title="{t}"><svg viewBox="0 0 24 24"><path d="{ICONS[k]}"/></svg></div>'
        + ('<div class="rsep"></div>' if sep else '')
        for k, t, sep in [("search", "Query", False), ("chart", "Dashboard", True),
                          ("tree", "Kafka Documentation", False),
                          ("db", "API Documentation", True),
                          ("book", "Acknowledgments", False), ("help", "Get Help", False)])

    def ml_scores(vv):
        tiles = "".join(tile(NICE[k], vv["proba"][k]) for k in CLS)
        return f'''
<section class="card">
  <div class="chead"><h3>ML Scores</h3><span class="sel">FLARE ▾</span></div>
  <div class="cbody">
    <div class="tabs"><span class="tab on">Current</span><span class="tab">Temporal</span></div>
    <div class="fam">FLARE</div>
    <div class="tiles">{tiles}</div>
    <div class="fam" style="margin-top:.6rem">BOOM classifiers</div>
    <div class="tiles">{"".join(f'<div class="tile off"><div class="trow"><span class="tname">{n}</span></div><div class="trow2"><span class="tpct">–</span></div></div>' for n in ("drb", "btsbot", "acai_h", "sgscore1"))}</div>
    <p class="foot">Tiles follow the widget's own rule: filled block, three-bucket
      colour at 0.7 and 0.4, integer per cent. BOOM's own classifiers are blank
      here because this page was rendered offline from the FLARE models alone.</p>
  </div>
</section>'''

    others = "".join(f'''
<section class="mini">
  <div class="mhead"><span class="mono">{oid}</span><span class="dim">{demo[oid]["truth"]}</span></div>
  <div class="mrow"><span class="mk">FLARE</span><span class="mv">{NICE[demo[oid]["conformal"]["predicted"]]}</span></div>
  <div class="mrow"><span class="mk">set (&alpha;=0.10)</span><span class="mv mono">{{{", ".join(NICE[x] for x in demo[oid]["conformal"]["set"])}}}</span></div>
  <div class="mrow"><span class="mk">coverage</span><span class="mv">90% &plusmn; 10% target · measured {demo[oid]["conformal"]["empirical_coverage"]*100:.1f}% &plusmn; {demo[oid]["conformal"]["coverage_err"]*100:.1f}%</span></div>
  <div class="mrow"><span class="mk">credibility</span><span class="mv">{demo[oid]["conformal"]["credibility"]:.3f}</span></div>
  <div class="mrow"><span class="mk">P(anomaly)</span><span class="mv">{demo[oid]["anom"]["p_anomaly"]*100:.1f}% ({demo[oid]["anom"]["p_lo"]*100:.0f}&ndash;{demo[oid]["anom"]["p_hi"]*100:.0f}%) · LR {demo[oid]["anom"]["likelihood_ratio"]:.1f}&times;</span></div>
</section>''' for oid in order[:-1])

    return f'''<title>FLARE inside BOOM</title>
<style>
:root {{
  --bg:{TOK_DARK["bg"]}; --card:{TOK_DARK["card"]}; --fg:{TOK_DARK["fg"]};
  --muted:{TOK_DARK["muted"]}; --mutedfg:{TOK_DARK["mutedfg"]};
  --border:{TOK_DARK["border"]}; --sidebar:{TOK_DARK["sidebar"]};
  --ztf:{ZTF}; --radius:0.5rem;
}}
:root[data-theme="light"] {{
  --bg:{TOK_LIGHT["bg"]}; --card:{TOK_LIGHT["card"]}; --fg:{TOK_LIGHT["fg"]};
  --muted:{TOK_LIGHT["muted"]}; --mutedfg:{TOK_LIGHT["mutedfg"]};
  --border:{TOK_LIGHT["border"]}; --sidebar:{TOK_LIGHT["sidebar"]};
}}
@media (prefers-color-scheme: light) {{
  :root:not([data-theme="dark"]) {{
    --bg:{TOK_LIGHT["bg"]}; --card:{TOK_LIGHT["card"]}; --fg:{TOK_LIGHT["fg"]};
    --muted:{TOK_LIGHT["muted"]}; --mutedfg:{TOK_LIGHT["mutedfg"]};
    --border:{TOK_LIGHT["border"]}; --sidebar:{TOK_LIGHT["sidebar"]};
  }}
}}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--fg); line-height:1.5;
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  font-size:14px }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace }}
.dim {{ color:var(--mutedfg) }}
.wrap {{ display:flex; min-height:100vh }}
/* icon rail: SIDEBAR_WIDTH_ICON = 2.5rem, collapsed by default */
.rail {{ width:2.5rem; flex:0 0 2.5rem; background:var(--sidebar);
  display:flex; flex-direction:column; align-items:center; gap:.4rem; padding:.5rem 0 }}
.brand {{ font-size:.85rem; text-align:center; line-height:1.1; margin-bottom:.5rem }}
.ic {{ width:1.75rem; height:1.75rem; display:grid; place-items:center; border-radius:.375rem;
  color:var(--mutedfg) }}
.ic svg {{ width:1rem; height:1rem; fill:none; stroke:currentColor; stroke-width:1.7;
  stroke-linecap:round; stroke-linejoin:round }}
.rsep {{ width:1.25rem; height:1px; background:var(--border); margin:.2rem 0 }}
/* SidebarInset: m-2 ml-0, rounded-xl, shadow */
.inset {{ flex:1; margin:.5rem .5rem .5rem 0; background:var(--bg);
  border:1px solid var(--border); border-radius:.75rem; overflow:hidden;
  box-shadow:0 1px 2px 0 rgb(0 0 0/.05) }}
.sheader {{ display:flex; align-items:center; gap:.5rem; height:3rem;
  padding:0 1rem; border-bottom:1px solid var(--border) }}
.sheader h1 {{ font-size:1rem; font-weight:500; margin:0 }}
.vsep {{ width:1px; height:1rem; background:var(--border) }}
.crumb {{ margin-left:auto; font-size:.78rem; color:var(--mutedfg) }}
.note {{ margin:.75rem 1rem 0; padding:.55rem .8rem; border:1px dashed var(--border);
  border-radius:.5rem; font-size:.78rem; color:var(--mutedfg) }}
.note b {{ color:var(--fg) }}
/* card grid: grid-cols-1 gap-4 px-4 lg:px-6, @xl 2, @5xl 4 */
.grid {{ display:grid; grid-template-columns:1fr; gap:1rem; padding:1rem 1.5rem 1.5rem }}
@media (min-width:56rem) {{ .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)) }} }}
@media (min-width:80rem) {{ .grid {{ grid-template-columns:repeat(4,minmax(0,1fr)) }}
  .span2 {{ grid-column:span 2 }} }}
.card {{ background:var(--card); border:1px solid var(--border); border-radius:.75rem;
  padding:1.25rem 0; box-shadow:0 1px 2px 0 rgb(0 0 0/.05); min-width:0 }}
.chead {{ display:flex; align-items:center; justify-content:space-between;
  gap:.5rem; padding:0 1.25rem .75rem }}
.chead h3 {{ font-size:1.05rem; font-weight:600; margin:0; line-height:1 }}
.cbody {{ padding:0 1.25rem }}
.sel, .pill {{ font-size:.7rem; color:var(--mutedfg); border:1px solid var(--border);
  border-radius:.375rem; padding:.1rem .45rem }}
.tabs {{ display:grid; grid-template-columns:1fr 1fr; gap:3px; background:var(--muted);
  border-radius:.5rem; padding:3px; margin-bottom:.6rem }}
.tab {{ text-align:center; font-size:.75rem; font-weight:500; padding:.25rem;
  border-radius:.375rem; color:var(--mutedfg) }}
.tab.on {{ background:var(--card); color:var(--fg); box-shadow:0 1px 2px 0 rgb(0 0 0/.05) }}
.fam {{ font-size:.75rem; font-weight:500; color:var(--mutedfg); margin-bottom:.25rem }}
.tiles {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.5rem }}
.tile {{ padding:.5rem .625rem; border-radius:.5rem; color:#fff; display:flex;
  flex-direction:column; min-width:0 }}
.tile.off {{ background:var(--muted); color:var(--mutedfg) }}
.trow {{ display:flex; justify-content:space-between; margin-bottom:.1rem }}
.tname {{ font-size:.75rem; font-weight:600; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis }}
.trow2 {{ display:flex; align-items:flex-end; justify-content:space-between; gap:.4rem }}
.tpct {{ font-size:1.25rem; font-weight:700; font-variant-numeric:tabular-nums;
  line-height:1 }}
.tsub {{ font-size:.7rem; opacity:.9 }}
.foot {{ font-size:.72rem; color:var(--mutedfg); margin:.7rem 0 0 }}
.lc {{ width:100%; height:auto; display:block }}
.gl {{ stroke:var(--border) }}
.tick {{ font-size:9px; fill:var(--mutedfg) }}
.axl {{ font-size:9px; fill:var(--mutedfg) }}
.acc {{ border-bottom:1px solid var(--border) }}
.acc summary {{ display:flex; align-items:center; justify-content:space-between;
  gap:.5rem; padding:.5rem 0; cursor:pointer; font-size:.82rem;
  font-family:ui-monospace,Menlo,monospace }}
.acc.empty {{ opacity:.4 }}
.acc.empty summary {{ cursor:not-allowed }}
.badge {{ font-size:.65rem; border:1px solid var(--border); border-radius:99px;
  padding:.05rem .45rem; color:var(--mutedfg); font-family:inherit }}
.xt {{ width:100%; border-collapse:collapse; margin:0 0 .5rem }}
.xt td {{ padding:.2rem 0; font-size:.78rem }}
.xt .k {{ color:var(--mutedfg) }}
.xt .v {{ text-align:right; font-variant-numeric:tabular-nums;
  font-family:ui-monospace,Menlo,monospace }}
.two {{ display:grid; grid-template-columns:1fr 1fr; gap:1.1rem }}
@media (max-width:640px) {{ .two {{ grid-template-columns:1fr }} }}
.lab {{ font-size:.72rem; letter-spacing:.06em; text-transform:uppercase;
  color:var(--mutedfg); margin-bottom:.35rem }}
.setbox {{ font-family:ui-monospace,Menlo,monospace; font-size:.95rem;
  background:var(--muted); border-radius:.5rem; padding:.4rem .6rem; margin-bottom:.5rem }}
.err {{ width:100%; border-collapse:collapse }}
.err td {{ padding:.18rem 0; font-size:.78rem; color:var(--mutedfg) }}
.err td.n {{ text-align:right; color:var(--fg); font-variant-numeric:tabular-nums;
  font-family:ui-monospace,Menlo,monospace }}
.pv {{ display:grid; grid-template-columns:3.4rem 1fr 2.6rem; align-items:center;
  gap:.4rem; margin:.15rem 0; font-size:.78rem }}
.pk {{ color:var(--mutedfg) }}
.pv.hit .pk {{ color:var(--fg); font-weight:600 }}
.pb {{ height:.55rem; background:var(--muted); border-radius:3px; overflow:hidden }}
.pb i {{ display:block; height:100%; background:var(--mutedfg); opacity:.5 }}
.pv.hit .pb i {{ background:var(--ztf); opacity:1 }}
.pn {{ text-align:right; font-variant-numeric:tabular-nums;
  font-family:ui-monospace,Menlo,monospace; color:var(--mutedfg) }}
.guar {{ background:var(--muted); border-radius:.5rem; padding:.5rem .7rem; margin-bottom:.6rem }}
.gnum {{ font-size:1.5rem; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.1 }}
.pm {{ font-size:.95rem; font-weight:500; color:var(--mutedfg) }}
.gsub {{ font-size:.72rem; color:var(--mutedfg); margin-top:.15rem }}
.anom {{ background:var(--muted); border-radius:.5rem; padding:.55rem .7rem;
  border-left:3px solid {GREEN} }}
.anom.alert {{ border-left-color:{RED} }}
.anum {{ font-size:1.5rem; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.1 }}
.anom.alert .anum {{ color:{RED} }}
.asub {{ font-size:.72rem; color:var(--mutedfg); margin-top:.1rem }}
.gauge {{ margin-top:.1rem }}
.gtrack {{ position:relative; height:.55rem; background:var(--muted); border-radius:3px }}
.gfill {{ height:100%; background:var(--ztf); border-radius:3px }}
.gthr {{ position:absolute; top:-3px; bottom:-3px; left:99%; width:2px; background:{RED} }}
.gcap {{ display:flex; justify-content:space-between; font-size:.72rem;
  color:var(--mutedfg); margin-top:.25rem }}
.gcap b {{ color:var(--fg); font-variant-numeric:tabular-nums }}
.gcap b.hot {{ color:{RED} }}
.verdict {{ margin-top:.8rem; padding:.5rem .7rem; border-radius:.5rem;
  font-size:.78rem; background:var(--muted) }}
.verdict.alert {{ border-left:3px solid {RED} }}
.verdict.ok {{ border-left:3px solid {GREEN} }}
.hdr-kv {{ display:grid; grid-template-columns:auto 1fr; gap:.2rem .8rem; font-size:.8rem }}
.hdr-kv .k {{ color:var(--mutedfg) }}
.hdr-kv .v {{ font-family:ui-monospace,Menlo,monospace }}
.minis {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; padding:0 1.5rem 1.5rem }}
@media (max-width:640px) {{ .minis {{ grid-template-columns:1fr }} }}
.mini {{ background:var(--card); border:1px solid var(--border); border-radius:.75rem;
  padding:.9rem 1rem }}
.mhead {{ display:flex; justify-content:space-between; gap:.5rem; margin-bottom:.5rem;
  font-size:.82rem }}
.mrow {{ display:flex; justify-content:space-between; gap:.6rem; font-size:.78rem;
  padding:.12rem 0 }}
.mk {{ color:var(--mutedfg) }}
.mv {{ text-align:right }}
.tail {{ padding:0 1.5rem 2rem; font-size:.78rem; color:var(--mutedfg); max-width:75ch }}
.tail code {{ font-family:ui-monospace,Menlo,monospace; font-size:.74rem }}
</style>
<div class="wrap">
  <nav class="rail">
    <div class="brand">𒁀𒁀<br>𒀯</div>
    {rail}
  </nav>
  <div class="inset">
    <div class="sheader">
      <span class="dim">☰</span><div class="vsep"></div><h1>Babamul</h1>
      <span class="crumb mono">/objects/ztf/{main_id}</span>
    </div>
    <div class="note"><b>Design proposal, not a live page.</b> The chrome, the card grid,
      the “ML Scores” tiles, the band colours and every token here are taken from
      boom-astro/boom’s own frontend source; the FLARE numbers are real outputs of the
      shipped <code>bts6</code> models. Nothing was screenshotted and no BOOM classifier
      score is invented.</div>
    <div class="grid">
      <section class="card">
        <div class="chead"><h3 class="mono">{main_id}</h3><span class="pill">ZTF</span></div>
        <div class="cbody hdr-kv">
          <span class="k">IAU</span><span class="v">{v["iau"]}</span>
          <span class="k">RA</span><span class="v">{v["ra"]}</span>
          <span class="k">Dec</span><span class="v">{v["dec"]}</span>
          <span class="k">detections</span><span class="v">{v["n_det"]}</span>
          <span class="k">peak mag</span><span class="v">{v["peakmag"]:.2f}</span>
          <span class="k">spectroscopic</span><span class="v">{v["truth"]}</span>
        </div>
      </section>
      <section class="card span2">
        <div class="chead"><h3>Lightcurve</h3><span class="pill">ZTF g, r</span></div>
        <div class="cbody">{lightcurve(v["mag"], v["mjd0"])}</div>
      </section>
      {ml_scores(v)}
      {flare_card(v)}
      <section class="card span2">
        <div class="chead"><h3>Cross-matches</h3></div>
        <div class="cbody">{xmatch(v["ctx"])}</div>
      </section>
    </div>
    <div class="minis">{others}</div>
    <div class="tail">
      <p><b>What the current UI could and could not show.</b> The six class scores drop
      straight into the existing “ML Scores” card — but only after
      <code>mapAlertClassifications</code> in <code>ClassificationsV2.tsx</code> stops
      being a hard-coded whitelist (<code>acai_*</code>, <code>btsbot</code>,
      <code>drb</code>, <code>reliability</code>, <code>sgscore1</code>,
      <code>LSPSC</code>), which silently drops every other key. The prediction set, the
      error it admits and the anomaly energy have no home in the widget at all: it draws
      one number per classifier with a three-bucket colour and no threshold lines, so the
      FLARE card above is a new component.</p>
      <p><b>Pipeline fit.</b> BOOM already runs its models as ONNX in enrichment workers
      (<code>ORT_DYLIB_PATH</code>, results written back to MongoDB) and crossmatches
      static catalogues per alert, which is exactly what FLARE consumes: PS1 host and
      pre-outburst counterpart, Legacy Surveys photo-z, Gaia, AllWISE. The classifier is
      LightGBM and converts to ONNX; the anomaly layer reads its own narrower
      181-column space.</p>
    </div>
  </div>
</div>'''


if __name__ == "__main__":
    demo = json.load(open(sys.argv[1]))
    open(sys.argv[2], "w").write(page(demo))
    print("wrote", sys.argv[2])
