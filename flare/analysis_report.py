"""Self-contained HTML for offline unified analysis results."""
from html import escape
import json


def curve_svg(bands):
    curves = [b for b in bands.values() if b.get('status') == 'ok']
    if not curves:
        return '<p>No curve available.</p>'
    xs = [x for b in curves for x in b['days']]
    ys = [y for b in curves for key in ('lower95', 'upper95') for y in b[key] if y is not None]
    if not ys:
        return '<p>No finite curve available.</p>'
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    X = lambda x: 50 + 590 * (x-x0) / max(x1-x0, 1e-6)
    Y = lambda y: 15 + 205 * (y-y0) / max(y1-y0, 1e-6)
    out = ['<svg viewBox="0 0 670 265" role="img" aria-label="Magnitude against days since first detection; shaded 95 percent interval">']
    for name, b in bands.items():
        if b.get('status') != 'ok':
            continue
        color = {'g': '#267d32', 'r': '#b63242'}.get(name, '#555')
        rows = [r for r in zip(b['days'], b['magnitude'], b['lower95'], b['upper95']) if all(v is not None for v in r)]
        poly = [(r[0], r[2]) for r in rows] + [(r[0], r[3]) for r in reversed(rows)]
        points = ' '.join(f'{X(x):.2f},{Y(y):.2f}' for x,y in poly)
        out.append(f'<polygon points="{points}" fill="{color}" opacity=".15"/>')
        points = ' '.join(f'{X(r[0]):.2f},{Y(r[1]):.2f}' for r in rows)
        out.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for i in range(5):
        x, y = x0+(x1-x0)*i/4, y0+(y1-y0)*i/4
        out.append(f'<text x="{X(x):.1f}" y="242" text-anchor="middle">{x:.0f}</text>')
        out.append(f'<text x="42" y="{Y(y):.1f}" text-anchor="end">{y:.1f}</text>')
    out.append('<text x="335" y="261" text-anchor="middle">Days since first detection · AB mag · g green / r red</text></svg>')
    return ''.join(out)


def render_analysis(records):
    panels = []
    for name, r in records.items():
        c, a = r['classification'], r['anomaly']
        p = a.get('p_anomaly')
        probability = f'{p:.1%}' if p is not None else 'unavailable'
        panels.append(f'<article><h2>{escape(name)}</h2><p>Observation cutoff: day {r["observation_cutoff_days"]:g}; {r["n_detections"]} detections.</p>')
        panels.append(f'<h3>Classification and anomaly detection</h3><p>{escape(c["predicted"])} · candidates: {escape(", ".join(c["set"]))}</p>')
        panels.append(f'<p>Anomaly probability: {probability}, assuming prevalence {a["base_rate"]:.1%}. Novelty p-value: {a.get("novelty_p", "unavailable")}</p><p>{escape(c["calibration_scope"])}</p>')
        panels.append(f'<p>AD calibration: {escape(a["calibration_scope"])}</p>')
        for task in ('reconstruction', 'forecast'):
            b = r[task]
            panels.append(f'<h3>{task.capitalize()}</h3>')
            if b.get('status') == 'ok':
                if 'model_horizon_days' in b:
                    panels.append(f'<p>Forecast issued at day {b["model_horizon_days"]:g}. Shading: nominal 95% interval.</p>')
                panels.append(curve_svg(b['bands']))
                panels.append(f'<p>{escape(b["calibration"])}</p>')
            else:
                panels.append(f'<p>{escape(b.get("reason", b["status"]))}</p>')
        pe = r['pe']
        panels.append('<h3>SN Ia parameter estimation</h3>')
        if pe['status'] == 'conditional':
            panels.append(f'<p>Conditional on SN Ia, model day {pe["model_horizon_days"]:g}. {escape(pe["calibration"])}</p>')
            panels.append('<table><tr><th>Parameter</th><th>Estimate</th><th>68% interval</th><th>Unit</th></tr>')
            for target, v in pe['parameters'].items():
                lo, hi = v['interval68']
                panels.append(f'<tr><td>{escape(target)}</td><td>{v["value"]:.3f}</td><td>{lo:.3f} – {hi:.3f}</td><td>{escape(pe["units"][target])}</td></tr>')
            panels.append('</table>')
        else:
            panels.append(f'<p>{escape(pe.get("reason", pe["status"]))}</p>')
        panels.append('<details><summary>Full numerical result</summary><pre>' + escape(json.dumps(r, indent=2, allow_nan=False)) + '</pre></details></article>')
    return '<!doctype html><html lang="en"><meta charset="utf-8"><title>FLARE analysis</title><style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#202632}article{border-top:1px solid #ddd;margin-top:2rem}svg{max-width:760px;width:100%;font:12px system-ui}td,th{text-align:left;padding:.4rem 1rem}pre{white-space:pre-wrap;overflow-wrap:anywhere}p{line-height:1.5}</style><h1>FLARE analysis</h1>' + ''.join(panels) + '</html>'
