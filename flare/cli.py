"""Command line: classify objects and write the report.

    flare predict ZTF19acbzgog ZTF22abkfhua           # JSON to stdout
    flare report  ZTF19acbzgog ZTF22abkfhua -o r.html # the console page

Photometry comes from BOOM when BOOM_URL and credentials are set, otherwise
from ALeRCE; the host, photo-z, Gaia/WISE and pre-outburst blocks are queried
per position. Anything unavailable is left missing, which the models are
trained to tolerate.
"""
import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flare", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in [("predict", "class, prediction set and anomaly probability as JSON"),
                        ("report", "the same as a self-contained HTML console")]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("ids", nargs="+", help="ZTF object identifiers")
        p.add_argument("-o", "--out", default=None,
                       help="output file (default: stdout for predict, "
                            "flare_report.html for report)")
        p.add_argument("--scheme", default="bts6", help="bts6 (default) or broad5")
        p.add_argument("--horizon", type=float, default=100.0,
                       help="days from first detection to use (default 100)")
    a = ap.parse_args(argv)

    from . import load_classifier
    from .report import build_record, render
    clf = load_classifier(a.scheme)

    records, failed = {}, {}
    for oid in a.ids:
        try:
            records[oid] = build_record(oid, clf=clf, horizon_days=a.horizon)
            print(f"  {oid}: {records[oid]['conformal']['predicted']}"
                  f"  set={{{', '.join(records[oid]['conformal']['set'])}}}"
                  f"  P(anomaly)={records[oid]['anom']['p']*100:.1f}%",
                  file=sys.stderr)
        except Exception as e:                       # noqa: BLE001
            failed[oid] = str(e)
            print(f"  {oid}: FAILED — {e}", file=sys.stderr)
    if not records:
        print("nothing to report", file=sys.stderr)
        return 1

    if a.cmd == "predict":
        payload = {o: {k: v for k, v in r.items() if k not in ("mag",)}
                   for o, r in records.items()}
        if failed:
            payload["_failed"] = failed
        text = json.dumps(payload, indent=1)
        (open(a.out, "w").write(text) if a.out else print(text))
    else:
        out = a.out or "flare_report.html"
        open(out, "w").write(render(records))
        print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
