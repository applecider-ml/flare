"""Thin wrapper: render docs/prod_demo.json with the packaged renderer.

Usage:  python make_production_console.py prod_demo.json out.html
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flare.report import render   # noqa: E402

if __name__ == "__main__":
    open(sys.argv[2], "w").write(render(json.load(open(sys.argv[1]))))
    print("wrote", sys.argv[2])
