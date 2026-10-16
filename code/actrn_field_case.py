"""ACTRN field decision example on the Volve record under a chosen candidate budget.

Refits the v4 field-case model (decision date, training blocks, seed 747 and validation-NMAE
selection unchanged) and ranks a library of historical injection blocks rescaled to a common
total injection. ``--budget recent_median`` reproduces the v4 example; ``--budget implemented``
uses the total of the schedule that was implemented (ADR-0007).

Usage (from the project root):
    .venv/bin/python code/actrn_field_case.py --budget recent_median --out _tmp/actrn_case_check
    .venv/bin/python code/actrn_field_case.py --budget implemented \
        --protocol docs/PROTOCOL.md \
        --out data/processed/actrn_v4_case
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

from action_conditioned_decision import _sha256
from actrn_experiments import environment, field_decision_example

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--budget", choices=["recent_median", "implemented"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default=None,
                    help="protocol file recorded in the manifest (relative to the project root)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    case = field_decision_example(out, grid_limit=8, budget_rule=args.budget)
    (out / "field_case_summary.json").write_text(json.dumps(case, indent=2) + "\n")
    protocol = ROOT / args.protocol if args.protocol else None
    files = sorted(p for p in out.glob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "protocol": args.protocol,
        "protocol_sha256": _sha256(protocol) if protocol else None,
        "budget_rule": args.budget, "environment": environment(),
        "wall_seconds": time.perf_counter() - t0,
        "seeds": {"field_case_model": 747},
        "outputs": {p.name: _sha256(p) for p in files},
        "command": " ".join([Path(sys.executable).name] + sys.argv),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()
