"""Command line: semidesign run recipe.toml [-o out] [--json]

Designed to be called by people and by agents: --json prints the summary
(steps, timings, doses, junction depths, output file names) to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .output import write_outputs
from .recipe import load_recipe, run_recipe


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="semidesign", description="2D/3D silicon process simulator")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a TOML recipe")
    r.add_argument("recipe", type=Path)
    r.add_argument("-o", "--out", type=Path, default=None,
                   help="output folder (default: out/<recipe name>)")
    r.add_argument("--probe", type=float, nargs="+", default=None,
                   help="lateral position(s) in nm of the column used for junction depths")
    r.add_argument("--no-snapshots", action="store_true", help="do not save per-step arrays")
    r.add_argument("--no-plot", action="store_true")
    r.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = p.parse_args(argv)

    recipe = load_recipe(args.recipe)
    name = recipe.get("name", args.recipe.stem)
    out = args.out or Path("out") / args.recipe.stem
    sim = run_recipe(recipe, keep_snapshots=not args.no_snapshots)
    probe = args.probe[0] if args.probe and len(args.probe) == 1 else args.probe
    summary = write_outputs(sim, out, name, probe, snapshots=not args.no_snapshots,
                            plot=not args.no_plot)

    if args.json:
        json.dump(summary, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return 0
    print(f"== {name} ==  grid {summary['grid']['shape']} ({summary['grid']['n_cells']} cells)")
    for s in summary["steps"]:
        print(f" {s['index']:>2}. {s['type']:<9} {s['seconds']:>7.2f}s  {s['info']}")
    print("dose (cm^-2):", {k: f"{v:.3e}" for k, v in summary["dose_cm2"].items()})
    print("junction depths at probe (nm):", summary["junction_depths_nm"] or "none")
    print(f"wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
