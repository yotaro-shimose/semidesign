"""Command line.

    semidesign run recipe.toml [-o out] [--json]
    semidesign parameters [recipe.toml] > fab.toml

Designed to be called by people and by agents: --json prints the summary
(steps, timings, doses, probe columns, comparisons, pixel figures, output
file names) to stdout. `parameters` prints the model parameters (defaults, or
those a recipe uses) as a file to edit for calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .output import write_outputs
from .recipe import comparisons, load_recipe, parameters, pixel_spec, probes, run_recipe


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="semidesign", description="2D/3D silicon process simulator")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a TOML recipe")
    r.add_argument("recipe", type=Path)
    r.add_argument("-o", "--out", type=Path, default=None,
                   help="output folder (default: out/<recipe name>)")
    r.add_argument("--probe", type=float, nargs="+", action="append", default=None,
                   help="lateral position in nm (x, or x y in 3D) of a column to report; "
                        "repeat for several (overrides [output] probes)")
    r.add_argument("--no-snapshots", action="store_true", help="do not save per-step arrays")
    r.add_argument("--no-plot", action="store_true")
    r.add_argument("--no-pixel", action="store_true", help="skip the [pixel] evaluation")
    r.add_argument("--json", action="store_true", help="print the summary as JSON")
    pp = sub.add_parser("parameters", help="print model parameters as TOML")
    pp.add_argument("recipe", type=Path, nargs="?", help="print those this recipe uses")
    args = p.parse_args(argv)

    if args.cmd == "parameters":
        print(parameters(load_recipe(args.recipe) if args.recipe else {}).to_toml())
        return 0

    recipe = load_recipe(args.recipe)
    name = recipe.get("name", args.recipe.stem)
    out = args.out or Path("out") / args.recipe.stem
    sim = run_recipe(recipe, keep_snapshots=not args.no_snapshots)
    columns = [p[0] if len(p) == 1 else p for p in args.probe] if args.probe else probes(recipe)
    spec = None if args.no_pixel else pixel_spec(recipe)
    summary = write_outputs(sim, out, name, columns, snapshots=not args.no_snapshots,
                            plot=not args.no_plot, pixel=spec, comparisons=comparisons(recipe))

    if args.json:
        json.dump(summary, sys.stdout, indent=2, ensure_ascii=False)
        print()
        return 0
    print(f"== {name} ==  grid {summary['grid']['shape']} ({summary['grid']['n_cells']} cells)")
    for s in summary["steps"]:
        print(f" {s['index']:>2}. {s['type']:<9} {s['seconds']:>7.2f}s  {s['info']}")
    print("dose (cm^-2):", {k: f"{v:.3e}" for k, v in summary["dose_cm2"].items()})
    for pr in summary["probes"]:
        stack = " / ".join(f"{l['material']} {l['thickness_nm']:.0f}" for l in pr["layers"])
        print(f"column {pr['at_nm']} nm: junctions {pr['junction_depths_nm'] or 'none'} nm | {stack} nm")
    for c in summary.get("compare", []):
        print(f"compare {c['label']} ({c['dopant']} at {c['probe_nm']} nm): within x{c['typical_factor']} "
              f"on average, dose {c['dose_cm2']['simulated']:.3g} vs measured {c['dose_cm2']['measured']:.3g}")
    if "pixel" in summary:
        px = summary["pixel"]
        unit = " per um" if px["units"].endswith("(2D)") else ""
        fw = next(v for k, v in px["full_well"].items() if k.startswith("electrons"))
        rs = next(v for k, v in px.items() if k.startswith("residual"))
        print(f"pixel: sensor potential {px['sensor_potential_v']:.2f} V | barrier TG off "
              f"{px['tg_off']['barrier_v']:.2f} V, TG on {px['tg_on']['barrier_v']:.2f} V | "
              f"full well {fw:.0f} e{unit} ({px['full_well']['limited_by']}) | "
              f"left after transfer {rs:.3g} e{unit} | from the bottom {px['collection_from_bottom']['barrier_v']:.2f} V "
              f"[{px['seconds']} s]")
    print(f"wrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
