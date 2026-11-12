"""Command line entry point: semidesign run recipe.toml"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .recipe import load_recipe, run_recipe
from .simulator import Wafer


def _write_csv(w: Wafer, path: Path) -> None:
    names = sorted(w.profiles)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["depth_nm", *[f"{n}_cm3" for n in names], "net_cm3"])
        net = w.net_doping()
        for i, x in enumerate(w.x):
            wr.writerow([f"{x:.3f}", *[f"{w.profiles[n][i]:.6e}" for n in names], f"{net[i]:.6e}"])


def _plot(w: Wafer, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in sorted(w.profiles):
        ax.semilogy(w.x, np.clip(w.profiles[name], 1e10, None), label=name)
    ax.semilogy(w.x, np.clip(np.abs(w.net_doping()), 1e10, None), "k--", lw=1, label="|net|")
    for xj in w.junction_depths():
        ax.axvline(xj, color="gray", lw=0.8, ls=":")
        ax.annotate(f"xj={xj:.0f} nm", (xj, 1e14), rotation=90, fontsize=8,
                    va="bottom", ha="right", color="gray")
    ax.set_xlabel("Depth from Si surface (nm)")
    ax.set_ylabel("Concentration (cm$^{-3}$)")
    ax.set_ylim(1e13, None)
    ax.set_title(title + f"   (oxide {w.oxide_nm:.1f} nm)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="semidesign", description="1D silicon process simulator")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a TOML recipe")
    r.add_argument("recipe", type=Path)
    r.add_argument("-o", "--out", type=Path, default=Path("out"), help="output directory")
    args = p.parse_args(argv)

    recipe = load_recipe(args.recipe)
    w, _ = run_recipe(recipe)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.recipe.stem
    _write_csv(w, args.out / f"{stem}.csv")
    _plot(w, args.out / f"{stem}.png", recipe.get("name", stem))

    print(f"== {recipe.get('name', stem)} ==")
    for line in w.history:
        print(" -", line)
    print(f"oxide thickness: {w.oxide_nm:.1f} nm")
    for name in sorted(w.profiles):
        print(f"{name} dose in Si: {w.sheet_dose(name):.3e} cm^-2")
    xjs = w.junction_depths()
    print("junction depth(s):", ", ".join(f"{x:.1f} nm" for x in xjs) if xjs else "none")
    print(f"wrote {args.out / (stem + '.csv')} and {args.out / (stem + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
