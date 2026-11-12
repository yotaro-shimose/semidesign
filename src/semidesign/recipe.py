"""Process recipes in TOML: build a Simulation and run its steps.

Example:

    name = "CIS pixel"
    [grid]
    width_nm = 1100          # one number = 2D, a list [x, y] = 3D
    depth_nm = 3000
    spacing_nm = 10
    [substrate]
    dopant = "B"
    conc = 1e15
    [[step]]
    type = "implant"
    dopant = "P"
    dose = 2e12
    energy_kev = 200
    openings = [[150, 950]]  # 3D: [[x0, x1, y0, y1]]
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .grid import Grid
from .simulation import Simulation

STEP_TYPES = {
    "etch": ("depth_nm",),
    "deposit": ("material", "thickness_nm"),
    "planarize": (),
    "implant": ("dopant", "dose", "energy_kev"),
    "anneal": ("temp_c", "time_min"),
    "oxidize": ("temp_c", "time_min"),
}


def load_recipe(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_simulation(recipe: dict[str, Any], keep_snapshots: bool = True) -> Simulation:
    g = dict(recipe.get("grid", {}))
    width = g.pop("width_nm", 1000.0)
    lateral = tuple(float(w) for w in (width if isinstance(width, list) else [width]))
    grid = Grid(lateral_nm=lateral, depth_nm=float(g.pop("depth_nm", 3000.0)),
                spacing_nm=float(g.pop("spacing_nm", 10.0)),
                headroom_nm=float(g.pop("headroom_nm", 200.0)),
                periodic=bool(g.pop("periodic", False)))
    if g:
        raise ValueError(f"unknown [grid] keys: {sorted(g)}")
    sub = recipe.get("substrate", {})
    return Simulation(grid, substrate_dopant=sub.get("dopant", "B"),
                      substrate_conc=float(sub.get("conc", 1e15)),
                      orientation=str(sub.get("orientation", "100")),
                      keep_snapshots=keep_snapshots)


def run_step(sim: Simulation, step: dict[str, Any], index: int) -> None:
    s = dict(step)
    kind = s.pop("type", None)
    if kind not in STEP_TYPES:
        raise ValueError(f"step {index}: unknown type {kind!r}; choose from {sorted(STEP_TYPES)}")
    missing = [k for k in STEP_TYPES[kind] if k not in s]
    if missing:
        raise ValueError(f"step {index} ({kind}): missing {missing}")
    s.pop("note", None)
    try:
        getattr(sim, kind)(**s)
    except TypeError as e:
        raise ValueError(f"step {index} ({kind}): {e}") from None


def run_recipe(recipe: dict[str, Any], keep_snapshots: bool = True) -> Simulation:
    sim = build_simulation(recipe, keep_snapshots)
    for i, step in enumerate(recipe.get("step", []), start=1):
        run_step(sim, step, i)
    return sim
