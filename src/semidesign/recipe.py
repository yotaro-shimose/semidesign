"""Process recipes in TOML: build a Simulation and run its steps.

Example:

    name = "CIS pixel"
    [grid]
    width_nm = 1100          # one number = 2D, a list [x, y] = 3D
    depth_nm = 3000
    spacing_nm = 10
    [[grid.refine]]          # optional: finer cells in a band
    axis = "depth"
    from_nm = -20
    to_nm = 20
    spacing_nm = 1
    [substrate]
    dopant = "B"
    conc = 1e15              # or profile = [[depth_nm, conc], ...] (an epitaxial layer)
    [parameters]             # optional: model parameters (see parameters.py)
    file = "fab.toml"        # relative to the recipe; tables below override it
    [parameters.dopant.B]
    d0 = 0.8
    [output]                 # optional
    probes = [450, 1000]     # columns reported like a measurement; 3D: [[x, y], ...]
    [[compare]]              # optional: a measured depth profile (SIMS) to compare with
    probe = 450
    dopant = "B"             # or "net"
    file = "sims_b.csv"      # two columns: depth (nm), concentration (cm^-3)
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

from .compare import Comparison
from .grid import DEPTH, Grid, Refinement
from .parameters import Parameters
from .pixel import PixelSpec
from .simulation import Simulation

STEP_TYPES = {
    "etch": ("depth_nm",),
    "deposit": ("material", "thickness_nm"),
    "planarize": (),
    "thin": ("si_thickness_nm",),
    "implant": ("dopant", "dose", "energy_kev"),
    "anneal": ("temp_c", "time_min"),
    "oxidize": ("temp_c", "time_min"),
}


def load_recipe(path: str | Path) -> dict[str, Any]:
    """Read a recipe; files it names are resolved relative to it."""
    path = Path(path)
    with open(path, "rb") as f:
        recipe = tomllib.load(f)
    params = recipe.get("parameters", {})
    if "file" in params:
        params["file"] = str(path.parent / params["file"])
    for c in recipe.get("compare", []):
        if "file" in c:
            c["file"] = str(path.parent / c["file"])
    return recipe


def parameters(recipe: dict[str, Any]) -> Parameters:
    """[parameters]: a file of model parameters, then the tables given inline."""
    table = dict(recipe.get("parameters", {}))
    file = table.pop("file", None)
    base = Parameters.load(file) if file else Parameters()
    return base.updated(table)


def build_simulation(recipe: dict[str, Any], keep_snapshots: bool = True) -> Simulation:
    g = dict(recipe.get("grid", {}))
    width = g.pop("width_nm", 1000.0)
    lateral = tuple(float(w) for w in (width if isinstance(width, list) else [width]))
    grid = Grid(lateral_nm=lateral, depth_nm=float(g.pop("depth_nm", 3000.0)),
                spacing_nm=float(g.pop("spacing_nm", 10.0)),
                headroom_nm=float(g.pop("headroom_nm", 200.0)),
                periodic=bool(g.pop("periodic", False)),
                refine=tuple(_refinement(r) for r in g.pop("refine", [])),
                growth=float(g.pop("growth", 1.3)))
    if g:
        raise ValueError(f"unknown [grid] keys: {sorted(g)}")
    sub = dict(recipe.get("substrate", {}))
    if "conc" in sub and "profile" in sub:
        raise ValueError("[substrate]: give either conc or profile, not both")
    profile = sub.pop("profile", None)
    sim = Simulation(grid, substrate_dopant=sub.pop("dopant", "B"),
                     substrate_conc=float(sub.pop("conc", 1e15)),
                     substrate_profile=[tuple(map(float, p)) for p in profile] if profile else None,
                     orientation=str(sub.pop("orientation", "100")),
                     params=parameters(recipe), keep_snapshots=keep_snapshots)
    if sub:
        raise ValueError(f"unknown [substrate] keys: {sorted(sub)}")
    return sim


_AXES = {"x": 0, "y": 1, "depth": DEPTH, "z": DEPTH}


def _refinement(r: dict[str, Any]) -> Refinement:
    """[[grid.refine]] entry: axis ("x", "y" or "depth"), from_nm, to_nm, spacing_nm."""
    r = dict(r)
    try:
        axis = _AXES[r.pop("axis")]
        ref = Refinement(axis, float(r.pop("from_nm")), float(r.pop("to_nm")), float(r.pop("spacing_nm")))
    except KeyError as e:
        raise ValueError(f"[[grid.refine]] needs axis (x/y/depth), from_nm, to_nm, spacing_nm; problem with {e}") from None
    if r:
        raise ValueError(f"unknown [[grid.refine]] keys: {sorted(r)}")
    return ref


def run_step(sim: Simulation, step: dict[str, Any], index: int) -> None:
    s = dict(step)
    kind = s.pop("type", None)
    if kind not in STEP_TYPES:
        raise ValueError(f"step {index}: unknown type {kind!r}; choose from {sorted(STEP_TYPES)}")
    missing = [k for k in STEP_TYPES[kind] if k not in s]
    if missing:
        raise ValueError(f"step {index} ({kind}): missing {missing}")
    if kind == "oxidize" and "openings" in s:
        raise ValueError(f"step {index} (oxidize): openings are not supported; "
                         "deposit and pattern a nitride mask instead")
    s.pop("note", None)
    try:
        getattr(sim, kind)(**s)
    except TypeError as e:
        raise ValueError(f"step {index} ({kind}): {e}") from None


def pixel_spec(recipe: dict[str, Any]) -> PixelSpec | None:
    """The [pixel] section: points inside the PD and FD (nm) and the biases."""
    if "pixel" not in recipe:
        return None
    px = dict(recipe["pixel"])
    try:
        spec = PixelSpec(pd_at=tuple(px.pop("pd_at")), fd_at=tuple(px.pop("fd_at")),
                         **{k: float(px.pop(k)) for k in ("tg_off_v", "tg_on_v", "fd_v", "temp_k") if k in px})
    except KeyError as e:
        raise ValueError(f"[pixel] needs pd_at and fd_at; missing {e}") from None
    if px:
        raise ValueError(f"unknown [pixel] keys: {sorted(px)}")
    return spec


def probes(recipe: dict[str, Any]) -> list | None:
    """[output] probes: lateral positions of the columns to report."""
    out = dict(recipe.get("output", {}))
    found = out.pop("probes", None)
    if out:
        raise ValueError(f"unknown [output] keys: {sorted(out)}")
    return list(found) if found else None


def comparisons(recipe: dict[str, Any]) -> list[Comparison]:
    """[[compare]] entries: measured depth profiles."""
    out = []
    for i, c in enumerate(recipe.get("compare", []), start=1):
        c = dict(c)
        try:
            out.append(Comparison.read(c.pop("file"), probe=c.pop("probe"), dopant=c.pop("dopant"),
                                       **{k: c.pop(k) for k in ("surface", "floor_cm3", "label") if k in c}))
        except KeyError as e:
            raise ValueError(f"[[compare]] {i}: needs file, probe and dopant; missing {e}") from None
        if c:
            raise ValueError(f"[[compare]] {i}: unknown keys {sorted(c)}")
    return out


def run_recipe(recipe: dict[str, Any], keep_snapshots: bool = True) -> Simulation:
    sim = build_simulation(recipe, keep_snapshots)
    for i, step in enumerate(recipe.get("step", []), start=1):
        run_step(sim, step, i)
    return sim
