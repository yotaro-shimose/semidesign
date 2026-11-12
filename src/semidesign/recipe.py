"""Load a process recipe (TOML) and run it on a Wafer."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .simulator import Wafer

_STEP_HANDLERS = {
    "implant": lambda w, s: w.implant(
        s["dopant"], float(s["dose"]), float(s["energy_kev"]),
        rp_nm=s.get("rp_nm"), drp_nm=s.get("drp_nm"),
    ),
    "anneal": lambda w, s: w.anneal(float(s["temp_c"]), float(s["time_min"]),
                                    steps=int(s.get("steps", 200))),
    "oxidize": lambda w, s: w.oxidize(float(s["temp_c"]), float(s["time_min"]),
                                      ambient=s.get("ambient", "dry"),
                                      steps=int(s.get("steps", 200))),
    "strip_oxide": lambda w, s: w.strip_oxide(),
}


def load_recipe(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def run_recipe(recipe: dict[str, Any]) -> tuple[Wafer, list[tuple[str, dict]]]:
    """Run every step; return the final wafer and per-step snapshots."""
    w = Wafer(**recipe.get("wafer", {}))
    snapshots: list[tuple[str, dict]] = []
    for i, step in enumerate(recipe.get("step", []), start=1):
        kind = step.get("type")
        if kind not in _STEP_HANDLERS:
            raise ValueError(f"step {i}: unknown type {kind!r}; choose from {sorted(_STEP_HANDLERS)}")
        _STEP_HANDLERS[kind](w, step)
        snapshots.append((w.history[-1], {k: v.copy() for k, v in w.profiles.items()}))
    return w, snapshots
