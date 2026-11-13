"""Materials a cell can hold."""

from __future__ import annotations

GAS, SI, OXIDE, POLY, NITRIDE = 0, 1, 2, 3, 4
COUNT = 5

NAMES = {GAS: "gas", SI: "Si", OXIDE: "SiO2", POLY: "poly-Si", NITRIDE: "Si3N4"}
_ALIASES = {name.lower(): code for code, name in NAMES.items()} | {
    "oxide": OXIDE, "poly": POLY, "nitride": NITRIDE, "sin": NITRIDE,
}

OXIDIZABLE = (SI, POLY)  # consumed by thermal oxidation


def material_code(name: str | int) -> int:
    if isinstance(name, int):
        return name
    try:
        return _ALIASES[name.lower()]
    except KeyError:
        raise ValueError(f"unknown material {name!r}; choose from {sorted(_ALIASES)}") from None
