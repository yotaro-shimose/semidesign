"""Materials a cell can hold."""

from __future__ import annotations

GAS, SI, OXIDE, POLY, NITRIDE, AL2O3, HFO2 = 0, 1, 2, 3, 4, 5, 6
COUNT = 7

# Al2O3 and HfO2 stand for the charged films on the back surface of a
# back-illuminated sensor: deposited at low temperature, dopants do not move in them.
NAMES = {GAS: "gas", SI: "Si", OXIDE: "SiO2", POLY: "poly-Si", NITRIDE: "Si3N4",
         AL2O3: "Al2O3", HFO2: "HfO2"}
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
