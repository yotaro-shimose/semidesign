"""Model parameters: the knobs a simulation is calibrated with, in one place.

Each physics module keeps its own parameters next to the equations that use
them (and the literature they come from); this module only gathers them into
one set that can be written to a file, edited, and read back:

    [dopant.B]                 # dopants.Dopant: diffusion and segregation
    d0 = 0.76
    ea = 3.46
    [defects]                  # defects.DefectModel: implant damage, TED
    tau_800c_s = 1000.0
    [oxidation.dry]            # oxidation.OxidationModel: Deal-Grove constants
    b_um2_h = 772.0
    [interface_charge_cm2]     # electrostatics: fixed charge at Si/insulator
    Al2O3 = -2e12

A file only needs the values it changes; everything else keeps its default.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .defects import DefectModel
from .dopants import DOPANTS, Dopant
from .materials import NAMES, material_code
from .oxidation import OxidationModel

# Dopant fields that describe the element rather than a calibration
_DOPANT_IDENTITY = {"name", "kind", "z", "mass_amu"}


@dataclass(frozen=True)
class Parameters:
    dopants: Mapping[str, Dopant] = field(default_factory=lambda: dict(DOPANTS))
    defects: DefectModel = DefectModel()
    oxidation: OxidationModel = OxidationModel()
    interface_charge_cm2: Mapping[str, float] = field(default_factory=dict)  # by material name

    # ------------------------------------------------------------- reading

    @classmethod
    def load(cls, path: str | Path, base: "Parameters | None" = None) -> "Parameters":
        with open(path, "rb") as f:
            return (base or cls()).updated(tomllib.load(f))

    def updated(self, table: Mapping[str, Any]) -> "Parameters":
        """A copy with the values in `table` (same layout as the file) replaced."""
        table = dict(table)
        dopants = dict(self.dopants)
        for name, values in table.pop("dopant", {}).items():
            if name not in dopants:
                raise ValueError(f"[dopant.{name}]: unknown dopant; choose from {sorted(dopants)}")
            bad = _DOPANT_IDENTITY & set(values)
            if bad:
                raise ValueError(f"[dopant.{name}]: {sorted(bad)} describe the element and cannot be changed")
            dopants[name] = _replace(dopants[name], values, f"dopant.{name}")
        charges = dict(self.interface_charge_cm2)
        for name, density in table.pop("interface_charge_cm2", {}).items():
            charges[NAMES[material_code(name)]] = float(density)
        out = replace(self, dopants=dopants, interface_charge_cm2=charges,
                      defects=_replace(self.defects, table.pop("defects", {}), "defects"),
                      oxidation=_replace(self.oxidation, table.pop("oxidation", {}), "oxidation"))
        if table:
            raise ValueError(f"unknown parameter sections: {sorted(table)}")
        return out

    # ------------------------------------------------------------- writing

    def to_dict(self) -> dict[str, Any]:
        return {
            "dopant": {n: {k: v for k, v in _as_dict(d).items() if k not in _DOPANT_IDENTITY}
                       for n, d in self.dopants.items()},
            "defects": _as_dict(self.defects),
            "oxidation": _as_dict(self.oxidation),
            "interface_charge_cm2": dict(self.interface_charge_cm2),
        }

    def to_toml(self) -> str:
        lines: list[str] = []

        def section(name: str, table: Mapping[str, Any]) -> None:
            scalars = {k: v for k, v in table.items() if not isinstance(v, Mapping)}
            if scalars or not table:
                lines.append(f"[{name}]")
                lines.extend(f"{k} = {_toml_value(v)}" for k, v in scalars.items())
                lines.append("")
            for k, v in table.items():
                if isinstance(v, Mapping):
                    section(f"{name}.{k}", v)

        for name, table in self.to_dict().items():
            section(name, table)
        return "\n".join(lines)


def _replace(obj, values: Mapping[str, Any], where: str):
    """dataclasses.replace for nested tables, with errors that name the bad key."""
    names = {f.name: f for f in fields(obj)}
    changes = {}
    for key, value in values.items():
        if key not in names:
            raise ValueError(f"[{where}]: unknown key {key!r}; choose from {sorted(names)}")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, Mapping):
                raise ValueError(f"[{where}.{key}] must be a table")
            changes[key] = _replace(current, value, f"{where}.{key}")
        else:
            changes[key] = float(value)
    return replace(obj, **changes)


def _as_dict(obj) -> dict[str, Any]:
    return {f.name: (_as_dict(v) if is_dataclass(v := getattr(obj, f.name)) else v) for f in fields(obj)}


def _toml_value(v) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    return repr(float(v)) if isinstance(v, float) else str(v)
