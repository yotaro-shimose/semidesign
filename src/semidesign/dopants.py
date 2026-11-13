"""Dopant species and how they move in each material.

Diffusion in a cell is described by two numbers per material:

- diffusivity D (cm^2/s)
- capacity w: how much dopant the material holds in equilibrium with silicon
  at the same "chemical" level. Silicon has w = 1; oxide has w = 1/m with the
  segregation coefficient m = C_Si / C_SiO2 at the interface.

Sources
- Intrinsic diffusivities in Si: R. B. Fair, in "Impurity Doping Processes in
  Silicon" (1981), as tabulated in process textbooks.
- Fraction of diffusion through interstitials, f_I: P. M. Fahey, P. B. Griffin,
  J. D. Plummer, Rev. Mod. Phys. 61, 289 (1989).
- Segregation coefficients at Si/SiO2: B 0.1-0.3, P/As/Sb 10 (compiled in
  R. Hollauer, PhD thesis, TU Wien 2007, Table 4.1).
- Diffusivity in SiO2 at 1100 C: B ~2e-16 cm^2/s (N2/O2), Ea ~3.5 eV;
  P ~3e-18 cm^2/s, Ea 4.4 eV (compiled by enigmatic-consulting.com from
  Shacham-Diamand & Oldham and others). No reliable values were found for As
  and Sb; they are given the P values (all three are very slow in oxide).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import kt_ev
from .materials import COUNT, GAS, NITRIDE, OXIDE, POLY, SI

_KT_1100 = kt_ev(1100.0)
POLY_SPEEDUP = 10.0  # grain-boundary transport in poly-Si, rough
NITRIDE_SLOWDOWN = 1e-2  # relative to oxide, rough


@dataclass(frozen=True)
class Dopant:
    name: str
    kind: str  # "p" (acceptor) or "n" (donor)
    z: int
    mass_amu: float  # implanted isotope
    d0: float  # cm^2/s, intrinsic diffusivity in Si
    ea: float  # eV
    f_interstitial: float  # share of diffusion through interstitials
    segregation: float  # m = C_Si / C_SiO2 in equilibrium
    d_oxide_1100: float  # cm^2/s, diffusivity in SiO2 at 1100 C
    ea_oxide: float  # eV

    def diffusivity(self, temp_c: float) -> float:
        """Intrinsic diffusivity in silicon, cm^2/s."""
        return self.d0 * math.exp(-self.ea / kt_ev(temp_c))

    def diffusivity_in_oxide(self, temp_c: float) -> float:
        return self.d_oxide_1100 * math.exp(-self.ea_oxide * (1 / kt_ev(temp_c) - 1 / _KT_1100))

    def diffusivity_by_material(self, temp_c: float) -> np.ndarray:
        """Diffusivity (cm^2/s) indexed by material code."""
        d = np.zeros(COUNT)
        d[SI] = self.diffusivity(temp_c)
        d[POLY] = POLY_SPEEDUP * d[SI]
        d[OXIDE] = self.diffusivity_in_oxide(temp_c)
        d[NITRIDE] = NITRIDE_SLOWDOWN * d[OXIDE]
        return d

    def capacity_by_material(self) -> np.ndarray:
        """Equilibrium capacity w indexed by material code (Si = 1)."""
        w = np.ones(COUNT)
        w[OXIDE] = w[NITRIDE] = 1.0 / self.segregation
        w[GAS] = 1.0  # holds nothing; any value keeps the equations regular
        return w

    def point_defect_factor(self, supersaturation: np.ndarray) -> np.ndarray:
        """Diffusivity multiplier at interstitial supersaturation S = C_I / C_I*.

        Vacancies are taken to be in local equilibrium with interstitials
        (C_V / C_V* = 1 / S), so interstitial diffusers speed up and vacancy
        diffusers slow down.
        """
        s = supersaturation
        return self.f_interstitial * s + (1.0 - self.f_interstitial) / s


DOPANTS: dict[str, Dopant] = {
    "B": Dopant("B", "p", 5, 11.009, d0=0.76, ea=3.46, f_interstitial=1.0,
                segregation=0.3, d_oxide_1100=2e-16, ea_oxide=3.5),
    "P": Dopant("P", "n", 15, 30.974, d0=3.85, ea=3.66, f_interstitial=1.0,
                segregation=10.0, d_oxide_1100=3e-18, ea_oxide=4.4),
    "As": Dopant("As", "n", 33, 74.922, d0=0.32, ea=3.56, f_interstitial=0.4,
                 segregation=10.0, d_oxide_1100=3e-18, ea_oxide=4.4),
    "Sb": Dopant("Sb", "n", 51, 120.904, d0=0.214, ea=3.65, f_interstitial=0.02,
                 segregation=10.0, d_oxide_1100=3e-18, ea_oxide=4.4),
}


def get_dopant(name: str) -> Dopant:
    try:
        return DOPANTS[name]
    except KeyError:
        raise ValueError(f"unknown dopant {name!r}; choose from {sorted(DOPANTS)}") from None
