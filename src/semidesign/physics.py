"""Physical models and material parameters for 1D silicon process simulation.

Units used throughout:
    depth          nm
    concentration  cm^-3
    dose           cm^-2
    temperature    degC (converted internally to K)
    time           min
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

K_BOLTZMANN_EV = 8.617333262e-5  # eV/K
NM_PER_CM = 1e7
SI_CONSUMED_PER_OXIDE = 0.44  # thickness of Si consumed per unit oxide grown


# ---------------------------------------------------------------------------
# Dopants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dopant:
    name: str
    kind: str  # "p" (acceptor) or "n" (donor)
    d0: float  # cm^2/s, intrinsic diffusivity prefactor
    ea: float  # eV, activation energy
    # Approximate implant range table in Si: energy keV -> (Rp nm, dRp nm)
    range_table: tuple[tuple[float, float, float], ...]

    def diffusivity(self, temp_c: float) -> float:
        """Intrinsic diffusivity in cm^2/s at the given temperature."""
        t_k = temp_c + 273.15
        return self.d0 * math.exp(-self.ea / (K_BOLTZMANN_EV * t_k))

    def projected_range(self, energy_kev: float) -> tuple[float, float]:
        """(Rp, dRp) in nm, log-log interpolated from the built-in table."""
        e = np.array([r[0] for r in self.range_table])
        rp = np.array([r[1] for r in self.range_table])
        drp = np.array([r[2] for r in self.range_table])
        le = math.log(energy_kev)
        rp_i = math.exp(np.interp(le, np.log(e), np.log(rp)))
        drp_i = math.exp(np.interp(le, np.log(e), np.log(drp)))
        if energy_kev < e[0] or energy_kev > e[-1]:
            # np.interp clamps; extrapolate on a power law from the end segment
            i, j = (0, 1) if energy_kev < e[0] else (-2, -1)
            k_rp = math.log(rp[j] / rp[i]) / math.log(e[j] / e[i])
            k_drp = math.log(drp[j] / drp[i]) / math.log(e[j] / e[i])
            ref = i if energy_kev < e[0] else j
            rp_i = rp[ref] * (energy_kev / e[ref]) ** k_rp
            drp_i = drp[ref] * (energy_kev / e[ref]) ** k_drp
        return float(rp_i), float(drp_i)


# Diffusivities: intrinsic values commonly quoted in process textbooks (Fair).
# Range tables: approximate values for amorphous-target implants into Si.
DOPANTS: dict[str, Dopant] = {
    "B": Dopant(
        "B", "p", d0=0.76, ea=3.46,
        range_table=((10, 33, 17), (30, 98, 37), (50, 161, 51),
                     (100, 307, 69), (200, 529, 89)),
    ),
    "P": Dopant(
        "P", "n", d0=3.85, ea=3.66,
        range_table=((10, 14, 6), (30, 42, 17), (50, 72, 27),
                     (100, 135, 46), (200, 255, 76)),
    ),
    "As": Dopant(
        "As", "n", d0=0.32, ea=3.56,
        range_table=((10, 10, 4), (30, 23, 9), (50, 34, 12),
                     (100, 62, 22), (200, 120, 40)),
    ),
    "Sb": Dopant(
        "Sb", "n", d0=0.214, ea=3.65,
        range_table=((10, 9, 3), (30, 19, 6), (50, 28, 9),
                     (100, 48, 15), (200, 88, 27)),
    ),
}


def get_dopant(name: str) -> Dopant:
    try:
        return DOPANTS[name]
    except KeyError:
        raise ValueError(f"unknown dopant {name!r}; choose from {sorted(DOPANTS)}") from None


# ---------------------------------------------------------------------------
# Ion implantation
# ---------------------------------------------------------------------------

def gaussian_implant(depth_nm: np.ndarray, dose: float, rp_nm: float, drp_nm: float) -> np.ndarray:
    """Gaussian implant profile in cm^-3 for a dose in cm^-2."""
    drp_cm = drp_nm / NM_PER_CM
    peak = dose / (math.sqrt(2 * math.pi) * drp_cm)
    return peak * np.exp(-((depth_nm - rp_nm) ** 2) / (2 * drp_nm**2))


# ---------------------------------------------------------------------------
# Diffusion (Crank-Nicolson, zero-flux boundaries)
# ---------------------------------------------------------------------------

def _thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system (sub a, diag b, super c)."""
    n = len(d)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def diffuse_step(conc: np.ndarray, d_nm2_s: float, dx_nm: float, dt_s: float) -> np.ndarray:
    """Advance one Crank-Nicolson step of dC/dt = D d2C/dx2 with no flux at both ends."""
    n = len(conc)
    r = d_nm2_s * dt_s / dx_nm**2
    # Implicit side: -r/2 C[i-1] + (1+r) C[i] - r/2 C[i+1]
    a = np.full(n, -r / 2)
    b = np.full(n, 1 + r)
    c = np.full(n, -r / 2)
    # Zero flux via mirrored ghost nodes
    c[0] = -r
    a[-1] = -r
    # Explicit side
    rhs = (1 - r) * conc
    rhs[1:-1] += r / 2 * (conc[:-2] + conc[2:])
    rhs[0] += r * conc[1]
    rhs[-1] += r * conc[-2]
    return _thomas(a, b, c, rhs)


# ---------------------------------------------------------------------------
# Thermal oxidation (Deal-Grove)
# ---------------------------------------------------------------------------

# Parameters for <111> Si (Plummer, Deal & Griffin); B in um^2/h, B/A in um/h
_DEAL_GROVE = {
    "dry": {"b_c": 7.72e2, "b_e": 1.23, "ba_c": 6.23e6, "ba_e": 2.0},
    "wet": {"b_c": 3.86e2, "b_e": 0.78, "ba_c": 1.63e8, "ba_e": 2.05},
}
_ORIENTATION_FACTOR = {"111": 1.0, "100": 1 / 1.68}


def deal_grove_coefficients(temp_c: float, ambient: str, orientation: str = "100") -> tuple[float, float]:
    """Return (B um^2/h, B/A um/h)."""
    if ambient not in _DEAL_GROVE:
        raise ValueError(f"ambient must be one of {sorted(_DEAL_GROVE)}")
    if orientation not in _ORIENTATION_FACTOR:
        raise ValueError(f"orientation must be one of {sorted(_ORIENTATION_FACTOR)}")
    p = _DEAL_GROVE[ambient]
    kt = K_BOLTZMANN_EV * (temp_c + 273.15)
    b = p["b_c"] * math.exp(-p["b_e"] / kt)
    ba = p["ba_c"] * math.exp(-p["ba_e"] / kt) * _ORIENTATION_FACTOR[orientation]
    return b, ba


def deal_grove_grow(tox_nm: float, dt_min: float, b_um2_h: float, ba_um_h: float) -> float:
    """Oxide thickness (nm) after growing for dt from an existing thickness."""
    a_um = b_um2_h / ba_um_h
    x0 = tox_nm / 1000
    dt_h = dt_min / 60
    rhs = x0**2 + a_um * x0 + b_um2_h * dt_h
    x1 = (-a_um + math.sqrt(a_um**2 + 4 * rhs)) / 2
    return x1 * 1000
