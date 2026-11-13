"""Ion ranges in amorphous silicon by Monte Carlo (binary collision approximation).

This produces the range table that implant.py reads
(data/implant_ranges.json). Regenerate it with

    python -m semidesign.bca

Each ion is followed collision by collision until it stops:

- Nuclear scattering: ZBL universal screened potential; the scattering angle
  from the MAGIC formula (J. P. Biersack, L. G. Haggmark, Nucl. Instrum.
  Methods 174, 257 (1980)); one collision per mean atomic spacing, impact
  parameter drawn uniformly over the corresponding area (as in TRIM).
- Electronic stopping: continuous along the path.
  Boron: SRIM-2013 tabulated values (at MeV energies boron is faster than
  v0 Z1^(2/3), where the Lindhard-Scharff form stops being valid); below the
  table it is proportional to velocity.
  P, As, Sb: Lindhard-Scharff (J. Lindhard, M. Scharff, Phys. Rev. 124, 128
  (1961)), proportional to velocity, valid for these ions up to 5 MeV and
  known to be within about 20 % (Z1 oscillations).

Checked against SRIM-2013 for boron: projected ranges agree within a few
percent from 10 keV to 3 MeV; the straggling comes out 10-20 % narrower than
SRIM's tables (which use SRIM's analytic range algorithm, not Monte Carlo).

The target is amorphous, so channeling is not included (equivalent to the
usual tilted implant into crystalline silicon, or into a pre-amorphized layer).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .constants import SI_ATOMS_PER_CM3
from .dopants import DOPANTS, Dopant, get_dopant

DATA_DIR = Path(__file__).parent / "data"
SRIM_BORON = DATA_DIR / "srim_b_in_si.csv"
RANGE_TABLE = DATA_DIR / "implant_ranges.json"

E2_EV_NM = 1.439964  # e^2 / (4 pi eps0)
BOHR_NM = 0.0529177
KEV_PER_AMU_AT_BOHR_VELOCITY = 24.8016
SI_Z, SI_MASS_AMU = 14, 28.0855
SI_ATOMS_PER_NM3 = SI_ATOMS_PER_CM3 * 1e-21
EV_NM2_PER_MEV_CM2_MG = 1e6 * SI_MASS_AMU * 1.66053907e-21 * 1e14  # stopping per Si atom
STOP_ENERGY_EV = 10.0

# ZBL universal screening function and MAGIC coefficients for it
_ZBL_C = np.array([0.18175, 0.50986, 0.28022, 0.028171])
_ZBL_D = np.array([3.1998, 0.94229, 0.4029, 0.20162])
_MAGIC = (0.99229, 0.011615, 0.0071222, 9.3066, 14.813)

TABLE_ENERGIES_KEV = (1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300,
                      500, 700, 1000, 1500, 2000, 3000, 5000)


# ---------------------------------------------------------------------------
# Electronic stopping
# ---------------------------------------------------------------------------

def velocity_au(energy_kev, mass_amu):
    """Ion velocity in units of the Bohr velocity."""
    return np.sqrt(np.asarray(energy_kev) / (mass_amu * KEV_PER_AMU_AT_BOHR_VELOCITY))


def read_srim_boron() -> dict[str, np.ndarray]:
    rows = np.loadtxt(SRIM_BORON, delimiter=",", comments="#", skiprows=4)
    keys = ("energy_kev", "electronic", "nuclear", "rp_nm", "drp_nm", "lateral_nm")
    return dict(zip(keys, rows.T))


class TabulatedStopping:
    """Electronic stopping from SRIM's boron table, in eV nm^2 per Si atom."""

    def __init__(self, dopant: Dopant):
        ref = read_srim_boron()
        self._log_v = np.log(velocity_au(ref["energy_kev"], dopant.mass_amu))
        self._log_s = np.log(ref["electronic"] * EV_NM2_PER_MEV_CM2_MG)
        self._mass = dopant.mass_amu

    def __call__(self, energy_ev: np.ndarray) -> np.ndarray:
        v = velocity_au(np.maximum(energy_ev, 1e-6) / 1e3, self._mass)
        s = np.exp(np.interp(np.log(v), self._log_v, self._log_s))
        slow = v < math.exp(self._log_v[0])
        s[slow] = math.exp(self._log_s[0]) * v[slow] / math.exp(self._log_v[0])
        return s


class LindhardScharffStopping:
    """Electronic stopping proportional to velocity, in eV nm^2 per Si atom."""

    def __init__(self, dopant: Dopant):
        z1 = dopant.z
        self._k = (8 * math.pi * E2_EV_NM * BOHR_NM * z1 ** (7 / 6) * SI_Z
                   / (z1 ** (2 / 3) + SI_Z ** (2 / 3)) ** 1.5)
        self._mass = dopant.mass_amu

    def __call__(self, energy_ev: np.ndarray) -> np.ndarray:
        return self._k * velocity_au(np.maximum(energy_ev, 0.0) / 1e3, self._mass)


def electronic_stopping(dopant: Dopant):
    return TabulatedStopping(dopant) if dopant.name == "B" else LindhardScharffStopping(dopant)


# ---------------------------------------------------------------------------
# Nuclear scattering
# ---------------------------------------------------------------------------

def _screening(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.exp(-np.multiply.outer(r, _ZBL_D))
    return e @ _ZBL_C, -(e @ (_ZBL_C * _ZBL_D))


def sin2_half_angle(eps: np.ndarray, b: np.ndarray) -> np.ndarray:
    """sin^2(theta/2) of the centre-of-mass scattering angle (MAGIC formula).

    eps: reduced energy, b: reduced impact parameter (both ZBL-reduced).
    """
    # distance of closest approach: root of 1 - phi(r)/(r eps) - (b/r)^2,
    # starting from the unscreened Coulomb value (an upper bound)
    r = 1 / (2 * eps) + np.sqrt(1 / (2 * eps) ** 2 + b**2)
    for _ in range(50):
        phi, dphi = _screening(r)
        f = 1 - phi / (r * eps) - (b / r) ** 2
        df = (phi - dphi * r) / (r * r * eps) + 2 * b**2 / r**3
        step = f / df
        r = np.where(r - step > 0, r - step, 0.5 * r)
        if np.max(np.abs(step) / r) < 1e-9:
            break
    phi, dphi = _screening(r)
    v, dv = phi / r, dphi / r - phi / r**2
    rho = 2 * (eps - v) / (-dv)  # radius of curvature at closest approach
    c1, c2, c3, c4, c5 = _MAGIC
    sq = np.sqrt(eps)
    alpha = 1 + c1 / sq
    beta = (c2 + sq) / (c3 + sq)
    gamma = (c4 + eps) / (c5 + eps)
    a = 2 * alpha * eps * b**beta
    g = 1 / (gamma * (np.sqrt(1 + a * a) - a))
    delta = a * (r - b) / (1 + g)
    cos_half = (b + rho + delta) / (r + rho)
    return np.clip(1 - cos_half**2, 0.0, 1.0)


def _deflect(d: np.ndarray, psi: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Rotate unit directions d by polar angle psi and azimuth phi."""
    cx, cy, cz = d.T
    sp, cp, sf, cf = np.sin(psi), np.cos(psi), np.sin(phi), np.cos(phi)
    st = np.sqrt(np.maximum(1 - cz**2, 0.0))
    along_z = st < 1e-9
    st = np.where(along_z, 1.0, st)
    nx = np.where(along_z, sp * cf, cx * cp + sp * (cx * cz * cf - cy * sf) / st)
    ny = np.where(along_z, sp * sf, cy * cp + sp * (cy * cz * cf + cx * sf) / st)
    nz = np.where(along_z, np.sign(cz) * cp, cz * cp - sp * cf * st)
    out = np.stack([nx, ny, nz], 1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

# Depth quantiles are stored at these probability levels (dense toward both tails).
QUANTILE_LEVELS = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, 129))


@dataclass(frozen=True)
class RangeDistribution:
    """Where ions of one species and energy come to rest (depth along the beam)."""

    rp_nm: float
    drp_nm: float
    skewness: float
    kurtosis: float
    lateral_nm: float  # standard deviation along one lateral axis
    retained: float  # fraction of ions that stay in the target
    depth_quantiles_nm: tuple[float, ...]  # at QUANTILE_LEVELS


def simulate(dopant: Dopant, energy_kev: float, n_ions: int = 5000, seed: int = 0) -> tuple[np.ndarray, int]:
    """Final positions (n, 3: x, y, depth in nm) of ions that stop in the target."""
    rng = np.random.default_rng(seed)
    m1, z1 = dopant.mass_amu, dopant.z
    screening_nm = 0.8854 * BOHR_NM / (z1**0.23 + SI_Z**0.23)
    eps_per_ev = screening_nm * SI_MASS_AMU / (z1 * SI_Z * E2_EV_NM * (m1 + SI_MASS_AMU))
    transfer = 4 * m1 * SI_MASS_AMU / (m1 + SI_MASS_AMU) ** 2
    flight = SI_ATOMS_PER_NM3 ** (-1 / 3)
    p_max = 1 / math.sqrt(math.pi * SI_ATOMS_PER_NM3 * flight)
    stopping = electronic_stopping(dopant)

    energy = np.full(n_ions, energy_kev * 1e3)
    pos = np.zeros((n_ions, 3))
    d = np.tile([0.0, 0.0, 1.0], (n_ions, 1))
    path = flight * rng.random(n_ions)  # first flight starts anywhere in a spacing
    stopped = []
    while energy.size:
        pos += d * path[:, None]
        energy = energy - SI_ATOMS_PER_NM3 * path * stopping(energy)
        moving = energy > STOP_ENERGY_EV
        if moving.any():
            e = energy[moving]
            b = p_max * np.sqrt(rng.random(e.size)) / screening_nm
            s2 = sin2_half_angle(e * eps_per_ev, b)
            cos_t, sin_t = 1 - 2 * s2, 2 * np.sqrt(s2 * (1 - s2))
            psi = np.arctan2(sin_t, cos_t + m1 / SI_MASS_AMU)  # lab-frame deflection
            d[moving] = _deflect(d[moving], psi, 2 * math.pi * rng.random(e.size))
            energy[moving] = e * (1 - transfer * s2)
        left = pos[:, 2] < 0
        done = (energy <= STOP_ENERGY_EV) | left
        stopped.append(pos[done & ~left])
        keep = ~done
        energy, pos, d = energy[keep], pos[keep], d[keep]
        path = np.full(energy.size, flight)
    return np.concatenate(stopped), n_ions


def distribution(positions: np.ndarray, launched: int) -> RangeDistribution:
    z = positions[:, 2]
    rp = z.mean()
    dz = z - rp
    var = (dz**2).mean()
    return RangeDistribution(
        rp_nm=float(rp), drp_nm=float(math.sqrt(var)),
        skewness=float((dz**3).mean() / var**1.5), kurtosis=float((dz**4).mean() / var**2),
        lateral_nm=float(math.sqrt((positions[:, 0].var() + positions[:, 1].var()) / 2)),
        retained=positions.shape[0] / launched,
        depth_quantiles_nm=tuple(float(q) for q in np.quantile(z, QUANTILE_LEVELS)),
    )


def range_distribution(dopant: str | Dopant, energy_kev: float, n_ions: int = 5000,
                       seed: int = 0) -> RangeDistribution:
    d = get_dopant(dopant) if isinstance(dopant, str) else dopant
    return distribution(*simulate(d, energy_kev, n_ions, seed))


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def _job(args):
    name, energy, n_ions = args
    return name, energy, asdict(range_distribution(name, energy, n_ions, seed=int(energy * 1000) % 2**31))


def build_table(dopants=tuple(DOPANTS), energies=TABLE_ENERGIES_KEV, n_ions: int = 5000,
                workers: int | None = None) -> dict:
    jobs = [(name, float(e), n_ions) for name in dopants for e in energies]
    jobs.sort(key=lambda j: -j[1] / get_dopant(j[0]).mass_amu)  # longest first
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_job, jobs))
    table = {"target": "amorphous Si", "ions_per_point": n_ions,
             "quantile_levels": [round(float(q), 8) for q in QUANTILE_LEVELS],
             "method": "semidesign.bca: ZBL/MAGIC nuclear scattering; electronic stopping "
                       "SRIM-2013 (B) or Lindhard-Scharff (P, As, Sb)",
             "ions": {}}
    for name in dopants:
        rows = sorted((e, m) for n, e, m in results if n == name)
        table["ions"][name] = {"energy_kev": [e for e, _ in rows]} | {
            k: [np.round(m[k], 4).tolist() for _, m in rows] for k in rows[0][1]}
    return table


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Regenerate the implant range table.")
    p.add_argument("--ions-per-point", type=int, default=5000)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("-o", "--out", type=Path, default=RANGE_TABLE)
    args = p.parse_args(argv)
    t0 = time.perf_counter()
    table = build_table(n_ions=args.ions_per_point, workers=args.workers)
    args.out.write_text(json.dumps(table, indent=1))
    print(f"wrote {args.out} in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
