"""Comparison with measured depth profiles (SIMS, spreading resistance).

A measurement is a list of (depth, concentration) in one column of the
wafer. The simulated column is read at the same place and compared on a log
scale, the way profiles are judged by eye on a SIMS plot:

- rms_log10: root mean square of log10(simulated / measured) where the
  measurement is above its floor (the detection limit; values below it are
  raised to it on both sides). 0.1 means the curves differ by about 26 %
  on average, 0.3 by a factor of 2.
- the peak (depth, concentration) of each curve, and the dose (area under
  each curve) over the measured depth range.

Depth is measured from the top of the silicon in that column by default
(surface = "silicon"; films on top are stripped before SIMS), or from the
initial front surface (surface = "initial").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dopants import DOPANTS
from .materials import SI


@dataclass
class Comparison:
    label: str
    probe: float | list[float]
    dopant: str  # a dopant name, or "net" (|donors - acceptors|)
    depth_nm: np.ndarray
    conc: np.ndarray
    surface: str = "silicon"
    floor_cm3: float | None = None

    def __post_init__(self):
        if self.dopant != "net" and self.dopant not in DOPANTS:
            raise ValueError(f"compare: dopant must be one of {sorted(DOPANTS)} or 'net'")
        if self.surface not in ("silicon", "initial"):
            raise ValueError("compare: surface must be 'silicon' or 'initial'")

    @classmethod
    def read(cls, path: str | Path, probe, dopant: str, surface: str = "silicon",
             floor_cm3: float | None = None, label: str | None = None) -> "Comparison":
        """Two numeric columns (depth nm, concentration cm^-3); comma, tab or space
        separated; lines that are not two numbers (headers, comments) are skipped."""
        rows = []
        for line in Path(path).read_text().splitlines():
            parts = line.replace(",", " ").replace("\t", " ").split()
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue
        if len(rows) < 2:
            raise ValueError(f"{path}: no depth/concentration rows found")
        depth, conc = np.array(sorted(rows)).T
        return cls(label or Path(path).stem, probe, dopant, depth, conc, surface,
                   None if floor_cm3 is None else float(floor_cm3))

    # ------------------------------------------------------------------

    def simulated(self, sim) -> tuple[np.ndarray, np.ndarray]:
        """The simulated curve in the silicon of the probe column, on the same depth scale."""
        prof = sim.depth_profile(self.probe)
        si = prof["material"] == SI
        if not si.any():
            raise ValueError(f"compare {self.label}: no silicon at {self.probe}")
        values = np.abs(prof["net"]) if self.dopant == "net" else prof.get(self.dopant, np.zeros(si.size))
        origin = sim.grid.z_edges[np.argmax(si)] if self.surface == "silicon" else 0.0
        return prof["depth_nm"][si] - origin, values[si]

    def evaluate(self, sim) -> dict:
        z_sim, c_sim = self.simulated(sim)
        floor = self.floor_cm3 or float(self.conc[self.conc > 0].min())
        inside = (self.depth_nm >= z_sim[0]) & (self.depth_nm <= z_sim[-1])
        z, measured = self.depth_nm[inside], self.conc[inside]
        if z.size < 2:
            raise ValueError(f"compare {self.label}: the measured depths are outside the simulated silicon")
        simulated = 10 ** np.interp(z, z_sim, np.log10(np.maximum(c_sim, 1e-30)))
        used = measured >= floor
        diff = np.log10(np.maximum(simulated, floor)) - np.log10(np.maximum(measured, floor))
        rms = float(np.sqrt(np.mean(diff[used] ** 2))) if used.any() else float("nan")
        in_range = (z_sim >= z[0]) & (z_sim <= z[-1])

        def peak(depths, values):
            k = int(np.argmax(values))
            return {"depth_nm": round(float(depths[k]), 1), "conc_cm3": float(f"{values[k]:.3g}")}

        def dose(depths, values):  # cm^-2
            return float(f"{np.trapezoid(values, depths) * 1e-7:.4g}") if depths.size > 1 else 0.0

        return {
            "label": self.label, "probe_nm": self.probe, "dopant": self.dopant, "surface": self.surface,
            "points_compared": int(used.sum()), "floor_cm3": floor,
            "rms_log10": round(rms, 4), "typical_factor": round(10 ** rms, 3),
            "peak": {"measured": peak(z, measured), "simulated": peak(z_sim[in_range], c_sim[in_range])},
            "dose_cm2": {"measured": dose(z, measured), "simulated": dose(z_sim[in_range], c_sim[in_range])},
        }

    def plot(self, sim, path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        z_sim, c_sim = self.simulated(sim)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.semilogy(self.depth_nm, self.conc, "o", ms=3, color="#444", label="measured")
        ax.semilogy(z_sim, np.maximum(c_sim, 1e10), "-", lw=1.6, color="#c0392b", label="simulated")
        floor = self.floor_cm3 or float(self.conc[self.conc > 0].min())
        ax.axhline(floor, ls=":", lw=0.8, color="#888")
        ax.set_xlim(min(0, self.depth_nm[0]), self.depth_nm[-1] * 1.05)
        ax.set_ylim(floor / 10, max(self.conc.max(), c_sim.max()) * 3)
        what = "|net doping|" if self.dopant == "net" else self.dopant
        ax.set_xlabel(f"depth from the {'silicon surface' if self.surface == 'silicon' else 'initial surface'} (nm)")
        ax.set_ylabel(f"{what} (cm$^{{-3}}$)")
        ax.set_title(f"{self.label} at x = {self.probe} nm", fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
