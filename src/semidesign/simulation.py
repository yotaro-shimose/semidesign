"""Process simulation on a 2D/3D grid: the wafer state and the process steps.

The state is a material per cell, a concentration field per dopant, and the
implant damage (defects.Defects). Each step changes the state through the
module that owns the physics; thermal steps share one time loop in which
oxidation, damage dissolution and dopant diffusion advance together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import geometry
from .defects import Defects, cluster_dissolution_time
from .diffusion import DopantTransport, segregate_at_new_oxide
from .dopants import get_dopant
from .grid import Grid
from .implant import calibrated, implant_profile, range_at
from .materials import GAS, NAMES, OXIDE, SI, material_code
from .oxidation import DealGrove, OxidationFront

DAMAGE_STEPS_PER_TAU = 4  # time steps per cluster dissolution time while TED is active


@dataclass
class StepRecord:
    index: int
    kind: str
    params: dict
    info: dict
    seconds: float

    def to_dict(self) -> dict:
        return {"index": self.index, "type": self.kind, "params": self.params,
                "info": self.info, "seconds": round(self.seconds, 3)}


@dataclass
class Snapshot:
    label: str
    mat: np.ndarray
    conc: dict[str, np.ndarray]
    clusters: np.ndarray


@dataclass
class Simulation:
    grid: Grid
    substrate_dopant: str = "B"
    substrate_conc: float = 1e15
    orientation: str = "100"
    keep_snapshots: bool = True
    mat: np.ndarray = field(init=False)
    conc: dict[str, np.ndarray] = field(init=False)
    defects: Defects = field(init=False)
    history: list[StepRecord] = field(init=False, default_factory=list)
    snapshots: list[Snapshot] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        get_dopant(self.substrate_dopant)
        below_surface = np.broadcast_to(self.grid.z_centers >= 0, self.grid.shape)
        self.mat = np.where(below_surface, SI, GAS).astype(np.int8)
        self.conc = {self.substrate_dopant: np.where(self.mat == SI, self.substrate_conc, 0.0)}
        self.defects = Defects(self.grid)
        self._snapshot("initial")

    # ------------------------------------------------------------- geometry

    def etch(self, depth_nm: float, openings=None, materials=None) -> None:
        def run():
            codes = None if materials is None else [material_code(m) for m in materials]
            removed = geometry.etch(self.grid, self.mat, openings, depth_nm, codes)
            self._forget(removed)
            return {"cells_removed": int(removed.sum())}
        self._run("etch", {"depth_nm": depth_nm, "openings": openings, "materials": materials}, run)

    def deposit(self, material: str, thickness_nm: float, conformal: bool = True) -> None:
        def run():
            new = geometry.deposit(self.grid, self.mat, material_code(material), thickness_nm, conformal)
            return {"cells_added": int(new.sum())}
        self._run("deposit", {"material": material, "thickness_nm": thickness_nm,
                              "conformal": conformal}, run)

    def planarize(self, level_nm: float = 0.0) -> None:
        def run():
            removed = geometry.planarize(self.grid, self.mat, level_nm)
            self._forget(removed)
            return {"cells_removed": int(removed.sum())}
        self._run("planarize", {"level_nm": level_nm}, run)

    # -------------------------------------------------------------- implant

    def implant(self, dopant: str, dose: float, energy_kev: float, openings=None,
                rp_nm: float | None = None, drp_nm: float | None = None,
                lateral_sigma_nm: float | None = None, damage_factor: float = 1.0) -> None:
        """Implant `dose` (cm^-2) at `energy_kev`.

        rp_nm / drp_nm / lateral_sigma_nm recalibrate the tabulated range
        distribution; damage_factor scales the '+1' excess interstitials
        (0 switches transient enhanced diffusion off for this implant).
        """
        def run():
            dist = calibrated(range_at(dopant, energy_kev), rp_nm, drp_nm, lateral_sigma_nm)
            added = implant_profile(self.grid, self.mat, dose, dist, openings)
            self.conc[dopant] = self.conc.get(dopant, np.zeros(self.grid.shape)) + added
            self.defects.add_implant_damage(added, self.mat, damage_factor)
            return {"rp_nm": round(dist.rp_nm, 1), "drp_nm": round(dist.drp_nm, 1),
                    "skewness": round(dist.skewness, 2), "kurtosis": round(dist.kurtosis, 2),
                    "lateral_nm": round(dist.lateral_nm, 1), "retained": round(dist.retained, 4),
                    "dose_in_structure": self._areal(added),
                    "excess_interstitials_cm2": self._areal(self.defects.clusters)}
        self._run("implant", {"dopant": dopant, "dose": dose, "energy_kev": energy_kev,
                              "openings": openings, "rp_nm": rp_nm, "drp_nm": drp_nm,
                              "lateral_sigma_nm": lateral_sigma_nm,
                              "damage_factor": damage_factor}, run)

    # -------------------------------------------------------- thermal steps

    def anneal(self, temp_c: float, time_min: float, steps: int = 40) -> None:
        """Inert anneal. `steps` sets the longest time step (time / steps)."""
        self._run("anneal", {"temp_c": temp_c, "time_min": time_min, "steps": steps},
                  lambda: self._thermal(temp_c, time_min * 60, steps))

    def oxidize(self, temp_c: float, time_min: float, ambient: str = "dry", steps: int = 40) -> None:
        """Thermal oxidation of every exposed Si/poly surface (mask with nitride)."""
        def run():
            front = OxidationFront(self.grid, DealGrove.at(temp_c, ambient, self.orientation))
            oxide_before = int((self.mat == OXIDE).sum())
            info = self._thermal(temp_c, time_min * 60, steps, front)
            info["oxide_cells_added"] = int((self.mat == OXIDE).sum()) - oxide_before
            info["headroom_exceeded"] = front.headroom_exceeded
            return info
        self._run("oxidize", {"temp_c": temp_c, "time_min": time_min, "ambient": ambient,
                              "steps": steps}, run)

    def _thermal(self, temp_c: float, duration_s: float, steps: int,
                 front: OxidationFront | None = None) -> dict:
        transport = DopantTransport(self.grid, temp_c)
        tau = cluster_dissolution_time(temp_c)
        damage_before = self.defects.clusters.sum()
        dt_max = duration_s / steps
        t, n, peak, method = 0.0, 0, 1.0, "none"
        while t < duration_s * (1 - 1e-12):
            dt = min(dt_max, duration_s - t)
            if self.defects.enhancing:
                dt = min(dt, tau / DAMAGE_STEPS_PER_TAU)
            if front is not None:
                consume, fill = front.rates(self.mat)
                dt = min(dt, front.stable_step(consume, fill))
                converted = front.advance(self.mat, consume, fill, dt)
                if converted.any():
                    segregate_at_new_oxide(self.conc, self.mat, converted, front.i, front.j)
                    self.defects.remove(converted)
            supersaturation = self.defects.advance(self.mat, temp_c, dt)
            if supersaturation is not None:
                peak = max(peak, float(supersaturation.max()))
            method = transport.step(self.conc, self.mat, dt, supersaturation)
            t += dt
            n += 1
        left = self.defects.clusters.sum() / damage_before if damage_before > 0 else 0.0
        return {"time_steps": n, "solver": method, "peak_supersaturation": round(peak, 2),
                "damage_left_fraction": round(float(left), 4),
                "damage_dissolution_time_s": round(tau, 3)}

    # ------------------------------------------------------------ internals

    def _run(self, kind: str, params: dict, fn) -> None:
        t0 = time.perf_counter()
        info = fn()
        rec = StepRecord(len(self.history) + 1, kind, params, info, time.perf_counter() - t0)
        self.history.append(rec)
        self._snapshot(f"{rec.index}: {kind}")

    def _snapshot(self, label: str) -> None:
        if self.keep_snapshots:
            self.snapshots.append(Snapshot(label, self.mat.copy(),
                                           {k: v.copy() for k, v in self.conc.items()},
                                           self.defects.clusters.copy()))

    def _forget(self, removed: np.ndarray) -> None:
        """Material removed: its dopant and damage go with it."""
        for c in self.conc.values():
            c[removed] = 0.0
        self.defects.remove(removed)

    def _lateral_area_cm2(self) -> float:
        return float(np.prod([w * 1e-7 for w in self.grid.lateral_nm]))

    def _areal(self, field_cm3: np.ndarray) -> float:
        """Amount per unit wafer area (cm^-2), averaged over the box."""
        return float(field_cm3.sum() * self.grid.cell_volume_cm / self._lateral_area_cm2())

    # ------------------------------------------------------------- analysis

    def net_doping(self) -> np.ndarray:
        """Donors minus acceptors in cm^-3 (positive = n-type); zero outside Si."""
        net = np.zeros(self.grid.shape)
        for name, c in self.conc.items():
            net += c if get_dopant(name).kind == "n" else -c
        net[self.mat != SI] = 0.0
        return net

    def column(self, pos_nm: list[float] | float | None = None) -> tuple[int, ...]:
        """Lateral index of the column at a position (default: box centre)."""
        lat = self.grid.lateral_nm
        if pos_nm is None:
            pos = [w / 2 for w in lat]
        else:
            pos = [pos_nm] if np.isscalar(pos_nm) else list(pos_nm)
        return tuple(self.grid.lateral_index(p, a) for a, p in enumerate(pos))

    def junctions_along_depth(self, pos_nm=None) -> list[float]:
        """Depths (nm, from the initial Si surface) where net doping changes sign."""
        col = self.column(pos_nm)
        net = self.net_doping()[col]
        si = self.mat[col] == SI
        z = self.grid.z_centers
        out = []
        for i in range(len(net) - 1):
            if si[i] and si[i + 1] and net[i] * net[i + 1] < 0:
                out.append(float(z[i] - net[i] * (z[i + 1] - z[i]) / (net[i + 1] - net[i])))
        return out

    def total_dose(self, dopant: str) -> float:
        c = self.conc.get(dopant)
        return 0.0 if c is None else self._areal(c)

    def oxide_thickness(self, pos_nm=None) -> float:
        """Oxide (nm) stacked directly on the silicon of one column."""
        col = self.mat[self.column(pos_nm)]
        si = np.flatnonzero(col == SI)
        top = si[0] if si.size else col.size
        return float((col[:top] == OXIDE).sum() * self.grid.spacing_nm)

    def material_fractions(self) -> dict[str, float]:
        return {name: float((self.mat == code).mean()) for code, name in NAMES.items()}

    def summary(self, probe_nm=None) -> dict:
        return {
            "grid": self.grid.describe(),
            "substrate": {"dopant": self.substrate_dopant, "conc": self.substrate_conc},
            "steps": [r.to_dict() for r in self.history],
            "total_seconds": round(sum(r.seconds for r in self.history), 3),
            "dose_cm2": {k: self.total_dose(k) for k in sorted(self.conc)},
            "excess_interstitials_cm2": self._areal(self.defects.clusters),
            "probe_column_nm": probe_nm if probe_nm is not None else [w / 2 for w in self.grid.lateral_nm],
            "junction_depths_nm": [round(x, 1) for x in self.junctions_along_depth(probe_nm)],
            "oxide_on_probe_column_nm": self.oxide_thickness(probe_nm),
            "material_fractions": self.material_fractions(),
        }
