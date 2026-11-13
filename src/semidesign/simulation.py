"""Process simulation on a 2D/3D grid: the wafer state and the process steps.

The state is a material per cell, a concentration field per dopant, and the
implant damage (defects.Defects). Each step changes the state through the
module that owns the physics; thermal steps share one time loop in which
oxidation, damage dissolution and dopant diffusion advance together. The
model parameters (parameters.Parameters) are fixed for the whole run.

Geometry steps take a side: "front" (default) or "back", the surface left
after the wafer is bonded face down and thinned (thin()). The back side is
processed at low temperature, so in practice only films, trenches and fixed
charge are added there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import geometry
from .defects import Defects
from .diffusion import DopantTransport, segregate_at_new_oxide
from .dopants import get_dopant
from .grid import Grid
from .implant import calibrated, implant_profile, range_at
from .materials import GAS, NAMES, OXIDE, SI, material_code
from .oxidation import DealGrove, OxidationFront
from .parameters import Parameters

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
    # depth profile of the starting wafer (an epitaxial layer): [(depth_nm, conc), ...],
    # interpolated on a log scale, constant beyond the ends; replaces substrate_conc
    substrate_profile: list[tuple[float, float]] | None = None
    orientation: str = "100"
    params: Parameters = field(default_factory=Parameters)
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
        start = np.broadcast_to(self._starting_doping(), self.grid.shape)
        self.conc = {self.substrate_dopant: np.where(self.mat == SI, start, 0.0)}
        self.defects = Defects(self.grid, self.params.defects)
        self._snapshot("initial")

    def _starting_doping(self) -> np.ndarray | float:
        if self.substrate_profile is None:
            return self.substrate_conc
        depth, conc = np.array(self.substrate_profile, dtype=float).T
        if depth.size < 1 or np.any(np.diff(depth) <= 0) or np.any(conc <= 0):
            raise ValueError("substrate profile needs increasing depths and positive concentrations")
        return 10 ** np.interp(self.grid.z_centers, depth, np.log10(conc))

    # ------------------------------------------------------------- geometry

    def etch(self, depth_nm: float, openings=None, materials=None, side: str = "front") -> None:
        """Straight etch through mask openings, `depth_nm` below the surface on `side`."""
        def run():
            codes = None if materials is None else [material_code(m) for m in materials]
            g, m = geometry.from_side(self.grid, self.mat, side)
            removed = geometry.unflip(geometry.etch(g, m, openings, depth_nm, codes), side)
            self._forget(removed)
            return {"cells_removed": int(removed.sum())}
        self._run("etch", {"depth_nm": depth_nm, "openings": openings, "materials": materials,
                           "side": side}, run)

    def deposit(self, material: str, thickness_nm: float, conformal: bool = True,
                side: str = "front") -> None:
        def run():
            g, m = geometry.from_side(self.grid, self.mat, side)
            new = geometry.deposit(g, m, material_code(material), thickness_nm, conformal)
            return {"cells_added": int(new.sum())}
        self._run("deposit", {"material": material, "thickness_nm": thickness_nm,
                              "conformal": conformal, "side": side}, run)

    def planarize(self, level_nm: float = 0.0, side: str = "front") -> None:
        """Polish flat at `level_nm` (depth from the initial front surface): the
        front side removes everything above it, the back side everything below."""
        self._run("planarize", {"level_nm": level_nm, "side": side},
                  lambda: self._polish(level_nm, side))

    def thin(self, si_thickness_nm: float) -> None:
        """Bond face down and grind the wafer from the back to `si_thickness_nm`
        (measured from the initial front surface); later back-side steps start
        from the new surface."""
        if not 0 < si_thickness_nm < self.grid.depth_nm:
            raise ValueError(f"thickness must be between 0 and the box depth ({self.grid.depth_nm} nm); "
                             "make depth_nm larger to leave room for the back-side films")
        self._run("thin", {"si_thickness_nm": si_thickness_nm},
                  lambda: self._polish(si_thickness_nm, "back"))

    def _polish(self, level_nm: float, side: str) -> dict:
        g, m = geometry.from_side(self.grid, self.mat, side)
        level = level_nm if side == "front" else -level_nm
        removed = geometry.unflip(geometry.planarize(g, m, level), side)
        self._forget(removed)
        return {"cells_removed": int(removed.sum())}

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
            model = DealGrove.at(temp_c, ambient, self.orientation, self.params.oxidation)
            front = OxidationFront(self.grid, model)
            oxide_before = int((self.mat == OXIDE).sum())
            info = self._thermal(temp_c, time_min * 60, steps, front)
            info["oxide_cells_added"] = int((self.mat == OXIDE).sum()) - oxide_before
            info["headroom_exceeded"] = front.headroom_exceeded
            return info
        self._run("oxidize", {"temp_c": temp_c, "time_min": time_min, "ambient": ambient,
                              "steps": steps}, run)

    def _thermal(self, temp_c: float, duration_s: float, steps: int,
                 front: OxidationFront | None = None) -> dict:
        transport = DopantTransport(self.grid, temp_c, self.params.dopants)
        tau = self.params.defects.dissolution_time(temp_c)
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
                    segregate_at_new_oxide(self.grid, self.conc, self.mat, converted, self.params.dopants)
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
        return float((field_cm3 * self.grid.volumes_cm).sum() / self._lateral_area_cm2())

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

    def layers(self, pos_nm=None) -> list[dict]:
        """The stack of materials in one column, top to bottom (gas left out)."""
        col = self.mat[self.column(pos_nm)]
        z = self.grid.z_edges
        starts = np.flatnonzero(np.diff(col, prepend=-1) != 0)
        ends = np.append(starts[1:], col.size)
        return [{"material": NAMES[int(col[a])], "from_nm": round(float(z[a]), 2),
                 "to_nm": round(float(z[b]), 2), "thickness_nm": round(float(z[b] - z[a]), 2)}
                for a, b in zip(starts, ends) if col[a] != GAS]

    def depth_profile(self, pos_nm=None) -> dict[str, np.ndarray]:
        """One column as a measurement would see it: depth, material and each dopant."""
        col = self.column(pos_nm)
        out = {"depth_nm": self.grid.z_centers, "material": self.mat[col]}
        out |= {name: c[col] for name, c in sorted(self.conc.items())}
        out["net"] = self.net_doping()[col]
        return out

    def total_dose(self, dopant: str) -> float:
        c = self.conc.get(dopant)
        return 0.0 if c is None else self._areal(c)

    def oxide_thickness(self, pos_nm=None) -> float:
        """Oxide (nm) stacked directly on the silicon of one column."""
        col = self.mat[self.column(pos_nm)]
        si = np.flatnonzero(col == SI)
        top = si[0] if si.size else col.size
        return float(self.grid.widths(-1)[:top][col[:top] == OXIDE].sum())

    def material_fractions(self) -> dict[str, float]:
        return {name: float((self.mat == code).mean()) for code, name in NAMES.items()}

    def probe(self, pos_nm=None) -> dict:
        """Figures of one column that are measured on real wafers."""
        return {"at_nm": pos_nm if pos_nm is not None else [w / 2 for w in self.grid.lateral_nm],
                "junction_depths_nm": [round(x, 1) for x in self.junctions_along_depth(pos_nm)],
                "layers": self.layers(pos_nm)}

    def summary(self, probes=None) -> dict:
        """probes: lateral positions (nm) of the columns to report; default the box centre."""
        return {
            "grid": self.grid.describe(),
            "substrate": {"dopant": self.substrate_dopant, "conc": self.substrate_conc,
                          "profile": self.substrate_profile},
            "steps": [r.to_dict() for r in self.history],
            "total_seconds": round(sum(r.seconds for r in self.history), 3),
            "dose_cm2": {k: self.total_dose(k) for k in sorted(self.conc)},
            "excess_interstitials_cm2": self._areal(self.defects.clusters),
            "probes": [self.probe(p) for p in (probes or [None])],
            "material_fractions": self.material_fractions(),
        }
