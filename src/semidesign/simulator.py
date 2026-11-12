"""1D process simulator: a silicon wafer state and the process steps applied to it."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import physics
from .physics import SI_CONSUMED_PER_OXIDE, get_dopant


@dataclass
class Wafer:
    """Silicon substrate seen as a 1D column below the Si surface.

    Depth 0 is always the current silicon surface (top of Si, below any oxide).
    """

    background_dopant: str = "B"
    background_conc: float = 1e15  # cm^-3
    depth_nm: float = 1000.0
    dx_nm: float = 1.0
    orientation: str = "100"
    oxide_nm: float = 1.0  # native oxide
    profiles: dict[str, np.ndarray] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        get_dopant(self.background_dopant)
        self.x = np.arange(0.0, self.depth_nm + self.dx_nm / 2, self.dx_nm)
        bg = np.full_like(self.x, self.background_conc)
        self.profiles.setdefault(self.background_dopant, bg)

    # -- derived quantities -------------------------------------------------

    def total(self, dopant: str) -> np.ndarray:
        return self.profiles.get(dopant, np.zeros_like(self.x))

    def net_doping(self) -> np.ndarray:
        """Donors minus acceptors (positive = n-type), cm^-3."""
        net = np.zeros_like(self.x)
        for name, c in self.profiles.items():
            net += c if get_dopant(name).kind == "n" else -c
        return net

    def junction_depths(self) -> list[float]:
        """Depths (nm) where the net doping changes sign, linearly interpolated."""
        net = self.net_doping()
        s = np.sign(net)
        idx = np.where(s[:-1] * s[1:] < 0)[0]
        out = []
        for i in idx:
            x0, x1, n0, n1 = self.x[i], self.x[i + 1], net[i], net[i + 1]
            out.append(float(x0 - n0 * (x1 - x0) / (n1 - n0)))
        return out

    def sheet_dose(self, dopant: str) -> float:
        """Integrated dopant in silicon, cm^-2."""
        return float(np.trapezoid(self.total(dopant), self.x) / physics.NM_PER_CM)

    # -- process steps --------------------------------------------------------

    def implant(self, dopant: str, dose: float, energy_kev: float,
                rp_nm: float | None = None, drp_nm: float | None = None) -> None:
        """Gaussian implant through the current oxide.

        The range is taken from the built-in table unless rp/drp are given.
        Oxide is treated as having the same stopping power as Si, and the part
        of the profile that stops in the oxide is discarded.
        """
        d = get_dopant(dopant)
        if rp_nm is None or drp_nm is None:
            rp_tab, drp_tab = d.projected_range(energy_kev)
            rp_nm = rp_nm if rp_nm is not None else rp_tab
            drp_nm = drp_nm if drp_nm is not None else drp_tab
        added = physics.gaussian_implant(self.x + self.oxide_nm, dose, rp_nm, drp_nm)
        self.profiles[dopant] = self.total(dopant) + added
        self.history.append(
            f"implant {dopant} {dose:.2e} cm^-2 @ {energy_kev:g} keV "
            f"(Rp={rp_nm:.1f} nm, dRp={drp_nm:.1f} nm, through {self.oxide_nm:.1f} nm oxide)"
        )

    def anneal(self, temp_c: float, time_min: float, steps: int = 200) -> None:
        """Inert anneal: intrinsic diffusion of every dopant."""
        self._thermal(temp_c, time_min, steps, ambient=None)
        self.history.append(f"anneal {temp_c:g} C, {time_min:g} min")

    def oxidize(self, temp_c: float, time_min: float, ambient: str = "dry",
                steps: int = 200) -> None:
        """Thermal oxidation (Deal-Grove) with simultaneous dopant diffusion.

        Silicon consumed by the oxide is removed from the top of the column;
        dopant in that silicon is discarded (no segregation in this version).
        """
        before = self.oxide_nm
        self._thermal(temp_c, time_min, steps, ambient=ambient)
        self.history.append(
            f"oxidize {ambient} {temp_c:g} C, {time_min:g} min: "
            f"oxide {before:.1f} -> {self.oxide_nm:.1f} nm"
        )

    def strip_oxide(self) -> None:
        self.history.append(f"strip oxide ({self.oxide_nm:.1f} nm)")
        self.oxide_nm = 0.0

    def _thermal(self, temp_c: float, time_min: float, steps: int,
                 ambient: str | None) -> None:
        dt_min = time_min / steps
        dt_s = dt_min * 60
        diff = {name: get_dopant(name).diffusivity(temp_c) * physics.NM_PER_CM**2
                for name in self.profiles}  # nm^2/s
        if ambient is not None:
            b, ba = physics.deal_grove_coefficients(temp_c, ambient, self.orientation)
        consumed_pending = 0.0
        for _ in range(steps):
            for name, c in self.profiles.items():
                self.profiles[name] = physics.diffuse_step(c, diff[name], self.dx_nm, dt_s)
            if ambient is not None:
                new_ox = physics.deal_grove_grow(self.oxide_nm, dt_min, b, ba)
                consumed_pending += SI_CONSUMED_PER_OXIDE * (new_ox - self.oxide_nm)
                self.oxide_nm = new_ox
                if consumed_pending >= self.dx_nm * 0.1:
                    self._consume_silicon(consumed_pending)
                    consumed_pending = 0.0
        if consumed_pending > 0:
            self._consume_silicon(consumed_pending)

    def _consume_silicon(self, thickness_nm: float) -> None:
        """Shift the column up by the consumed thickness; refill the bottom."""
        for name, c in self.profiles.items():
            fill = self.background_conc if name == self.background_dopant else 0.0
            self.profiles[name] = np.interp(self.x + thickness_nm, self.x, c, right=fill)
