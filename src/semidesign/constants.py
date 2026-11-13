"""Physical constants and unit conversions used across the package."""

K_BOLTZMANN_EV = 8.617333262e-5  # eV/K
NM_PER_CM = 1e7
NM2_PER_CM2 = 1e14
SI_ATOMS_PER_CM3 = 4.995e22  # crystalline silicon


def kelvin(temp_c: float) -> float:
    return temp_c + 273.15


def kt_ev(temp_c: float) -> float:
    return K_BOLTZMANN_EV * kelvin(temp_c)
