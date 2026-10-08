"""Units and physical constants (from scipy.constants / CODATA where available)."""

from __future__ import annotations

from scipy import constants

# Julian year (IAU): 365.25 days. Half-lives are converted with this value; the
# conversion cancels in every ratio, and the tests state their units.
YEAR_S = 365.25 * 86400.0

TIME_UNITS_S = {
    "saniye": 1.0,
    "dakika": 60.0,
    "saat": 3600.0,
    "gun": 86400.0,
    "yil": YEAR_S,
}

AVOGADRO = constants.N_A
CURIE_BQ = 3.7e10  # exact by definition
ELECTRON_MASS_KEV = constants.physical_constants["electron mass energy equivalent in MeV"][0] * 1e3
U_KEV = constants.physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e3


def to_seconds(value: float, unit: str) -> float:
    try:
        return value * TIME_UNITS_S[unit]
    except KeyError:
        raise ValueError(f"bilinmeyen zaman birimi: {unit!r} (seçenekler: {', '.join(TIME_UNITS_S)})") from None
