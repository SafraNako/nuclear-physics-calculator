"""Evaluated decay data for a few nuclides: half-lives and Q-values from the
LNHB/DDEP "Table de Radionucleides" (the same evaluation used for the gamma
lines in gamma-spectrum-analyzer).

Each entry: (half-life value, uncertainty, unit), Q-value (keV, uncertainty,
kind) and, for Mo-99, the fraction p of decays that feed the Tc-99m isomer.
The Q-values are used to cross-check the Q-values computed from AME2020 mass
excesses: two independent evaluations of the same quantity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DdepEntry:
    nuclide: str
    z: int
    a: int
    half_life: float
    half_life_unc: float
    unit: str  # one of constants.TIME_UNITS_S
    q_kev: float | None = None
    q_unc_kev: float | None = None
    q_kind: str | None = None  # 'alpha' | 'beta-minus' | 'ec'


DDEP = {
    "co-60": DdepEntry("Co-60", 27, 60, 5.2711, 0.0008, "yil", 2823.07, 0.21, "beta-minus"),
    "cs-137": DdepEntry("Cs-137", 55, 137, 30.018, 0.022, "yil", 1175.63, 0.17, "beta-minus"),
    "na-22": DdepEntry("Na-22", 11, 22, 2.6029, 0.0008, "yil", 2843.02, 0.21, "ec"),
    "am-241": DdepEntry("Am-241", 95, 241, 432.6, 0.6, "yil", 5637.82, 0.12, "alpha"),
    "ba-133": DdepEntry("Ba-133", 56, 133, 10.539, 0.006, "yil", 517.3, 1.0, "ec"),
    "mo-99": DdepEntry("Mo-99", 42, 99, 2.7479, 0.0006, "gun", 1357.2, 1.0, "beta-minus"),
    "tc-99m": DdepEntry("Tc-99m", 43, 99, 6.0067, 0.0010, "saat"),
}

# Fraction of Mo-99 decays that populate the 142 keV isomer Tc-99m (DDEP: p = 0.876(19)).
MO99_TO_TC99M_BRANCHING = 0.876
MO99_TO_TC99M_BRANCHING_UNC = 0.019
# DDEP's own statement of the equilibrium: A(Tc-99m)/A(Mo-99) = p * T1/(T1 - T2) = 0.963(21),
# with T1/(T1 - T2) = 1.1005(8).
MO99_TC99M_EQUILIBRIUM_RATIO = 0.963
MO99_TC99M_EQUILIBRIUM_RATIO_UNC = 0.021
MO99_TC99M_EQUILIBRIUM_FACTOR = 1.1005
MO99_TC99M_EQUILIBRIUM_FACTOR_UNC = 0.0008


def lookup(name: str) -> DdepEntry:
    key = name.lower().replace(" ", "")
    if key not in DDEP:
        raise ValueError(f"DDEP tablosunda yok: {name!r} (seçenekler: {', '.join(DDEP)})")
    return DDEP[key]
