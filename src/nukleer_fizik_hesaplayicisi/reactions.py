"""Decay and reaction Q-values from AME2020 mass excesses.

For atomic masses (electron masses cancel or are accounted for explicitly):

    Q_alpha = Delta(Z,A) - Delta(Z-2,A-4) - Delta(4He)
    Q_beta- = Delta(Z,A) - Delta(Z+1,A)
    Q_EC    = Delta(Z,A) - Delta(Z-1,A)
    Q_beta+ = Q_EC - 2 m_e c^2          (the positron plus the extra atomic electron)

A decay is energetically allowed when Q > 0. These are Q-values to the daughter
ground state.
"""

from __future__ import annotations

import math

from . import ame
from .constants import ELECTRON_MASS_KEV


def _delta(z: int, a: int) -> float:
    return ame.get(z, a).mass_excess_kev


def _unc(*pairs) -> float:
    return math.sqrt(sum(ame.get(z, a).mass_excess_unc_kev ** 2 for z, a in pairs))


def q_alpha_kev(z: int, a: int) -> float:
    return _delta(z, a) - _delta(z - 2, a - 4) - _delta(2, 4)


def q_beta_minus_kev(z: int, a: int) -> float:
    return _delta(z, a) - _delta(z + 1, a)


def q_ec_kev(z: int, a: int) -> float:
    return _delta(z, a) - _delta(z - 1, a)


def q_beta_plus_kev(z: int, a: int) -> float:
    return q_ec_kev(z, a) - 2.0 * ELECTRON_MASS_KEV


def q_uncertainty_kev(kind: str, z: int, a: int) -> float:
    """Propagated (uncorrelated) uncertainty of the Q-value, keV."""
    if kind == "alpha":
        return _unc((z, a), (z - 2, a - 4), (2, 4))
    if kind == "beta-minus":
        return _unc((z, a), (z + 1, a))
    if kind in ("ec", "beta-plus"):
        return _unc((z, a), (z - 1, a))
    raise ValueError(f"bilinmeyen bozunma türü: {kind!r}")


def neutron_separation_energy_kev(z: int, a: int) -> float:
    """S_n = Delta(Z, A-1) + Delta(n) - Delta(Z, A)."""
    return _delta(z, a - 1) + _delta(0, 1) - _delta(z, a)


def proton_separation_energy_kev(z: int, a: int) -> float:
    """S_p = Delta(Z-1, A-1) + Delta(1H) - Delta(Z, A)."""
    return _delta(z - 1, a - 1) + _delta(1, 1) - _delta(z, a)


Q_FUNCTIONS = {
    "alpha": q_alpha_kev,
    "beta-minus": q_beta_minus_kev,
    "ec": q_ec_kev,
    "beta-plus": q_beta_plus_kev,
}
