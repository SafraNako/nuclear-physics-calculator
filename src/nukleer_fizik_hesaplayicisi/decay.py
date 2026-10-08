"""Radioactive decay law: N(t) = N0 * exp(-lambda*t), lambda = ln2 / T_half.

All times in seconds unless a function says otherwise. A half-life of math.inf
means a stable nuclide (lambda = 0).
"""

from __future__ import annotations

import math

from .constants import AVOGADRO, CURIE_BQ


def decay_constant(half_life_s: float) -> float:
    if half_life_s <= 0:
        raise ValueError("yarı ömür pozitif olmalı")
    return 0.0 if math.isinf(half_life_s) else math.log(2.0) / half_life_s


def mean_life(half_life_s: float) -> float:
    lam = decay_constant(half_life_s)
    return math.inf if lam == 0.0 else 1.0 / lam


def fraction_remaining(t_s: float, half_life_s: float) -> float:
    return math.exp(-decay_constant(half_life_s) * t_s)


def activity_bq(a0_bq: float, t_s: float, half_life_s: float) -> float:
    return a0_bq * fraction_remaining(t_s, half_life_s)


def atoms_from_activity(activity_bq_value: float, half_life_s: float) -> float:
    lam = decay_constant(half_life_s)
    if lam == 0.0:
        raise ValueError("kararlı nüklidin aktivitesi sıfırdır")
    return activity_bq_value / lam


def time_to_fraction(fraction: float, half_life_s: float) -> float:
    """Time at which the fraction `fraction` of the initial atoms is left."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("oran (0, 1] aralığında olmalı")
    return -math.log(fraction) / decay_constant(half_life_s)


def atoms_in_sample(mass_g: float, molar_mass_g_per_mol: float) -> float:
    return mass_g / molar_mass_g_per_mol * AVOGADRO


def specific_activity_bq_per_g(half_life_s: float, molar_mass_g_per_mol: float) -> float:
    """A / m = lambda * N_A / M."""
    return decay_constant(half_life_s) * AVOGADRO / molar_mass_g_per_mol


def bq_to_curie(activity_bq_value: float) -> float:
    return activity_bq_value / CURIE_BQ
