"""Neutron activation (single product, target burn-up neglected).

Production at the constant rate R = N_target * sigma * phi balances decay of the
product, dN/dt = R - lambda*N, so after irradiation time t_irr and cooling time
t_cool the activity is

    A = N_target * sigma * phi * (1 - exp(-lambda t_irr)) * exp(-lambda t_cool).

The saturation activity N*sigma*phi is approached as 1 - 2^(-t_irr/T_half):
50 % after one half-life, 75 % after two, 90 % after 3.32.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm

from .decay import decay_constant

BARN_CM2 = 1.0e-24


def production_rate_per_s(target_atoms: float, cross_section_barn: float, flux_cm2_s: float) -> float:
    return target_atoms * cross_section_barn * BARN_CM2 * flux_cm2_s


def saturation_fraction(t_irr_s: float, half_life_s: float) -> float:
    return 1.0 - math.exp(-decay_constant(half_life_s) * t_irr_s)


def time_to_saturation_fraction(fraction: float, half_life_s: float) -> float:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("oran [0, 1) aralığında olmalı")
    return -math.log(1.0 - fraction) / decay_constant(half_life_s)


def activation_activity_bq(
    target_atoms: float, cross_section_barn: float, flux_cm2_s: float, half_life_s: float, t_irr_s: float, t_cool_s: float = 0.0
) -> float:
    if t_irr_s < 0 or t_cool_s < 0:
        raise ValueError("süreler negatif olamaz")
    lam = decay_constant(half_life_s)
    rate = production_rate_per_s(target_atoms, cross_section_barn, flux_cm2_s)
    return rate * saturation_fraction(t_irr_s, half_life_s) * math.exp(-lam * t_cool_s)


def activation_activity_numeric_bq(
    target_atoms: float, cross_section_barn: float, flux_cm2_s: float, half_life_s: float, t_irr_s: float, t_cool_s: float = 0.0
) -> float:
    """Independent check: the augmented linear system d/dt [N, 1] = [[-lambda, R], [0, 0]] [N, 1]
    integrated with a matrix exponential, irradiation then cooling (no source)."""
    lam = decay_constant(half_life_s)
    rate = production_rate_per_s(target_atoms, cross_section_barn, flux_cm2_s)
    irradiate = expm(np.array([[-lam, rate], [0.0, 0.0]]) * t_irr_s)
    cool = expm(np.array([[-lam, 0.0], [0.0, 0.0]]) * t_cool_s)
    n_product = (cool @ irradiate @ np.array([0.0, 1.0]))[0]
    return lam * n_product
