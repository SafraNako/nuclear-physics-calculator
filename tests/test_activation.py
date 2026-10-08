from __future__ import annotations

import math

import pytest

from nukleer_fizik_hesaplayicisi.activation import (
    BARN_CM2,
    activation_activity_bq,
    activation_activity_numeric_bq,
    production_rate_per_s,
    saturation_fraction,
    time_to_saturation_fraction,
)
from nukleer_fizik_hesaplayicisi.decay import decay_constant

T_HALF = 65.9496 * 3600.0


def test_production_rate_arithmetic():
    assert math.isclose(production_rate_per_s(1e20, 1.0, 1e14), 1e10)
    assert BARN_CM2 == 1e-24


@pytest.mark.parametrize(
    "t_irr_h,t_cool_h", [(10.0, 0.0), (100.0, 10.0), (500.0, 24.0), (1.0, 100.0), (0.0, 5.0)]
)
def test_closed_form_matches_an_independent_matrix_exponential_integration(t_irr_h, t_cool_h):
    args = (3e19, 0.13, 8e13, T_HALF, t_irr_h * 3600.0, t_cool_h * 3600.0)
    assert math.isclose(activation_activity_bq(*args), activation_activity_numeric_bq(*args), rel_tol=1e-10, abs_tol=1e-9)


def test_saturation_fraction_after_whole_half_lives():
    assert math.isclose(saturation_fraction(T_HALF, T_HALF), 0.5)
    assert math.isclose(saturation_fraction(2 * T_HALF, T_HALF), 0.75)
    assert math.isclose(saturation_fraction(math.log2(10) * T_HALF, T_HALF), 0.9)  # 3.32 half-lives


def test_time_to_saturation_inverts_the_saturation_fraction():
    for f in (0.0, 0.5, 0.9, 0.99):
        assert math.isclose(saturation_fraction(time_to_saturation_fraction(f, T_HALF), T_HALF), f, abs_tol=1e-12)
    assert math.isclose(time_to_saturation_fraction(0.5, T_HALF), T_HALF)
    assert math.isclose(time_to_saturation_fraction(0.99, T_HALF) / T_HALF, math.log2(100), rel_tol=1e-12)
    with pytest.raises(ValueError):
        time_to_saturation_fraction(1.0, T_HALF)


def test_long_irradiation_approaches_the_saturation_activity_n_sigma_phi():
    a = activation_activity_bq(1e20, 2.0, 1e13, T_HALF, 40 * T_HALF)
    assert math.isclose(a, production_rate_per_s(1e20, 2.0, 1e13), rel_tol=1e-11)


def test_short_irradiation_activity_grows_linearly_with_time():
    """For t << T_half, N_product = R t, so A = lambda R t."""
    rate = production_rate_per_s(1e20, 2.0, 1e13)
    t = 1e-4 * T_HALF
    expected = decay_constant(T_HALF) * rate * t
    assert math.isclose(activation_activity_bq(1e20, 2.0, 1e13, T_HALF, t), expected, rel_tol=1e-4)


def test_cooling_decays_the_activity_exponentially():
    base = activation_activity_bq(1e20, 2.0, 1e13, T_HALF, 100 * 3600.0)
    cooled = activation_activity_bq(1e20, 2.0, 1e13, T_HALF, 100 * 3600.0, T_HALF)
    assert math.isclose(cooled, 0.5 * base)


def test_activity_is_proportional_to_target_cross_section_and_flux():
    a = activation_activity_bq(1e20, 1.0, 1e13, T_HALF, 1e5)
    assert math.isclose(activation_activity_bq(2e20, 1.0, 1e13, T_HALF, 1e5), 2 * a)
    assert math.isclose(activation_activity_bq(1e20, 3.0, 1e13, T_HALF, 1e5), 3 * a)
    assert math.isclose(activation_activity_bq(1e20, 1.0, 5e13, T_HALF, 1e5), 5 * a)


def test_negative_times_are_rejected():
    with pytest.raises(ValueError):
        activation_activity_bq(1e20, 1.0, 1e13, T_HALF, -1.0)
    with pytest.raises(ValueError):
        activation_activity_bq(1e20, 1.0, 1e13, T_HALF, 1.0, -1.0)
