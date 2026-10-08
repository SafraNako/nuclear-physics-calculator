from __future__ import annotations

import math

import pytest

from nukleer_fizik_hesaplayicisi import ame
from nukleer_fizik_hesaplayicisi.constants import CURIE_BQ, TIME_UNITS_S, YEAR_S, to_seconds
from nukleer_fizik_hesaplayicisi.decay import (
    activity_bq,
    atoms_from_activity,
    atoms_in_sample,
    bq_to_curie,
    decay_constant,
    fraction_remaining,
    mean_life,
    specific_activity_bq_per_g,
    time_to_fraction,
)


def test_half_life_mean_life_and_decay_constant_are_related_by_ln2():
    t_half = 8.0233 * 86400.0  # I-131, days -> s
    lam = decay_constant(t_half)
    assert math.isclose(lam * t_half, math.log(2.0))
    assert math.isclose(mean_life(t_half), t_half / math.log(2.0))
    assert math.isclose(mean_life(t_half) * lam, 1.0)


def test_one_half_life_halves_one_mean_life_leaves_one_over_e():
    t_half = 1234.5
    assert math.isclose(fraction_remaining(t_half, t_half), 0.5)
    assert math.isclose(fraction_remaining(mean_life(t_half), t_half), math.exp(-1.0))
    assert math.isclose(fraction_remaining(10 * t_half, t_half), 2.0**-10)


def test_decay_factors_multiply_over_consecutive_intervals():
    t_half = 500.0
    assert math.isclose(fraction_remaining(300.0 + 700.0, t_half), fraction_remaining(300.0, t_half) * fraction_remaining(700.0, t_half))


def test_time_to_fraction_inverts_fraction_remaining():
    t_half = 4321.0
    for f in (0.9, 0.5, 0.1, 1e-3):
        assert math.isclose(fraction_remaining(time_to_fraction(f, t_half), t_half), f)
    assert math.isclose(time_to_fraction(0.5, t_half), t_half)
    assert time_to_fraction(1.0, t_half) == 0.0


def test_activity_follows_the_same_law_and_atoms_follow_from_activity():
    t_half = 3600.0
    assert math.isclose(activity_bq(1e6, t_half, t_half), 5e5)
    n = atoms_from_activity(1e6, t_half)
    assert math.isclose(n * decay_constant(t_half), 1e6)


def test_stable_nuclide_never_decays():
    assert decay_constant(math.inf) == 0.0
    assert fraction_remaining(1e30, math.inf) == 1.0
    assert mean_life(math.inf) == math.inf
    with pytest.raises(ValueError):
        atoms_from_activity(1.0, math.inf)


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_nonpositive_half_life_is_rejected(bad):
    with pytest.raises(ValueError):
        decay_constant(bad)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_fraction_outside_the_unit_interval_is_rejected(bad):
    with pytest.raises(ValueError):
        time_to_fraction(bad, 100.0)


def test_cobalt_60_specific_activity_is_about_1130_curie_per_gram():
    x = ame.parse_nuclide("Co-60")
    spec = specific_activity_bq_per_g(5.2711 * YEAR_S, x.atomic_mass_u)
    assert 1.10e3 < bq_to_curie(spec) < 1.16e3
    assert math.isclose(spec, 4.187e13, rel_tol=2e-3)


def test_radium_226_specific_activity_is_about_one_curie_per_gram():
    """The curie was originally the activity of 1 g of Ra-226 (3.7e10 Bq); modern half-life: 3.66e10 Bq/g."""
    x = ame.parse_nuclide("Ra-226")
    spec = specific_activity_bq_per_g(1600.0 * YEAR_S, x.atomic_mass_u)
    assert math.isclose(spec, 3.66e10, rel_tol=0.01)
    assert math.isclose(bq_to_curie(spec), 0.989, abs_tol=0.003)


def test_specific_activity_equals_activity_per_mass_from_atom_counting():
    t_half, molar = 1e8, 100.0
    n = atoms_in_sample(2.0, molar)
    assert math.isclose(specific_activity_bq_per_g(t_half, molar) * 2.0, decay_constant(t_half) * n)


def test_curie_is_3_7e10_becquerel():
    assert CURIE_BQ == 3.7e10
    assert math.isclose(bq_to_curie(3.7e10), 1.0)


def test_time_units():
    assert to_seconds(2.0, "saat") == 7200.0
    assert to_seconds(1.0, "gun") == 86400.0
    assert math.isclose(to_seconds(1.0, "yil"), 365.25 * 86400.0)
    assert set(TIME_UNITS_S) == {"saniye", "dakika", "saat", "gun", "yil"}
    with pytest.raises(ValueError):
        to_seconds(1.0, "ay")
