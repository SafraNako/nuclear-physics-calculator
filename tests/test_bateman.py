from __future__ import annotations

import math

import numpy as np
import pytest

from nukleer_fizik_hesaplayicisi import ddep
from nukleer_fizik_hesaplayicisi.bateman import (
    bateman_closed_form,
    bateman_reference,
    chain_activities,
    chain_populations,
    decay_matrix,
    time_of_maximum_daughter_activity,
    transient_equilibrium_ratio,
)
from nukleer_fizik_hesaplayicisi.constants import to_seconds
from nukleer_fizik_hesaplayicisi.decay import decay_constant

MO = to_seconds(2.7479, "gun")
TC = to_seconds(6.0067, "saat")


def _rel(a, b):
    return abs(a - b) / abs(b)


def test_single_member_chain_is_the_plain_exponential():
    t_half = 1000.0
    n = chain_populations([t_half], [0.0, t_half, 3 * t_half])
    assert np.allclose(n[:, 0], [1.0, 0.5, 0.125])


def test_matrix_columns_sum_to_zero_for_unit_branching_so_atoms_are_conserved():
    m = decay_matrix([10.0, 3.0, 50.0, math.inf])
    assert np.allclose(m.sum(axis=0)[:3], 0.0)
    n = chain_populations([10.0, 3.0, 50.0, math.inf], [0.0, 5.0, 40.0, 1000.0])
    assert np.allclose(n.sum(axis=1), 1.0, atol=1e-12)


def test_two_member_chain_against_the_textbook_formula():
    l1, l2 = decay_constant(MO), decay_constant(TC)
    for t in (3600.0, 86400.0, 5 * 86400.0):
        n2 = chain_populations([MO, TC], [t])[0, 1]
        textbook = l1 / (l2 - l1) * (math.exp(-l1 * t) - math.exp(-l2 * t))
        assert math.isclose(n2, textbook, rel_tol=1e-12)


def test_branching_scales_the_daughter_by_the_branching_fraction():
    b = 0.876
    full = chain_populations([MO, TC], [86400.0])[0, 1]
    branched = chain_populations([MO, TC], [86400.0], branching=[b])[0, 1]
    assert math.isclose(branched, b * full, rel_tol=1e-12)
    n = chain_populations([MO, TC], [86400.0], branching=[b])[0]
    assert n.sum() < 1.0  # the rest fed another channel


def test_matrix_exponential_and_bateman_agree_with_an_80_digit_reference_for_well_separated_chains():
    for half_lives, branching in (([MO, TC], [0.876]), ([500.0, 120.0, 30.0, 7.0], None), ([1e5, 10.0, 1.0], [0.4, 1.0])):
        for t in (1.0, 50.0, 400.0):
            ref = [float(x) for x in bateman_reference(half_lives, t, branching)]
            numeric = chain_populations(half_lives, [t], branching)[0]
            closed = bateman_closed_form(half_lives, t, branching)
            for i in range(len(ref)):
                assert _rel(numeric[i], ref[i]) < 1e-12
                assert _rel(closed[i], ref[i]) < 1e-9  # looser: see the early-time test below


def test_bateman_also_loses_precision_for_deep_chain_members_at_early_times():
    """At t = 1 the fourth member of a 4-chain has barely been fed (~(lambda t)^3/6), yet the Bateman terms are
    each of order one and cancel: relative error ~3.5e-11 (measured), against ~2e-16 for the matrix exponential."""
    half_lives = [500.0, 120.0, 30.0, 7.0]
    ref = float(bateman_reference(half_lives, 1.0)[3])
    assert ref < 1e-5
    closed_error = _rel(bateman_closed_form(half_lives, 1.0)[3], ref)
    expm_error = _rel(chain_populations(half_lives, [1.0])[0, 3], ref)
    assert closed_error > 1e-12
    assert expm_error < 1e-14
    assert closed_error > 100 * expm_error


def test_stable_end_member_accumulates_everything():
    n = chain_populations([100.0, 10.0, math.inf], [1e5])[0]
    assert math.isclose(n[2], 1.0, abs_tol=1e-9)
    ref = [float(x) for x in bateman_reference([100.0, 10.0, math.inf], 250.0)]
    assert math.isclose(chain_populations([100.0, 10.0, math.inf], [250.0])[0, 2], ref[2], rel_tol=1e-11)


def test_equal_half_lives_have_the_closed_form_lambda_t_exp_minus_lambda_t():
    t_half, t = 10.0, 20.0
    lam = decay_constant(t_half)
    n2 = chain_populations([t_half, t_half], [t])[0, 1]
    assert math.isclose(n2, lam * t * math.exp(-lam * t), rel_tol=1e-12)
    with pytest.raises(ZeroDivisionError):
        bateman_closed_form([t_half, t_half], t)


def test_bateman_loses_precision_as_decay_constants_approach_each_other_but_the_matrix_exponential_does_not():
    """Relative error of the float Bateman formula grows like eps_machine/|l2 - l1|."""
    t_half_1 = 6.931471805599453  # lambda_1 = 0.1
    errors_closed, errors_expm = [], []
    for eps in (1e-3, 1e-6, 1e-9):
        t_half_2 = t_half_1 / (1.0 + eps)
        t = 10.0
        ref = float(bateman_reference([t_half_1, t_half_2], t)[1])
        errors_closed.append(_rel(bateman_closed_form([t_half_1, t_half_2], t)[1], ref))
        errors_expm.append(_rel(chain_populations([t_half_1, t_half_2], [t])[0, 1], ref))
    assert errors_closed[2] > 1e3 * errors_closed[0]
    assert errors_closed[2] > 1e-9
    assert max(errors_expm) < 1e-12


def test_transient_equilibrium_ratio_of_the_technetium_generator_matches_ddep():
    """DDEP states T1/(T1 - T2) = 1.1005(8) and, with p = 0.876(19), A(Tc-99m)/A(Mo-99) = 0.963(21)."""
    factor = transient_equilibrium_ratio(MO, TC)
    assert abs(factor - ddep.MO99_TC99M_EQUILIBRIUM_FACTOR) < ddep.MO99_TC99M_EQUILIBRIUM_FACTOR_UNC
    with_branching = transient_equilibrium_ratio(MO, TC, branching=ddep.MO99_TO_TC99M_BRANCHING)
    assert abs(with_branching - ddep.MO99_TC99M_EQUILIBRIUM_RATIO) < ddep.MO99_TC99M_EQUILIBRIUM_RATIO_UNC


def test_transient_equilibrium_is_reached_numerically():
    ratio = transient_equilibrium_ratio(MO, TC, 0.876)
    t = 25 * TC
    a = chain_activities([MO, TC], [t], branching=[0.876])[0]
    assert math.isclose(a[1] / a[0], ratio, rel_tol=1e-4)


def test_secular_equilibrium_when_the_parent_is_much_longer_lived():
    parent, daughter = 1e6 * TC, TC
    a = chain_activities([parent, daughter], [30 * daughter])[0]
    assert math.isclose(a[1] / a[0], 1.0, rel_tol=1e-5)
    assert math.isclose(transient_equilibrium_ratio(parent, daughter), 1.0, rel_tol=2e-6)


def test_time_of_maximum_activity_is_where_the_derivative_vanishes():
    t_max = time_of_maximum_daughter_activity(MO, TC)
    assert math.isclose(t_max / 3600.0, 22.84, abs_tol=0.01)  # the familiar ~23 h generator elution peak
    h = 1.0
    a = chain_activities([MO, TC], [t_max - h, t_max, t_max + h])[:, 1]
    assert a[1] > a[0] and a[1] > a[2]
    assert abs(a[2] - a[0]) / a[1] < 1e-6  # flat at the top


def test_time_of_maximum_for_equal_half_lives_is_one_mean_life():
    assert math.isclose(time_of_maximum_daughter_activity(100.0, 100.0), 1.0 / decay_constant(100.0))


def test_daughter_activity_peaks_below_the_parent_and_the_peak_lies_on_the_parent_curve():
    """At t_max the daughter and parent activities are equal (unit branching): A2 = A1 where A2 is maximal."""
    t_max = time_of_maximum_daughter_activity(MO, TC)
    a = chain_activities([MO, TC], [t_max])[0]
    assert math.isclose(a[0], a[1], rel_tol=1e-9)


def test_activities_are_lambda_times_populations():
    a = chain_activities([MO, TC], [10000.0], a0=7.0)[0]
    n = chain_populations([MO, TC], [10000.0], n0=7.0 / decay_constant(MO))[0]
    assert np.allclose(a, n * np.array([decay_constant(MO), decay_constant(TC)]))
    assert math.isclose(chain_activities([MO, TC], [0.0], a0=7.0)[0, 0], 7.0)


@pytest.mark.parametrize("branching", [[0.5, 0.5], [1.5], [-0.1]])
def test_bad_branching_is_rejected(branching):
    with pytest.raises(ValueError):
        decay_matrix([MO, TC], branching)


def test_transient_equilibrium_needs_a_longer_lived_parent():
    with pytest.raises(ValueError):
        transient_equilibrium_ratio(TC, MO)


def test_stable_parent_has_no_activity():
    with pytest.raises(ValueError):
        chain_activities([math.inf, 10.0], [1.0])
