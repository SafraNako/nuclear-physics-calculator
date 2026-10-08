from __future__ import annotations

import math

import pytest

from nukleer_fizik_hesaplayicisi import ame, ddep, reactions
from nukleer_fizik_hesaplayicisi.constants import ELECTRON_MASS_KEV


@pytest.mark.parametrize("key", [k for k, e in ddep.DDEP.items() if e.q_kind])
def test_q_values_from_ame_masses_agree_with_the_independent_ddep_evaluation(key):
    """Two independent evaluations of the same quantity: the Q-value from AME2020 mass differences and
    the one in the LNHB/DDEP decay-data table (observed differences: 0.00 to 0.56 keV)."""
    entry = ddep.DDEP[key]
    q = reactions.Q_FUNCTIONS[entry.q_kind](entry.z, entry.a)
    unc = reactions.q_uncertainty_kev(entry.q_kind, entry.z, entry.a)
    sigma = math.hypot(unc, entry.q_unc_kev)
    assert abs(q - entry.q_kev) < 2.5 * sigma
    assert abs(q - entry.q_kev) < 1.0  # keV


def test_beta_plus_q_is_the_ec_q_minus_two_electron_masses():
    assert math.isclose(ELECTRON_MASS_KEV, 510.99895, abs_tol=1e-4)
    for z, a in ((11, 22), (9, 18), (27, 56)):
        assert math.isclose(reactions.q_beta_plus_kev(z, a), reactions.q_ec_kev(z, a) - 1021.9979, abs_tol=1e-3)


def test_sodium_22_positron_endpoints_match_the_ddep_transition_table():
    """DDEP lists the beta+ transition to the ground state at 1821.02(21) keV and to the 1274.537 keV
    level at 546.44(21) keV (the latter is Q_beta+ minus the level energy)."""
    q_plus = reactions.q_beta_plus_kev(11, 22)
    assert abs(q_plus - 1821.02) < 0.8
    assert abs((q_plus - 1274.537) - 546.44) < 0.8


@pytest.mark.parametrize(
    "name,kind,textbook_kev,tolerance_kev",
    [
        ("H-3", "beta-minus", 18.59, 0.05),
        ("C-14", "beta-minus", 156.5, 0.1),
        ("K-40", "ec", 1504.4, 0.5),
        ("K-40", "beta-minus", 1311.0, 0.5),
        ("Po-210", "alpha", 5407.5, 0.5),
        ("Ra-226", "alpha", 4870.6, 0.5),
        ("U-238", "alpha", 4269.7, 0.5),
    ],
)
def test_textbook_q_values(name, kind, textbook_kev, tolerance_kev):
    x = ame.parse_nuclide(name)
    assert abs(reactions.Q_FUNCTIONS[kind](x.z, x.a) - textbook_kev) < tolerance_kev


@pytest.mark.parametrize("name", ["Fe-56", "Pb-208", "C-12", "O-16", "Ca-40", "Ni-62", "Si-28"])
def test_stable_nuclides_cannot_decay_by_beta_or_electron_capture(name):
    x = ame.parse_nuclide(name)
    assert reactions.q_beta_minus_kev(x.z, x.a) < 0
    assert reactions.q_ec_kev(x.z, x.a) < 0
    assert reactions.q_beta_plus_kev(x.z, x.a) < 0


def test_alpha_decay_is_allowed_for_heavy_nuclei_and_forbidden_for_light_stable_ones():
    assert reactions.q_alpha_kev(92, 238) > 0
    assert reactions.q_alpha_kev(83, 209) > 0  # Bi-209 is (very weakly) alpha unstable
    assert reactions.q_alpha_kev(26, 56) < 0
    assert reactions.q_alpha_kev(20, 40) < 0


def test_deuteron_separation_energy_is_its_binding_energy():
    assert math.isclose(reactions.neutron_separation_energy_kev(1, 2), ame.binding_energy_mev(1, 2) * 1e3, abs_tol=1e-6)
    assert math.isclose(reactions.proton_separation_energy_kev(1, 2), ame.binding_energy_mev(1, 2) * 1e3, abs_tol=1e-6)


def test_separation_energy_is_a_difference_of_total_binding_energies():
    for z, a in ((26, 56), (82, 208), (4, 9), (8, 17)):
        expected = (ame.binding_energy_mev(z, a) - ame.binding_energy_mev(z, a - 1)) * 1e3
        assert math.isclose(reactions.neutron_separation_energy_kev(z, a), expected, abs_tol=1e-6)


def test_beryllium_9_neutron_separation_energy_is_the_photoneutron_threshold():
    assert abs(reactions.neutron_separation_energy_kev(4, 9) - 1665.0) < 1.0


def test_q_uncertainty_is_the_quadrature_sum_of_the_mass_excess_uncertainties():
    z, a = 27, 60
    expected = math.hypot(ame.get(27, 60).mass_excess_unc_kev, ame.get(28, 60).mass_excess_unc_kev)
    assert math.isclose(reactions.q_uncertainty_kev("beta-minus", z, a), expected)
    with pytest.raises(ValueError):
        reactions.q_uncertainty_kev("gamma", z, a)


def test_q_value_of_a_decay_chain_adds_up():
    """Q(A -> C) = Q(A -> B) + Q(B -> C): alpha then beta- from Ra-226 to ... checked on beta- twice."""
    total = reactions.q_beta_minus_kev(82, 214) + reactions.q_beta_minus_kev(83, 214)  # Pb-214 -> Bi-214 -> Po-214
    direct = ame.get(82, 214).mass_excess_kev - ame.get(84, 214).mass_excess_kev
    assert math.isclose(total, direct, abs_tol=1e-6)
