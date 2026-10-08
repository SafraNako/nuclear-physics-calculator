from __future__ import annotations

import math

import numpy as np
import pytest

from nukleer_fizik_hesaplayicisi import ame
from nukleer_fizik_hesaplayicisi.constants import U_KEV


def test_table_size_and_estimated_flag():
    assert len(ame.all_nuclides()) == 3558
    assert len(ame.all_nuclides(experimental_only=True)) == 3558 - 1008


def test_carbon_12_defines_the_atomic_mass_unit():
    c12 = ame.get(6, 12)
    assert c12.mass_excess_kev == 0.0
    assert c12.atomic_mass_u == 12.0


def test_neutron_and_hydrogen_mass_excesses():
    assert math.isclose(ame.get(0, 1).mass_excess_kev, 8071.31806, abs_tol=1e-5)
    assert math.isclose(ame.get(1, 1).mass_excess_kev, 7288.971064, abs_tol=1e-6)


def test_atomic_masses_are_consistent_with_mass_excesses():
    """M = A*u + Delta, i.e. M/u = A + Delta/(u c^2), for every experimental nuclide."""
    worst = max(abs(x.atomic_mass_u - (x.a + x.mass_excess_kev / U_KEV)) for x in ame.all_nuclides(experimental_only=True))
    assert worst < 3e-9


def test_binding_energy_from_mass_excess_matches_the_published_binding_per_nucleon():
    """Independent check of the parsing and of B = Z*D(1H) + N*D(n) - D(Z,A): the file lists B/A separately."""
    diffs = [
        abs(ame.binding_energy_mev(x.z, x.a) * 1e3 / x.a - x.binding_per_a_kev)
        for x in ame.all_nuclides(experimental_only=True)
        if x.a >= 2
    ]
    assert len(diffs) > 2500
    assert max(diffs) < 1e-3  # keV per nucleon


@pytest.mark.parametrize(
    "name,total_mev",
    [("H-2", 2.2246), ("He-4", 28.296), ("C-12", 92.162), ("Fe-56", 492.26)],
)
def test_well_known_total_binding_energies(name, total_mev):
    x = ame.parse_nuclide(name)
    assert math.isclose(ame.binding_energy_mev(x.z, x.a), total_mev, abs_tol=2e-3)


def test_nickel_62_is_the_most_tightly_bound_nucleus_per_nucleon_with_iron_behind():
    ranked = sorted(ame.all_nuclides(experimental_only=True), key=lambda x: -x.binding_per_a_kev)
    assert [x.name for x in ranked[:3]] == ["Ni-62", "Fe-58", "Fe-56"]
    assert math.isclose(ranked[0].binding_per_a_kev / 1e3, 8.7946, abs_tol=1e-4)


def test_estimated_values_are_flagged_and_far_less_certain():
    estimated = [x.mass_excess_unc_kev for x in ame.all_nuclides() if x.estimated]
    measured = [x.mass_excess_unc_kev for x in ame.all_nuclides(experimental_only=True)]
    assert np.median(estimated) > 20 * np.median(measured)


@pytest.mark.parametrize("text", ["Co-60", "co60", "60Co", "60-Co", " CO-60 "])
def test_nuclide_names_are_parsed_in_several_spellings(text):
    x = ame.parse_nuclide(text)
    assert (x.z, x.a, x.name) == (27, 60, "Co-60")


@pytest.mark.parametrize("text", ["", "Xx-99", "Co", "60", "Co-1000"])
def test_bad_or_missing_nuclides_are_rejected(text):
    with pytest.raises(ValueError):
        ame.parse_nuclide(text)


def test_lookup_by_charge_and_mass_number_rejects_unknown():
    with pytest.raises(ValueError):
        ame.get(200, 400)
