from __future__ import annotations

import math

import pytest

from nukleer_fizik_hesaplayicisi import ddep
from nukleer_fizik_hesaplayicisi.constants import TIME_UNITS_S, to_seconds


def test_lookup_is_case_and_space_insensitive():
    assert ddep.lookup("Co-60") is ddep.DDEP["co-60"]
    assert ddep.lookup(" TC-99m ").nuclide == "Tc-99m"


def test_unknown_nuclide_is_rejected():
    with pytest.raises(ValueError, match="DDEP tablosunda yok"):
        ddep.lookup("u-235")


def test_units_are_known_and_uncertainties_positive():
    for entry in ddep.DDEP.values():
        assert entry.unit in TIME_UNITS_S
        assert entry.half_life > 0 and entry.half_life_unc > 0
        assert (entry.q_kev is None) == (entry.q_kind is None)


def test_molybdenum_99_half_life_in_hours_matches_the_equilibrium_statement_t_over_60_h():
    hours = to_seconds(ddep.DDEP["mo-99"].half_life, "gun") / 3600.0
    assert math.isclose(hours, 65.9496, rel_tol=1e-9)
    assert hours > 60.0


def test_ddep_equilibrium_numbers_are_internally_consistent():
    """DDEP's own p x factor = ratio: 0.876 x 1.1005 = 0.964 against the stated 0.963(21)."""
    product = ddep.MO99_TO_TC99M_BRANCHING * ddep.MO99_TC99M_EQUILIBRIUM_FACTOR
    assert abs(product - ddep.MO99_TC99M_EQUILIBRIUM_RATIO) < ddep.MO99_TC99M_EQUILIBRIUM_RATIO_UNC
