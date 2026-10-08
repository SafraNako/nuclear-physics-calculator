from __future__ import annotations

import math

import numpy as np

from nukleer_fizik_hesaplayicisi import ame, semf


def test_pairing_sign_by_nucleon_parity():
    assert semf.pairing_sign(30, 26) == 1.0  # even-even
    assert semf.pairing_sign(31, 27) == -1.0  # odd-odd
    assert semf.pairing_sign(30, 27) == 0.0  # odd A
    assert semf.pairing_sign(31, 26) == 0.0


def test_formula_matches_a_hand_computation_for_iron_56():
    c = semf.DEFAULT
    z, a = 26, 56
    n = a - z
    by_hand = (
        c.a_v * a
        - c.a_s * a ** (2 / 3)
        - c.a_c * z * (z - 1) / a ** (1 / 3)
        - c.a_a * (n - z) ** 2 / a
        + c.a_p / math.sqrt(a)  # even-even
    )
    assert math.isclose(semf.semf_binding_energy_mev(z, a), by_hand, rel_tol=1e-12)


def test_scalar_input_gives_a_float_and_array_input_an_array():
    assert isinstance(semf.semf_binding_energy_mev(26, 56), float)
    out = semf.semf_binding_energy_mev(np.array([26, 28]), np.array([56, 62]))
    assert out.shape == (2,)


def test_odd_odd_nucleus_has_the_pairing_term_subtracted_even_even_added():
    c = semf.DEFAULT
    z, a = 27, 60  # odd-odd Co-60
    no_pair = semf.SemfCoefficients(c.a_v, c.a_s, c.a_c, c.a_a, 0.0)
    assert math.isclose(
        semf.semf_binding_energy_mev(z, a, c) - semf.semf_binding_energy_mev(z, a, no_pair), -c.a_p / math.sqrt(a), rel_tol=1e-9
    )


def test_fit_recovers_known_coefficients_exactly_from_noise_free_data():
    truth = semf.SemfCoefficients(15.5, 17.0, 0.70, 23.0, 12.5)
    nuclides = ame.all_nuclides(experimental_only=True)
    z = np.array([x.z for x in nuclides if x.a >= 16])
    a = np.array([x.a for x in nuclides if x.a >= 16])
    fitted = semf.fit_semf(z, a, semf.semf_binding_energy_mev(z, a, truth))
    assert np.allclose(fitted.as_array(), truth.as_array(), rtol=1e-9)


def test_fit_recovers_coefficients_from_noisy_data_within_expected_scatter():
    truth = semf.SemfCoefficients(15.5, 17.0, 0.70, 23.0, 12.5)
    nuclides = ame.all_nuclides(experimental_only=True)
    z = np.array([x.z for x in nuclides if x.a >= 16])
    a = np.array([x.a for x in nuclides if x.a >= 16])
    noisy = semf.semf_binding_energy_mev(z, a, truth) + np.random.default_rng(0).normal(0.0, 1.0, len(z))
    fitted = semf.fit_semf(z, a, noisy)
    assert np.allclose(fitted.as_array(), truth.as_array(), rtol=0.03)


def test_valley_of_stability_formula_hits_the_stable_charges():
    for a, z_stable in ((56, 26), (120, 50), (208, 82), (238, 92)):
        assert abs(float(semf.valley_z(a)) - z_stable) < 1.0


def test_stored_default_coefficients_are_the_fit_to_the_near_valley_data():
    z, a, b = semf.experimental_dataset(min_a=16, valley_width=3.0)
    assert len(z) == 1367
    assert np.allclose(semf.fit_semf(z, a, b).as_array(), semf.DEFAULT.as_array(), atol=1.5e-3)


def test_fit_to_valley_data_reproduces_published_coefficients():
    """An independent least-squares fit to AME2020 lands on the literature values:
    a_V 15.83 / 15.8, a_S 18.25 / 18.3, a_C 0.721 / 0.714, a_A 23.40 / 23.2, and a_P 11.17,
    which is Rohlf's 11.18 (the other literature fit quotes 12)."""
    z, a, b = semf.experimental_dataset(min_a=16, valley_width=3.0)
    c = semf.fit_semf(z, a, b)
    lit = semf.LITERATURE_FIT
    for fitted, quoted in ((c.a_v, lit.a_v), (c.a_s, lit.a_s), (c.a_c, lit.a_c), (c.a_a, lit.a_a)):
        assert abs(fitted / quoted - 1.0) < 0.03
    assert abs(c.a_p - semf.ROHLF.a_p) < 0.1
    assert abs(c.a_p / lit.a_p - 1.0) < 0.09


def test_fitting_the_whole_chart_instead_pulls_the_coefficients_away_from_the_textbook_values():
    z, a, b = semf.experimental_dataset(min_a=16)
    c = semf.fit_semf(z, a, b)
    assert c.a_s < semf.LITERATURE_FIT.a_s * 0.95
    assert len(z) == 2484


def test_formula_is_accurate_near_the_valley_but_not_to_the_experimental_precision():
    z, a, b = semf.experimental_dataset(min_a=16, valley_width=3.0)
    resid = b - semf.semf_binding_energy_mev(z, a)
    assert np.sqrt(np.mean(resid**2)) < 3.0  # MeV
    assert np.median(np.abs(resid) / b) < 0.005
    assert np.sqrt(np.mean(resid**2)) > 1.0  # shell effects: nowhere near the keV precision of the masses


def test_the_formula_misses_the_shell_closures():
    """Residual (experimental - SEMF) is large and positive at doubly magic nuclei
    (Sn-100 +15, Sn-132 +14, Pb-208 +10 MeV) and ~0 elsewhere."""
    z, a, b = semf.experimental_dataset(min_a=16)
    n = a - z
    resid = b - semf.semf_binding_energy_mev(z, a)
    doubly = np.array([semf.is_doubly_magic(zi, ni) and ai >= 40 for zi, ni, ai in zip(z, n, a)])
    near_magic = np.array([(int(zi) in semf.MAGIC_NUMBERS) or (int(ni) in semf.MAGIC_NUMBERS) for zi, ni in zip(z, n)])
    assert doubly.sum() >= 5
    assert resid[doubly].mean() > 5.0
    assert resid[doubly].mean() > resid[~near_magic].mean() + 10 * resid[~near_magic].std(ddof=1) / math.sqrt((~near_magic).sum())
    for name in ("Sn-100", "Sn-132", "Pb-208"):
        x = ame.parse_nuclide(name)
        assert ame.binding_energy_mev(x.z, x.a) - semf.semf_binding_energy_mev(x.z, x.a) > 8.0


def test_binding_energy_per_nucleon_saturates_near_8_mev_for_medium_mass_nuclei():
    for z, a in ((26, 56), (28, 62), (42, 98), (50, 120)):
        assert abs(semf.semf_binding_energy_mev(z, a) / a - ame.binding_energy_mev(z, a) / a) / (ame.binding_energy_mev(z, a) / a) < 0.01


def test_doubly_magic_helper():
    assert semf.is_doubly_magic(82, 126)
    assert semf.is_doubly_magic(20, 28)
    assert not semf.is_doubly_magic(26, 30)
