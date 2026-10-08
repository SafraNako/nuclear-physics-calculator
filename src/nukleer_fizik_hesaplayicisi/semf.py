"""Semi-empirical (Bethe-Weizsacker) mass formula.

    B(Z, A) = a_V*A - a_S*A^(2/3) - a_C*Z(Z-1)/A^(1/3) - a_A*(N-Z)^2/A + delta

with the pairing term delta = +a_P*A^k_P for even-even, -a_P*A^k_P for odd-odd and
0 for odd A (k_P = -1/2 here; -3/4 is the other common choice). The formula has
no shell structure: its residuals against AME2020 are what shows the magic
numbers.

fit_semf() is an ordinary least-squares fit of the five coefficients (the model
is linear in them). The *default* coefficients below come from fitting the 1367
experimental AME2020 nuclides with A >= 16 that lie within three charge units of
the valley of beta stability (|Z - Z0(A)| <= 3); fitting the whole chart, with its
neutron-rich and proton-rich extremes, pulls the surface and volume coefficients
away from the usual textbook values.

Literature values (least-squares fits quoted in the review literature, e.g.
Wikipedia's "Semi-empirical mass formula" table, k_P = -1/2): a_V = 15.8,
a_S = 18.3, a_C = 0.714, a_A = 23.2, a_P = 12 MeV; Rohlf: 15.75, 17.8, 0.711,
23.7, 11.18 MeV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ame

MAGIC_NUMBERS = (2, 8, 20, 28, 50, 82, 126)


@dataclass(frozen=True)
class SemfCoefficients:
    a_v: float
    a_s: float
    a_c: float
    a_a: float
    a_p: float
    k_p: float = -0.5

    def as_array(self) -> np.ndarray:
        return np.array([self.a_v, self.a_s, self.a_c, self.a_a, self.a_p])


LITERATURE_FIT = SemfCoefficients(15.8, 18.3, 0.714, 23.2, 12.0)
ROHLF = SemfCoefficients(15.75, 17.8, 0.711, 23.7, 11.18)
DEFAULT = SemfCoefficients(15.827, 18.250, 0.721, 23.397, 11.165)


def pairing_sign(n, z):
    n, z = np.asarray(n), np.asarray(z)
    return np.where((n % 2 == 0) & (z % 2 == 0), 1.0, np.where((n % 2 == 1) & (z % 2 == 1), -1.0, 0.0))


def design_matrix(z, a, k_p: float = -0.5) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    n = a - z
    return np.column_stack(
        [a, -(a ** (2.0 / 3.0)), -z * (z - 1.0) / a ** (1.0 / 3.0), -((n - z) ** 2) / a, pairing_sign(n, z) * a**k_p]
    )


def semf_binding_energy_mev(z, a, coefficients: SemfCoefficients = DEFAULT):
    """B(Z, A) in MeV: a float for scalar input, an array otherwise."""
    result = design_matrix(np.atleast_1d(z), np.atleast_1d(a), coefficients.k_p) @ coefficients.as_array()
    return float(result[0]) if np.ndim(z) == 0 and np.ndim(a) == 0 else result


def valley_z(a, a_c: float = 0.711, a_a: float = 23.7):
    """Most bound charge for given A: minimizing -B over Z gives Z0 = A / (2 + a_C A^(2/3) / (2 a_A))."""
    a = np.asarray(a, dtype=np.float64)
    return a / (2.0 + (a_c / (2.0 * a_a)) * a ** (2.0 / 3.0))


def fit_semf(z, a, binding_mev, k_p: float = -0.5) -> SemfCoefficients:
    x = design_matrix(z, a, k_p)
    coef, *_ = np.linalg.lstsq(x, np.asarray(binding_mev, dtype=np.float64), rcond=None)
    return SemfCoefficients(*(float(c) for c in coef), k_p=k_p)


def experimental_dataset(min_a: int = 16, valley_width: float | None = None):
    """(z, a, binding_mev) arrays of experimental AME2020 nuclides, optionally near the valley."""
    nuclides = [x for x in ame.all_nuclides(experimental_only=True) if x.a >= min_a]
    z = np.array([x.z for x in nuclides])
    a = np.array([x.a for x in nuclides])
    binding = np.array([ame.binding_energy_mev(x.z, x.a) for x in nuclides])
    if valley_width is not None:
        keep = np.abs(z - valley_z(a)) <= valley_width
        z, a, binding = z[keep], a[keep], binding[keep]
    return z, a, binding


def is_doubly_magic(z, n) -> bool:
    return int(z) in MAGIC_NUMBERS and int(n) in MAGIC_NUMBERS
