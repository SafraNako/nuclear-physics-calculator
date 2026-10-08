"""AME2020 atomic masses (mass excess, binding energy) for 3558 nuclides.

Source: Huang, Wang, Kondev, Audi, Naimi, Chin. Phys. C 45, 030002 (2021) and
Wang et al., Chin. Phys. C 45, 030003 (2021); table from the IAEA AMDC. The
packaged CSV is built by tools/build_ame_csv.py. Values AME marks as estimated
(not experimental) are flagged `estimated` and excluded from every analysis
here that claims to compare with measurements.

Total binding energy from atomic masses (electron binding energy neglected):

    B(Z, N) = Z*Delta(1H) + N*Delta(n) - Delta(Z, A)       [mass excesses]

because the nucleon-number terms of the masses cancel. The tests check this
against the binding energy per nucleon published in the same file.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Nuclide:
    n: int
    z: int
    a: int
    symbol: str
    mass_excess_kev: float
    mass_excess_unc_kev: float
    binding_per_a_kev: float
    atomic_mass_micro_u: float
    estimated: bool

    @property
    def name(self) -> str:
        return f"{self.symbol}-{self.a}"

    @property
    def atomic_mass_u(self) -> float:
        return self.atomic_mass_micro_u * 1e-6


@lru_cache(maxsize=1)
def _table() -> dict:
    path = Path(__file__).parent / "data" / "ame2020.csv"
    table = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            nuclide = Nuclide(
                n=int(row["n"]),
                z=int(row["z"]),
                a=int(row["a"]),
                symbol=row["symbol"],
                mass_excess_kev=float(row["mass_excess_kev"]),
                mass_excess_unc_kev=float(row["mass_excess_unc_kev"]),
                binding_per_a_kev=float(row["binding_per_a_kev"]),
                atomic_mass_micro_u=float(row["atomic_mass_micro_u"]),
                estimated=bool(int(row["estimated"])),
            )
            table[(nuclide.z, nuclide.a)] = nuclide
    return table


def all_nuclides(experimental_only: bool = False) -> list:
    nuclides = list(_table().values())
    return [x for x in nuclides if not x.estimated] if experimental_only else nuclides


def get(z: int, a: int) -> Nuclide:
    try:
        return _table()[(z, a)]
    except KeyError:
        raise ValueError(f"AME2020'de (Z={z}, A={a}) nüklidi yok") from None


@lru_cache(maxsize=1)
def _symbol_index() -> dict:
    return {(x.symbol.lower(), x.a): x for x in _table().values()}


_NAME = re.compile(r"^\s*(?:(\d+)\s*-?\s*([A-Za-z]{1,2})|([A-Za-z]{1,2})\s*-?\s*(\d+))\s*$")


def parse_nuclide(text: str) -> Nuclide:
    """Accepts 'Co-60', 'co60', '60Co', '60-Co'."""
    match = _NAME.match(text)
    if not match:
        raise ValueError(f"nüklid adı anlaşılamadı: {text!r} (örnek: Co-60)")
    a_text, sym1, sym2, a_text2 = match.groups()
    symbol = (sym1 or sym2).lower()
    mass_number = int(a_text or a_text2)
    try:
        return _symbol_index()[(symbol, mass_number)]
    except KeyError:
        raise ValueError(f"AME2020'de {text!r} nüklidi yok") from None


def binding_energy_mev(z: int, a: int) -> float:
    """Total binding energy from mass excesses (see module docstring)."""
    nuclide = get(z, a)
    hydrogen, neutron = get(1, 1), get(0, 1)
    return (z * hydrogen.mass_excess_kev + (a - z) * neutron.mass_excess_kev - nuclide.mass_excess_kev) / 1e3


def mass_excess_kev(z: int, a: int) -> float:
    return get(z, a).mass_excess_kev
