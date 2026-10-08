"""Build src/nukleer_fizik_hesaplayicisi/data/ame2020.csv from the AME2020 mass table.

Usage:
    curl -O https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt
    python tools/build_ame_csv.py mass_1.mas20.txt

Source: W.J. Huang, M. Wang, F.G. Kondev, G. Audi, S. Naimi, "The AME 2020 atomic
mass evaluation (I)", Chinese Physics C 45, 030002 (2021); M. Wang et al., part (II),
Chinese Physics C 45, 030003 (2021). Table distributed by the IAEA Atomic Mass Data
Center (AMDC).

The fixed-width layout in the file header is not followed exactly (columns shift
with the number of decimals), so the three leading integers are sliced and the rest
is tokenized around the 'B-' label. Values containing '#' are AME's estimated
(non-experimental) values and are flagged, not dropped.
"""

import csv
import math
import sys
from pathlib import Path


def _float(token: str) -> float:
    if token == "*":
        return math.nan
    return float(token.replace("#", "."))


def parse(path: str):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines()[36:]:
        if not line.strip():
            continue
        n, z, a = int(line[4:9]), int(line[9:14]), int(line[14:19])
        tokens = line[19:].split()
        label = tokens.index("B-")
        element = tokens[0]
        mass, mass_unc, binding, _binding_unc = tokens[label - 4 : label]
        am_int, am_frac, _am_unc = tokens[-3:]
        rows.append(
            {
                "n": n,
                "z": z,
                "a": a,
                "symbol": element,
                "mass_excess_kev": _float(mass),
                "mass_excess_unc_kev": _float(mass_unc),
                "binding_per_a_kev": _float(binding),
                "atomic_mass_micro_u": int(am_int) * 1_000_000 + _float(am_frac),
                "estimated": int("#" in mass),
            }
        )
    return rows


def main(argv):
    rows = parse(argv[1])
    out = Path(__file__).resolve().parent.parent / "src" / "nukleer_fizik_hesaplayicisi" / "data" / "ame2020.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows written to {out}")


if __name__ == "__main__":
    main(sys.argv)
