# nuclear-physics-calculator

A nuclear physics calculator: radioactive decay and decay chains (including the
Mo-99/Tc-99m generator), binding energies and Q-values from the AME2020 atomic
mass evaluation, a semi-empirical mass formula fitted to AME2020, specific
activity, and neutron activation. CLI and internal package names are in Turkish
(`nukleer-fizik-hesaplayicisi` / `nukleer_fizik_hesaplayicisi`), consistent with
the other tools in this project series; code and this document are in English.

The calculator part is small. What the project is really about is that every
number it prints is checked against something independent: a second solver, a
high-precision reference, a closed form, or published evaluated data.

> Educational and reference use. Not a substitute for evaluated nuclear data
> files in safety-relevant work.

## Data and provenance

- **Atomic masses: AME2020.** W. J. Huang, M. Wang, F. G. Kondev, G. Audi,
  S. Naimi, *The AME 2020 atomic mass evaluation (I)*, Chinese Physics C 45,
  030002 (2021); M. Wang et al., *(II)*, Chinese Physics C 45, 030003 (2021). The
  papers are open access (CC BY 3.0); the table is distributed by the IAEA Atomic
  Mass Data Center. The CSV in
  [`data/`](src/nukleer_fizik_hesaplayicisi/data/ame2020.csv) is derived from that
  table with [`tools/build_ame_csv.py`](tools/build_ame_csv.py) and is
  redistributed here with attribution. It has 3558 nuclides; the 1008 values AME
  marks as *estimated* (not experimental) are flagged and excluded from every
  comparison with measurements.
- **Half-lives and Q-values: LNHB/DDEP** ("Table de Radionucleides") for Co-60,
  Cs-137, Na-22, Am-241, Ba-133, Mo-99 and Tc-99m, including DDEP's own
  statement of the Mo-99/Tc-99m equilibrium. These serve as the independent
  reference for the Q-values computed from masses.
- **SEMF literature coefficients:** the least-squares fits and Rohlf's values
  quoted in Wikipedia's "Semi-empirical mass formula" table.

## What is checked, and against what

### Binding energies and Q-values (AME2020)

- **Parsing and the binding-energy formula.** `B = Z*D(1H) + N*D(n) - D(Z,A)`
  from mass excesses `D` reproduces the binding energy per nucleon *published in
  the same file* to `1.1e-4` keV for all 2548 experimental nuclides with
  `A >= 2`. (The file's fixed-width layout is not followed exactly, so the columns
  are tokenized around the `B-` label; this check is what shows the parse is right.)
  Carbon-12 has mass excess exactly 0 (it defines the atomic mass unit), and
  `M = A + D/u` holds to `1.5e-9` u.
- **Known facts reproduced.** The most tightly bound nuclei per nucleon are
  Ni-62 (8.7946 MeV), then Fe-58 (8.7923) and Fe-56 (8.7904): Ni-62, not Fe-56, has
  the highest binding energy per nucleon. The deuteron binds with 2224.566 keV, which is also its neutron
  separation energy; `S_n(Be-9) = 1664.54` keV.
- **Q-values against an independent evaluation.** Computed from AME2020 mass
  differences versus the LNHB/DDEP table:

  | decay | AME2020 (keV) | DDEP (keV) | difference |
  |---|---|---|---|
  | Co-60 beta- | 2822.81 | 2823.07(21) | -0.26 |
  | Cs-137 beta- | 1175.63 | 1175.63(17) | 0.00 |
  | Mo-99 beta- | 1357.76 | 1357.2(10) | +0.56 |
  | Na-22 EC | 2843.33 | 2843.02(21) | +0.31 |
  | Ba-133 EC | 517.43 | 517.3(10) | +0.13 |
  | Am-241 alpha | 5637.82 | 5637.82(12) | 0.00 |

  All within 1 keV and 2.5 sigma. For Na-22, DDEP also lists the beta+ transition
  to the ground state at 1821.02(21) keV and to the 1274.5 keV level at 546.44(21)
  keV; `Q_beta+ = Q_EC - 2*m_e*c^2` from AME2020 gives 1821.33 and 546.79 keV.
  Textbook values are reproduced too (H-3 18.59, C-14 156.48, K-40 EC 1504.40,
  Po-210 alpha 5407.53, Ra-226 alpha 4870.70 keV).

### Decay chains ([`bateman.py`](src/nukleer_fizik_hesaplayicisi/bateman.py))

The chain `dN/dt = M N` is solved with the matrix exponential, which is exact
for any decay constants (equal ones, a stable end member, branching). It is
checked against the textbook Bateman formula and against the same formula in
80-digit decimal arithmetic.

- Well-separated chains agree with the reference to `~1e-16` (matrix exponential)
  and mostly to better than `1e-12` (float Bateman).
- **The float Bateman formula loses precision in two regimes**, both from
  cancellation of large terms. Near-equal decay constants: relative error `9e-14`,
  `7e-11`, `1.2e-7`, `1.1e-4` for relative separations `1e-3`, `1e-6`, `1e-9`,
  `1e-12`, while the matrix exponential stays at `~1e-16`. And deep chain members at
  early times: `3.5e-11` for the fourth member of a 4-chain at `t = 1`.
- Equal half-lives (where Bateman divides by zero): `N2 = lambda*t*exp(-lambda*t)`,
  reproduced exactly.

**Mo-99/Tc-99m generator.** `t_max = ln(l2/l1)/(l2 - l1) = 22.844` h (the familiar
~23 h peak); the equilibrium factor `T1/(T1 - T2)` is `1.10024` against DDEP's
`1.1005(8)`, and with the branching `p = 0.876(19)` the equilibrium activity ratio
is `0.9638` against DDEP's `0.963(21)`.

### Semi-empirical mass formula ([`semf.py`](src/nukleer_fizik_hesaplayicisi/semf.py))

Fitted by least squares to experimental AME2020 nuclides with `A >= 16`
(coefficients in MeV, pairing exponent -1/2):

| fit | n | a_V | a_S | a_C | a_A | a_P | rms |
|---|---|---|---|---|---|---|---|
| near the valley of stability (abs(Z-Z0) <= 3) | 1367 | 15.827 | 18.250 | 0.7214 | 23.397 | 11.165 | 2.77 MeV |
| whole experimental table | 2484 | 15.414 | 16.860 | 0.6952 | 22.497 | 12.028 | 3.31 MeV |
| literature (least squares) | | 15.8 | 18.3 | 0.714 | 23.2 | 12 | |
| Rohlf | | 15.75 | 17.8 | 0.711 | 23.7 | 11.18 | |

The valley fit lands on the literature values to within about 1% (a_P equals
Rohlf's 11.18); fitting the whole chart, with its exotic nuclei, pulls the
surface and volume terms away from them, which is why the near-valley fit is the
default. The fit routine itself is checked by recovering known coefficients
exactly from noise-free synthetic data.

**What the formula misses is the physics.** The residual (experimental - SEMF) at
doubly magic nuclei is `+15.1` (Sn-100), `+14.3` (Sn-132), `+10.3` (Pb-208),
`+5.8` (Ni-56), `+3.5` (O-16), `+2.3` (Ca-48) and `+0.8` MeV (Ca-40): a mean of
`+7.4` MeV, against `+0.6 +- 0.07` MeV for nuclei with no magic number. Plotted
over (N, Z), the shell closures at 50, 82 and 126 show up as bands of extra
binding (`semf-uydur --grafik`).

### Specific activity and activation

Co-60: `4.187e13` Bq/g = `1131.6` Ci/g (textbook ~1130). Ra-226 (`T1/2 = 1600` a):
`3.66e10` Bq/g, `0.989` Ci/g (the curie was originally the activity of 1 g of Ra-226).
Neutron activation `A = N*sigma*phi*(1 - exp(-lambda t_irr))*exp(-lambda t_cool)` agrees
with an independent integration of the augmented linear system to `~1e-16`;
50%, 75% and 90% of saturation are reached after 1, 2 and 3.32 half-lives.

## Things found along the way

- The AME file's fixed-width format description does not match the file. Slicing by
  column shifted values into neighbouring fields; the internal binding-energy
  consistency check above is what makes the tokenized parse trustworthy.
- I first assumed the float Bateman formula was only a problem for nearly equal
  decay constants. A test failure showed it also loses precision for deep chain
  members at early times; the tolerance and the documentation now say so.
- Reading the `aktivasyon` code before its first run showed a saturation-activity
  expression that evaluated `inf * 0` (NaN); it is now computed directly as `N*sigma*phi`.
- A recalled K-40 beta- Q-value (1311.07 keV) was 0.17 keV off the AME2020 value
  (1310.90 keV), so that textbook test uses a 0.5 keV tolerance.

## Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

nukleer-fizik-hesaplayicisi bozunma --nuklid co-60 --sure 10 --sure-birimi yil --aktivite 3.7e10
nukleer-fizik-hesaplayicisi ozgul-aktivite --nuklid co-60
nukleer-fizik-hesaplayicisi jenerator --grafik jenerator.png
nukleer-fizik-hesaplayicisi zincir --yari-omurler 65.9496,6.0067,0 --sure 100 --dallanma 0.876,1.0
nukleer-fizik-hesaplayicisi baglanma-enerjisi --nuklid fe-56
nukleer-fizik-hesaplayicisi semf-uydur --grafik semf.png
nukleer-fizik-hesaplayicisi q-degeri --nuklid co-60 --tur beta-minus
nukleer-fizik-hesaplayicisi aktivasyon --nuklid mo-99 --atom-sayisi 1e20 --kesit-barn 0.13 --aki 1e14 --isinma-suresi 100
```

Each subcommand prints the cross-check next to the result (DDEP comparison, three-way
chain comparison, independent activation integration), so correctness is visible on
any machine.

## Testing

```bash
pytest -v
```

133 tests in about 6 seconds, no mocking. The one stochastic test uses a fixed
seed. Pure numerical Python (numpy, scipy, matplotlib), no Linux/Docker dependency.

## Scope and limitations

- Atomic masses are used as given: electron binding energies are neglected. Q-values
  are to the daughter ground state.
- Decay chains are linear (no cycles); only the DDEP data tabulated in
  [`ddep.py`](src/nukleer_fizik_hesaplayicisi/ddep.py) is built in.
- The mass formula has no shell structure by construction.
- Activation neglects burn-up of the target and of the product by further neutron
  capture, and a single product is considered.
- Not a source of evaluated decay data, emission probabilities or cross sections.

## License

MIT -- see [LICENSE](LICENSE). The AME2020 data keeps its own terms (CC BY 3.0).
