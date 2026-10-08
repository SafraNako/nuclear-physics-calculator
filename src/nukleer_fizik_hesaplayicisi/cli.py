from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from . import ame, ddep, reactions, semf
from .activation import (
    activation_activity_bq,
    activation_activity_numeric_bq,
    production_rate_per_s,
    saturation_fraction,
    time_to_saturation_fraction,
)
from .bateman import (
    bateman_closed_form,
    bateman_reference,
    chain_activities,
    chain_populations,
    time_of_maximum_daughter_activity,
    transient_equilibrium_ratio,
)
from .constants import TIME_UNITS_S, to_seconds
from .decay import (
    activity_bq,
    atoms_from_activity,
    bq_to_curie,
    decay_constant,
    fraction_remaining,
    mean_life,
    specific_activity_bq_per_g,
)
from .render import plot_binding_energy, plot_chain

UNITS = list(TIME_UNITS_S)


def _half_life_s(args) -> tuple:
    """(half_life_s, label) from --nuklid (DDEP) or --yari-omur/--birim."""
    if getattr(args, "nuklid", None):
        entry = ddep.lookup(args.nuklid)
        return to_seconds(entry.half_life, entry.unit), f"{entry.nuclide} (DDEP: {entry.half_life} ± {entry.half_life_unc} {entry.unit})"
    if args.yari_omur is None:
        raise ValueError("--nuklid ya da --yari-omur verilmeli")
    return to_seconds(args.yari_omur, args.birim), f"yarı ömür {args.yari_omur} {args.birim}"


def _positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} pozitif olmalı")


def _cmd_bozunma(args: argparse.Namespace) -> int:
    _positive(aktivite=args.aktivite)
    t_half, label = _half_life_s(args)
    t = to_seconds(args.sure, args.sure_birimi)
    lam = decay_constant(t_half)
    print(label)
    print(f"Bozunma sabiti λ = ln2/T½ = {lam:.6e} 1/s;  ortalama ömür τ = {mean_life(t_half):.6e} s")
    print(f"t = {args.sure} {args.sure_birimi}: kalan oran N/N₀ = {fraction_remaining(t, t_half):.6g}  ({t / t_half:.4g} yarı ömür)")
    a = activity_bq(args.aktivite, t, t_half)
    print(f"Aktivite: {args.aktivite:.6g} Bq -> {a:.6g} Bq ({bq_to_curie(a):.4g} Ci);  atom sayısı N₀ = {atoms_from_activity(args.aktivite, t_half):.6g}")
    return 0


def _cmd_ozgul_aktivite(args: argparse.Namespace) -> int:
    t_half, label = _half_life_s(args)
    if args.nuklid:
        nuclide = ame.parse_nuclide(args.nuklid)
        molar = nuclide.atomic_mass_u
        source = f"AME2020 atomik kütlesi {molar:.6f} u"
    elif args.molar_kutle:
        molar, source = args.molar_kutle, "verilen molar kütle"
    else:
        raise ValueError("--nuklid ya da --molar-kutle verilmeli")
    spec = specific_activity_bq_per_g(t_half, molar)
    print(label)
    print(f"Özgül aktivite = λ·N_A/M = {spec:.6e} Bq/g = {bq_to_curie(spec):.6g} Ci/g  (M: {source})")
    return 0


def _cmd_jenerator(args: argparse.Namespace) -> int:
    _positive(sure_saat=args.sure_saat)
    t1 = to_seconds(ddep.DDEP["mo-99"].half_life, "gun")
    t2 = to_seconds(ddep.DDEP["tc-99m"].half_life, "saat")
    b = ddep.MO99_TO_TC99M_BRANCHING
    hours = np.linspace(0.0, args.sure_saat, 400)
    activities = chain_activities([t1, t2], hours * 3600.0, branching=[b])
    t_max_h = time_of_maximum_daughter_activity(t1, t2) / 3600.0
    ratio = transient_equilibrium_ratio(t1, t2, b)
    print(f"Mo-99 T½ = {t1 / 3600:.4f} saat, Tc-99m T½ = {t2 / 3600:.4f} saat (DDEP), dallanma p = {b}")
    print(f"Tc-99m aktivitesi en yüksek: t_max = ln(λ₂/λ₁)/(λ₂-λ₁) = {t_max_h:.3f} saat (kapalı-form)")
    peak_numeric = hours[int(np.argmax(activities[:, 1]))]
    print(f"  sayısal tarama (matris üstel): {peak_numeric:.2f} saat (ızgara adımı {hours[1]:.2f} saat)")
    print(f"Geçici denge: A(Tc-99m)/A(Mo-99) = p·T₁/(T₁-T₂) = {ratio:.4f}  (DDEP: {ddep.MO99_TC99M_EQUILIBRIUM_RATIO} ± {ddep.MO99_TC99M_EQUILIBRIUM_RATIO_UNC})")
    print(f"{'t (saat)':>9} | {'A(Mo-99)':>9} | {'A(Tc-99m)':>10} | {'oran':>7}")
    for h in (6, 12, 23, 48, 72, 120):
        if h > args.sure_saat:
            continue
        a = chain_activities([t1, t2], [h * 3600.0], branching=[b])[0]
        print(f"{h:>9} | {a[0]:>9.4f} | {a[1]:>10.4f} | {a[1] / a[0]:>7.4f}")
    if args.grafik:
        plot_chain(hours, activities, ["Mo-99", "Tc-99m"], args.grafik, "t (saat)", marker_time=t_max_h)
        print(f"Grafik kaydedildi: {args.grafik}")
    return 0


def _cmd_zincir(args: argparse.Namespace) -> int:
    values = [float(v) for v in args.yari_omurler.split(",")]
    half_lives = [math.inf if v == 0 else to_seconds(v, args.birim) for v in values]
    branching = [float(v) for v in args.dallanma.split(",")] if args.dallanma else None
    t = to_seconds(args.sure, args.sure_birimi)
    populations = chain_populations(half_lives, [t], branching)[0]
    print(f"{len(half_lives)} üyeli zincir, t = {args.sure} {args.sure_birimi} (0 yarı ömür = kararlı)")
    print("Kalan atom oranları N_i(t)/N₁(0) (matris üstel): " + ", ".join(f"{p:.6g}" for p in populations))
    try:
        closed = bateman_closed_form(half_lives, t, branching)
        reference = [float(x) for x in bateman_reference(half_lives, t, branching)]
        print("Bateman kapalı-formuna göre en büyük göreli fark: %.2e" % max(abs(c - p) / max(abs(p), 1e-300) for c, p in zip(closed, populations)))
        print("80 basamaklı referansa göre göreli hata: matris üstel %.2e, kapalı-form %.2e" % (
            max(abs(p - r) / max(abs(r), 1e-300) for p, r in zip(populations, reference)),
            max(abs(c - r) / max(abs(r), 1e-300) for c, r in zip(closed, reference)),
        ))
    except (ZeroDivisionError, ValueError, ArithmeticError) as exc:
        print(f"Bateman kapalı-formu bu zincirde uygulanamıyor ({type(exc).__name__}): eşit/kararlı bozunma sabitleri; matris üstel sonucu geçerli.")
    if args.grafik:
        times = np.linspace(0.0, t, 300)
        plot_chain(times / TIME_UNITS_S[args.sure_birimi], chain_activities(half_lives, times, branching),
                   [f"üye {i + 1}" for i in range(len(half_lives))], args.grafik, f"t ({args.sure_birimi})")
        print(f"Grafik kaydedildi: {args.grafik}")
    return 0


def _cmd_baglanma_enerjisi(args: argparse.Namespace) -> int:
    if args.nuklid:
        x = ame.parse_nuclide(args.nuklid)
        b_ame = ame.binding_energy_mev(x.z, x.a)
        b_semf = semf.semf_binding_energy_mev(x.z, x.a)
        print(f"{x.name} (Z={x.z}, N={x.n}, A={x.a}){'  [AME tahmini değer]' if x.estimated else ''}")
        print(f"AME2020: B = {b_ame:.3f} MeV, B/A = {b_ame / x.a:.4f} MeV;  dosyada yayımlanan B/A = {x.binding_per_a_kev / 1e3:.4f} MeV")
        print(f"SEMF   : B = {b_semf:.3f} MeV, B/A = {b_semf / x.a:.4f} MeV;  artık (AME - SEMF) = {b_ame - b_semf:+.2f} MeV")
        return 0
    top = sorted(ame.all_nuclides(experimental_only=True), key=lambda x: -x.binding_per_a_kev)[:5]
    print("Nükleon başına bağlanma enerjisi en yüksek 5 deneysel nüklid (AME2020):")
    for x in top:
        print(f"  {x.name:>7}: B/A = {x.binding_per_a_kev / 1e3:.4f} MeV")
    return 0


def _cmd_semf_uydur(args: argparse.Namespace) -> int:
    rows = []
    for label, width in (("kararlılık vadisi (|Z-Z₀|≤3)", 3.0), ("tüm deneysel tablo", None)):
        z, a, b = semf.experimental_dataset(min_a=16, valley_width=width)
        c = semf.fit_semf(z, a, b)
        resid = b - semf.semf_binding_energy_mev(z, a, c)
        rows.append((label, len(z), c, float(np.sqrt(np.mean(resid**2)))))
    lit = semf.LITERATURE_FIT
    print("SEMF katsayıları (MeV, k_P = -1/2):")
    print(f"{'':32} {'n':>5} | {'a_V':>7} {'a_S':>7} {'a_C':>7} {'a_A':>7} {'a_P':>7} | rms artık")
    for label, n, c, rms in rows:
        print(f"{label:32} {n:>5} | {c.a_v:7.3f} {c.a_s:7.3f} {c.a_c:7.4f} {c.a_a:7.3f} {c.a_p:7.3f} | {rms:.2f} MeV")
    print(f"{'literatür (en küçük kareler)':32} {'':>5} | {lit.a_v:7.3f} {lit.a_s:7.3f} {lit.a_c:7.4f} {lit.a_a:7.3f} {lit.a_p:7.3f} |")

    z, a, b = semf.experimental_dataset(min_a=16)
    resid = b - semf.semf_binding_energy_mev(z, a)
    n = a - z
    magic = np.array([semf.is_doubly_magic(zi, ni) for zi, ni in zip(z, n)])
    near_magic = np.array([(int(zi) in semf.MAGIC_NUMBERS) or (int(ni) in semf.MAGIC_NUMBERS) for zi, ni in zip(z, n)])
    print("\nÇift sihirli çekirdeklerde artık (deneysel - SEMF): " + ", ".join(f"{ame.get(int(zi), int(ai)).name} {r:+.1f}" for zi, ai, r in zip(z[magic], a[magic], resid[magic])) + " MeV")
    print(f"Ortalama artık: çift sihirli {resid[magic].mean():+.2f} MeV, hiç sihirli sayısı olmayanlar {resid[~near_magic].mean():+.2f} ± {resid[~near_magic].std(ddof=1) / math.sqrt((~near_magic).sum()):.2f} MeV (std {resid[~near_magic].std(ddof=1):.2f})")
    best = int(np.argmax(b / a))
    print(f"En yüksek B/A: {ame.get(int(z[best]), int(a[best])).name} = {b[best] / a[best]:.4f} MeV")
    if args.grafik:
        grid = np.arange(16, 260)
        z0 = np.rint(semf.valley_z(grid)).astype(int)
        plot_binding_energy(a, b / a, grid, semf.semf_binding_energy_mev(z0, grid) / grid, z, n, resid, semf.MAGIC_NUMBERS, args.grafik)
        print(f"Grafik kaydedildi: {args.grafik}")
    return 0


def _cmd_q_degeri(args: argparse.Namespace) -> int:
    x = ame.parse_nuclide(args.nuklid)
    q = reactions.Q_FUNCTIONS[args.tur](x.z, x.a)
    unc = reactions.q_uncertainty_kev(args.tur, x.z, x.a)
    print(f"{x.name} {args.tur}: Q = {q:.2f} ± {unc:.2f} keV (AME2020 kütle farkı){'  [bozunma enerjetik olarak yasak: Q < 0]' if q <= 0 else ''}")
    key = f"{x.symbol.lower()}-{x.a}"
    entry = ddep.DDEP.get(key)
    if entry and entry.q_kind == args.tur:
        diff = q - entry.q_kev
        sigma = math.hypot(unc, entry.q_unc_kev)
        print(f"DDEP değerlendirmesi: {entry.q_kev} ± {entry.q_unc_kev} keV -> fark {diff:+.2f} keV ({diff / sigma:+.2f}σ)")
    return 0


def _cmd_aktivasyon(args: argparse.Namespace) -> int:
    _positive(kesit_barn=args.kesit_barn, aki=args.aki)
    t_half, label = _half_life_s(args)
    if args.atom_sayisi:
        atoms = args.atom_sayisi
    elif args.kutle_g and args.molar_kutle:
        atoms = args.kutle_g / args.molar_kutle * 6.02214076e23
    else:
        raise ValueError("--atom-sayisi ya da (--kutle-g ve --molar-kutle) verilmeli")
    t_irr = to_seconds(args.isinma_suresi, args.isinma_birimi)
    t_cool = to_seconds(args.bekleme_suresi, args.bekleme_birimi)
    a = activation_activity_bq(atoms, args.kesit_barn, args.aki, t_half, t_irr, t_cool)
    a_numeric = activation_activity_numeric_bq(atoms, args.kesit_barn, args.aki, t_half, t_irr, t_cool)
    a_sat = production_rate_per_s(atoms, args.kesit_barn, args.aki)
    print(label)
    print(f"Doyum aktivitesi N·σ·φ = {a_sat:.6g} Bq;  ışınlama sonunda doyum oranı = {saturation_fraction(t_irr, t_half):.4f}")
    print(f"Aktivite (ışınlama {args.isinma_suresi} {args.isinma_birimi}, bekleme {args.bekleme_suresi} {args.bekleme_birimi}): {a:.6g} Bq ({bq_to_curie(a):.4g} Ci)")
    print(f"Bağımsız sayısal kontrol (matris üstel): {a_numeric:.6g} Bq  (göreli fark {abs(a - a_numeric) / max(a, 1e-300):.1e})")
    print(f"Doyumun %50 / %90 / %99'una ulaşma süresi: {time_to_saturation_fraction(0.5, t_half) / TIME_UNITS_S[args.isinma_birimi]:.4g} / {time_to_saturation_fraction(0.9, t_half) / TIME_UNITS_S[args.isinma_birimi]:.4g} / {time_to_saturation_fraction(0.99, t_half) / TIME_UNITS_S[args.isinma_birimi]:.4g} {args.isinma_birimi}")
    return 0


def _add_half_life_arguments(p) -> None:
    p.add_argument("--nuklid", default=None, help=f"DDEP tablosundan yarı ömür ({', '.join(ddep.DDEP)})")
    p.add_argument("--yari-omur", type=float, default=None)
    p.add_argument("--birim", default="saniye", choices=UNITS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nukleer-fizik-hesaplayicisi", description="Nükleer fizik hesaplayıcısı")
    sub = parser.add_subparsers(dest="komut", required=True)

    p = sub.add_parser("bozunma", help="Radyoaktif bozunma yasası")
    _add_half_life_arguments(p)
    p.add_argument("--aktivite", type=float, default=3.7e10, help="başlangıç aktivitesi (Bq)")
    p.add_argument("--sure", type=float, required=True)
    p.add_argument("--sure-birimi", default="gun", choices=UNITS)
    p.set_defaults(func=_cmd_bozunma)

    p = sub.add_parser("ozgul-aktivite", help="Özgül aktivite (Bq/g, Ci/g)")
    _add_half_life_arguments(p)
    p.add_argument("--molar-kutle", type=float, default=None, help="g/mol (--nuklid verilirse AME2020 atomik kütlesi kullanılır)")
    p.set_defaults(func=_cmd_ozgul_aktivite)

    p = sub.add_parser("jenerator", help="Mo-99/Tc-99m jeneratörü (geçici denge)")
    p.add_argument("--sure-saat", type=float, default=120.0)
    p.add_argument("--grafik", default=None)
    p.set_defaults(func=_cmd_jenerator)

    p = sub.add_parser("zincir", help="Genel bozunma zinciri: matris üstel, Bateman, 80 basamaklı referans")
    p.add_argument("--yari-omurler", required=True, help="virgülle ayrılmış; 0 = kararlı")
    p.add_argument("--birim", default="saat", choices=UNITS)
    p.add_argument("--dallanma", default=None, help="üye i'nin üye i+1'i beslediği oranlar, virgülle")
    p.add_argument("--sure", type=float, required=True)
    p.add_argument("--sure-birimi", default="saat", choices=UNITS)
    p.add_argument("--grafik", default=None)
    p.set_defaults(func=_cmd_zincir)

    p = sub.add_parser("baglanma-enerjisi", help="AME2020 bağlanma enerjisi ve SEMF; --nuklid yoksa en yüksek B/A")
    p.add_argument("--nuklid", default=None)
    p.set_defaults(func=_cmd_baglanma_enerjisi)

    p = sub.add_parser("semf-uydur", help="SEMF'i AME2020'ye uydur, literatürle ve sihirli sayılarla karşılaştır")
    p.add_argument("--grafik", default=None)
    p.set_defaults(func=_cmd_semf_uydur)

    p = sub.add_parser("q-degeri", help="Kütle farkından Q-değeri; DDEP ile çapraz kontrol")
    p.add_argument("--nuklid", required=True)
    p.add_argument("--tur", required=True, choices=list(reactions.Q_FUNCTIONS))
    p.set_defaults(func=_cmd_q_degeri)

    p = sub.add_parser("aktivasyon", help="Nötron aktivasyonu")
    _add_half_life_arguments(p)
    p.add_argument("--atom-sayisi", type=float, default=None)
    p.add_argument("--kutle-g", type=float, default=None)
    p.add_argument("--molar-kutle", type=float, default=None)
    p.add_argument("--kesit-barn", type=float, required=True)
    p.add_argument("--aki", type=float, required=True, help="nötron akısı (cm^-2 s^-1)")
    p.add_argument("--isinma-suresi", type=float, required=True, help="ışınlama süresi")
    p.add_argument("--isinma-birimi", default="saat", choices=UNITS)
    p.add_argument("--bekleme-suresi", type=float, default=0.0)
    p.add_argument("--bekleme-birimi", default="saat", choices=UNITS)
    p.set_defaults(func=_cmd_aktivasyon)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"Hata: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
