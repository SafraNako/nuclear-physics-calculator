from __future__ import annotations

import re

from nukleer_fizik_hesaplayicisi.cli import main


def _run(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr().out


def test_bozunma_with_ddep_half_life(capsys):
    code, out = _run(capsys, "bozunma", "--nuklid", "co-60", "--sure", "5.2711", "--sure-birimi", "yil", "--aktivite", "1e6")
    assert code == 0
    assert "N/N₀ = 0.5  (1 yarı ömür)" in out
    assert "-> 500000 Bq" in out


def test_bozunma_with_explicit_half_life(capsys):
    code, out = _run(capsys, "bozunma", "--yari-omur", "10", "--birim", "saniye", "--sure", "20", "--sure-birimi", "saniye")
    assert code == 0
    assert "N/N₀ = 0.25" in out


def test_bozunma_needs_a_half_life(capsys):
    code, out = _run(capsys, "bozunma", "--sure", "1")
    assert code == 1 and "Hata" in out


def test_ozgul_aktivite_for_cobalt_60(capsys):
    code, out = _run(capsys, "ozgul-aktivite", "--nuklid", "co-60")
    assert code == 0
    ci = float(re.search(r"= ([\d.]+) Ci/g", out).group(1))
    assert 1100 < ci < 1160


def test_jenerator_reports_tmax_and_the_ddep_equilibrium(capsys, tmp_path):
    png = tmp_path / "g.png"
    code, out = _run(capsys, "jenerator", "--grafik", str(png))
    assert code == 0
    assert "22.844 saat" in out
    assert "0.9638" in out and "0.963 ± 0.021" in out
    assert png.exists() and png.stat().st_size > 0


def test_zincir_compares_all_three_methods(capsys, tmp_path):
    png = tmp_path / "z.png"
    code, out = _run(capsys, "zincir", "--yari-omurler", "65.9496,6.0067", "--sure", "100", "--dallanma", "0.876", "--grafik", str(png))
    assert code == 0
    assert "Bateman kapalı-formuna göre en büyük göreli fark" in out
    assert "80 basamaklı referansa göre" in out
    assert png.exists()


def test_zincir_with_equal_half_lives_reports_that_bateman_does_not_apply(capsys):
    code, out = _run(capsys, "zincir", "--yari-omurler", "10,10", "--sure", "20")
    assert code == 0
    assert "0.346574" in out
    assert "uygulanamıyor" in out


def test_zincir_rejects_bad_branching(capsys):
    code, out = _run(capsys, "zincir", "--yari-omurler", "10,5", "--sure", "1", "--dallanma", "1.5")
    assert code == 1 and "Hata" in out


def test_baglanma_enerjisi_for_iron_56_and_the_top_list(capsys):
    code, out = _run(capsys, "baglanma-enerjisi", "--nuklid", "fe-56")
    assert code == 0
    assert "B/A = 8.7904 MeV" in out
    assert "artık" in out
    code, out = _run(capsys, "baglanma-enerjisi")
    assert code == 0
    assert out.index("Ni-62") < out.index("Fe-58") < out.index("Fe-56")


def test_semf_uydur_prints_fit_magic_residuals_and_plots(capsys, tmp_path):
    png = tmp_path / "s.png"
    code, out = _run(capsys, "semf-uydur", "--grafik", str(png))
    assert code == 0
    assert "kararlılık vadisi" in out and "literatür" in out
    assert "Pb-208" in out and "Sn-132" in out
    assert "En yüksek B/A: Ni-62" in out
    assert png.exists() and png.stat().st_size > 0


def test_q_degeri_crosschecks_against_ddep(capsys):
    code, out = _run(capsys, "q-degeri", "--nuklid", "co-60", "--tur", "beta-minus")
    assert code == 0
    assert "2822.81" in out
    assert "DDEP değerlendirmesi: 2823.07" in out
    code, out = _run(capsys, "q-degeri", "--nuklid", "fe-56", "--tur", "beta-minus")
    assert code == 0
    assert "yasak" in out


def test_q_degeri_rejects_unknown_nuclide(capsys):
    code, out = _run(capsys, "q-degeri", "--nuklid", "xx-99", "--tur", "alpha")
    assert code == 1 and "Hata" in out


def test_aktivasyon_numeric_check_agrees(capsys):
    code, out = _run(
        capsys, "aktivasyon", "--nuklid", "mo-99", "--atom-sayisi", "1e20", "--kesit-barn", "0.13", "--aki", "1e14",
        "--isinma-suresi", "100", "--isinma-birimi", "saat", "--bekleme-suresi", "10", "--bekleme-birimi", "saat",
    )
    assert code == 0
    assert "Doyum aktivitesi N·σ·φ = 1.3e+09 Bq" in out
    assert "doyum oranı = 0.6504" in out
    diff = float(re.search(r"göreli fark ([\d.e+-]+)\)", out).group(1))
    assert diff < 1e-12


def test_aktivasyon_needs_a_target(capsys):
    code, out = _run(capsys, "aktivasyon", "--yari-omur", "1", "--birim", "saat", "--kesit-barn", "1", "--aki", "1e12", "--isinma-suresi", "1")
    assert code == 1 and "Hata" in out
