"""Plotting helpers. Each function writes a PNG and returns nothing."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_chain(times, activities, labels, path: str, time_label: str, marker_time=None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for column, label in enumerate(labels):
        ax.plot(times, activities[:, column], label=label)
    if marker_time is not None:
        ax.axvline(marker_time, color="gray", ls=":", label="t_max (kapalı-form)")
    ax.set_xlabel(time_label)
    ax.set_ylabel("aktivite (ana çekirdeğin başlangıç aktivitesi = 1)")
    ax.set_title("Bozunma zinciri")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_binding_energy(a, binding_per_a, semf_a, semf_per_a, residual_z, residual_n, residual, magic, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(a, binding_per_a, ".", ms=2, alpha=0.5, label="AME2020 (deneysel)")
    axes[0].plot(semf_a, semf_per_a, "r-", lw=1.5, label="SEMF (kararlılık vadisi)")
    axes[0].set_xlabel("kütle numarası A")
    axes[0].set_ylabel("B/A (MeV)")
    axes[0].set_title("Nükleon başına bağlanma enerjisi")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    scatter = axes[1].scatter(residual_n, residual_z, c=residual, s=4, cmap="coolwarm", vmin=-8, vmax=8)
    for m in magic:
        axes[1].axvline(m, color="k", lw=0.4, alpha=0.5)
        axes[1].axhline(m, color="k", lw=0.4, alpha=0.5)
    axes[1].set_xlabel("N")
    axes[1].set_ylabel("Z")
    axes[1].set_title("Artık (deneysel - SEMF), MeV; çizgiler sihirli sayılar")
    fig.colorbar(scatter, ax=axes[1])
    _save(fig, path)
