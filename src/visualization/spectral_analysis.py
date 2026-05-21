"""Spectral post-processing utilities for comparing CFD and DRRN velocity fields in k-space."""

from __future__ import annotations

import numpy as np


def plot_kinetic_energy_spectrum(
    u: np.ndarray,
    v: np.ndarray,
    dx: float,
    label: str,
    ax,
    kolmogorov_line: bool = True,
) -> None:
    """
    Compute and plot the azimuthally-averaged 2D kinetic energy spectrum E(k).

    Algorithm:
      1. Compute 2D FFT of u and v fields.
      2. Compute E(kx, ky) = 0.5 * (|FFT(u)|^2 + |FFT(v)|^2) / N^4
         (divide by N^4 = N^2 * N^2 to normalise the DFT convention).
      3. Compute radial wavenumber k = sqrt(kx^2 + ky^2) for each (kx, ky) mode.
      4. Bin E values by integer wavenumber shell k_int = round(k).
      5. Sum E within each shell → E(k).
      6. Plot E(k) vs k on log-log axes.

    The Kolmogorov k^(-5/3) line is anchored at the peak energy wavenumber.

    Parameters
    ----------
    u, v : (H, W) arrays — velocity components on the fine grid.
    dx   : grid spacing (= 1/(N-1) for a unit-square domain).
    label, ax, kolmogorov_line : standard plotting arguments.
    """
    H, W = u.shape
    N = H  # assume square grid

    # Step 1: 2D FFT
    Fu = np.fft.fft2(u)
    Fv = np.fft.fft2(v)

    # Step 2: Energy in spectral space, normalised by N^4
    E2d = 0.5 * (np.abs(Fu) ** 2 + np.abs(Fv) ** 2) / (N**4)

    # Step 3: Radial wavenumber array
    # np.fft.fftfreq returns frequencies in cycles/sample; multiply by N to get
    # wavenumber in cycles/domain (integer modes 0..N/2)
    kx = np.fft.fftfreq(W) * W  # shape (W,)
    ky = np.fft.fftfreq(H) * H  # shape (H,)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)  # shape (H, W)

    # Step 4: Bin into integer shells
    k_int = np.round(K).astype(int)
    k_max = N // 2

    Ek = np.zeros(k_max + 1)
    for ki in range(k_max + 1):
        mask = k_int == ki
        if mask.any():
            Ek[ki] = E2d[mask].sum()

    # Step 5: Only plot shells with nonzero energy
    k_range = np.arange(1, k_max + 1)  # skip DC (k=0)
    Ek_range = Ek[1 : k_max + 1]
    valid = Ek_range > 0

    ax.plot(k_range[valid], Ek_range[valid], label=label, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k) — Kinetic Energy")

    # Step 6: Kolmogorov k^(-5/3) reference line
    if kolmogorov_line and valid.any():
        k_peak = k_range[valid][np.argmax(Ek_range[valid])]
        E_peak = Ek_range[valid].max()
        k_ref = k_range[valid]
        kolmo = E_peak * (k_ref / k_peak) ** (-5.0 / 3.0)
        ax.plot(
            k_ref,
            kolmo,
            "k--",
            linewidth=0.8,
            alpha=0.6,
            label=r"$k^{-5/3}$ Kolmogorov",
        )
