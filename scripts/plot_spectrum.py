"""
Compares the kinetic energy spectrum E(k) of the true CFD solution against
the DRRN upsampled prediction. Reveals whether the DRRN damps high-frequency
components (spectral roll-off before the grid cutoff) or injects non-physical
high-frequency noise (energy above the CFD curve at high k).
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import glob
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.ml.drrn import DRRN
from src.ml.drrn_config import DRRNConfig
from src.visualization.spectral_analysis import plot_kinetic_energy_spectrum

config = DRRNConfig()
model = DRRN(
    n_resblocks=config.n_resblocks,
    n_feats=config.n_feats,
    scale=config.scale,
)

# Load the best / latest checkpoint
ckpts = sorted(glob.glob("checkpoints/drrn_warmup_epoch*.pt"))
assert ckpts, "No checkpoint found — run Stage 1 (warmup) first"
print(f"Loading checkpoint: {ckpts[-1]}")
model.load_state_dict(torch.load(ckpts[-1], map_location="cpu"))
model.eval()

fig, ax = plt.subplots(figsize=(8, 5))
dx = 1.0 / 127.0

kolmogorov_drawn = False
for Re in [500, 1000]:
    h5_path = f"data/fields/Re_{Re:05.0f}.h5"
    with h5py.File(h5_path, "r") as f:
        u_true = f["u_128"][:]
        v_true = f["v_128"][:]
        u_c = f["u_32"][:]
        v_c = f["v_32"][:]

    # True CFD field
    draw_kolmogorov = not kolmogorov_drawn
    plot_kinetic_energy_spectrum(
        u_true, v_true, dx,
        label=f"CFD Re={Re}",
        ax=ax,
        kolmogorov_line=draw_kolmogorov,
    )
    kolmogorov_drawn = True

    # DRRN prediction: upsample coarse 32×32 → 128×128
    coarse = torch.from_numpy(np.stack([u_c, v_c], axis=0)).float().unsqueeze(0)
    with torch.no_grad():
        pred = model(coarse).squeeze(0).numpy()
    plot_kinetic_energy_spectrum(
        pred[0], pred[1], dx,
        label=f"DRRN Re={Re}",
        ax=ax,
        kolmogorov_line=False,
    )

ax.set_xlabel("Wavenumber k")
ax.set_ylabel("E(k) — Kinetic Energy")
ax.set_title("Kinetic Energy Spectrum: CFD vs DRRN")
ax.legend()
plt.tight_layout()

out_path = pathlib.Path("outputs/figures")
out_path.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path / "energy_spectrum.png", dpi=200)
print("Saved outputs/figures/energy_spectrum.png")
