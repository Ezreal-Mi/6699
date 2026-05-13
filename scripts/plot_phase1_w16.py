"""Regenerate a Phase 1 W=16 fit figure from saved models."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.io_utils import load_models
from src.targets import sawtooth_target

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
MOD_DIR = ROOT / "results" / "models" / "phase1"

per_seed = json.loads((TAB_DIR / "phase1_per_seed.json").read_text())
deep_losses = per_seed["deep_final_mse"]
shallow_losses = per_seed["shallow_final_mse"]
best_deep = int(np.argmin(deep_losses))
best_shallow = int(np.argmin(shallow_losses))

deep_models, deep_meta = load_models(MOD_DIR, "deep", device="cpu")
shallow_models, shallow_meta = load_models(MOD_DIR, "shallow", device="cpu")

x = torch.linspace(0, 1, 2000).view(-1, 1)
y = sawtooth_target(x, k=4)

with torch.no_grad():
    yd = deep_models[best_deep](x)
    ys = shallow_models[best_shallow](x)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(x.squeeze().numpy(), y.squeeze().numpy(), "k--", lw=1.2, alpha=0.55, label="target $f_4$")
ax.plot(
    x.squeeze().numpy(),
    yd.squeeze().numpy(),
    color="tab:green",
    lw=2.0,
    label=f"deep L=9, W=16 (best seed {per_seed['seeds'][best_deep]})",
)
ax.plot(
    x.squeeze().numpy(),
    ys.squeeze().numpy(),
    color="tab:red",
    lw=1.7,
    label=f"shallow L=2, W=651 (best seed {per_seed['seeds'][best_shallow]})",
)
ax.set_title("Phase 1 W=16 rescue: best-seed fits")
ax.set_xlabel("x")
ax.set_ylabel("$f_4(x)$")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()

out = FIG_DIR / "phase1_w16_rescue_fits.png"
fig.savefig(out, dpi=150)
print(out)
