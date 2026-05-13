"""Re-run Phase 1 with current train.py (curriculum + restart logic).

Fixes:
  - deep model uses W=16, which is stable enough to demonstrate H1
  - deep model gets up to 5 curriculum restarts (collapse_threshold=0.02)
  - shallow model gets NO restarts (collapse_threshold=None) -- fair comparison
  - saves new phase1_summary.csv and model files

Run from EECS6699_Final_Project/:
    python scripts/run_phase1_fixed.py
"""
import sys, os, csv, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.models import make_matched_pair, count_parameters
from src.train import TrainConfig, multi_seed_run
from src.targets import sawtooth_target
from src.io_utils import save_models

FIG_DIR  = ROOT / "results" / "figures"
TAB_DIR  = ROOT / "results" / "tables"
MOD_DIR  = ROOT / "results" / "models" / "phase1"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
MOD_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS  = [0, 1, 2, 3, 4]
K      = 4
DEEP_DEPTH = 9
DEEP_WIDTH = 16

# --- deep config: curriculum + restart logic ---
deep_cfg = TrainConfig(
    epochs=30_000,
    lr=5e-3,
    lr_min=1e-5,
    use_cosine_lr=True,
    grad_clip=1.0,
    n_train=1200,
    k=K,
    curriculum_ks=[1, 2, 3, 4],
    curriculum_weights=[1, 1, 2, 4],
    reset_lr_per_stage=True,
    log_every=3_000,
    device=device,
    max_curriculum_restarts=5,
    collapse_threshold=0.02,
)

# --- shallow config: same training but NO restarts (fair comparison) ---
shallow_cfg = TrainConfig(
    epochs=30_000,
    lr=5e-3,
    lr_min=1e-5,
    use_cosine_lr=True,
    grad_clip=1.0,
    n_train=1200,
    k=K,
    curriculum_ks=[1, 2, 3, 4],
    curriculum_weights=[1, 1, 2, 4],
    reset_lr_per_stage=True,
    log_every=3_000,
    device=device,
    max_curriculum_restarts=0,
    collapse_threshold=None,
)

print(f"Running Phase 1 (fixed) — device={device}, seeds={SEEDS}")
print(f"Architecture: deep L={DEEP_DEPTH}, W={DEEP_WIDTH}; shallow L=2, parameter matched")
print(f"Deep:    30k epochs, up to 5 curriculum restarts (collapse_threshold=0.02)")
print(f"Shallow: 30k epochs, NO restarts (collapse_threshold=None)")
print()

results = multi_seed_run(
    SEEDS, deep_cfg,
    deep_depth=DEEP_DEPTH, deep_width=DEEP_WIDTH,
    shallow_cfg=shallow_cfg,
)

# --- aggregate ---
deep_losses    = results["deep"]["final_loss"]
shallow_losses = results["shallow"]["final_loss"]
deep_mu,    deep_sd    = float(np.mean(deep_losses)),    float(np.std(deep_losses))
shallow_mu, shallow_sd = float(np.mean(shallow_losses)), float(np.std(shallow_losses))

print("\n" + "="*60)
print("PHASE 1 FIXED RESULTS")
print("="*60)
for s, d, sh in zip(SEEDS, deep_losses, shallow_losses):
    tag = "[OK]" if d < 0.02 else "[--]"
    print(f"  seed {s}: deep={d:.6f}{tag}  shallow={sh:.6f}")
print(f"\n  deep    mean={deep_mu:.4e} ± {deep_sd:.4e}")
print(f"  shallow mean={shallow_mu:.4e} ± {shallow_sd:.4e}")
if deep_mu > 0:
    ratio = shallow_mu / deep_mu
    log10adv = np.log10(ratio)
    print(f"  ratio (shallow/deep): {ratio:.2f}x  (log10 = {log10adv:.2f})")
    print(f"  H1 {'SUPPORTED' if log10adv >= 1 else 'NOT supported at log10>=1'} (need >= 1)")

# --- save CSV ---
csv_path = TAB_DIR / "phase1_summary.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "params", "final_mse_mean", "final_mse_std"])
    w.writerow(["deep",    results["deep"]["info"][0]["deep_params"],    deep_mu,    deep_sd])
    w.writerow(["shallow", results["shallow"]["info"][0]["shallow_params"], shallow_mu, shallow_sd])
print(f"\nSaved: {csv_path}")

# --- save per-seed JSON ---
per_seed = {
    "seeds": SEEDS,
    "deep_final_mse":    deep_losses,
    "shallow_final_mse": shallow_losses,
    "deep_mean":    deep_mu,    "deep_std":    deep_sd,
    "shallow_mean": shallow_mu, "shallow_std": shallow_sd,
}
json_path = TAB_DIR / "phase1_per_seed.json"
json_path.write_text(json.dumps(per_seed, indent=2))
print(f"Saved: {json_path}")

# --- save models ---
save_models(MOD_DIR, "deep",
            results["deep"]["models"],   results["deep"]["info"],   SEEDS)
save_models(MOD_DIR, "shallow",
            results["shallow"]["models"], results["shallow"]["info"], SEEDS)
print(f"Saved models: {MOD_DIR}")
print("\nDone. Re-run phase1_baseline.ipynb plots cell to regenerate figures.")
