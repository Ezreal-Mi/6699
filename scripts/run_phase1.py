"""Reproduce Phase 1 with the current curriculum and restart logic.

Default settings reproduce the conservative H1 W=16 rescue run:

    python scripts/run_phase1.py

For a cheap smoke run that does not overwrite the default interpretation:

    python scripts/run_phase1.py --quick
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import save_models
from src.train import TrainConfig, multi_seed_run


FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
MOD_DIR = ROOT / "results" / "models" / "phase1"


def parse_seed_list(value: str) -> list[int]:
    """Parse a comma-separated seed list such as ``0,1,2,3,4``."""
    seeds = [part.strip() for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    try:
        return [int(seed) for seed in seeds]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be integers") from exc


def resolve_device(value: str) -> str:
    """Resolve ``auto`` to CUDA when available, otherwise CPU."""
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def build_train_configs(args: argparse.Namespace, device: str) -> tuple[TrainConfig, TrainConfig]:
    """Build deep and shallow training configs with the same curriculum schedule."""
    common = dict(
        epochs=args.epochs,
        lr=args.lr,
        lr_min=args.lr_min,
        use_cosine_lr=True,
        grad_clip=args.grad_clip,
        n_train=args.n_train,
        k=args.k,
        curriculum_ks=[1, 2, 3, 4],
        curriculum_weights=[1, 1, 2, 4],
        reset_lr_per_stage=True,
        log_every=args.log_every,
        device=device,
    )
    deep_cfg = TrainConfig(
        **common,
        max_curriculum_restarts=args.max_restarts,
        collapse_threshold=args.collapse_threshold,
    )
    shallow_cfg = TrainConfig(
        **common,
        max_curriculum_restarts=0,
        collapse_threshold=None,
    )
    return deep_cfg, shallow_cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30_000)
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("0,1,2,3,4"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--deep-depth", type=int, default=9)
    parser.add_argument("--deep-width", type=int, default=16)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=3000)
    parser.add_argument("--max-restarts", type=int, default=5)
    parser.add_argument("--collapse-threshold", type=float, default=0.02)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one seed, fewer epochs, no deep restarts, and quiet logging.",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.seeds = [0]
        args.epochs = min(args.epochs, 1000)
        args.max_restarts = 0
        args.log_every = 0
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    MOD_DIR.mkdir(parents=True, exist_ok=True)

    deep_cfg, shallow_cfg = build_train_configs(args, device)

    print(f"Running Phase 1: device={device}, seeds={args.seeds}")
    print(f"Architecture: deep L={args.deep_depth}, W={args.deep_width}; shallow L=2, parameter matched")
    print(f"Deep:    {args.epochs} epochs, up to {args.max_restarts} curriculum restarts")
    print("Shallow: same epochs, no curriculum restarts")
    print()

    results = multi_seed_run(
        args.seeds,
        deep_cfg,
        deep_depth=args.deep_depth,
        deep_width=args.deep_width,
        shallow_cfg=shallow_cfg,
    )

    deep_losses = results["deep"]["final_loss"]
    shallow_losses = results["shallow"]["final_loss"]
    deep_mu, deep_sd = float(np.mean(deep_losses)), float(np.std(deep_losses))
    shallow_mu, shallow_sd = float(np.mean(shallow_losses)), float(np.std(shallow_losses))

    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS")
    print("=" * 60)
    for seed, deep_loss, shallow_loss in zip(args.seeds, deep_losses, shallow_losses):
        tag = "[OK]" if deep_loss < args.collapse_threshold else "[--]"
        print(f"  seed {seed}: deep={deep_loss:.6f}{tag}  shallow={shallow_loss:.6f}")
    print(f"\n  deep    mean={deep_mu:.4e} +/- {deep_sd:.4e}")
    print(f"  shallow mean={shallow_mu:.4e} +/- {shallow_sd:.4e}")
    if deep_mu > 0:
        ratio = shallow_mu / deep_mu
        log10adv = np.log10(ratio)
        print(f"  ratio (shallow/deep): {ratio:.2f}x  (log10 = {log10adv:.2f})")
        print(f"  H1 {'SUPPORTED' if log10adv >= 1 else 'NOT supported at log10>=1'} (need >= 1)")

    csv_path = TAB_DIR / "phase1_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "params", "final_mse_mean", "final_mse_std"])
        writer.writerow(["deep", results["deep"]["info"][0]["deep_params"], deep_mu, deep_sd])
        writer.writerow(["shallow", results["shallow"]["info"][0]["shallow_params"], shallow_mu, shallow_sd])
    print(f"\nSaved: {csv_path}")

    per_seed = {
        "seeds": args.seeds,
        "deep_final_mse": deep_losses,
        "shallow_final_mse": shallow_losses,
        "deep_mean": deep_mu,
        "deep_std": deep_sd,
        "shallow_mean": shallow_mu,
        "shallow_std": shallow_sd,
    }
    json_path = TAB_DIR / "phase1_per_seed.json"
    json_path.write_text(json.dumps(per_seed, indent=2))
    print(f"Saved: {json_path}")

    save_models(MOD_DIR, "deep", results["deep"]["models"], results["deep"]["info"], args.seeds)
    save_models(MOD_DIR, "shallow", results["shallow"]["models"], results["shallow"]["info"], args.seeds)
    print(f"Saved models: {MOD_DIR}")
    print("\nDone. Run scripts/plot_phase1_w16.py to regenerate the Phase 1 figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
