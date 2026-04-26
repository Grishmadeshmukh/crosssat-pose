#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path, default_dataset_root, default_output_root

add_src_to_path()

import torch

from common import set_seed
from overfit.benchmark_refiner import train_benchmark_refiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a shortlist-based benchmark-style pose refiner.")
    parser.add_argument("--dataset-root", default=default_dataset_root())
    parser.add_argument("--query-satellite", required=True)
    parser.add_argument("--mesh-satellite", required=True)
    parser.add_argument("--candidate-satellite")
    parser.add_argument("--train-split", default="paper_train", choices=["all", "paper_train", "paper_val", "paper_trainval"])
    parser.add_argument("--eval-split", default="paper_val", choices=["all", "paper_train", "paper_val", "paper_trainval"])
    parser.add_argument("--max-train-samples", type=int, default=256)
    parser.add_argument("--max-eval-samples", type=int, default=64)
    parser.add_argument("--max-candidate-samples", type=int, default=256)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-padding", type=int, default=12)
    parser.add_argument("--shortlist-size", type=int, default=16)
    parser.add_argument("--close-pool-size", type=int, default=32)
    parser.add_argument("--score-temperature-deg", type=float, default=20.0)
    parser.add_argument("--refine-top-m", type=int, default=4)
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--refine-weight", type=float, default=1.0)
    parser.add_argument("--use-dataset-bank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-structured-bank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid-azimuth-bins", type=int, default=24)
    parser.add_argument("--grid-elevation-bins", type=int, default=9)
    parser.add_argument("--grid-roll-bins", type=int, default=8)
    parser.add_argument("--grid-radius-samples", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=default_output_root("benchmark_refiner"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    train_benchmark_refiner(
        args.dataset_root,
        query_satellite=args.query_satellite,
        mesh_satellite=args.mesh_satellite,
        candidate_satellite=args.candidate_satellite,
        train_split=args.train_split,
        eval_split=args.eval_split,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_candidate_samples=args.max_candidate_samples,
        samples_per_epoch=args.samples_per_epoch,
        eval_samples=args.eval_samples,
        image_size=args.image_size,
        crop_padding=args.crop_padding,
        shortlist_size=args.shortlist_size,
        close_pool_size=args.close_pool_size,
        score_temperature_deg=args.score_temperature_deg,
        refine_top_m=args.refine_top_m,
        base_width=args.base_width,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        score_weight=args.score_weight,
        refine_weight=args.refine_weight,
        use_dataset_bank=args.use_dataset_bank,
        use_structured_bank=args.use_structured_bank,
        grid_azimuth_bins=args.grid_azimuth_bins,
        grid_elevation_bins=args.grid_elevation_bins,
        grid_roll_bins=args.grid_roll_bins,
        grid_radius_samples=args.grid_radius_samples,
        num_workers=args.num_workers,
        device=device,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
