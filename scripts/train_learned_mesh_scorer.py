#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path, default_dataset_root, default_output_root

add_src_to_path()

import torch

from common import set_seed
from overfit.learned_mesh_scorer import train_learned_mesh_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learned mesh-conditioned pose scorer.")
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
    parser.add_argument("--base-width", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--positive-rotation-sigma-deg", type=float, default=4.0)
    parser.add_argument("--positive-translation-sigma", type=float, default=0.01)
    parser.add_argument("--negative-rotation-sigma-deg", type=float, default=40.0)
    parser.add_argument("--negative-translation-sigma", type=float, default=0.08)
    parser.add_argument("--min-negative-rotation-deg", type=float, default=25.0)
    parser.add_argument("--bank-negative-fraction", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=default_output_root("learned_pose_scorer"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    train_learned_mesh_scorer(
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
        base_width=args.base_width,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        positive_fraction=args.positive_fraction,
        positive_rotation_sigma_deg=args.positive_rotation_sigma_deg,
        positive_translation_sigma=args.positive_translation_sigma,
        negative_rotation_sigma_deg=args.negative_rotation_sigma_deg,
        negative_translation_sigma=args.negative_translation_sigma,
        min_negative_rotation_deg=args.min_negative_rotation_deg,
        bank_negative_fraction=args.bank_negative_fraction,
        num_workers=args.num_workers,
        device=device,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
