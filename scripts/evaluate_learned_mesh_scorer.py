#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path, default_dataset_root, default_output_root

add_src_to_path()

import torch

from common import set_seed
from overfit.learned_mesh_scorer import evaluate_learned_mesh_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the learned mesh-conditioned pose scorer.")
    parser.add_argument("--dataset-root", default=default_dataset_root())
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query-satellite", required=True)
    parser.add_argument("--mesh-satellite", required=True)
    parser.add_argument("--candidate-satellite")
    parser.add_argument("--query-split", default="paper_val", choices=["all", "paper_train", "paper_val", "paper_trainval"])
    parser.add_argument("--candidate-split", default="paper_train", choices=["all", "paper_train", "paper_val", "paper_trainval"])
    parser.add_argument("--max-query-samples", type=int, default=64)
    parser.add_argument("--max-candidate-samples", type=int, default=256)
    parser.add_argument("--candidate-strategy", default="even", choices=["even", "random"])
    parser.add_argument("--candidate-mode", default="hybrid", choices=["dataset", "structured", "hybrid"])
    parser.add_argument("--grid-azimuth-bins", type=int, default=24)
    parser.add_argument("--grid-elevation-bins", type=int, default=9)
    parser.add_argument("--grid-roll-bins", type=int, default=8)
    parser.add_argument("--grid-radius-samples", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument("--refine-samples-per-round", type=int, default=24)
    parser.add_argument("--rotation-sigma-deg", type=float, default=10.0)
    parser.add_argument("--translation-sigma", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-visualizations", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=default_output_root("learned_mesh_pose_search"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    evaluate_learned_mesh_scorer(
        args.dataset_root,
        checkpoint=args.checkpoint,
        query_satellite=args.query_satellite,
        mesh_satellite=args.mesh_satellite,
        candidate_satellite=args.candidate_satellite,
        query_split=args.query_split,
        candidate_split=args.candidate_split,
        max_query_samples=args.max_query_samples,
        max_candidate_samples=args.max_candidate_samples,
        candidate_strategy=args.candidate_strategy,
        candidate_mode=args.candidate_mode,
        grid_azimuth_bins=args.grid_azimuth_bins,
        grid_elevation_bins=args.grid_elevation_bins,
        grid_roll_bins=args.grid_roll_bins,
        grid_radius_samples=args.grid_radius_samples,
        top_k=args.top_k,
        refine_rounds=args.refine_rounds,
        refine_samples_per_round=args.refine_samples_per_round,
        rotation_sigma_deg=args.rotation_sigma_deg,
        translation_sigma=args.translation_sigma,
        batch_size=args.batch_size,
        num_visualizations=args.num_visualizations,
        device=device,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
