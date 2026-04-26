#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path, default_dataset_root, default_output_root

add_src_to_path()

import torch

from common import set_seed
from overfit.benchmark_refiner import evaluate_benchmark_refiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the shortlist-based benchmark-style refiner.")
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
    parser.add_argument("--use-dataset-bank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-structured-bank", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid-azimuth-bins", type=int, default=24)
    parser.add_argument("--grid-elevation-bins", type=int, default=9)
    parser.add_argument("--grid-roll-bins", type=int, default=8)
    parser.add_argument("--grid-radius-samples", type=int, default=1)
    parser.add_argument("--coarse-shortlist-size", type=int, default=32)
    parser.add_argument("--keep-top-k", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--num-visualizations", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=default_output_root("benchmark_pose_search"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    evaluate_benchmark_refiner(
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
        use_dataset_bank=args.use_dataset_bank,
        use_structured_bank=args.use_structured_bank,
        grid_azimuth_bins=args.grid_azimuth_bins,
        grid_elevation_bins=args.grid_elevation_bins,
        grid_roll_bins=args.grid_roll_bins,
        grid_radius_samples=args.grid_radius_samples,
        coarse_shortlist_size=args.coarse_shortlist_size,
        keep_top_k=args.keep_top_k,
        iterations=args.iterations,
        num_visualizations=args.num_visualizations,
        device=device,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
