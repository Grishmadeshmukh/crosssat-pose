#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path, default_dataset_root, default_output_root

add_src_to_path()

from overfit.geometry_search import run_geometry_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Geometry-aware same-satellite and cross-satellite pose experiments.")
    parser.add_argument("--dataset-root", default=default_dataset_root())
    parser.add_argument("--query-satellite", required=True)
    parser.add_argument("--mesh-satellite", required=True)
    parser.add_argument("--candidate-satellite")
    parser.add_argument("--query-split", default="paper_val", choices=["all", "paper_train", "paper_val", "paper_trainval"])
    parser.add_argument("--candidate-split", default="paper_train", choices=["all", "paper_train", "paper_val", "paper_trainval"])
    parser.add_argument("--max-query-samples", type=int, default=64)
    parser.add_argument("--max-candidate-samples", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--refine-rounds", type=int, default=2)
    parser.add_argument("--refine-samples-per-round", type=int, default=24)
    parser.add_argument("--rotation-sigma-deg", type=float, default=10.0)
    parser.add_argument("--translation-sigma", type=float, default=0.03)
    parser.add_argument("--crop-padding", type=int, default=12)
    parser.add_argument("--output-dir", default=default_output_root("geometry_search"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_geometry_experiment(
        args.dataset_root,
        query_satellite=args.query_satellite,
        mesh_satellite=args.mesh_satellite,
        candidate_satellite=args.candidate_satellite,
        query_split=args.query_split,
        candidate_split=args.candidate_split,
        max_query_samples=args.max_query_samples,
        max_candidate_samples=args.max_candidate_samples,
        top_k=args.top_k,
        refine_rounds=args.refine_rounds,
        refine_samples_per_round=args.refine_samples_per_round,
        rotation_sigma_deg=args.rotation_sigma_deg,
        translation_sigma=args.translation_sigma,
        crop_padding=args.crop_padding,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
