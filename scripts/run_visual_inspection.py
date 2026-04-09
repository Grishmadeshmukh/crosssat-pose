#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path, default_dataset_root, default_output_root

add_src_to_path()

from visual_inspection.inspection import run_visual_inspection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual inspection on one or more SPE3R satellites.")
    parser.add_argument("--dataset-root", default=default_dataset_root())
    parser.add_argument("--satellites", nargs="+", required=True)
    parser.add_argument("--split", default="all", choices=["all", "paper_train", "paper_val", "paper_trainval"])
    parser.add_argument("--samples-per-satellite", type=int, default=12)
    parser.add_argument("--correlation-samples", type=int, default=96)
    parser.add_argument("--pair-samples", type=int, default=400)
    parser.add_argument("--raw-image-size", type=int, default=64)
    parser.add_argument("--output-dir", default=default_output_root("visual_inspection"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_visual_inspection(
        args.dataset_root,
        args.satellites,
        split=args.split,
        samples_per_satellite=args.samples_per_satellite,
        correlation_samples=args.correlation_samples,
        pair_samples=args.pair_samples,
        raw_image_size=args.raw_image_size,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
