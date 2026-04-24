#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path

add_src_to_path()

from classification import predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict architecture class for one satellite image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = predict_image(args.checkpoint, args.image, top_k=args.top_k)
    for class_name, probability in predictions:
        print(f"{class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
