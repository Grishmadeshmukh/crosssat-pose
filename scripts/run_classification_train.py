#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _bootstrap import add_src_to_path, default_dataset_root, default_output_root, project_root

add_src_to_path()

from classification import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 6-class satellite architecture classifier.")
    parser.add_argument("--dataset-root", default=default_dataset_root())
    parser.add_argument("--classification-csv", default=str(project_root() / "classification.csv"))
    parser.add_argument("--output-dir", default=default_output_root("classification"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_training(
        args.dataset_root,
        args.classification_csv,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_fraction=args.train_fraction,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
    print(f"Checkpoint: {metrics['checkpoint']}")


if __name__ == "__main__":
    main()
