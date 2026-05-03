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
    parser.add_argument("--split-mode", default="image", choices=["image", "satellite"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--strong-augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-masks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--crop-to-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--class-weight-power", type=float, default=1.0)
    parser.add_argument("--use-class-weighted-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-weighted-sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-metric", default="macro_acc", choices=["macro_acc", "val_acc"])
    parser.add_argument("--early-stopping-patience", type=int, default=5)
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
        split_mode=args.split_mode,
        image_size=args.image_size,
        strong_augmentation=args.strong_augmentation,
        use_masks=args.use_masks,
        crop_to_mask=args.crop_to_mask,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        class_weight_power=args.class_weight_power,
        use_class_weighted_loss=args.use_class_weighted_loss,
        use_weighted_sampler=args.use_weighted_sampler,
        checkpoint_metric=args.checkpoint_metric,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
    )
    print(f"Best validation accuracy: {metrics['best_val_acc']:.4f}")
    print(f"Checkpoint: {metrics['checkpoint']}")


if __name__ == "__main__":
    main()
