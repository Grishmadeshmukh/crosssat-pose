#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models

from _bootstrap import add_src_to_path, default_dataset_root, project_root

add_src_to_path()

from classification import build_validation_dataset
from common import load_torch_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classifier on held-out validation split.")
    parser.add_argument("--dataset-root", default=default_dataset_root())
    parser.add_argument("--classification-csv", default=str(project_root() / "classification.csv"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-mode", default="satellite", choices=["image", "satellite"])
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--strong-augmentation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-masks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--crop-to-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def build_model(num_classes: int) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def main() -> None:
    args = parse_args()
    checkpoint = load_torch_checkpoint(args.checkpoint, map_location="cpu")
    class_names = checkpoint["class_names"]
    model = build_model(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_set, metadata = build_validation_dataset(
        args.dataset_root,
        args.classification_csv,
        split_mode=args.split_mode,
        train_fraction=args.train_fraction,
        image_size=args.image_size,
        strong_augmentation=args.strong_augmentation,
        use_masks=args.use_masks,
        crop_to_mask=args.crop_to_mask,
        seed=args.seed,
    )
    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            for true_label, pred_label in zip(labels.tolist(), preds.tolist()):
                confusion[int(true_label), int(pred_label)] += 1

    per_class_metrics: list[dict[str, float | int | str]] = []
    total_correct = int(np.trace(confusion))
    total_samples = int(confusion.sum())
    overall_acc = float(total_correct / max(total_samples, 1))
    macro_acc_values: list[float] = []
    macro_f1_values: list[float] = []
    print(f"Overall accuracy: {overall_acc:.4f}")
    print("Per-class metrics:")
    for class_idx, class_name in enumerate(class_names):
        true_count = int(confusion[class_idx].sum())
        pred_count = int(confusion[:, class_idx].sum())
        correct = int(confusion[class_idx, class_idx])
        acc = float(correct / true_count) if true_count > 0 else 0.0
        precision = float(correct / pred_count) if pred_count > 0 else 0.0
        recall = acc
        f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) > 0.0 else 0.0
        macro_acc_values.append(acc)
        macro_f1_values.append(f1)
        per_class_metrics.append(
            {
                "class_id": class_idx,
                "class_name": class_name,
                "correct": correct,
                "total_true": true_count,
                "total_pred": pred_count,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        print(
            f"  [{class_idx}] {class_name}: "
            f"acc={acc:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
            f"({correct}/{true_count})"
        )

    macro_acc = float(np.mean(macro_acc_values)) if macro_acc_values else 0.0
    macro_f1 = float(np.mean(macro_f1_values)) if macro_f1_values else 0.0
    print(f"\nMacro accuracy: {macro_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "     " + " ".join([f"{idx:>5d}" for idx in range(len(class_names))])
    print(header)
    for row_idx in range(len(class_names)):
        values = " ".join([f"{int(value):>5d}" for value in confusion[row_idx]])
        print(f"{row_idx:>3d}: {values}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "overall_accuracy": overall_acc,
            "macro_accuracy": macro_acc,
            "macro_f1": macro_f1,
            "per_class_metrics": per_class_metrics,
            "class_names": class_names,
            "split_mode": args.split_mode,
            "split_info": metadata["split_info"],
            "checkpoint": args.checkpoint,
            "val_samples": total_samples,
        }
        (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
        np.savetxt(output_dir / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
        print(f"\nSaved metrics to: {output_dir / 'evaluation_metrics.json'}")
        print(f"Saved confusion matrix to: {output_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
