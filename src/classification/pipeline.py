from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from common import ensure_dir, set_seed
from data_utils import SPE3RSatellite


@dataclass(frozen=True)
class ClassificationRow:
    satellite_name: str
    class_id: int
    class_name: str


def load_classification_rows(csv_path: str | Path) -> list[ClassificationRow]:
    rows: list[ClassificationRow] = []
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            satellite_name = (row.get("satellite_name") or "").strip()
            class_id_text = (row.get("architecture_class") or "").strip()
            class_name = (row.get("architecture_label") or "").strip()
            if not satellite_name or not class_id_text:
                continue
            rows.append(
                ClassificationRow(
                    satellite_name=satellite_name,
                    class_id=int(class_id_text) - 1,
                    class_name=class_name,
                )
            )
    if not rows:
        raise ValueError(f"No valid rows found in classification csv: {csv_path}")
    return rows


def build_class_metadata(rows: list[ClassificationRow]) -> dict[str, Any]:
    class_names: dict[int, str] = {}
    satellite_to_class: dict[str, int] = {}
    for row in rows:
        satellite_to_class[row.satellite_name] = row.class_id
        class_names[row.class_id] = row.class_name

    num_classes = max(class_names.keys()) + 1
    id_to_name = [class_names[idx] for idx in range(num_classes)]
    return {
        "num_classes": num_classes,
        "satellite_to_class": satellite_to_class,
        "class_names": id_to_name,
    }


class SatelliteClassificationDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        dataset_root: str | Path,
        satellite_to_class: dict[str, int],
        *,
        split: str,
        train_fraction: float,
        seed: int,
        image_size: int,
        strong_augmentation: bool = True,
        use_masks: bool = False,
        crop_to_mask: bool = False,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")
        self.transform = _build_transform(split=split, image_size=image_size, strong_augmentation=strong_augmentation)
        self.use_masks = use_masks
        self.crop_to_mask = crop_to_mask
        rng = np.random.default_rng(seed)
        self.samples: list[tuple[str, str, int]] = []
        self.satellites: dict[str, SPE3RSatellite] = {}

        for satellite_name, class_id in sorted(satellite_to_class.items()):
            satellite = SPE3RSatellite(dataset_root, satellite_name)
            image_names = [record.image_name for record in satellite.records]
            if not image_names:
                continue
            indices = rng.permutation(len(image_names))
            split_idx = int(len(indices) * train_fraction)
            if split == "train":
                selected = indices[:split_idx]
            else:
                selected = indices[split_idx:]
            for index in selected:
                self.samples.append((satellite_name, image_names[int(index)], class_id))
            self.satellites[satellite_name] = satellite

        if not self.samples:
            raise ValueError(f"No samples loaded for split '{split}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        satellite_name, image_name, class_id = self.samples[index]
        satellite = self.satellites[satellite_name]
        image = satellite.load_image(image_name)
        if self.use_masks:
            mask_name = image_name.replace(".jpg", ".png")
            mask = satellite.load_mask(mask_name)
            image = _apply_mask_options(image, mask, crop_to_mask=self.crop_to_mask)
        return self.transform(image), class_id


def _build_transform(*, split: str, image_size: int, strong_augmentation: bool) -> transforms.Compose:
    if split == "train":
        if strong_augmentation:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=20),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                    transforms.RandomGrayscale(p=0.05),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _apply_mask_options(image: Image.Image, mask: Image.Image, *, crop_to_mask: bool) -> Image.Image:
    image_arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    if not np.any(mask_arr):
        return image
    masked = image_arr.copy()
    masked[~mask_arr] = 0
    if not crop_to_mask:
        return Image.fromarray(masked, mode="RGB")
    ys, xs = np.where(mask_arr)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    cropped = masked[y_min : y_max + 1, x_min : x_max + 1]
    return Image.fromarray(cropped, mode="RGB")


def _split_satellites_by_class(
    satellite_to_class: dict[str, int],
    train_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    class_to_satellites: dict[int, list[str]] = {}
    for satellite_name, class_id in satellite_to_class.items():
        class_to_satellites.setdefault(class_id, []).append(satellite_name)

    rng = np.random.default_rng(seed)
    train_satellites: set[str] = set()
    val_satellites: set[str] = set()
    for satellites in class_to_satellites.values():
        shuffled = list(sorted(satellites))
        rng.shuffle(shuffled)
        train_count = int(math.floor(len(shuffled) * train_fraction))
        train_count = max(1, min(train_count, len(shuffled) - 1))
        train_satellites.update(shuffled[:train_count])
        val_satellites.update(shuffled[train_count:])
    return train_satellites, val_satellites


def _build_datasets(
    dataset_root: str | Path,
    satellite_to_class: dict[str, int],
    *,
    split_mode: str,
    train_fraction: float,
    seed: int,
    image_size: int,
    strong_augmentation: bool,
    use_masks: bool,
    crop_to_mask: bool,
) -> tuple[SatelliteClassificationDataset, SatelliteClassificationDataset, dict[str, Any]]:
    if split_mode == "image":
        train_set = SatelliteClassificationDataset(
            dataset_root,
            satellite_to_class,
            split="train",
            train_fraction=train_fraction,
            seed=seed,
            image_size=image_size,
            strong_augmentation=strong_augmentation,
            use_masks=use_masks,
            crop_to_mask=crop_to_mask,
        )
        val_set = SatelliteClassificationDataset(
            dataset_root,
            satellite_to_class,
            split="val",
            train_fraction=train_fraction,
            seed=seed,
            image_size=image_size,
            strong_augmentation=strong_augmentation,
            use_masks=use_masks,
            crop_to_mask=crop_to_mask,
        )
        split_info = {
            "split_mode": split_mode,
            "train_satellites": sorted(satellite_to_class.keys()),
            "val_satellites": sorted(satellite_to_class.keys()),
        }
        return train_set, val_set, split_info

    if split_mode == "satellite":
        train_satellites, val_satellites = _split_satellites_by_class(
            satellite_to_class, train_fraction=train_fraction, seed=seed
        )
        train_set = SatelliteClassificationDataset(
            dataset_root,
            {name: satellite_to_class[name] for name in sorted(train_satellites)},
            split="train",
            train_fraction=1.0,
            seed=seed,
            image_size=image_size,
            strong_augmentation=strong_augmentation,
            use_masks=use_masks,
            crop_to_mask=crop_to_mask,
        )
        val_set = SatelliteClassificationDataset(
            dataset_root,
            {name: satellite_to_class[name] for name in sorted(val_satellites)},
            split="val",
            train_fraction=0.0,
            seed=seed,
            image_size=image_size,
            strong_augmentation=strong_augmentation,
            use_masks=use_masks,
            crop_to_mask=crop_to_mask,
        )
        split_info = {
            "split_mode": split_mode,
            "train_satellites": sorted(train_satellites),
            "val_satellites": sorted(val_satellites),
        }
        return train_set, val_set, split_info

    raise ValueError(f"Unsupported split mode: {split_mode}")


def _build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    *,
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def _macro_accuracy(confusion: np.ndarray) -> float:
    per_class: list[float] = []
    for class_idx in range(confusion.shape[0]):
        total = int(confusion[class_idx].sum())
        correct = int(confusion[class_idx, class_idx])
        per_class.append(float(correct / total) if total > 0 else 0.0)
    return float(np.mean(per_class)) if per_class else 0.0


def _train_class_counts(train_set: SatelliteClassificationDataset, num_classes: int) -> np.ndarray:
    counts = np.zeros((num_classes,), dtype=np.int64)
    for _, _, class_id in train_set.samples:
        counts[int(class_id)] += 1
    return counts


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, np.ndarray]:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_true: list[int] = []
    all_pred: list[int] = []
    num_classes = int(getattr(model, "fc").out_features)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((preds == labels).sum().item())
        total += int(images.size(0))
        all_true.extend(labels.detach().cpu().tolist())
        all_pred.extend(preds.detach().cpu().tolist())
    confusion = _confusion_matrix(all_true, all_pred, num_classes=num_classes)
    return total_loss / max(total, 1), total_correct / max(total, 1), confusion


def run_training(
    dataset_root: str | Path,
    classification_csv: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    train_fraction: float = 0.8,
    split_mode: str = "image",
    image_size: int = 224,
    strong_augmentation: bool = True,
    use_masks: bool = False,
    crop_to_mask: bool = False,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    class_weight_power: float = 1.0,
    use_class_weighted_loss: bool = True,
    use_weighted_sampler: bool = True,
    checkpoint_metric: str = "macro_acc",
    early_stopping_patience: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    if checkpoint_metric not in {"macro_acc", "val_acc"}:
        raise ValueError(f"Unsupported checkpoint metric: {checkpoint_metric}")
    set_seed(seed)
    rows = load_classification_rows(classification_csv)
    metadata = build_class_metadata(rows)
    train_set, val_set, split_info = _build_datasets(
        dataset_root,
        metadata["satellite_to_class"],
        split_mode=split_mode,
        train_fraction=train_fraction,
        seed=seed,
        image_size=image_size,
        strong_augmentation=strong_augmentation,
        use_masks=use_masks,
        crop_to_mask=crop_to_mask,
    )
    train_counts = _train_class_counts(train_set, metadata["num_classes"])
    if use_weighted_sampler:
        class_weights = np.power(np.maximum(train_counts, 1), -class_weight_power)
        sample_weights = [float(class_weights[class_id]) for _, _, class_id in train_set.samples]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(metadata["num_classes"]).to(device)
    if use_class_weighted_loss:
        class_weights = np.power(np.maximum(train_counts, 1), -class_weight_power).astype(np.float32)
        criterion = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device),
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_val_macro_acc = -1.0
    best_score = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    output_root = ensure_dir(output_dir)
    best_model_path = output_root / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc, val_confusion = _run_epoch(model, val_loader, criterion, device, None)
        val_macro_acc = _macro_accuracy(val_confusion)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_acc": val_macro_acc,
                "learning_rate": current_lr,
            }
        )
        current_score = val_macro_acc if checkpoint_metric == "macro_acc" else val_acc
        if current_score > best_score:
            best_score = current_score
            best_val_acc = val_acc
            best_val_macro_acc = val_macro_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": metadata["class_names"],
                    "image_size": image_size,
                    "use_masks": use_masks,
                    "crop_to_mask": crop_to_mask,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1
        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_acc={val_macro_acc:.4f} "
            f"lr={current_lr:.6f}"
        )
        scheduler.step()
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:02d} "
                f"(best {checkpoint_metric}={best_score:.4f} at epoch {best_epoch:02d})"
            )
            break

    metrics = {
        "best_val_acc": best_val_acc,
        "best_val_macro_acc": best_val_macro_acc,
        "best_score": best_score,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "class_weight_power": class_weight_power,
        "use_class_weighted_loss": use_class_weighted_loss,
        "use_weighted_sampler": use_weighted_sampler,
        "checkpoint_metric": checkpoint_metric,
        "strong_augmentation": strong_augmentation,
        "use_masks": use_masks,
        "crop_to_mask": crop_to_mask,
        "early_stopping_patience": early_stopping_patience,
        "best_epoch": best_epoch,
        "split_mode": split_mode,
        "split_info": split_info,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "checkpoint": str(best_model_path),
        "class_names": metadata["class_names"],
        "history": history,
    }
    (output_root / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def predict_image(
    checkpoint_path: str | Path,
    image_path: str | Path,
    *,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint["image_size"])
    model = _build_model(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    batch = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1).squeeze(0)
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(class_names)))
    return [
        (class_names[int(class_idx)], float(prob))
        for prob, class_idx in zip(top_probs.tolist(), top_indices.tolist())
    ]


def build_validation_dataset(
    dataset_root: str | Path,
    classification_csv: str | Path,
    *,
    split_mode: str = "satellite",
    train_fraction: float = 0.8,
    image_size: int = 224,
    strong_augmentation: bool = True,
    use_masks: bool = False,
    crop_to_mask: bool = False,
    seed: int = 42,
) -> tuple[SatelliteClassificationDataset, dict[str, Any]]:
    rows = load_classification_rows(classification_csv)
    metadata = build_class_metadata(rows)
    _, val_set, split_info = _build_datasets(
        dataset_root,
        metadata["satellite_to_class"],
        split_mode=split_mode,
        train_fraction=train_fraction,
        seed=seed,
        image_size=image_size,
        strong_augmentation=strong_augmentation,
        use_masks=use_masks,
        crop_to_mask=crop_to_mask,
    )
    return val_set, {"class_names": metadata["class_names"], "split_info": split_info}
