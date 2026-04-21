from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5 if split == "train" else 0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
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
        image = self.satellites[satellite_name].load_image(image_name)
        return self.transform(image), class_id


def _build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(images.size(0))
    return total_loss / max(total, 1), total_correct / max(total, 1)


def run_training(
    dataset_root: str | Path,
    classification_csv: str | Path,
    output_dir: str | Path,
    *,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    train_fraction: float = 0.8,
    image_size: int = 224,
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)
    rows = load_classification_rows(classification_csv)
    metadata = build_class_metadata(rows)
    train_set = SatelliteClassificationDataset(
        dataset_root,
        metadata["satellite_to_class"],
        split="train",
        train_fraction=train_fraction,
        seed=seed,
        image_size=image_size,
    )
    val_set = SatelliteClassificationDataset(
        dataset_root,
        metadata["satellite_to_class"],
        split="val",
        train_fraction=train_fraction,
        seed=seed,
        image_size=image_size,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(metadata["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    output_root = ensure_dir(output_dir)
    best_model_path = output_root / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device, None)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": metadata["class_names"],
                    "image_size": image_size,
                },
                best_model_path,
            )
        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    metrics = {
        "best_val_acc": best_val_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
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
