from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

PAPER_TRAIN_RANGES = ((1, 400), (501, 900))
PAPER_VAL_RANGES = ((401, 500), (901, 1000))


@dataclass
class PoseRecord:
    filename: str
    quaternion_xyzw: np.ndarray
    translation: np.ndarray

    @property
    def image_name(self) -> str:
        return f"{self.filename}.jpg"

    @property
    def mask_name(self) -> str:
        return f"{self.filename}.png"

    @property
    def image_index(self) -> int:
        return int(self.filename.replace("img", ""))


def load_camera_config(dataset_root: str | Path) -> dict[str, object]:
    path = Path(dataset_root) / "camera.json"
    return json.loads(path.read_text())


def _index_in_ranges(index: int, ranges: Sequence[tuple[int, int]]) -> bool:
    return any(start <= index <= end for start, end in ranges)


def subsample_records(
    records: Sequence[PoseRecord],
    max_samples: int | None,
    *,
    strategy: str = "even",
    seed: int = 42,
) -> list[PoseRecord]:
    if max_samples is None or max_samples >= len(records):
        return list(records)
    if max_samples <= 0:
        return []
    if strategy == "even":
        indices = np.linspace(0, len(records) - 1, num=max_samples, dtype=int)
        return [records[index] for index in indices]
    if strategy == "random":
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(len(records), size=max_samples, replace=False).tolist())
        return [records[index] for index in indices]
    raise ValueError(f"Unsupported subsample strategy: {strategy}")


class SPE3RSatellite:
    def __init__(self, dataset_root: str | Path, satellite: str):
        self.dataset_root = Path(dataset_root)
        self.satellite = satellite
        self.root = self.dataset_root / satellite
        self.labels_path = self.root / "labels.json"
        self.images_zip_path = self.root / "images.zip"
        self.masks_zip_path = self.root / "masks.zip"
        self.model_path = self.root / "models" / "model_normalized.obj"
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self._images_zip: zipfile.ZipFile | None = None
        self._masks_zip: zipfile.ZipFile | None = None
        self.records = self._load_records()

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_images_zip"] = None
        state["_masks_zip"] = None
        return state

    def _load_records(self) -> list[PoseRecord]:
        payload = json.loads(self.labels_path.read_text())
        return [
            PoseRecord(
                filename=row["filename"],
                quaternion_xyzw=np.asarray(row["q_vbs2tango_true"], dtype=np.float32),
                translation=np.asarray(row["r_Vo2To_vbs_true"], dtype=np.float32),
            )
            for row in payload
        ]

    def select_records(self, split: str = "all") -> list[PoseRecord]:
        if split == "all":
            return list(self.records)
        if split == "paper_train":
            return [record for record in self.records if _index_in_ranges(record.image_index, PAPER_TRAIN_RANGES)]
        if split == "paper_val":
            return [record for record in self.records if _index_in_ranges(record.image_index, PAPER_VAL_RANGES)]
        if split == "paper_trainval":
            return list(self.records)
        raise ValueError(f"Unsupported split: {split}")

    def _get_images_zip(self) -> zipfile.ZipFile:
        if self._images_zip is None:
            self._images_zip = zipfile.ZipFile(self.images_zip_path)
        return self._images_zip

    def _get_masks_zip(self) -> zipfile.ZipFile:
        if self._masks_zip is None:
            self._masks_zip = zipfile.ZipFile(self.masks_zip_path)
        return self._masks_zip

    def load_image(self, record: PoseRecord | str) -> Image.Image:
        image_name = record.image_name if isinstance(record, PoseRecord) else record
        extracted = self.images_dir / image_name
        if extracted.exists():
            return Image.open(extracted).convert("RGB")
        with self._get_images_zip().open(image_name) as handle:
            return Image.open(io.BytesIO(handle.read())).convert("RGB")

    def load_mask(self, record: PoseRecord | str) -> Image.Image:
        mask_name = record.mask_name if isinstance(record, PoseRecord) else record
        extracted = self.masks_dir / mask_name
        if extracted.exists():
            return Image.open(extracted).convert("L")
        with self._get_masks_zip().open(mask_name) as handle:
            return Image.open(io.BytesIO(handle.read())).convert("L")

