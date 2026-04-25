from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyrender
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from common import ensure_dir, write_csv, write_json, write_text
from data_utils import PoseRecord, SPE3RSatellite, load_camera_config, subsample_records
from geometry_utils import perturb_pose, pose_matrix, quaternion_multiply, rotation_error_degrees, translation_distance


def mask_contour(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, iterations=1, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


@dataclass
class RenderResult:
    color: np.ndarray
    depth: np.ndarray
    mask: np.ndarray


class MeshRenderer:
    def __init__(self, mesh_path: str | Path, camera_config: dict[str, object]):
        raw_mesh = trimesh.load(Path(mesh_path), force="mesh")
        if isinstance(raw_mesh, trimesh.Scene):
            raw_mesh = trimesh.util.concatenate(tuple(raw_mesh.geometry.values()))
        self.mesh = pyrender.Mesh.from_trimesh(raw_mesh, smooth=False)
        self.camera = pyrender.IntrinsicsCamera(
            fx=float(camera_config["cameraMatrix"][0][0]),
            fy=float(camera_config["cameraMatrix"][1][1]),
            cx=float(camera_config["cameraMatrix"][0][2]),
            cy=float(camera_config["cameraMatrix"][1][2]),
        )
        self.scene = pyrender.Scene(bg_color=np.zeros(4), ambient_light=np.asarray([0.35, 0.35, 0.35]))
        self.mesh_node = self.scene.add(self.mesh, pose=np.eye(4))
        self.camera_node = self.scene.add(self.camera, pose=np.eye(4))
        self.light_node = self.scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=np.eye(4))
        self.cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=int(camera_config["Nu"]),
            viewport_height=int(camera_config["Nv"]),
        )

    def render(self, quaternion_xyzw: np.ndarray, translation: np.ndarray) -> RenderResult:
        transform = self.cv_to_gl @ pose_matrix(quaternion_xyzw, translation)
        self.scene.set_pose(self.mesh_node, pose=transform)
        self.scene.set_pose(self.camera_node, pose=np.eye(4))
        self.scene.set_pose(self.light_node, pose=np.eye(4))
        color, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        return RenderResult(color=color[:, :, :3], depth=depth, mask=depth > 0)

    def close(self) -> None:
        self.renderer.delete()


def _resize_rgb(array: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8))
    resized = image.resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _resize_single_channel(array: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(array.astype(np.float32), mode="F")
    resized = image.resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def render_to_feature_stack(render: RenderResult, image_size: int) -> np.ndarray:
    render_rgb = render.color.astype(np.float32) / 255.0
    render_mask = render.mask.astype(np.float32)
    render_contour = mask_contour(render.mask).astype(np.float32)
    depth = render.depth.astype(np.float32)
    if np.any(render.mask):
        valid_depth = depth[render.mask]
        depth_min = float(valid_depth.min())
        depth_max = float(valid_depth.max())
        scale = max(depth_max - depth_min, 1e-6)
        normalized_depth = np.zeros_like(depth, dtype=np.float32)
        normalized_depth[render.mask] = (depth[render.mask] - depth_min) / scale
    else:
        normalized_depth = np.zeros_like(depth, dtype=np.float32)

    return np.concatenate(
        [
            _resize_rgb(render_rgb, image_size),
            _resize_single_channel(render_mask, image_size)[..., None],
            _resize_single_channel(render_contour, image_size)[..., None],
            _resize_single_channel(normalized_depth, image_size)[..., None],
        ],
        axis=-1,
    )


def query_image_to_array(image: Image.Image, image_size: int) -> np.ndarray:
    return np.asarray(image.resize((image_size, image_size), resample=Image.BILINEAR), dtype=np.float32) / 255.0


def build_pair_tensor(query_image: Image.Image, render: RenderResult, image_size: int) -> torch.Tensor:
    query_rgb = query_image_to_array(query_image, image_size=image_size)
    render_stack = render_to_feature_stack(render, image_size=image_size)
    stacked = np.concatenate([query_rgb, render_stack], axis=-1)
    return torch.from_numpy(stacked.transpose(2, 0, 1).astype(np.float32))


class MeshPoseScoringModel(nn.Module):
    def __init__(self, *, input_channels: int = 9, base_width: int = 32, hidden_dim: int = 256):
        super().__init__()
        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]
        layers = []
        in_channels = input_channels
        for index, out_channels in enumerate(widths):
            stride = 2 if index > 0 else 1
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[-1], hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, pair_tensor: torch.Tensor) -> torch.Tensor:
        features = self.encoder(pair_tensor)
        pooled = self.pool(features)
        embedding = self.projection(pooled)
        return self.score_head(embedding).squeeze(-1)


class MeshPosePairDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        query_satellite: SPE3RSatellite,
        query_records: Sequence[PoseRecord],
        mesh_satellite: SPE3RSatellite,
        camera_config: dict[str, object],
        *,
        image_size: int,
        samples_per_epoch: int,
        positive_fraction: float,
        positive_rotation_sigma_deg: float,
        positive_translation_sigma: float,
        negative_rotation_sigma_deg: float,
        negative_translation_sigma: float,
        min_negative_rotation_deg: float,
        candidate_records: Sequence[PoseRecord] | None,
        bank_negative_fraction: float,
        seed: int,
    ):
        self.query_satellite = query_satellite
        self.query_records = list(query_records)
        self.mesh_satellite = mesh_satellite
        self.camera_config = camera_config
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.positive_fraction = positive_fraction
        self.positive_rotation_sigma_deg = positive_rotation_sigma_deg
        self.positive_translation_sigma = positive_translation_sigma
        self.negative_rotation_sigma_deg = negative_rotation_sigma_deg
        self.negative_translation_sigma = negative_translation_sigma
        self.min_negative_rotation_deg = min_negative_rotation_deg
        self.candidate_records = list(candidate_records or [])
        self.bank_negative_fraction = bank_negative_fraction
        self.seed = seed
        self._renderer: MeshRenderer | None = None

    def _get_renderer(self) -> MeshRenderer:
        if self._renderer is None:
            self._renderer = MeshRenderer(self.mesh_satellite.model_path, self.camera_config)
        return self._renderer

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, object]:
        rng = np.random.default_rng(self.seed + index)
        query_record = self.query_records[int(rng.integers(0, len(self.query_records)))]
        query_image = self.query_satellite.load_image(query_record)
        is_positive = bool(rng.random() < self.positive_fraction)

        if is_positive:
            candidate_quaternion, candidate_translation = perturb_pose(
                query_record.quaternion_xyzw,
                query_record.translation,
                rotation_sigma_deg=self.positive_rotation_sigma_deg,
                translation_sigma=self.positive_translation_sigma,
                rng=rng,
            )
        else:
            use_bank_negative = bool(self.candidate_records) and rng.random() < self.bank_negative_fraction
            if use_bank_negative:
                candidate = self.candidate_records[int(rng.integers(0, len(self.candidate_records)))]
                candidate_quaternion = candidate.quaternion_xyzw.astype(np.float64)
                candidate_translation = candidate.translation.astype(np.float64)
            else:
                candidate_quaternion = query_record.quaternion_xyzw.astype(np.float64)
                candidate_translation = query_record.translation.astype(np.float64)
                for _ in range(8):
                    candidate_quaternion, candidate_translation = perturb_pose(
                        query_record.quaternion_xyzw,
                        query_record.translation,
                        rotation_sigma_deg=self.negative_rotation_sigma_deg,
                        translation_sigma=self.negative_translation_sigma,
                        rng=rng,
                    )
                    if rotation_error_degrees(candidate_quaternion, query_record.quaternion_xyzw) >= self.min_negative_rotation_deg:
                        break

        render = self._get_renderer().render(candidate_quaternion, candidate_translation)
        return {
            "pair": build_pair_tensor(query_image, render, self.image_size),
            "label": torch.tensor(1.0 if is_positive else 0.0, dtype=torch.float32),
            "rotation_error_deg": torch.tensor(
                rotation_error_degrees(candidate_quaternion, query_record.quaternion_xyzw),
                dtype=torch.float32,
            ),
            "translation_error": torch.tensor(
                translation_distance(candidate_translation, query_record.translation),
                dtype=torch.float32,
            ),
        }

    def __del__(self) -> None:  # pragma: no cover
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _spherical_to_cartesian(azimuth_deg: float, elevation_deg: float, radius: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    cos_elevation = math.cos(elevation)
    return np.asarray(
        [
            radius * cos_elevation * math.cos(azimuth),
            radius * cos_elevation * math.sin(azimuth),
            radius * math.sin(elevation),
        ],
        dtype=np.float64,
    )


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle_radians: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    return (
        vector * cos_angle
        + np.cross(axis, vector) * sin_angle
        + axis * np.dot(axis, vector) * (1.0 - cos_angle)
    )


def _pose_from_camera_position(camera_position_body: np.ndarray, *, roll_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    camera_position_body = np.asarray(camera_position_body, dtype=np.float64)
    distance = np.linalg.norm(camera_position_body)
    z_axis_body = -camera_position_body / distance
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(up, z_axis_body))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis_body = np.cross(up, z_axis_body)
    x_axis_body /= np.linalg.norm(x_axis_body)
    y_axis_body = np.cross(z_axis_body, x_axis_body)
    if roll_deg != 0.0:
        roll_radians = math.radians(roll_deg)
        x_axis_body = _rotate_about_axis(x_axis_body, z_axis_body, roll_radians)
        y_axis_body = _rotate_about_axis(y_axis_body, z_axis_body, roll_radians)
    rotation = np.stack([x_axis_body, y_axis_body, z_axis_body], axis=0)
    translation = -(rotation @ camera_position_body)
    return _matrix_to_quaternion(rotation), translation.astype(np.float64)


def build_structured_pose_bank(
    candidates: Sequence[PoseRecord],
    *,
    azimuth_bins: int,
    elevation_bins: int,
    roll_bins: int,
    radius_samples: int,
) -> list[PoseRecord]:
    if not candidates:
        return []
    positions = []
    for candidate in candidates:
        rotation = pose_matrix(candidate.quaternion_xyzw, candidate.translation)[:3, :3]
        translation = np.asarray(candidate.translation, dtype=np.float64)
        positions.append(-(rotation.T @ translation))
    positions = np.asarray(positions, dtype=np.float64)
    radii = np.linalg.norm(positions, axis=1)
    azimuth = np.degrees(np.arctan2(positions[:, 1], positions[:, 0]))
    safe_radii = np.where(radii == 0, 1.0, radii)
    elevation = np.degrees(np.arcsin(np.clip(positions[:, 2] / safe_radii, -1.0, 1.0)))
    radius_values = np.unique(np.quantile(radii, np.linspace(0.0, 1.0, num=max(radius_samples, 1))).round(6))
    azimuth_values = np.linspace(-180.0, 180.0, num=max(azimuth_bins, 1), endpoint=False)
    roll_values = np.linspace(-180.0, 180.0, num=max(roll_bins, 1), endpoint=False)
    if elevation_bins <= 1:
        elevation_values = np.asarray([float(np.median(elevation))], dtype=np.float64)
    else:
        margin = 3.0
        elevation_values = np.linspace(float(elevation.min()) + margin, float(elevation.max()) - margin, num=elevation_bins)

    structured = []
    index = 0
    for radius in radius_values:
        for elev in elevation_values:
            for azim in azimuth_values:
                camera_position = _spherical_to_cartesian(float(azim), float(elev), float(radius))
                for roll in roll_values:
                    quaternion_xyzw, translation = _pose_from_camera_position(camera_position, roll_deg=float(roll))
                    structured.append(
                        PoseRecord(
                            filename=f"grid_pose_{index:05d}",
                            quaternion_xyzw=quaternion_xyzw.astype(np.float32),
                            translation=translation.astype(np.float32),
                        )
                    )
                    index += 1
    return structured


@dataclass
class LearnedCandidateScore:
    quaternion_xyzw: np.ndarray
    translation: np.ndarray
    score: float
    source_filename: str | None = None
    render: RenderResult | None = None


def score_pose_candidates(
    model: MeshPoseScoringModel,
    renderer: MeshRenderer,
    query_image: Image.Image,
    candidates: Sequence[PoseRecord],
    *,
    image_size: int,
    device: str,
    batch_size: int = 32,
) -> list[LearnedCandidateScore]:
    query_rgb = query_image_to_array(query_image, image_size=image_size)
    model.eval()
    scored: list[LearnedCandidateScore] = []
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            batch_candidates = candidates[start : start + batch_size]
            pair_tensors = []
            renders = []
            for candidate in batch_candidates:
                render = renderer.render(candidate.quaternion_xyzw, candidate.translation)
                render_stack = render_to_feature_stack(render, image_size=image_size)
                stacked = np.concatenate([query_rgb, render_stack], axis=-1)
                pair_tensors.append(torch.from_numpy(stacked.transpose(2, 0, 1).astype(np.float32)))
                renders.append(render)
            batch_tensor = torch.stack(pair_tensors, dim=0).to(device)
            scores = torch.sigmoid(model(batch_tensor)).cpu().tolist()
            for candidate, render, score in zip(batch_candidates, renders, scores):
                scored.append(
                    LearnedCandidateScore(
                        quaternion_xyzw=np.asarray(candidate.quaternion_xyzw, dtype=np.float64),
                        translation=np.asarray(candidate.translation, dtype=np.float64),
                        score=float(score),
                        source_filename=candidate.filename,
                        render=render,
                    )
                )
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored


def refine_learned_candidates(
    model: MeshPoseScoringModel,
    renderer: MeshRenderer,
    query_image: Image.Image,
    top_candidates: Sequence[LearnedCandidateScore],
    *,
    image_size: int,
    device: str,
    rounds: int,
    samples_per_round: int,
    rotation_sigma_deg: float,
    translation_sigma: float,
    seed: int,
) -> LearnedCandidateScore:
    rng = np.random.default_rng(seed)
    current = list(top_candidates)
    for _ in range(rounds):
        proposals: list[PoseRecord] = []
        for candidate_index, candidate in enumerate(current):
            proposals.append(
                PoseRecord(
                    filename=candidate.source_filename or f"candidate_{candidate_index}",
                    quaternion_xyzw=candidate.quaternion_xyzw.astype(np.float32),
                    translation=candidate.translation.astype(np.float32),
                )
            )
            for sample_index in range(samples_per_round):
                quat, trans = perturb_pose(
                    candidate.quaternion_xyzw,
                    candidate.translation,
                    rotation_sigma_deg=rotation_sigma_deg,
                    translation_sigma=translation_sigma,
                    rng=rng,
                )
                proposals.append(
                    PoseRecord(
                        filename=f"{candidate.source_filename or f'candidate_{candidate_index}'}__refine_{sample_index}",
                        quaternion_xyzw=quat.astype(np.float32),
                        translation=trans.astype(np.float32),
                    )
                )
        current = score_pose_candidates(
            model,
            renderer,
            query_image,
            proposals,
            image_size=image_size,
            device=device,
        )[: max(1, len(top_candidates))]
    return current[0]


def _build_loader(dataset: MeshPosePairDataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _classification_metrics(score_logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    loss = F.binary_cross_entropy_with_logits(score_logits, labels)
    probabilities = torch.sigmoid(score_logits)
    predictions = (probabilities >= 0.5).float()
    accuracy = float((predictions == labels).float().mean().detach().cpu())
    positive_mask = labels > 0.5
    negative_mask = ~positive_mask
    return loss, {
        "accuracy": accuracy,
        "positive_score_mean": float(probabilities[positive_mask].mean().detach().cpu()) if positive_mask.any() else 0.0,
        "negative_score_mean": float(probabilities[negative_mask].mean().detach().cpu()) if negative_mask.any() else 0.0,
    }


def _run_epoch(model: MeshPoseScoringModel, loader: DataLoader, *, device: str, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    totals = {
        "loss": 0.0,
        "accuracy": 0.0,
        "positive_score_mean": 0.0,
        "negative_score_mean": 0.0,
        "candidate_rotation_error_mean_deg": 0.0,
        "candidate_translation_error_mean": 0.0,
    }
    count = 0
    for batch in loader:
        pairs = batch["pair"].to(device)
        labels = batch["label"].to(device)
        score_logits = model(pairs)
        loss, metrics = _classification_metrics(score_logits, labels)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = pairs.shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["accuracy"] += metrics["accuracy"] * batch_size
        totals["positive_score_mean"] += metrics["positive_score_mean"] * batch_size
        totals["negative_score_mean"] += metrics["negative_score_mean"] * batch_size
        totals["candidate_rotation_error_mean_deg"] += float(batch["rotation_error_deg"].mean()) * batch_size
        totals["candidate_translation_error_mean"] += float(batch["translation_error"].mean()) * batch_size
        count += batch_size
    return {key: value / count for key, value in totals.items()}


def save_training_curves(history_rows: list[dict[str, object]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history_rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history_rows], label="train")
    axes[0].plot(epochs, [float(row["eval_loss"]) for row in history_rows], label="eval")
    axes[0].set_title("BCE loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [float(row["train_accuracy"]) for row in history_rows], label="train acc")
    axes[1].plot(epochs, [float(row["eval_accuracy"]) for row in history_rows], label="eval acc")
    axes[1].plot(epochs, [float(row["train_positive_score_mean"]) for row in history_rows], label="train pos")
    axes[1].plot(epochs, [float(row["eval_positive_score_mean"]) for row in history_rows], label="eval pos")
    axes[1].plot(epochs, [float(row["train_negative_score_mean"]) for row in history_rows], label="train neg")
    axes[1].plot(epochs, [float(row["eval_negative_score_mean"]) for row in history_rows], label="eval neg")
    axes[1].set_title("Score separation")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def train_learned_mesh_scorer(
    dataset_root: str | Path,
    *,
    query_satellite: str,
    mesh_satellite: str,
    candidate_satellite: str | None,
    train_split: str,
    eval_split: str,
    max_train_samples: int,
    max_eval_samples: int,
    max_candidate_samples: int,
    samples_per_epoch: int,
    eval_samples: int,
    image_size: int,
    base_width: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    positive_fraction: float,
    positive_rotation_sigma_deg: float,
    positive_translation_sigma: float,
    negative_rotation_sigma_deg: float,
    negative_translation_sigma: float,
    min_negative_rotation_deg: float,
    bank_negative_fraction: float,
    num_workers: int,
    device: str,
    output_dir: str | Path,
    seed: int,
) -> None:
    candidate_satellite_name = candidate_satellite or mesh_satellite
    output_root = ensure_dir(Path(output_dir) / f"{query_satellite}__mesh_{mesh_satellite}__train")
    camera_config = load_camera_config(dataset_root)
    query_sat_obj = SPE3RSatellite(dataset_root, query_satellite)
    mesh_sat_obj = SPE3RSatellite(dataset_root, mesh_satellite)
    candidate_sat_obj = SPE3RSatellite(dataset_root, candidate_satellite_name)

    train_records = subsample_records(query_sat_obj.select_records(train_split), max_train_samples, strategy="even", seed=seed)
    eval_records = subsample_records(query_sat_obj.select_records(eval_split), max_eval_samples, strategy="even", seed=seed)
    candidate_records = subsample_records(candidate_sat_obj.select_records(train_split), max_candidate_samples, strategy="even", seed=seed)

    train_dataset = MeshPosePairDataset(
        query_sat_obj,
        train_records,
        mesh_sat_obj,
        camera_config,
        image_size=image_size,
        samples_per_epoch=samples_per_epoch,
        positive_fraction=positive_fraction,
        positive_rotation_sigma_deg=positive_rotation_sigma_deg,
        positive_translation_sigma=positive_translation_sigma,
        negative_rotation_sigma_deg=negative_rotation_sigma_deg,
        negative_translation_sigma=negative_translation_sigma,
        min_negative_rotation_deg=min_negative_rotation_deg,
        candidate_records=candidate_records,
        bank_negative_fraction=bank_negative_fraction,
        seed=seed,
    )
    eval_dataset = MeshPosePairDataset(
        query_sat_obj,
        eval_records,
        mesh_sat_obj,
        camera_config,
        image_size=image_size,
        samples_per_epoch=eval_samples,
        positive_fraction=positive_fraction,
        positive_rotation_sigma_deg=positive_rotation_sigma_deg,
        positive_translation_sigma=positive_translation_sigma,
        negative_rotation_sigma_deg=negative_rotation_sigma_deg,
        negative_translation_sigma=negative_translation_sigma,
        min_negative_rotation_deg=min_negative_rotation_deg,
        candidate_records=candidate_records,
        bank_negative_fraction=bank_negative_fraction,
        seed=seed + 10000,
    )

    train_loader = _build_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = _build_loader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MeshPoseScoringModel(input_channels=9, base_width=base_width, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history_rows = []
    best_state = None
    best_eval_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        train_summary = _run_epoch(model, train_loader, device=device, optimizer=optimizer)
        eval_summary = _run_epoch(model, eval_loader, device=device, optimizer=None)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_summary['loss']:.6f}",
                "train_accuracy": f"{train_summary['accuracy']:.6f}",
                "train_positive_score_mean": f"{train_summary['positive_score_mean']:.6f}",
                "train_negative_score_mean": f"{train_summary['negative_score_mean']:.6f}",
                "train_candidate_rotation_error_mean_deg": f"{train_summary['candidate_rotation_error_mean_deg']:.6f}",
                "train_candidate_translation_error_mean": f"{train_summary['candidate_translation_error_mean']:.6f}",
                "eval_loss": f"{eval_summary['loss']:.6f}",
                "eval_accuracy": f"{eval_summary['accuracy']:.6f}",
                "eval_positive_score_mean": f"{eval_summary['positive_score_mean']:.6f}",
                "eval_negative_score_mean": f"{eval_summary['negative_score_mean']:.6f}",
                "eval_candidate_rotation_error_mean_deg": f"{eval_summary['candidate_rotation_error_mean_deg']:.6f}",
                "eval_candidate_translation_error_mean": f"{eval_summary['candidate_translation_error_mean']:.6f}",
            }
        )
        if eval_summary["loss"] < best_eval_loss:
            best_eval_loss = eval_summary["loss"]
            best_epoch = epoch
            best_state = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "image_size": image_size,
                    "query_satellite": query_satellite,
                    "mesh_satellite": mesh_satellite,
                    "candidate_satellite": candidate_satellite_name,
                },
                "input_channels": 9,
                "base_width": base_width,
                "hidden_dim": hidden_dim,
            }

    checkpoint_path = output_root / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    write_csv(
        output_root / "history.csv",
        [
            "epoch",
            "train_loss",
            "train_accuracy",
            "train_positive_score_mean",
            "train_negative_score_mean",
            "train_candidate_rotation_error_mean_deg",
            "train_candidate_translation_error_mean",
            "eval_loss",
            "eval_accuracy",
            "eval_positive_score_mean",
            "eval_negative_score_mean",
            "eval_candidate_rotation_error_mean_deg",
            "eval_candidate_translation_error_mean",
        ],
        history_rows,
    )
    save_training_curves(history_rows, output_root / "training_curves.png")
    write_json(
        output_root / "experiment_summary.json",
        {
            "query_satellite": query_satellite,
            "mesh_satellite": mesh_satellite,
            "candidate_satellite": candidate_satellite_name,
            "num_train_records": len(train_records),
            "num_eval_records": len(eval_records),
            "num_candidate_records": len(candidate_records),
            "best_eval_loss": best_eval_loss,
            "best_epoch": best_epoch,
            "checkpoint_path": str(checkpoint_path),
        },
    )


def _overlay_mask_on_image(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy()
    tint = np.asarray([1.0, 0.2, 0.2], dtype=np.float32)
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * tint
    return np.clip(overlay, 0.0, 1.0)


def _save_gallery(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    figure, axes = plt.subplots(len(rows), 4, figsize=(12, 3.3 * len(rows)))
    axes = np.atleast_2d(axes)
    for axis in axes.ravel():
        axis.axis("off")
    for row_index, row in enumerate(rows):
        axes[row_index, 0].imshow(row["query_rgb"])
        axes[row_index, 0].set_title(f"query\n{row['query_filename']}", fontsize=9)
        axes[row_index, 1].imshow(row["render_mask"], cmap="gray")
        axes[row_index, 1].set_title("best render mask", fontsize=9)
        axes[row_index, 2].imshow(row["render_rgb"])
        axes[row_index, 2].set_title(f"best render\nscore={row['score']:.3f}", fontsize=9)
        axes[row_index, 3].imshow(_overlay_mask_on_image(row["query_rgb"], row["render_mask"]))
        axes[row_index, 3].set_title(
            f"overlay\nrot={row['rotation_error_deg']:.1f} | trans={row['translation_error']:.3f}",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def evaluate_learned_mesh_scorer(
    dataset_root: str | Path,
    *,
    checkpoint: str | Path,
    query_satellite: str,
    mesh_satellite: str,
    candidate_satellite: str | None,
    query_split: str,
    candidate_split: str,
    max_query_samples: int,
    max_candidate_samples: int,
    candidate_strategy: str,
    candidate_mode: str,
    grid_azimuth_bins: int,
    grid_elevation_bins: int,
    grid_roll_bins: int,
    grid_radius_samples: int,
    top_k: int,
    refine_rounds: int,
    refine_samples_per_round: int,
    rotation_sigma_deg: float,
    translation_sigma: float,
    batch_size: int,
    num_visualizations: int,
    device: str,
    output_dir: str | Path,
    seed: int,
) -> None:
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    model = MeshPoseScoringModel(
        input_channels=int(checkpoint_payload.get("input_channels", 9)),
        base_width=int(checkpoint_payload.get("base_width", 32)),
        hidden_dim=int(checkpoint_payload.get("hidden_dim", 256)),
    )
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.to(device)
    model.eval()
    image_size = int(checkpoint_payload.get("config", {}).get("image_size", 224))

    candidate_satellite_name = candidate_satellite or mesh_satellite
    output_root = ensure_dir(Path(output_dir) / f"{query_satellite}__mesh_{mesh_satellite}")
    camera_config = load_camera_config(dataset_root)
    query_sat_obj = SPE3RSatellite(dataset_root, query_satellite)
    mesh_sat_obj = SPE3RSatellite(dataset_root, mesh_satellite)
    candidate_sat_obj = SPE3RSatellite(dataset_root, candidate_satellite_name)

    query_records = subsample_records(query_sat_obj.select_records(query_split), max_query_samples, strategy="even", seed=seed)
    dataset_candidates = subsample_records(
        candidate_sat_obj.select_records(candidate_split),
        max_candidate_samples,
        strategy=candidate_strategy,
        seed=seed,
    )
    structured_candidates = build_structured_pose_bank(
        dataset_candidates,
        azimuth_bins=grid_azimuth_bins,
        elevation_bins=grid_elevation_bins,
        roll_bins=grid_roll_bins,
        radius_samples=grid_radius_samples,
    )
    if candidate_mode == "dataset":
        search_candidates = dataset_candidates
    elif candidate_mode == "structured":
        search_candidates = structured_candidates
    else:
        search_candidates = list(dataset_candidates) + structured_candidates

    renderer = MeshRenderer(mesh_sat_obj.model_path, camera_config)
    rotation_errors = []
    translation_errors = []
    scores = []
    retrieval_rows = []
    gallery_rows = []
    try:
        for query_record in query_records:
            query_image = query_sat_obj.load_image(query_record)
            ranked = score_pose_candidates(
                model,
                renderer,
                query_image,
                search_candidates,
                image_size=image_size,
                device=device,
                batch_size=batch_size,
            )
            prediction = refine_learned_candidates(
                model,
                renderer,
                query_image,
                ranked[:top_k],
                image_size=image_size,
                device=device,
                rounds=refine_rounds,
                samples_per_round=refine_samples_per_round,
                rotation_sigma_deg=rotation_sigma_deg,
                translation_sigma=translation_sigma,
                seed=seed + query_record.image_index,
            )
            rotation_error = rotation_error_degrees(prediction.quaternion_xyzw, query_record.quaternion_xyzw)
            translation_error = translation_distance(prediction.translation, query_record.translation)
            rotation_errors.append(rotation_error)
            translation_errors.append(translation_error)
            scores.append(prediction.score)
            retrieval_rows.append(
                {
                    "query_filename": query_record.filename,
                    "best_source_filename": prediction.source_filename or "",
                    "score": f"{prediction.score:.6f}",
                    "rotation_error_deg": f"{rotation_error:.6f}",
                    "translation_error": f"{translation_error:.6f}",
                    "pred_quaternion": " ".join(f"{value:.6f}" for value in prediction.quaternion_xyzw.tolist()),
                    "pred_translation": " ".join(f"{value:.6f}" for value in prediction.translation.tolist()),
                    "target_quaternion": " ".join(f"{value:.6f}" for value in query_record.quaternion_xyzw.tolist()),
                    "target_translation": " ".join(f"{value:.6f}" for value in query_record.translation.tolist()),
                }
            )
            if len(gallery_rows) < num_visualizations and prediction.render is not None:
                gallery_rows.append(
                    {
                        "query_filename": query_record.filename,
                        "query_rgb": np.asarray(query_image, dtype=np.float32) / 255.0,
                        "render_rgb": prediction.render.color.astype(np.float32) / 255.0,
                        "render_mask": prediction.render.mask,
                        "score": prediction.score,
                        "rotation_error_deg": rotation_error,
                        "translation_error": translation_error,
                    }
                )
    finally:
        renderer.close()

    metrics = {
        "query_satellite": query_satellite,
        "mesh_satellite": mesh_satellite,
        "candidate_satellite": candidate_satellite_name,
        "candidate_mode": candidate_mode,
        "num_queries": len(query_records),
        "num_candidates": len(search_candidates),
        "num_dataset_candidates": len(dataset_candidates),
        "num_structured_candidates": len(structured_candidates),
        "rotation_error_mean_deg": float(np.mean(rotation_errors)),
        "rotation_error_median_deg": float(np.median(rotation_errors)),
        "rotation_error_max_deg": float(np.max(rotation_errors)),
        "translation_error_mean": float(np.mean(translation_errors)),
        "translation_error_median": float(np.median(translation_errors)),
        "translation_error_max": float(np.max(translation_errors)),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "checkpoint": str(checkpoint),
    }
    write_csv(
        output_root / "retrievals.csv",
        [
            "query_filename",
            "best_source_filename",
            "score",
            "rotation_error_deg",
            "translation_error",
            "pred_quaternion",
            "pred_translation",
            "target_quaternion",
            "target_translation",
        ],
        retrieval_rows,
    )
    write_json(output_root / "metrics.json", metrics)
    write_text(
        output_root / "summary.md",
        "\n".join(
            [
                "# Learned mesh-conditioned pose scoring",
                "",
                f"- Query satellite: `{query_satellite}`",
                f"- Mesh satellite: `{mesh_satellite}`",
                f"- Candidate satellite: `{candidate_satellite_name}`",
                f"- Candidate mode: `{candidate_mode}`",
                f"- Query images evaluated: {len(query_records)}",
                f"- Candidate bank size: {len(search_candidates)}",
                f"- Rotation mean / median / max (deg): {metrics['rotation_error_mean_deg']:.3f} / {metrics['rotation_error_median_deg']:.3f} / {metrics['rotation_error_max_deg']:.3f}",
                f"- Translation mean / median / max: {metrics['translation_error_mean']:.3f} / {metrics['translation_error_median']:.3f} / {metrics['translation_error_max']:.3f}",
                f"- Mean learned score: {metrics['score_mean']:.3f}",
            ]
        ),
    )
    _save_gallery(gallery_rows, output_root / "qualitative_gallery.png")
