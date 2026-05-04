from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset, get_worker_info

from common import ensure_dir, write_csv, write_json, write_text
from data_utils import PoseRecord, SPE3RSatellite, load_camera_config, subsample_records
from geometry_utils import (
    camera_positions_in_body_frame,
    cartesian_to_spherical,
    fold_halfturn_rotation_error_degrees,
    matrix_to_quaternion,
    pose_from_camera_position,
    quaternion_inverse,
    quaternion_multiply,
    rotation_error_degrees,
    spherical_to_cartesian,
    symmetry_group_rotation_error_degrees,
    translation_distance,
)
from models import MeshPoseScoringModel
from overfit.geometry_search import MeshRenderer, make_observation, mask_contour, rank_candidates


@dataclass
class RefinedCandidate:
    quaternion_xyzw: np.ndarray
    translation: np.ndarray
    score: float
    source_filename: str | None = None
    render_rgb: np.ndarray | None = None
    render_mask: np.ndarray | None = None


def crop_box_from_masks(mask_a: np.ndarray, mask_b: np.ndarray, *, padding: int = 12) -> tuple[int, int, int, int]:
    union = np.logical_or(mask_a, mask_b)
    if not np.any(union):
        height, width = union.shape
        return 0, height, 0, width
    ys, xs = np.nonzero(union)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(union.shape[0], int(ys.max()) + padding + 1)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(union.shape[1], int(xs.max()) + padding + 1)
    return y0, y1, x0, x1


def _resize_rgb(array: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8))
    resized = image.resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _resize_single_channel(array: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(array.astype(np.float32), mode="F")
    resized = image.resize((image_size, image_size), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def build_cropped_pair_tensor(
    query_rgb: np.ndarray,
    query_mask: np.ndarray,
    render_rgb_u8: np.ndarray,
    render_mask: np.ndarray,
    *,
    image_size: int,
    crop_padding: int,
) -> torch.Tensor:
    render_rgb = render_rgb_u8.astype(np.float32) / 255.0
    render_mask = render_mask.astype(bool)
    render_contour = mask_contour(render_mask).astype(np.float32)
    y0, y1, x0, x1 = crop_box_from_masks(query_mask, render_mask, padding=crop_padding)

    query_rgb_crop = _resize_rgb(query_rgb[y0:y1, x0:x1], image_size)
    query_mask_crop = _resize_single_channel(query_mask[y0:y1, x0:x1].astype(np.float32), image_size)[..., None]
    render_rgb_crop = _resize_rgb(render_rgb[y0:y1, x0:x1], image_size)
    render_mask_crop = _resize_single_channel(render_mask[y0:y1, x0:x1].astype(np.float32), image_size)[..., None]
    render_contour_crop = _resize_single_channel(render_contour[y0:y1, x0:x1], image_size)[..., None]

    stacked = np.concatenate(
        [
            query_rgb_crop,
            query_mask_crop,
            render_rgb_crop,
            render_mask_crop,
            render_contour_crop,
        ],
        axis=-1,
    )
    return torch.from_numpy(stacked.transpose(2, 0, 1).astype(np.float32))


def apply_delta_quaternion_torch(base_quaternion_xyzw: torch.Tensor, delta_quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = F.normalize(delta_quaternion_xyzw, dim=-1).unbind(dim=-1)
    bx, by, bz, bw = F.normalize(base_quaternion_xyzw, dim=-1).unbind(dim=-1)
    updated = torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )
    return F.normalize(updated, dim=-1)


def summarize_pose_errors(
    pred_quaternions: torch.Tensor,
    pred_translations: torch.Tensor,
    target_quaternions: torch.Tensor,
    target_translations: torch.Tensor,
) -> dict[str, float]:
    pred = F.normalize(pred_quaternions, dim=-1)
    target = F.normalize(target_quaternions, dim=-1)
    dots = (pred * target).sum(dim=-1).abs().clamp(0.0, 1.0)
    rotation_errors = 2.0 * torch.rad2deg(torch.acos(dots))
    translation_errors = torch.linalg.norm(pred_translations - target_translations, dim=-1)
    return {
        "rotation_mean_deg": float(rotation_errors.mean().cpu()),
        "translation_mean": float(translation_errors.mean().cpu()),
    }


def load_obj_vertices(path: str | Path) -> np.ndarray:
    vertices = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            _, x_value, y_value, z_value, *_ = line.split()
            vertices.append((float(x_value), float(y_value), float(z_value)))
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(vertices, dtype=np.float32)


def _center_scale_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
    aligned = centered @ right_singular_vectors.T
    radius = np.linalg.norm(aligned, axis=1).max()
    if radius > 0:
        aligned = aligned / radius
    return aligned, right_singular_vectors


def build_mesh_rotation_symmetry_group(
    mesh_path: str | Path,
    *,
    threshold: float,
    max_points: int = 5000,
    seed: int = 42,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    points = load_obj_vertices(mesh_path)
    if len(points) > max_points:
        selection = rng.choice(len(points), size=max_points, replace=False)
        points = points[selection]
    aligned, pca_basis = _center_scale_pca(points)
    tree = cKDTree(aligned)

    aligned_rotations = {
        "x": np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
        "y": np.asarray([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64),
        "z": np.asarray([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    }

    generator_quaternions: list[np.ndarray] = []
    generator_axes: list[str] = []
    generator_scores: dict[str, float] = {}
    for axis_name, aligned_rotation in aligned_rotations.items():
        rotated = aligned @ aligned_rotation.T
        distances, _ = tree.query(rotated, k=1)
        score = float(distances.mean())
        generator_scores[axis_name] = score
        if score <= threshold:
            body_rotation = pca_basis.T @ aligned_rotation @ pca_basis
            generator_quaternions.append(matrix_to_quaternion(body_rotation).astype(np.float32))
            generator_axes.append(axis_name)

    identity = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    group = [identity]
    labels = ["identity"]
    queue = [(identity, "identity")]
    eps = 1e-4

    while queue:
        current, current_label = queue.pop(0)
        for generator, axis_name in zip(generator_quaternions, generator_axes):
            candidate = quaternion_multiply(current, generator).astype(np.float32)
            exists = any(abs(float(np.dot(existing, candidate))) >= 1.0 - eps for existing in group)
            if exists:
                continue
            label = f"{current_label}*rot180_{axis_name}"
            group.append(candidate)
            labels.append(label)
            queue.append((candidate, label))

    return {
        "threshold": threshold,
        "generator_axes": generator_axes,
        "generator_scores": generator_scores,
        "quaternions_xyzw": group,
        "labels": labels,
    }


def shortlist_soft_targets(rotation_errors_deg: np.ndarray, *, temperature_deg: float) -> np.ndarray:
    scaled = -np.asarray(rotation_errors_deg, dtype=np.float64) / max(temperature_deg, 1e-6)
    scaled -= scaled.max()
    weights = np.exp(scaled)
    return weights / weights.sum()


def relative_delta_quaternion(seed_quaternion_xyzw: np.ndarray, target_quaternion_xyzw: np.ndarray) -> np.ndarray:
    return quaternion_multiply(target_quaternion_xyzw, quaternion_inverse(seed_quaternion_xyzw)).astype(np.float32)


def fold_halfturn_rotation_error_torch(rotation_errors_deg: torch.Tensor) -> torch.Tensor:
    return torch.minimum(rotation_errors_deg, 180.0 - rotation_errors_deg)


def quaternion_multiply_torch(quaternion_a: torch.Tensor, quaternion_b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = F.normalize(quaternion_a, dim=-1).unbind(dim=-1)
    bx, by, bz, bw = F.normalize(quaternion_b, dim=-1).unbind(dim=-1)
    return F.normalize(
        torch.stack(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dim=-1,
        ),
        dim=-1,
    )


def symmetry_group_rotation_errors_torch(
    predicted_quaternion_xyzw: torch.Tensor,
    target_quaternion_xyzw: torch.Tensor,
    symmetry_group_quaternions_xyzw: torch.Tensor | None,
) -> torch.Tensor:
    predicted = F.normalize(predicted_quaternion_xyzw, dim=-1)
    target = F.normalize(target_quaternion_xyzw, dim=-1)
    if symmetry_group_quaternions_xyzw is None or symmetry_group_quaternions_xyzw.numel() == 0:
        dots = (predicted * target).sum(dim=-1).abs().clamp(0.0, 1.0)
        return 2.0 * torch.rad2deg(torch.acos(dots))

    errors = []
    for symmetry_quaternion in symmetry_group_quaternions_xyzw:
        expanded = symmetry_quaternion.view(*([1] * (target.dim() - 1)), 4).expand_as(target)
        equivalent_target = quaternion_multiply_torch(target, expanded)
        dots = (predicted * equivalent_target).sum(dim=-1).abs().clamp(0.0, 1.0)
        errors.append(2.0 * torch.rad2deg(torch.acos(dots)))
    return torch.stack(errors, dim=-1).min(dim=-1).values


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

    camera_positions = camera_positions_in_body_frame(
        [candidate.quaternion_xyzw for candidate in candidates],
        [candidate.translation for candidate in candidates],
    )
    azimuth, elevation, radius = cartesian_to_spherical(camera_positions)
    radius_values = np.quantile(radius, np.linspace(0.0, 1.0, num=max(radius_samples, 1)))
    radius_values = np.unique(radius_values.round(decimals=6))
    roll_values = np.linspace(-180.0, 180.0, num=max(roll_bins, 1), endpoint=False)
    azimuth_values = np.linspace(-180.0, 180.0, num=max(azimuth_bins, 1), endpoint=False)
    if elevation_bins <= 1:
        elevation_values = np.asarray([float(np.median(elevation))], dtype=np.float64)
    else:
        elevation_margin = 3.0
        elevation_values = np.linspace(
            float(elevation.min()) + elevation_margin,
            float(elevation.max()) - elevation_margin,
            num=elevation_bins,
        )

    structured: list[PoseRecord] = []
    index = 0
    for radius_value in radius_values:
        for elevation_value in elevation_values:
            for azimuth_value in azimuth_values:
                camera_position = spherical_to_cartesian(float(azimuth_value), float(elevation_value), float(radius_value))
                for roll_value in roll_values:
                    quaternion_xyzw, translation = pose_from_camera_position(camera_position, roll_deg=float(roll_value))
                    structured.append(
                        PoseRecord(
                            filename=f"grid_pose_{index:05d}",
                            quaternion_xyzw=quaternion_xyzw.astype(np.float32),
                            translation=translation.astype(np.float32),
                        )
                    )
                    index += 1
    return structured


def build_coarse_candidate_bank(
    base_candidates: Sequence[PoseRecord],
    *,
    use_dataset_bank: bool,
    use_structured_bank: bool,
    grid_azimuth_bins: int,
    grid_elevation_bins: int,
    grid_roll_bins: int,
    grid_radius_samples: int,
) -> list[PoseRecord]:
    candidates: list[PoseRecord] = []
    if use_dataset_bank:
        candidates.extend(base_candidates)
    if use_structured_bank:
        candidates.extend(
            build_structured_pose_bank(
                base_candidates,
                azimuth_bins=grid_azimuth_bins,
                elevation_bins=grid_elevation_bins,
                roll_bins=grid_roll_bins,
                radius_samples=grid_radius_samples,
            )
        )
    return candidates


def _log(message: str) -> None:
    print(f"[benchmark_refiner] {message}", flush=True)


class ShortlistRefinerDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        query_satellite: SPE3RSatellite,
        query_records: Sequence[PoseRecord],
        mesh_satellite: SPE3RSatellite,
        camera_config: dict[str, object],
        coarse_candidates: Sequence[PoseRecord],
        *,
        image_size: int = 224,
        crop_padding: int = 12,
        shortlist_size: int = 16,
        close_pool_size: int = 32,
        score_temperature_deg: float = 20.0,
        refine_top_m: int = 4,
        fold_halfturn_symmetry_targets: bool = False,
        mesh_symmetry_group_quaternions_xyzw: Sequence[np.ndarray] | None = None,
        samples_per_epoch: int = 2048,
        seed: int = 42,
        enable_debug_logging: bool = False,
        debug_log_first_n_samples: int = 0,
        debug_name: str = "dataset",
    ):
        self.query_satellite = query_satellite
        self.query_records = list(query_records)
        self.mesh_satellite = mesh_satellite
        self.camera_config = camera_config
        self.coarse_candidates = list(coarse_candidates)
        self.image_size = image_size
        self.crop_padding = crop_padding
        self.shortlist_size = shortlist_size
        self.close_pool_size = close_pool_size
        self.score_temperature_deg = score_temperature_deg
        self.refine_top_m = refine_top_m
        self.fold_halfturn_symmetry_targets = fold_halfturn_symmetry_targets
        self.mesh_symmetry_group_quaternions_xyzw = [
            np.asarray(quaternion_xyzw, dtype=np.float32)
            for quaternion_xyzw in (mesh_symmetry_group_quaternions_xyzw or [])
        ]
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.enable_debug_logging = enable_debug_logging
        self.debug_log_first_n_samples = debug_log_first_n_samples
        self.debug_name = debug_name
        self._renderer: MeshRenderer | None = None
        self._debug_logged_samples = 0

    def _dataset_log(self, message: str) -> None:
        if not self.enable_debug_logging:
            return
        worker = get_worker_info()
        worker_suffix = f" worker={worker.id}" if worker is not None else " worker=main"
        print(f"[benchmark_refiner:{self.debug_name}{worker_suffix}] {message}", flush=True)

    def _get_renderer(self) -> MeshRenderer:
        if self._renderer is None:
            self._dataset_log(
                "creating renderer "
                f"mesh={self.mesh_satellite.model_path} viewport="
                f"{self.camera_config['Nu']}x{self.camera_config['Nv']}"
            )
            self._renderer = MeshRenderer(self.mesh_satellite.model_path, self.camera_config)
            self._dataset_log("renderer created successfully")
        return self._renderer

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_shortlist(self, query_image: Image.Image, query_mask_image: Image.Image) -> list[PoseRecord]:
        observation = make_observation(query_image, query_mask_image)
        renderer = self._get_renderer()
        coarse_ranked = rank_candidates(
            renderer,
            observation,
            self.coarse_candidates,
            top_k=max(self.shortlist_size, self.close_pool_size),
            crop_padding=self.crop_padding,
        )
        shortlist = coarse_ranked[: self.shortlist_size]
        return [
            PoseRecord(
                filename=item.source_filename or f"candidate_{idx}",
                quaternion_xyzw=item.quaternion_xyzw.astype(np.float32),
                translation=item.translation.astype(np.float32),
            )
            for idx, item in enumerate(shortlist)
        ]

    def __getitem__(self, index: int) -> dict[str, object]:
        should_log_sample = self.enable_debug_logging and self._debug_logged_samples < self.debug_log_first_n_samples
        try:
            rng = np.random.default_rng(self.seed + index)
            query_record = self.query_records[int(rng.integers(0, len(self.query_records)))]
            if should_log_sample:
                self._dataset_log(f"getitem start index={index} query={query_record.filename}")
            query_image = self.query_satellite.load_image(query_record)
            query_mask_image = self.query_satellite.load_mask(query_record)
            query_mask = np.asarray(query_mask_image, dtype=np.uint8) > 127
            query_rgb = np.asarray(query_image, dtype=np.float32) / 255.0
            if should_log_sample:
                self._dataset_log(
                    "loaded query image "
                    f"shape={query_rgb.shape} mask_pixels={int(query_mask.sum())}"
                )
            renderer = self._get_renderer()

            if should_log_sample:
                self._dataset_log("building shortlist from geometry")
            shortlist = self._sample_shortlist(query_image, query_mask_image)
            if should_log_sample:
                self._dataset_log(f"shortlist ready size={len(shortlist)}")
            pair_tensors = []
            seed_quaternions = []
            seed_translations = []
            rotation_errors = []
            symmetry_rotation_errors = []
            delta_quaternions = []

            for candidate_index, candidate in enumerate(shortlist):
                if should_log_sample and candidate_index == 0:
                    self._dataset_log(
                        "rendering first shortlist candidate "
                        f"source={candidate.filename}"
                    )
                render_rgb_u8, render_mask = renderer.render(candidate.quaternion_xyzw, candidate.translation)
                pair_tensors.append(
                    build_cropped_pair_tensor(
                        query_rgb,
                        query_mask,
                        render_rgb_u8,
                        render_mask,
                        image_size=self.image_size,
                        crop_padding=self.crop_padding,
                    )
                )
                seed_quaternions.append(np.asarray(candidate.quaternion_xyzw, dtype=np.float32))
                seed_translations.append(np.asarray(candidate.translation, dtype=np.float32))
                rotation_errors.append(rotation_error_degrees(candidate.quaternion_xyzw, query_record.quaternion_xyzw))
                symmetry_error = symmetry_group_rotation_error_degrees(
                    candidate.quaternion_xyzw,
                    query_record.quaternion_xyzw,
                    self.mesh_symmetry_group_quaternions_xyzw,
                )
                symmetry_rotation_errors.append(symmetry_error)
                if self.mesh_symmetry_group_quaternions_xyzw:
                    equivalent_targets = [
                        quaternion_multiply(query_record.quaternion_xyzw, symmetry_quaternion)
                        for symmetry_quaternion in self.mesh_symmetry_group_quaternions_xyzw
                    ]
                    best_target = min(
                        equivalent_targets,
                        key=lambda equivalent: rotation_error_degrees(candidate.quaternion_xyzw, equivalent),
                    )
                else:
                    best_target = query_record.quaternion_xyzw
                delta_quaternions.append(relative_delta_quaternion(candidate.quaternion_xyzw, best_target))

            rotation_errors_np = np.asarray(rotation_errors, dtype=np.float32)
            symmetry_rotation_errors_np = np.asarray(symmetry_rotation_errors, dtype=np.float32)
            folded_rotation_errors_np = np.asarray(
                [fold_halfturn_rotation_error_degrees(error) for error in rotation_errors_np],
                dtype=np.float32,
            )
            supervision_rotation_errors = (
                symmetry_rotation_errors_np if self.mesh_symmetry_group_quaternions_xyzw else (
                    folded_rotation_errors_np if self.fold_halfturn_symmetry_targets else rotation_errors_np
                )
            )
            soft_targets = shortlist_soft_targets(
                supervision_rotation_errors,
                temperature_deg=self.score_temperature_deg,
            ).astype(np.float32)
            refine_order = np.argsort(supervision_rotation_errors)
            refine_weights = np.zeros_like(rotation_errors_np, dtype=np.float32)
            refine_weights[refine_order[: min(self.refine_top_m, len(refine_order))]] = 1.0
            refine_weights_sum = float(refine_weights.sum())
            if refine_weights_sum > 0:
                refine_weights /= refine_weights_sum

            output = {
                "pairs": torch.stack(pair_tensors, dim=0),
                "seed_quaternions": torch.from_numpy(np.stack(seed_quaternions, axis=0)),
                "seed_translations": torch.from_numpy(np.stack(seed_translations, axis=0)),
                "target_quaternion": torch.from_numpy(np.asarray(query_record.quaternion_xyzw, dtype=np.float32)),
                "target_translation": torch.from_numpy(np.asarray(query_record.translation, dtype=np.float32)),
                "target_delta_quaternions": torch.from_numpy(np.stack(delta_quaternions, axis=0)),
                "score_targets": torch.from_numpy(soft_targets),
                "refine_weights": torch.from_numpy(refine_weights),
                "candidate_rotation_errors_deg": torch.from_numpy(rotation_errors_np),
                "candidate_folded_rotation_errors_deg": torch.from_numpy(folded_rotation_errors_np),
                "candidate_mesh_symmetry_rotation_errors_deg": torch.from_numpy(symmetry_rotation_errors_np),
                "filename": query_record.filename,
            }
            if should_log_sample:
                self._dataset_log(
                    "getitem success "
                    f"index={index} pairs_shape={tuple(output['pairs'].shape)} "
                    f"rot_min={float(rotation_errors_np.min()):.2f} "
                    f"rot_max={float(rotation_errors_np.max()):.2f}"
                )
                self._debug_logged_samples += 1
            return output
        except Exception as exc:
            self._dataset_log(f"getitem failure index={index} error={type(exc).__name__}: {exc}")
            raise

    def __del__(self) -> None:  # pragma: no cover
        self.close()

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception as exc:  # pragma: no cover
                self._dataset_log(f"renderer close warning: {type(exc).__name__}: {exc}")
            self._renderer = None


def shortlist_refinement_loss(
    score_logits: torch.Tensor,
    predicted_refined_quaternions: torch.Tensor,
    target_quaternions: torch.Tensor,
    score_targets: torch.Tensor,
    refine_weights: torch.Tensor,
    *,
    score_weight: float,
    refine_weight: float,
    fold_halfturn_symmetry: bool = False,
    mesh_symmetry_group_quaternions_xyzw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    log_probs = F.log_softmax(score_logits, dim=-1)
    score_loss = -(score_targets * log_probs).sum(dim=-1).mean()

    if mesh_symmetry_group_quaternions_xyzw is not None and mesh_symmetry_group_quaternions_xyzw.numel() > 0:
        rotation_errors = symmetry_group_rotation_errors_torch(
            predicted_refined_quaternions,
            target_quaternions,
            mesh_symmetry_group_quaternions_xyzw,
        )
    else:
        pred = F.normalize(predicted_refined_quaternions, dim=-1)
        target = F.normalize(target_quaternions, dim=-1)
        dots = (pred * target).sum(dim=-1).abs().clamp(0.0, 1.0)
        rotation_errors = 2.0 * torch.rad2deg(torch.acos(dots))
    if fold_halfturn_symmetry and (mesh_symmetry_group_quaternions_xyzw is None or mesh_symmetry_group_quaternions_xyzw.numel() == 0):
        rotation_errors = fold_halfturn_rotation_error_torch(rotation_errors)
    weighted_rotation = (rotation_errors * refine_weights).sum(dim=-1).mean() / 180.0

    total = score_weight * score_loss + refine_weight * weighted_rotation
    return total, {
        "score_loss": float(score_loss.detach().cpu()),
        "refine_loss": float(weighted_rotation.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }


def flatten_shortlist_batch(batch: dict[str, torch.Tensor], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs = batch["pairs"].to(device)
    batch_size, shortlist_size = pairs.shape[:2]
    flat_pairs = pairs.view(batch_size * shortlist_size, *pairs.shape[2:])
    seed_quaternions = batch["seed_quaternions"].to(device).view(batch_size * shortlist_size, -1)
    seed_translations = batch["seed_translations"].to(device).view(batch_size * shortlist_size, -1)
    return flat_pairs, seed_quaternions, seed_translations


def _build_loader(dataset: ShortlistRefinerDataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _run_epoch(
    model: MeshPoseScoringModel,
    loader: DataLoader,
    *,
    device: str,
    score_weight: float,
    refine_weight: float,
    fold_halfturn_symmetry_loss: bool,
    mesh_symmetry_group_tensor: torch.Tensor | None,
    epoch_name: str,
    debug_log_first_n_batches: int,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    totals = {
        "loss": 0.0,
        "score_loss": 0.0,
        "refine_loss": 0.0,
        "score_accuracy": 0.0,
        "candidate_rotation_mean_deg": 0.0,
        "refined_rotation_mean_deg": 0.0,
        "refined_rotation_folded_mean_deg": 0.0,
    }
    total_count = 0

    for batch_index, batch in enumerate(loader):
        if batch_index < debug_log_first_n_batches:
            _log(
                f"{epoch_name} batch_start index={batch_index} "
                f"pairs_shape={tuple(batch['pairs'].shape)}"
            )
        batch_size, shortlist_size = batch["pairs"].shape[:2]
        flat_pairs, seed_quaternions, seed_translations = flatten_shortlist_batch(batch, device)
        if batch_index < debug_log_first_n_batches:
            _log(
                f"{epoch_name} batch_flattened index={batch_index} "
                f"flat_pairs_shape={tuple(flat_pairs.shape)}"
            )
        score_logits, refinement = model(flat_pairs)
        assert refinement is not None
        delta_quaternions, _ = refinement
        score_logits = score_logits.view(batch_size, shortlist_size)
        delta_quaternions = delta_quaternions.view(batch_size, shortlist_size, 4)
        refined_quaternions = apply_delta_quaternion_torch(
            seed_quaternions.view(batch_size, shortlist_size, 4),
            delta_quaternions,
        )
        loss, terms = shortlist_refinement_loss(
            score_logits,
            refined_quaternions,
            batch["target_quaternion"].to(device).unsqueeze(1).expand_as(refined_quaternions),
            batch["score_targets"].to(device),
            batch["refine_weights"].to(device),
            score_weight=score_weight,
            refine_weight=refine_weight,
            fold_halfturn_symmetry=fold_halfturn_symmetry_loss,
            mesh_symmetry_group_quaternions_xyzw=mesh_symmetry_group_tensor,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            weights = batch["refine_weights"].to(device)
            target_quaternion = batch["target_quaternion"].to(device).unsqueeze(1).expand_as(refined_quaternions)
            target_translation = batch["target_translation"].to(device).unsqueeze(1).expand_as(
                seed_translations.view(batch_size, shortlist_size, 3)
            )
            refined_summary = summarize_pose_errors(
                refined_quaternions.reshape(-1, 4),
                seed_translations.view(batch_size, shortlist_size, 3).reshape(-1, 3),
                target_quaternion.reshape(-1, 4),
                target_translation.reshape(-1, 3),
            )
            dots = (refined_quaternions * target_quaternion).sum(dim=-1).abs().clamp(0.0, 1.0)
            refined_rotation_errors = 2.0 * torch.rad2deg(torch.acos(dots))
            refined_rotation_folded = fold_halfturn_rotation_error_torch(refined_rotation_errors)
            score_pred = score_logits.argmax(dim=-1)
            score_target = batch["score_targets"].to(device).argmax(dim=-1)
            score_accuracy = float((score_pred == score_target).float().mean().cpu())
            candidate_rotation_source = (
                batch["candidate_mesh_symmetry_rotation_errors_deg"].to(device)
                if mesh_symmetry_group_tensor is not None
                else (
                    batch["candidate_folded_rotation_errors_deg"].to(device)
                    if fold_halfturn_symmetry_loss
                    else batch["candidate_rotation_errors_deg"].to(device)
                )
            )
            candidate_rotation = (candidate_rotation_source * weights).sum(dim=-1).mean()

        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["score_loss"] += terms["score_loss"] * batch_size
        totals["refine_loss"] += terms["refine_loss"] * batch_size
        totals["score_accuracy"] += score_accuracy * batch_size
        totals["candidate_rotation_mean_deg"] += float(candidate_rotation.cpu()) * batch_size
        totals["refined_rotation_mean_deg"] += refined_summary["rotation_mean_deg"] * batch_size
        totals["refined_rotation_folded_mean_deg"] += float(refined_rotation_folded.mean().cpu()) * batch_size
        total_count += batch_size
        if batch_index < debug_log_first_n_batches:
            _log(
                f"{epoch_name} batch_done index={batch_index} "
                f"loss={float(loss.detach().cpu()):.4f} "
                f"score_acc={score_accuracy:.4f}"
            )

    return {key: value / total_count for key, value in totals.items()}


def _save_history_plot(history_rows: list[dict[str, object]], output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history_rows]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in history_rows], label="train")
    axes[0].plot(epochs, [float(row["eval_loss"]) for row in history_rows], label="eval")
    axes[0].set_title("Total loss")
    axes[0].legend()
    axes[1].plot(epochs, [float(row["train_score_accuracy"]) for row in history_rows], label="train")
    axes[1].plot(epochs, [float(row["eval_score_accuracy"]) for row in history_rows], label="eval")
    axes[1].set_title("Shortlist accuracy")
    axes[1].legend()
    axes[2].plot(epochs, [float(row["train_refined_rotation_mean_deg"]) for row in history_rows], label="train")
    axes[2].plot(epochs, [float(row["eval_refined_rotation_mean_deg"]) for row in history_rows], label="eval")
    axes[2].plot(epochs, [float(row["train_refined_rotation_folded_mean_deg"]) for row in history_rows], label="train folded")
    axes[2].plot(epochs, [float(row["eval_refined_rotation_folded_mean_deg"]) for row in history_rows], label="eval folded")
    axes[2].set_title("Refined rotation error")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def train_benchmark_refiner(
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
    crop_padding: int,
    shortlist_size: int,
    close_pool_size: int,
    score_temperature_deg: float,
    refine_top_m: int,
    base_width: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    score_weight: float,
    refine_weight: float,
    fold_halfturn_symmetry_loss: bool,
    use_mesh_symmetry_group: bool,
    mesh_symmetry_threshold: float,
    mesh_symmetry_max_points: int,
    use_dataset_bank: bool,
    use_structured_bank: bool,
    grid_azimuth_bins: int,
    grid_elevation_bins: int,
    grid_roll_bins: int,
    grid_radius_samples: int,
    num_workers: int,
    device: str,
    output_dir: str | Path,
    seed: int,
    debug_log_first_n_samples: int = 2,
    debug_log_first_n_batches: int = 2,
    debug_render_smoke_test: bool = False,
) -> None:
    _log(
        f"startup dataset_root={dataset_root} device={device} "
        f"cuda_available={torch.cuda.is_available()}"
    )
    candidate_satellite_name = candidate_satellite or mesh_satellite
    output_root = ensure_dir(Path(output_dir) / f"{query_satellite}__mesh_{mesh_satellite}__train")
    _log(f"output_dir={output_root}")

    camera_config = load_camera_config(dataset_root)
    query_ds = SPE3RSatellite(dataset_root, query_satellite)
    mesh_ds = SPE3RSatellite(dataset_root, mesh_satellite)
    candidate_ds = SPE3RSatellite(dataset_root, candidate_satellite_name)
    _log(
        f"satellites query={query_satellite} mesh={mesh_satellite} "
        f"candidate={candidate_satellite_name}"
    )
    mesh_symmetry_group = None
    mesh_symmetry_group_tensor = None
    if use_mesh_symmetry_group:
        _log("building mesh symmetry group")
        mesh_symmetry_group = build_mesh_rotation_symmetry_group(
            query_ds.model_path,
            threshold=mesh_symmetry_threshold,
            max_points=mesh_symmetry_max_points,
            seed=seed,
        )
        _log(f"mesh symmetry group ready size={len(mesh_symmetry_group['quaternions_xyzw'])}")
        mesh_symmetry_group_tensor = torch.as_tensor(
            np.asarray(mesh_symmetry_group["quaternions_xyzw"], dtype=np.float32),
            device=device,
        )

    train_records = subsample_records(query_ds.select_records(train_split), max_train_samples, strategy="even", seed=seed)
    eval_records = subsample_records(query_ds.select_records(eval_split), max_eval_samples, strategy="even", seed=seed)
    base_candidates = subsample_records(candidate_ds.select_records(train_split), max_candidate_samples, strategy="even", seed=seed)
    _log(
        f"record_counts train={len(train_records)} eval={len(eval_records)} "
        f"base_candidates={len(base_candidates)}"
    )
    coarse_candidates = build_coarse_candidate_bank(
        base_candidates,
        use_dataset_bank=use_dataset_bank,
        use_structured_bank=use_structured_bank,
        grid_azimuth_bins=grid_azimuth_bins,
        grid_elevation_bins=grid_elevation_bins,
        grid_roll_bins=grid_roll_bins,
        grid_radius_samples=grid_radius_samples,
    )
    _log(
        f"coarse_candidate_bank size={len(coarse_candidates)} "
        f"use_dataset_bank={use_dataset_bank} use_structured_bank={use_structured_bank}"
    )

    train_dataset = ShortlistRefinerDataset(
        query_ds,
        train_records,
        mesh_ds,
        camera_config,
        coarse_candidates,
        image_size=image_size,
        crop_padding=crop_padding,
        shortlist_size=shortlist_size,
        close_pool_size=close_pool_size,
        score_temperature_deg=score_temperature_deg,
        refine_top_m=refine_top_m,
        fold_halfturn_symmetry_targets=fold_halfturn_symmetry_loss,
        mesh_symmetry_group_quaternions_xyzw=(mesh_symmetry_group or {}).get("quaternions_xyzw"),
        samples_per_epoch=samples_per_epoch,
        seed=seed,
        enable_debug_logging=debug_log_first_n_samples > 0,
        debug_log_first_n_samples=debug_log_first_n_samples,
        debug_name="train",
    )
    eval_dataset = ShortlistRefinerDataset(
        query_ds,
        eval_records,
        mesh_ds,
        camera_config,
        coarse_candidates,
        image_size=image_size,
        crop_padding=crop_padding,
        shortlist_size=shortlist_size,
        close_pool_size=close_pool_size,
        score_temperature_deg=score_temperature_deg,
        refine_top_m=refine_top_m,
        fold_halfturn_symmetry_targets=fold_halfturn_symmetry_loss,
        mesh_symmetry_group_quaternions_xyzw=(mesh_symmetry_group or {}).get("quaternions_xyzw"),
        samples_per_epoch=eval_samples,
        seed=seed + 10000,
        enable_debug_logging=debug_log_first_n_samples > 0,
        debug_log_first_n_samples=debug_log_first_n_samples,
        debug_name="eval",
    )
    _log(
        f"datasets ready train_len={len(train_dataset)} eval_len={len(eval_dataset)} "
        f"shortlist_size={shortlist_size} image_size={image_size}"
    )
    if debug_render_smoke_test:
        _log("running dataset smoke test on first training sample")
        sample = train_dataset[0]
        _log(
            f"smoke test success pairs_shape={tuple(sample['pairs'].shape)} "
            f"filename={sample['filename']}"
        )
    train_loader = _build_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    eval_loader = _build_loader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    _log(
        f"dataloaders ready batch_size={batch_size} num_workers={num_workers} "
        f"pin_memory={torch.cuda.is_available()}"
    )

    model = MeshPoseScoringModel(
        input_channels=9,
        base_width=base_width,
        hidden_dim=hidden_dim,
        predict_refinement=True,
        rotation_only_refinement=True,
        translation_refinement_scale=0.0,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _log(f"model ready parameters={parameter_count}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    _log(f"optimizer ready lr={learning_rate} weight_decay={weight_decay}")

    history_rows = []
    best_state = None
    best_eval_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        _log(f"epoch_start epoch={epoch}")
        train_summary = _run_epoch(
            model,
            train_loader,
            device=device,
            score_weight=score_weight,
            refine_weight=refine_weight,
            fold_halfturn_symmetry_loss=fold_halfturn_symmetry_loss,
            mesh_symmetry_group_tensor=mesh_symmetry_group_tensor,
            epoch_name=f"train/epoch_{epoch}",
            debug_log_first_n_batches=debug_log_first_n_batches,
            optimizer=optimizer,
        )
        eval_summary = _run_epoch(
            model,
            eval_loader,
            device=device,
            score_weight=score_weight,
            refine_weight=refine_weight,
            fold_halfturn_symmetry_loss=fold_halfturn_symmetry_loss,
            mesh_symmetry_group_tensor=mesh_symmetry_group_tensor,
            epoch_name=f"eval/epoch_{epoch}",
            debug_log_first_n_batches=debug_log_first_n_batches,
            optimizer=None,
        )
        row = {
            "epoch": epoch,
            "train_loss": f"{train_summary['loss']:.6f}",
            "train_score_loss": f"{train_summary['score_loss']:.6f}",
            "train_refine_loss": f"{train_summary['refine_loss']:.6f}",
            "train_score_accuracy": f"{train_summary['score_accuracy']:.6f}",
            "train_candidate_rotation_mean_deg": f"{train_summary['candidate_rotation_mean_deg']:.6f}",
            "train_refined_rotation_mean_deg": f"{train_summary['refined_rotation_mean_deg']:.6f}",
            "train_refined_rotation_folded_mean_deg": f"{train_summary['refined_rotation_folded_mean_deg']:.6f}",
            "eval_loss": f"{eval_summary['loss']:.6f}",
            "eval_score_loss": f"{eval_summary['score_loss']:.6f}",
            "eval_refine_loss": f"{eval_summary['refine_loss']:.6f}",
            "eval_score_accuracy": f"{eval_summary['score_accuracy']:.6f}",
            "eval_candidate_rotation_mean_deg": f"{eval_summary['candidate_rotation_mean_deg']:.6f}",
            "eval_refined_rotation_mean_deg": f"{eval_summary['refined_rotation_mean_deg']:.6f}",
            "eval_refined_rotation_folded_mean_deg": f"{eval_summary['refined_rotation_folded_mean_deg']:.6f}",
        }
        history_rows.append(row)
        _log(
            f"epoch_done epoch={epoch} "
            f"train_loss={train_summary['loss']:.4f} "
            f"eval_loss={eval_summary['loss']:.4f} "
            f"train_acc={train_summary['score_accuracy']:.4f} "
            f"eval_acc={eval_summary['score_accuracy']:.4f} "
            f"eval_rot={eval_summary['refined_rotation_mean_deg']:.2f} "
            f"eval_fold={eval_summary['refined_rotation_folded_mean_deg']:.2f}"
        )
        if eval_summary["loss"] < best_eval_loss:
            best_eval_loss = eval_summary["loss"]
            best_epoch = epoch
            best_state = {
                "model_state_dict": model.state_dict(),
                "query_satellite": query_satellite,
                "mesh_satellite": mesh_satellite,
                "candidate_satellite": candidate_satellite_name,
                "config": {
                    "train_split": train_split,
                    "eval_split": eval_split,
                    "max_train_samples": max_train_samples,
                    "max_eval_samples": max_eval_samples,
                    "max_candidate_samples": max_candidate_samples,
                    "samples_per_epoch": samples_per_epoch,
                    "eval_samples": eval_samples,
                    "image_size": image_size,
                    "crop_padding": crop_padding,
                    "shortlist_size": shortlist_size,
                    "close_pool_size": close_pool_size,
                    "score_temperature_deg": score_temperature_deg,
                    "refine_top_m": refine_top_m,
                    "grid_azimuth_bins": grid_azimuth_bins,
                    "grid_elevation_bins": grid_elevation_bins,
                    "grid_roll_bins": grid_roll_bins,
                    "grid_radius_samples": grid_radius_samples,
                },
                "input_channels": 9,
                "base_width": base_width,
                "hidden_dim": hidden_dim,
                "predict_refinement": True,
                "rotation_only_refinement": True,
                "translation_refinement_scale": 0.0,
                "crop_padding": crop_padding,
                "best_epoch": epoch,
                "mesh_symmetry_group": mesh_symmetry_group,
            }
            _log(f"new_best epoch={epoch} eval_loss={best_eval_loss:.4f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    checkpoint_path = output_root / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    _log(f"saved checkpoint path={checkpoint_path}")
    write_csv(output_root / "history.csv", list(history_rows[0].keys()), history_rows)
    _save_history_plot(history_rows, output_root / "training_curves.png")
    write_json(
        output_root / "experiment_summary.json",
        {
            "query_satellite": query_satellite,
            "mesh_satellite": mesh_satellite,
            "candidate_satellite": candidate_satellite_name,
            "best_epoch": best_epoch,
            "best_eval_loss": best_eval_loss,
            "output_dir": str(output_root),
            "mesh_symmetry_group": mesh_symmetry_group,
            "config": best_state["config"],
        },
    )
    train_dataset.close()
    eval_dataset.close()
    _log(f"finished best_epoch={best_epoch} best_eval_loss={best_eval_loss:.4f}")


def build_coarse_shortlist_from_geometry(
    renderer: MeshRenderer,
    query_image: Image.Image,
    query_mask_image: Image.Image,
    candidates: Sequence[PoseRecord],
    *,
    shortlist_size: int,
    crop_padding: int,
) -> list[PoseRecord]:
    observation = make_observation(query_image, query_mask_image)
    ranked = rank_candidates(
        renderer,
        observation,
        candidates,
        top_k=shortlist_size,
        crop_padding=crop_padding,
    )
    return [
        PoseRecord(
            filename=item.source_filename or f"candidate_{idx}",
            quaternion_xyzw=item.quaternion_xyzw.astype(np.float32),
            translation=item.translation.astype(np.float32),
        )
        for idx, item in enumerate(ranked)
    ]


@torch.no_grad()
def run_shortlist_refinement(
    model: MeshPoseScoringModel,
    renderer: MeshRenderer,
    query_image: Image.Image,
    query_mask: np.ndarray,
    coarse_shortlist: Sequence[PoseRecord],
    *,
    image_size: int,
    crop_padding: int,
    device: str,
    iterations: int,
    keep_top_k: int,
) -> list[RefinedCandidate]:
    query_rgb = np.asarray(query_image, dtype=np.float32) / 255.0
    current = list(coarse_shortlist)
    refined: list[RefinedCandidate] = []

    for _ in range(iterations):
        pair_tensors = []
        seed_quaternions = []
        for candidate in current:
            render_rgb_u8, render_mask = renderer.render(candidate.quaternion_xyzw, candidate.translation)
            pair_tensors.append(
                build_cropped_pair_tensor(
                    query_rgb,
                    query_mask,
                    render_rgb_u8,
                    render_mask,
                    image_size=image_size,
                    crop_padding=crop_padding,
                )
            )
            seed_quaternions.append(torch.from_numpy(np.asarray(candidate.quaternion_xyzw, dtype=np.float32)))

        pair_tensor = torch.stack(pair_tensors, dim=0).to(device)
        seed_quaternion_tensor = torch.stack(seed_quaternions, dim=0).to(device)
        score_logits, refinement = model(pair_tensor)
        assert refinement is not None
        delta_quaternion, _ = refinement
        refined_quaternion = apply_delta_quaternion_torch(seed_quaternion_tensor, delta_quaternion).cpu().numpy()
        score_values = torch.softmax(score_logits, dim=0).cpu().numpy()

        refined = []
        for candidate, quaternion_xyzw, score in zip(current, refined_quaternion, score_values):
            render_rgb_u8, render_mask = renderer.render(quaternion_xyzw, candidate.translation)
            refined.append(
                RefinedCandidate(
                    quaternion_xyzw=np.asarray(quaternion_xyzw, dtype=np.float64),
                    translation=np.asarray(candidate.translation, dtype=np.float64),
                    score=float(score),
                    source_filename=candidate.filename,
                    render_rgb=render_rgb_u8.astype(np.float32) / 255.0,
                    render_mask=render_mask,
                )
            )
        refined.sort(key=lambda item: item.score, reverse=True)
        current = [
            PoseRecord(
                filename=item.source_filename or f"candidate_{idx}",
                quaternion_xyzw=item.quaternion_xyzw.astype(np.float32),
                translation=item.translation.astype(np.float32),
            )
            for idx, item in enumerate(refined[:keep_top_k])
        ]
    return refined


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
        axes[row_index, 1].set_title("refined mask", fontsize=9)
        axes[row_index, 2].imshow(row["render_rgb"])
        axes[row_index, 2].set_title(f"refined render\nscore={row['score']:.3f}", fontsize=9)
        axes[row_index, 3].imshow(_overlay_mask_on_image(row["query_rgb"], row["render_mask"]))
        axes[row_index, 3].set_title(
            f"overlay\nrot={row['rotation_error_deg']:.1f} | fold={row['folded_halfturn_rotation_error_deg']:.1f} | trans={row['translation_error']:.3f}",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _build_markdown_summary(metrics: dict[str, float], *, query_satellite: str, mesh_satellite: str, candidate_satellite: str) -> str:
    return "\n".join(
        [
            "# Benchmark-style shortlist refiner",
            "",
            f"- Query satellite: `{query_satellite}`",
            f"- Mesh satellite: `{mesh_satellite}`",
            f"- Candidate satellite: `{candidate_satellite}`",
            f"- Query images evaluated: {int(metrics['num_queries'])}",
            f"- Candidate bank size: {int(metrics['num_candidates'])}",
            f"- Coarse shortlist size: {int(metrics['coarse_shortlist_size'])}",
            f"- Refinement iterations: {int(metrics['iterations'])}",
            f"- Rotation mean / median / max (deg): {metrics['rotation_error_mean_deg']:.3f} / {metrics['rotation_error_median_deg']:.3f} / {metrics['rotation_error_max_deg']:.3f}",
            f"- Folded half-turn rotation mean / median / max (deg): {metrics['folded_halfturn_rotation_error_mean_deg']:.3f} / {metrics['folded_halfturn_rotation_error_median_deg']:.3f} / {metrics['folded_halfturn_rotation_error_max_deg']:.3f}",
            f"- Mesh-symmetry rotation mean / median / max (deg): {metrics['mesh_symmetry_rotation_error_mean_deg']:.3f} / {metrics['mesh_symmetry_rotation_error_median_deg']:.3f} / {metrics['mesh_symmetry_rotation_error_max_deg']:.3f}",
            f"- Translation mean / median / max: {metrics['translation_error_mean']:.3f} / {metrics['translation_error_median']:.3f} / {metrics['translation_error_max']:.3f}",
            f"- Mean shortlist score: {metrics['score_mean']:.3f}",
            f"- Potential symmetry-driven failures (rot >= 120 deg but folded <= 30 deg): {int(metrics['potential_halfturn_symmetry_count'])}",
            f"- Potential mesh-symmetry failures (rot >= 120 deg but mesh-sym <= 30 deg): {int(metrics['potential_mesh_symmetry_count'])}",
        ]
    )


def evaluate_benchmark_refiner(
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
    use_dataset_bank: bool,
    use_structured_bank: bool,
    grid_azimuth_bins: int,
    grid_elevation_bins: int,
    grid_roll_bins: int,
    grid_radius_samples: int,
    coarse_shortlist_size: int,
    keep_top_k: int,
    iterations: int,
    use_mesh_symmetry_group: bool,
    mesh_symmetry_threshold: float,
    mesh_symmetry_max_points: int,
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
        predict_refinement=bool(checkpoint_payload.get("predict_refinement", True)),
        rotation_only_refinement=bool(checkpoint_payload.get("rotation_only_refinement", True)),
        translation_refinement_scale=float(checkpoint_payload.get("translation_refinement_scale", 0.0)),
    )
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.to(device)
    model.eval()

    candidate_satellite_name = candidate_satellite or mesh_satellite
    output_root = ensure_dir(Path(output_dir) / f"{query_satellite}__mesh_{mesh_satellite}")
    query_ds = SPE3RSatellite(dataset_root, query_satellite)
    mesh_ds = SPE3RSatellite(dataset_root, mesh_satellite)
    candidate_ds = SPE3RSatellite(dataset_root, candidate_satellite_name)
    camera_config = load_camera_config(dataset_root)
    mesh_symmetry_group = None
    if use_mesh_symmetry_group:
        mesh_symmetry_group = build_mesh_rotation_symmetry_group(
            query_ds.model_path,
            threshold=mesh_symmetry_threshold,
            max_points=mesh_symmetry_max_points,
            seed=seed,
        )

    query_records = subsample_records(query_ds.select_records(query_split), max_query_samples, strategy="even", seed=seed)
    base_candidates = subsample_records(candidate_ds.select_records(candidate_split), max_candidate_samples, strategy=candidate_strategy, seed=seed)
    coarse_candidates = build_coarse_candidate_bank(
        base_candidates,
        use_dataset_bank=use_dataset_bank,
        use_structured_bank=use_structured_bank,
        grid_azimuth_bins=grid_azimuth_bins,
        grid_elevation_bins=grid_elevation_bins,
        grid_roll_bins=grid_roll_bins,
        grid_radius_samples=grid_radius_samples,
    )

    renderer = MeshRenderer(mesh_ds.model_path, camera_config)
    retrieval_rows = []
    gallery_rows = []
    rotation_errors = []
    folded_rotation_errors = []
    mesh_symmetry_rotation_errors = []
    translation_errors = []
    scores = []
    crop_padding = int(checkpoint_payload.get("crop_padding", checkpoint_payload.get("config", {}).get("crop_padding", 12)))
    image_size = int(checkpoint_payload.get("config", {}).get("image_size", 224))

    try:
        for query_record in query_records:
            query_image = query_ds.load_image(query_record)
            query_mask_image = query_ds.load_mask(query_record)
            query_mask = np.asarray(query_mask_image, dtype=np.uint8) > 127
            coarse_shortlist = build_coarse_shortlist_from_geometry(
                renderer,
                query_image,
                query_mask_image,
                coarse_candidates,
                shortlist_size=coarse_shortlist_size,
                crop_padding=crop_padding,
            )
            refined = run_shortlist_refinement(
                model,
                renderer,
                query_image,
                query_mask,
                coarse_shortlist,
                image_size=image_size,
                crop_padding=crop_padding,
                device=device,
                iterations=iterations,
                keep_top_k=keep_top_k,
            )
            prediction = refined[0]
            rotation_error = rotation_error_degrees(prediction.quaternion_xyzw, query_record.quaternion_xyzw)
            folded_rotation_error = fold_halfturn_rotation_error_degrees(rotation_error)
            mesh_symmetry_rotation_error = symmetry_group_rotation_error_degrees(
                prediction.quaternion_xyzw,
                query_record.quaternion_xyzw,
                (mesh_symmetry_group or {}).get("quaternions_xyzw"),
            )
            translation_error = translation_distance(prediction.translation, query_record.translation)
            rotation_errors.append(rotation_error)
            folded_rotation_errors.append(folded_rotation_error)
            mesh_symmetry_rotation_errors.append(mesh_symmetry_rotation_error)
            translation_errors.append(translation_error)
            scores.append(prediction.score)
            retrieval_rows.append(
                {
                    "query_filename": query_record.filename,
                    "best_source_filename": prediction.source_filename or "",
                    "score": f"{prediction.score:.6f}",
                    "rotation_error_deg": f"{rotation_error:.6f}",
                    "folded_halfturn_rotation_error_deg": f"{folded_rotation_error:.6f}",
                    "halfturn_symmetry_gain_deg": f"{rotation_error - folded_rotation_error:.6f}",
                    "mesh_symmetry_rotation_error_deg": f"{mesh_symmetry_rotation_error:.6f}",
                    "mesh_symmetry_gain_deg": f"{rotation_error - mesh_symmetry_rotation_error:.6f}",
                    "translation_error": f"{translation_error:.6f}",
                    "pred_quaternion": " ".join(f"{value:.6f}" for value in prediction.quaternion_xyzw.tolist()),
                    "pred_translation": " ".join(f"{value:.6f}" for value in prediction.translation.tolist()),
                    "target_quaternion": " ".join(f"{value:.6f}" for value in query_record.quaternion_xyzw.tolist()),
                    "target_translation": " ".join(f"{value:.6f}" for value in query_record.translation.tolist()),
                }
            )
            if len(gallery_rows) < num_visualizations and prediction.render_rgb is not None and prediction.render_mask is not None:
                gallery_rows.append(
                    {
                        "query_filename": query_record.filename,
                        "query_rgb": np.asarray(query_image, dtype=np.float32) / 255.0,
                        "render_rgb": prediction.render_rgb,
                        "render_mask": prediction.render_mask,
                        "score": prediction.score,
                        "rotation_error_deg": rotation_error,
                        "folded_halfturn_rotation_error_deg": folded_rotation_error,
                        "translation_error": translation_error,
                    }
                )
    finally:
        renderer.close()

    metrics = {
        "query_satellite": query_satellite,
        "mesh_satellite": mesh_satellite,
        "candidate_satellite": candidate_satellite_name,
        "num_queries": len(query_records),
        "num_candidates": len(coarse_candidates),
        "coarse_shortlist_size": coarse_shortlist_size,
        "iterations": iterations,
        "rotation_error_mean_deg": float(np.mean(rotation_errors)),
        "rotation_error_median_deg": float(np.median(rotation_errors)),
        "rotation_error_max_deg": float(np.max(rotation_errors)),
        "folded_halfturn_rotation_error_mean_deg": float(np.mean(folded_rotation_errors)),
        "folded_halfturn_rotation_error_median_deg": float(np.median(folded_rotation_errors)),
        "folded_halfturn_rotation_error_max_deg": float(np.max(folded_rotation_errors)),
        "mesh_symmetry_rotation_error_mean_deg": float(np.mean(mesh_symmetry_rotation_errors)),
        "mesh_symmetry_rotation_error_median_deg": float(np.median(mesh_symmetry_rotation_errors)),
        "mesh_symmetry_rotation_error_max_deg": float(np.max(mesh_symmetry_rotation_errors)),
        "halfturn_symmetry_gain_mean_deg": float(np.mean(np.asarray(rotation_errors) - np.asarray(folded_rotation_errors))),
        "mesh_symmetry_gain_mean_deg": float(np.mean(np.asarray(rotation_errors) - np.asarray(mesh_symmetry_rotation_errors))),
        "potential_halfturn_symmetry_count": int(
            np.sum((np.asarray(rotation_errors) >= 120.0) & (np.asarray(folded_rotation_errors) <= 30.0))
        ),
        "potential_mesh_symmetry_count": int(
            np.sum((np.asarray(rotation_errors) >= 120.0) & (np.asarray(mesh_symmetry_rotation_errors) <= 30.0))
        ),
        "translation_error_mean": float(np.mean(translation_errors)),
        "translation_error_median": float(np.median(translation_errors)),
        "translation_error_max": float(np.max(translation_errors)),
        "under_20_deg": int(np.sum(np.asarray(rotation_errors) < 20.0)),
        "under_45_deg": int(np.sum(np.asarray(rotation_errors) < 45.0)),
        "under_90_deg": int(np.sum(np.asarray(rotation_errors) < 90.0)),
        "ge_150_deg": int(np.sum(np.asarray(rotation_errors) >= 150.0)),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "checkpoint": str(checkpoint),
        "mesh_symmetry_group": mesh_symmetry_group,
    }
    write_csv(
        output_root / "retrievals.csv",
        [
            "query_filename",
            "best_source_filename",
            "score",
            "rotation_error_deg",
            "folded_halfturn_rotation_error_deg",
            "halfturn_symmetry_gain_deg",
            "mesh_symmetry_rotation_error_deg",
            "mesh_symmetry_gain_deg",
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
        _build_markdown_summary(
            metrics,
            query_satellite=query_satellite,
            mesh_satellite=mesh_satellite,
            candidate_satellite=candidate_satellite_name,
        ),
    )
    _save_gallery(gallery_rows, output_root / "qualitative_gallery.png")
