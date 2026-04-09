from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyrender
import trimesh
from scipy import ndimage

from common import ensure_dir, write_csv, write_json, write_text
from data_utils import SPE3RSatellite, load_camera_config, subsample_records
from geometry_utils import perturb_pose, pose_matrix, rotation_error_degrees, translation_distance


def image_to_numpy(image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def grayscale(rgb: np.ndarray) -> np.ndarray:
    return (0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]).astype(np.float32)


def sobel_edges(gray: np.ndarray, *, sigma: float = 1.0, threshold_quantile: float = 0.85) -> np.ndarray:
    blurred = ndimage.gaussian_filter(gray, sigma=sigma)
    grad_x = ndimage.sobel(blurred, axis=1)
    grad_y = ndimage.sobel(blurred, axis=0)
    magnitude = np.hypot(grad_x, grad_y)
    threshold = float(np.quantile(magnitude, threshold_quantile))
    return magnitude >= threshold


def mask_contour(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, iterations=1, border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def crop_box_from_masks(mask_a: np.ndarray, mask_b: np.ndarray, *, padding: int = 12) -> tuple[int, int, int, int]:
    union = np.logical_or(mask_a, mask_b)
    if not np.any(union):
        h, w = union.shape
        return 0, h, 0, w
    ys, xs = np.nonzero(union)
    y0 = max(0, int(ys.min()) - padding)
    y1 = min(union.shape[0], int(ys.max()) + padding + 1)
    x0 = max(0, int(xs.min()) - padding)
    x1 = min(union.shape[1], int(xs.max()) + padding + 1)
    return y0, y1, x0, x1


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(mask_a, mask_b).sum() / union)


def edge_iou(edges_a: np.ndarray, edges_b: np.ndarray) -> float:
    return mask_iou(ndimage.binary_dilation(edges_a, iterations=1), ndimage.binary_dilation(edges_b, iterations=1))


def contour_similarity(contour_a: np.ndarray, contour_b: np.ndarray) -> float:
    if not np.any(contour_a) and not np.any(contour_b):
        return 1.0
    if not np.any(contour_a) or not np.any(contour_b):
        return 0.0
    dt_a = ndimage.distance_transform_edt(~contour_a)
    dt_b = ndimage.distance_transform_edt(~contour_b)
    mean_ab = float(dt_b[contour_a].mean())
    mean_ba = float(dt_a[contour_b].mean())
    symmetric = 0.5 * (mean_ab + mean_ba)
    diagonal = float(np.hypot(*contour_a.shape))
    normalized = min(1.0, symmetric / max(diagonal * 0.08, 1e-6))
    return 1.0 - normalized


def cropped_rgb_score(obs_rgb: np.ndarray, ren_rgb: np.ndarray, obs_mask: np.ndarray, ren_mask: np.ndarray, *, padding: int) -> float:
    y0, y1, x0, x1 = crop_box_from_masks(obs_mask, ren_mask, padding=padding)
    union = np.logical_or(obs_mask[y0:y1, x0:x1], ren_mask[y0:y1, x0:x1])
    if not np.any(union):
        return 1.0
    diff = np.abs(obs_rgb[y0:y1, x0:x1] - ren_rgb[y0:y1, x0:x1])
    return 1.0 - float(diff[union].mean())


@dataclass
class Observation:
    rgb: np.ndarray
    mask: np.ndarray
    edges: np.ndarray
    contour: np.ndarray


def make_observation(image, mask_image) -> Observation:
    rgb = image_to_numpy(image)
    mask = np.asarray(mask_image, dtype=np.uint8) > 127
    edges = sobel_edges(grayscale(rgb))
    contour = mask_contour(mask)
    return Observation(rgb=rgb, mask=mask, edges=edges, contour=contour)


@dataclass
class ScoredPose:
    quaternion_xyzw: np.ndarray
    translation: np.ndarray
    score: float
    mask_iou: float
    edge_iou: float
    contour_score: float
    crop_rgb_score: float
    source_filename: str | None = None
    render_rgb: np.ndarray | None = None
    render_mask: np.ndarray | None = None


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
        self.renderer = pyrender.OffscreenRenderer(viewport_width=int(camera_config["Nu"]), viewport_height=int(camera_config["Nv"]))

    def render(self, quaternion_xyzw: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transform = self.cv_to_gl @ pose_matrix(quaternion_xyzw, translation)
        self.scene.set_pose(self.mesh_node, pose=transform)
        self.scene.set_pose(self.camera_node, pose=np.eye(4))
        self.scene.set_pose(self.light_node, pose=np.eye(4))
        color, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        return color[:, :, :3], depth > 0

    def close(self) -> None:
        self.renderer.delete()


def score_pose(renderer: MeshRenderer, observation: Observation, quaternion_xyzw: np.ndarray, translation: np.ndarray, *, source_filename: str | None, crop_padding: int, store_render: bool = False) -> ScoredPose:
    render_rgb_u8, render_mask = renderer.render(quaternion_xyzw, translation)
    render_rgb = render_rgb_u8.astype(np.float32) / 255.0
    render_edges = sobel_edges(grayscale(render_rgb))
    render_contour = mask_contour(render_mask)
    metrics = {
        "mask_iou": mask_iou(observation.mask, render_mask),
        "edge_iou": edge_iou(observation.edges, render_edges),
        "contour_score": contour_similarity(observation.contour, render_contour),
        "crop_rgb_score": cropped_rgb_score(observation.rgb, render_rgb, observation.mask, render_mask, padding=crop_padding),
    }
    score = (
        0.20 * metrics["mask_iou"]
        + 0.20 * metrics["edge_iou"]
        + 0.35 * metrics["contour_score"]
        + 0.25 * metrics["crop_rgb_score"]
    )
    return ScoredPose(
        quaternion_xyzw=np.asarray(quaternion_xyzw, dtype=np.float64),
        translation=np.asarray(translation, dtype=np.float64),
        score=float(score),
        mask_iou=float(metrics["mask_iou"]),
        edge_iou=float(metrics["edge_iou"]),
        contour_score=float(metrics["contour_score"]),
        crop_rgb_score=float(metrics["crop_rgb_score"]),
        source_filename=source_filename,
        render_rgb=render_rgb if store_render else None,
        render_mask=render_mask if store_render else None,
    )


def rank_candidates(renderer: MeshRenderer, observation: Observation, candidates, *, top_k: int, crop_padding: int):
    scored = [
        score_pose(renderer, observation, record.quaternion_xyzw, record.translation, source_filename=record.filename, crop_padding=crop_padding)
        for record in candidates
    ]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def refine_candidates(renderer: MeshRenderer, observation: Observation, top_candidates, *, rounds: int, samples_per_round: int, rotation_sigma_deg: float, translation_sigma: float, crop_padding: int, seed: int):
    rng = np.random.default_rng(seed)
    best = top_candidates[0]
    current = list(top_candidates)
    for _ in range(rounds):
        proposals = list(current)
        for candidate in current:
            for _ in range(samples_per_round):
                quat, trans = perturb_pose(
                    candidate.quaternion_xyzw,
                    candidate.translation,
                    rotation_sigma_deg=rotation_sigma_deg,
                    translation_sigma=translation_sigma,
                    rng=rng,
                )
                proposals.append(score_pose(renderer, observation, quat, trans, source_filename=candidate.source_filename, crop_padding=crop_padding))
        proposals.sort(key=lambda item: item.score, reverse=True)
        best = proposals[0]
        current = proposals[: len(current)]
    return score_pose(renderer, observation, best.quaternion_xyzw, best.translation, source_filename=best.source_filename, crop_padding=crop_padding, store_render=True)


def save_gallery(rows, output_path: Path) -> None:
    if not rows:
        return
    figure, axes = plt.subplots(len(rows), 4, figsize=(12, 3.3 * len(rows)))
    axes = np.atleast_2d(axes)
    for axis in axes.ravel():
        axis.axis("off")
    for idx, row in enumerate(rows):
        overlay = row["query_rgb"].copy()
        overlay[row["render_mask"]] = 0.55 * overlay[row["render_mask"]] + 0.45 * np.asarray([1.0, 0.2, 0.2], dtype=np.float32)
        axes[idx, 0].imshow(row["query_rgb"])
        axes[idx, 0].set_title(f"query\n{row['query_filename']}", fontsize=9)
        axes[idx, 1].imshow(row["query_mask"], cmap="gray")
        axes[idx, 1].set_title("query mask", fontsize=9)
        axes[idx, 2].imshow(row["render_rgb"])
        axes[idx, 2].set_title(f"render\nscore={row['score']:.3f}", fontsize=9)
        axes[idx, 3].imshow(overlay)
        axes[idx, 3].set_title(f"overlay\nrot={row['rotation_error_deg']:.1f}", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_geometry_experiment(
    dataset_root: str | Path,
    *,
    query_satellite: str,
    mesh_satellite: str,
    candidate_satellite: str | None,
    query_split: str,
    candidate_split: str,
    max_query_samples: int,
    max_candidate_samples: int,
    top_k: int,
    refine_rounds: int,
    refine_samples_per_round: int,
    rotation_sigma_deg: float,
    translation_sigma: float,
    crop_padding: int,
    output_dir: str | Path,
    seed: int,
) -> None:
    output_root = ensure_dir(Path(output_dir) / f"{query_satellite}__mesh_{mesh_satellite}")
    query_ds = SPE3RSatellite(dataset_root, query_satellite)
    mesh_ds = SPE3RSatellite(dataset_root, mesh_satellite)
    candidate_name = candidate_satellite or mesh_satellite
    candidate_ds = SPE3RSatellite(dataset_root, candidate_name)
    camera_config = load_camera_config(dataset_root)
    query_records = subsample_records(query_ds.select_records(query_split), max_query_samples, strategy="even", seed=seed)
    candidate_records = subsample_records(candidate_ds.select_records(candidate_split), max_candidate_samples, strategy="even", seed=seed)

    renderer = MeshRenderer(mesh_ds.model_path, camera_config)
    rows = []
    gallery = []
    rotation_errors = []
    translation_errors = []
    scores = []
    try:
        for record in query_records:
            observation = make_observation(query_ds.load_image(record), query_ds.load_mask(record))
            top_candidates = rank_candidates(renderer, observation, candidate_records, top_k=top_k, crop_padding=crop_padding)
            prediction = refine_candidates(
                renderer,
                observation,
                top_candidates,
                rounds=refine_rounds,
                samples_per_round=refine_samples_per_round,
                rotation_sigma_deg=rotation_sigma_deg,
                translation_sigma=translation_sigma,
                crop_padding=crop_padding,
                seed=seed + record.image_index,
            )
            rot_err = rotation_error_degrees(prediction.quaternion_xyzw, record.quaternion_xyzw)
            trans_err = translation_distance(prediction.translation, record.translation)
            rotation_errors.append(rot_err)
            translation_errors.append(trans_err)
            scores.append(prediction.score)
            rows.append(
                {
                    "query_filename": record.filename,
                    "best_source_filename": prediction.source_filename or "",
                    "score": f"{prediction.score:.6f}",
                    "mask_iou": f"{prediction.mask_iou:.6f}",
                    "edge_iou": f"{prediction.edge_iou:.6f}",
                    "contour_score": f"{prediction.contour_score:.6f}",
                    "crop_rgb_score": f"{prediction.crop_rgb_score:.6f}",
                    "rotation_error_deg": f"{rot_err:.6f}",
                    "translation_error": f"{trans_err:.6f}",
                }
            )
            if len(gallery) < 8 and prediction.render_rgb is not None and prediction.render_mask is not None:
                gallery.append(
                    {
                        "query_filename": record.filename,
                        "query_rgb": observation.rgb,
                        "query_mask": observation.mask.astype(np.float32),
                        "render_rgb": prediction.render_rgb,
                        "render_mask": prediction.render_mask,
                        "score": prediction.score,
                        "rotation_error_deg": rot_err,
                    }
                )
    finally:
        renderer.close()

    write_csv(
        output_root / "retrievals.csv",
        ["query_filename", "best_source_filename", "score", "mask_iou", "edge_iou", "contour_score", "crop_rgb_score", "rotation_error_deg", "translation_error"],
        rows,
    )
    metrics = {
        "query_satellite": query_satellite,
        "mesh_satellite": mesh_satellite,
        "candidate_satellite": candidate_name,
        "num_queries": len(query_records),
        "num_candidates": len(candidate_records),
        "rotation_error_mean_deg": float(np.mean(rotation_errors)),
        "rotation_error_median_deg": float(np.median(rotation_errors)),
        "rotation_error_max_deg": float(np.max(rotation_errors)),
        "translation_error_mean": float(np.mean(translation_errors)),
        "translation_error_median": float(np.median(translation_errors)),
        "translation_error_max": float(np.max(translation_errors)),
        "score_mean": float(np.mean(scores)),
    }
    write_json(output_root / "metrics.json", metrics)
    write_text(
        output_root / "summary.md",
        "\n".join(
            [
                "# Geometry-overfit experiment",
                "",
                f"- Query satellite: `{query_satellite}`",
                f"- Mesh satellite: `{mesh_satellite}`",
                f"- Candidate satellite: `{candidate_name}`",
                f"- Rotation mean / median / max (deg): {metrics['rotation_error_mean_deg']:.3f} / {metrics['rotation_error_median_deg']:.3f} / {metrics['rotation_error_max_deg']:.3f}",
                f"- Translation mean / median / max: {metrics['translation_error_mean']:.3f} / {metrics['translation_error_median']:.3f} / {metrics['translation_error_max']:.3f}",
            ]
        ),
    )
    save_gallery(gallery, output_root / "qualitative_gallery.png")

