from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from data_utils import SPE3RSatellite, subsample_records
from geometry_utils import camera_position_in_body_frame, quaternion_to_euler_degrees, rotation_error_degrees
from common import ensure_dir, write_json, write_text


def raw_pixel_feature(image, image_size: int = 64) -> np.ndarray:
    resized = image.resize((image_size, image_size))
    return (np.asarray(resized, dtype=np.float32) / 255.0).reshape(-1)


def pairwise_pose_image_correlation(records, satellite: SPE3RSatellite, *, image_size: int, num_pairs: int, seed: int):
    rng = np.random.default_rng(seed)
    features = np.vstack([raw_pixel_feature(satellite.load_image(record), image_size=image_size) for record in records])
    all_pairs = [(i, j) for i in range(len(records)) for j in range(i + 1, len(records))]
    chosen = rng.choice(len(all_pairs), size=min(num_pairs, len(all_pairs)), replace=False)
    rotation_gaps = []
    translation_gaps = []
    image_distances = []
    for pair_idx in chosen:
        i, j = all_pairs[int(pair_idx)]
        rotation_gaps.append(rotation_error_degrees(records[i].quaternion_xyzw, records[j].quaternion_xyzw))
        translation_gaps.append(float(np.linalg.norm(records[i].translation - records[j].translation)))
        image_distances.append(float(np.linalg.norm(features[i] - features[j])))
    rotation_gaps = np.asarray(rotation_gaps)
    translation_gaps = np.asarray(translation_gaps)
    image_distances = np.asarray(image_distances)
    corr = float(np.corrcoef(rotation_gaps, image_distances)[0, 1]) if len(rotation_gaps) > 1 else float("nan")
    return rotation_gaps, translation_gaps, image_distances, corr


def save_contact_sheet(satellite: SPE3RSatellite, records, output_path: Path) -> None:
    cols = min(4, max(1, int(np.ceil(np.sqrt(len(records))))))
    rows = int(np.ceil(len(records) / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, record in zip(axes.ravel(), records):
        image = satellite.load_image(record)
        roll, pitch, yaw = quaternion_to_euler_degrees(record.quaternion_xyzw)
        axis.imshow(image)
        axis.set_title(
            f"{record.filename}\n"
            f"r/p/y=({roll:.1f}, {pitch:.1f}, {yaw:.1f})\n"
            f"t={record.translation.tolist()}",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_viewpoints_plot(satellite: SPE3RSatellite, records, output_path: Path) -> None:
    positions = np.asarray(
        [camera_position_in_body_frame(record.quaternion_xyzw, record.translation) for record in records],
        dtype=np.float32,
    )
    figure = plt.figure(figsize=(7, 6))
    axis = figure.add_subplot(111, projection="3d")
    scatter = axis.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=np.linalg.norm(positions, axis=1), cmap="viridis", s=18)
    extent = float(np.abs(positions).max())
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_zlim(-extent, extent)
    axis.set_xlabel("body x")
    axis.set_ylabel("body y")
    axis.set_zlabel("body z")
    figure.colorbar(scatter, ax=axis, shrink=0.7, label="view radius")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_correlation_plot(rotation_gaps, translation_gaps, image_distances, satellite_name: str, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    scatter = axis.scatter(rotation_gaps, image_distances, c=translation_gaps, cmap="magma", s=18, alpha=0.75)
    axis.set_xlabel("rotation gap (deg)")
    axis.set_ylabel("raw image distance")
    axis.set_title(f"Pose/image correlation: {satellite_name}")
    figure.colorbar(scatter, ax=axis, label="translation gap")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_visual_inspection(
    dataset_root: str | Path,
    satellites: list[str],
    *,
    split: str,
    samples_per_satellite: int,
    correlation_samples: int,
    pair_samples: int,
    raw_image_size: int,
    output_dir: str | Path,
    seed: int,
) -> None:
    output_root = ensure_dir(output_dir)
    for satellite_name in satellites:
        satellite = SPE3RSatellite(dataset_root, satellite_name)
        records = satellite.select_records(split)
        sampled = subsample_records(records, samples_per_satellite, strategy="even", seed=seed)
        corr_records = subsample_records(records, correlation_samples, strategy="even", seed=seed)
        rotation_gaps, translation_gaps, image_distances, corr = pairwise_pose_image_correlation(
            corr_records,
            satellite,
            image_size=raw_image_size,
            num_pairs=pair_samples,
            seed=seed,
        )
        satellite_dir = ensure_dir(Path(output_root) / satellite_name)
        save_contact_sheet(satellite, sampled, satellite_dir / "contact_sheet.png")
        save_viewpoints_plot(satellite, records, satellite_dir / "viewpoints.png")
        save_correlation_plot(rotation_gaps, translation_gaps, image_distances, satellite_name, satellite_dir / "pose_image_correlation.png")
        write_json(
            satellite_dir / "summary.json",
            {
                "satellite": satellite_name,
                "split": split,
                "num_records": len(records),
                "sampled_filenames": [record.filename for record in sampled],
                "pairwise_correlation": corr,
            },
        )
        write_text(
            satellite_dir / "summary.md",
            "\n".join(
                [
                    f"# Visual inspection: {satellite_name}",
                    "",
                    f"- Split: `{split}`",
                    f"- Number of images: {len(records)}",
                    f"- Pairwise image/pose correlation: {corr:.3f}",
                ]
            ),
        )

