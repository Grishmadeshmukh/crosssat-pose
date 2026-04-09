from __future__ import annotations

import math

import numpy as np


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Quaternion has zero norm.")
    return quaternion / norm


def quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.asarray(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quaternion_to_euler_degrees(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.degrees([roll, pitch, yaw])


def pose_matrix(quaternion_xyzw: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_matrix(quaternion_xyzw)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64)
    return transform


def quaternion_multiply(quaternion_a: np.ndarray, quaternion_b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = normalize_quaternion(quaternion_a)
    bx, by, bz, bw = normalize_quaternion(quaternion_b)
    return normalize_quaternion(
        np.asarray(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float64,
        )
    )


def axis_angle_to_quaternion(axis: np.ndarray, angle_radians: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / norm
    half = angle_radians / 2.0
    xyz = axis * math.sin(half)
    return normalize_quaternion(np.asarray([xyz[0], xyz[1], xyz[2], math.cos(half)], dtype=np.float64))


def perturb_pose(
    quaternion_xyzw: np.ndarray,
    translation: np.ndarray,
    *,
    rotation_sigma_deg: float,
    translation_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    axis = rng.normal(size=3)
    angle = math.radians(rng.normal(0.0, rotation_sigma_deg))
    delta = axis_angle_to_quaternion(axis, angle)
    new_quaternion = quaternion_multiply(delta, quaternion_xyzw)
    new_translation = np.asarray(translation, dtype=np.float64) + rng.normal(0.0, translation_sigma, size=3)
    return normalize_quaternion(new_quaternion), new_translation.astype(np.float64)


def rotation_error_degrees(quaternion_a: np.ndarray, quaternion_b: np.ndarray) -> float:
    quat_a = normalize_quaternion(quaternion_a)
    quat_b = normalize_quaternion(quaternion_b)
    dot = float(np.clip(abs(np.dot(quat_a, quat_b)), 0.0, 1.0))
    return math.degrees(2.0 * math.acos(dot))


def translation_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(vector_a) - np.asarray(vector_b)))


def camera_position_in_body_frame(quaternion_xyzw: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rotation_matrix = quaternion_to_matrix(quaternion_xyzw)
    translation = np.asarray(translation, dtype=np.float64)
    return -(rotation_matrix.T @ translation)

