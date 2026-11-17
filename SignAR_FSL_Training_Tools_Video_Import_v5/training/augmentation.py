"""Conservative sign-preserving augmentation for temporal FSL landmark sequences."""
from __future__ import annotations
import numpy as np

SEQ_LEN = 30
FEATURES = 456
MISSING_THRESHOLD = -2.75

def augment_training_data(x: np.ndarray, y: np.ndarray, seed: int, seq_len: int = SEQ_LEN, features: int = FEATURES):
    """Add one conservative, sign-preserving variation per training clip only.

    No mirroring or label-changing augmentation is used. Test and validation clips
    remain untouched so the unseen-signer evaluation stays honest.
    """
    rng = np.random.default_rng(seed)
    coordinate_mask = np.ones(features, dtype=bool)
    coordinate_mask[np.arange(3, min(132, features), 4)] = False  # Keep pose visibility unchanged.
    visibility_indices = np.arange(3, min(132, features), 4)
    xy_pairs = []
    # Pose x/y pairs (4 values per landmark: x,y,z,visibility).
    xy_pairs.extend((i, i + 1) for i in range(0, 132, 4))
    # Hand features: global xyz followed by wrist-relative xyz.
    for start in range(132, min(384, features), 6):
        xy_pairs.append((start, start + 1))
        xy_pairs.append((start + 3, start + 4))
    # Selected face features: xyz.
    xy_pairs.extend((i, i + 1) for i in range(384, FEATURES, 3))

    augmented = []
    for sequence in x:
        # Mild timing warp, within about six percent; the model still receives 30 frames.
        gamma = float(rng.uniform(0.94, 1.06))
        base = np.linspace(0, seq_len - 1, seq_len, dtype=np.float32)
        positions = np.power(base / (seq_len - 1), gamma) * (seq_len - 1)
        lo = np.floor(positions).astype(np.int32)
        hi = np.minimum(lo + 1, seq_len - 1)
        weight = (positions - lo)[:, None]
        sample = sequence[lo] * (1.0 - weight) + sequence[hi] * weight

        # Small scale and rotation variations after torso-relative normalization.
        sample = sample.copy()
        scale_factor = float(rng.uniform(0.985, 1.015))
        valid_coordinates = sample > MISSING_THRESHOLD
        # Pose xyz values are usable only when their joint visibility channel is positive.
        for joint in range(33):
            joint_visible = sample[:, joint * 4 + 3] > 0.05
            valid_coordinates[:, joint * 4:joint * 4 + 3] &= joint_visible[:, None]
        scale_mask = valid_coordinates & coordinate_mask[None, :]
        sample[scale_mask] *= scale_factor
        theta = float(rng.uniform(-2.0, 2.0) * np.pi / 180.0)
        cosine, sine = np.cos(theta), np.sin(theta)
        for xi, yi in xy_pairs:
            old_x = sample[:, xi].copy()
            old_y = sample[:, yi].copy()
            pair_valid = (old_x > MISSING_THRESHOLD) & (old_y > MISSING_THRESHOLD)
            rotated_x = cosine * old_x - sine * old_y
            rotated_y = sine * old_x + cosine * old_y
            sample[:, xi] = np.where(pair_valid, rotated_x, old_x)
            sample[:, yi] = np.where(pair_valid, rotated_y, old_y)

        # Tiny coordinate noise only where a real coordinate exists; missing markers and visibility stay exact.
        noise = rng.normal(0.0, 0.0015, size=sample.shape).astype(np.float32)
        noise[:, ~coordinate_mask] = 0.0
        valid_after_rotation = sample > MISSING_THRESHOLD
        for joint in range(33):
            joint_visible = sample[:, joint * 4 + 3] > 0.05
            valid_after_rotation[:, joint * 4:joint * 4 + 3] &= joint_visible[:, None]
        noise[~valid_after_rotation] = 0.0
        sample += noise
        sample[:, visibility_indices] = np.clip(sample[:, visibility_indices], 0.0, 1.0)
        augmented.append(sample.astype(np.float32))

    x_out = np.concatenate([x.astype(np.float32), np.stack(augmented)], axis=0)
    y_out = np.concatenate([y.astype(np.int32), y.astype(np.int32)], axis=0)
    order = rng.permutation(len(y_out))
    return x_out[order], y_out[order]
