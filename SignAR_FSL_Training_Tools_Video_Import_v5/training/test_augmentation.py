"""Unit tests for training-only temporal augmentation (NumPy only)."""
import unittest
import numpy as np
from augmentation import augment_training_data


class AugmentationTests(unittest.TestCase):
    def test_augmented_dataset_doubles_and_preserves_shape(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(4, 30, 456)).astype(np.float32)
        y = np.array([0, 1, 1, 2], dtype=np.int32)
        original_x = x.copy()
        original_y = y.copy()
        result_x, result_y = augment_training_data(x, y, seed=42)
        self.assertEqual(result_x.shape, (8, 30, 456))
        self.assertEqual(result_y.shape, (8,))
        self.assertTrue(np.isfinite(result_x).all())
        self.assertTrue(np.array_equal(x, original_x))
        self.assertTrue(np.array_equal(y, original_y))
        self.assertEqual(sorted(result_y.tolist()), sorted([0, 1, 1, 2, 0, 1, 1, 2]))

    def test_same_seed_is_reproducible(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(2, 30, 456)).astype(np.float32)
        y = np.array([2, 7], dtype=np.int32)
        a_x, a_y = augment_training_data(x, y, seed=77)
        b_x, b_y = augment_training_data(x, y, seed=77)
        self.assertTrue(np.array_equal(a_x, b_x))
        self.assertTrue(np.array_equal(a_y, b_y))

    def test_missing_landmark_sentinel_is_preserved(self):
        x = np.zeros((2, 30, 456), dtype=np.float32)
        y = np.array([0, 1], dtype=np.int32)
        x[:, :, 132:258] = -3.0  # Left hand absent.
        x[:, :, 384:456] = -3.0  # Selected face points absent.
        result_x, _ = augment_training_data(x, y, seed=9)
        self.assertTrue(np.isin(result_x[:, :, 132:258], [-3.0]).all())
        self.assertTrue(np.isin(result_x[:, :, 384:456], [-3.0]).all())


if __name__ == "__main__":
    unittest.main()
