import unittest

from colpali_edge.pooling import hierarchical_pool
from colpali_edge.quantize import (
    binary_dot_approx,
    binary_hamming,
    binary_quantize,
)


class QuantizationTests(unittest.TestCase):
    def test_sign_bits_are_packed_lsb_first(self):
        self.assertEqual(binary_quantize([0.1, -0.2, 0.3, -0.4]), b"\x05")
        self.assertEqual(binary_quantize([-1.0, -1.0]), b"\x00")
        self.assertEqual(binary_quantize([1.0, 1.0, 1.0]), b"\x07")

    def test_hamming_and_dot(self):
        left = binary_quantize([1.0, -1.0, 1.0, -1.0])
        right = binary_quantize([-1.0, -1.0, 1.0, 1.0])
        self.assertEqual(binary_hamming(left, left), 0)
        self.assertEqual(binary_hamming(left, right), 2)
        self.assertEqual(binary_dot_approx(left, right, 4), 0)
        self.assertEqual(binary_dot_approx(left, left, 4), 4)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            binary_quantize([])
        with self.assertRaises(ValueError):
            binary_quantize([float("nan")])
        with self.assertRaises(ValueError):
            binary_hamming(b"\x00", b"\x00\x00")
        with self.assertRaises(ValueError):
            binary_dot_approx(b"\x00", b"\x00", 9)


class PoolingTests(unittest.TestCase):
    def test_mean_pools_consecutive_vectors(self):
        embeddings = [[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]
        self.assertEqual(hierarchical_pool(embeddings, 2), [[2.0, 0.0], [5.0, 0.0]])
        self.assertEqual(hierarchical_pool(embeddings, 1), embeddings)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            hierarchical_pool([], 2)
        with self.assertRaises(ValueError):
            hierarchical_pool([[1.0]], 0)
        with self.assertRaises(ValueError):
            hierarchical_pool([[1.0], [1.0, 2.0]], 2)


if __name__ == "__main__":
    unittest.main()
