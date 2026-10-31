import math
import unittest

from colpali_edge.loss import contrastive_loss, in_batch_contrastive_loss


def softplus(value):
    return math.log1p(math.exp(value))


class ContrastiveLossTests(unittest.TestCase):
    def test_known_softplus_loss(self):
        self.assertAlmostEqual(
            contrastive_loss([1.0], [0.0]),
            softplus(-1.0),
            places=12,
        )
        self.assertAlmostEqual(
            contrastive_loss([0.0], [1.0]),
            softplus(1.0),
            places=12,
        )

    def test_in_batch_positive_and_negative(self):
        queries = [[[1.0, 0.0]], [[0.0, 1.0]]]
        documents = [[[1.0, 0.0]], [[0.0, 1.0]]]
        result = in_batch_contrastive_loss(queries, documents)
        self.assertEqual(result.positives, (1.0, 1.0))
        self.assertEqual(result.negatives, (0.0, 0.0))
        self.assertAlmostEqual(result.loss, softplus(-1.0), places=12)

    def test_wrong_alignment_has_larger_loss(self):
        queries = [[[1.0, 0.0]], [[0.0, 1.0]]]
        documents = [[[0.0, 1.0]], [[1.0, 0.0]]]
        result = in_batch_contrastive_loss(queries, documents)
        self.assertEqual(result.positives, (0.0, 0.0))
        self.assertEqual(result.negatives, (1.0, 1.0))
        self.assertAlmostEqual(result.loss, softplus(1.0), places=12)

    def test_batch_requires_a_negative(self):
        with self.assertRaises(ValueError):
            in_batch_contrastive_loss([[[1.0]]], [[[1.0]]])
        with self.assertRaises(ValueError):
            contrastive_loss([], [])
        with self.assertRaises(ValueError):
            contrastive_loss([1.0], [])


if __name__ == "__main__":
    unittest.main()
