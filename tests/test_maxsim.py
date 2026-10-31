import unittest

from colpali_edge.maxsim import maxsim, maxsim_batch


class MaxSimTests(unittest.TestCase):
    def test_known_score(self):
        query = [[1.0, 0.0], [0.0, 1.0]]
        document = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        self.assertEqual(maxsim(query, document), 2.0)

    def test_scale_and_max(self):
        query = [[2.0, 0.0]]
        document = [[1.0, 0.0], [0.0, 1.0]]
        self.assertEqual(maxsim(query, document), 2.0)

    def test_batch_scores(self):
        query = [[1.0, 0.0]]
        docs = [
            [[1.0, 0.0]],
            [[-1.0, 0.0]],
            [[0.5, 0.5]],
        ]
        self.assertEqual(maxsim_batch(query, docs), [1.0, -1.0, 0.5])

    def test_rejects_empty_and_mismatched_dimensions(self):
        with self.assertRaises(ValueError):
            maxsim([], [[1.0]])
        with self.assertRaises(ValueError):
            maxsim([[1.0]], [])
        with self.assertRaises(ValueError):
            maxsim([[1.0, 2.0]], [[1.0]])
        with self.assertRaises(ValueError):
            maxsim([[1.0], [1.0, 2.0]], [[1.0]])
        with self.assertRaises(ValueError):
            maxsim_batch([[1.0]], [])


if __name__ == "__main__":
    unittest.main()
