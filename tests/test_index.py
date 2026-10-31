import unittest

from colpali_edge.index import MultiVectorIndex


class IndexTests(unittest.TestCase):
    def test_two_stage_search_keeps_coarse_and_refined_scores(self):
        index = MultiVectorIndex()
        index.add("a", [[1.0, 0.0], [0.0, 1.0]])
        index.add("b", [[-1.0, 0.0], [0.0, -1.0]])
        index.add("c", [[0.5, 0.5]])

        hits = index.search([[1.0, 0.0]], top_k=2, candidate_k=3)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].doc_id, "a")
        self.assertEqual(hits[0].rank, 1)
        self.assertEqual(hits[0].score, 1.0)
        self.assertIsInstance(hits[0].coarse_score, int)
        self.assertEqual(hits[1].doc_id, "c")
        self.assertEqual(hits[1].score, 0.5)

    def test_candidate_limit_and_validation(self):
        index = MultiVectorIndex()
        index.add("a", [[1.0, 0.0]])
        index.add("b", [[0.0, 1.0]])
        with self.assertRaises(ValueError):
            index.search([[1.0, 0.0]], top_k=0)
        with self.assertRaises(ValueError):
            index.search([[1.0, 0.0]], top_k=2, candidate_k=1)
        with self.assertRaises(ValueError):
            index.search([[1.0, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            index.add("a", [[1.0, 0.0]])
        with self.assertRaises(ValueError):
            index.add("c", [[1.0, 0.0, 0.0]])

    def test_empty_index_rejects_search(self):
        with self.assertRaises(ValueError):
            MultiVectorIndex().search([[1.0]])


if __name__ == "__main__":
    unittest.main()
