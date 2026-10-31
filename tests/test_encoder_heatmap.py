import json
import unittest

from colpali_edge.encoder import (
    EncoderUnavailableError,
    SyntheticEncoder,
    TransformersColPaliEncoder,
)
from colpali_edge.heatmap import heatmap_payload, maxsim_attention
from colpali_edge.maxsim import maxsim


class SyntheticEncoderTests(unittest.TestCase):
    def test_deterministic_shapes_and_prefixes(self):
        encoder = SyntheticEncoder(dimension=8, tokens=4, seed="test")
        query = encoder.encode_query("revenue")
        page = encoder.encode_page("page-image-bytes")
        self.assertEqual(encoder.dimension, 8)
        self.assertEqual(len(query), 4)
        self.assertEqual(len(query[0]), 8)
        self.assertEqual(query, encoder.encode_query("revenue"))
        self.assertNotEqual(query, page)

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            SyntheticEncoder(dimension=0)
        with self.assertRaises(ValueError):
            SyntheticEncoder(tokens=0)
        with self.assertRaises(ValueError):
            SyntheticEncoder(seed="")


class TransformersAdapterTests(unittest.TestCase):
    def test_unavailable_runtime_fails_clearly(self):
        if TransformersColPaliEncoder.available():
            self.skipTest("torch/transformers are installed; model weights are not")
        encoder = TransformersColPaliEncoder()
        with self.assertRaises(EncoderUnavailableError):
            encoder.load()
        with self.assertRaises(EncoderUnavailableError):
            _ = encoder.dimension


class AttentionMapTests(unittest.TestCase):
    def setUp(self):
        self.query = [[1.0, 0.0], [0.0, 1.0]]
        self.document = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    def test_scores_and_best_patch(self):
        attention = maxsim_attention(self.query, self.document)
        self.assertEqual(attention.scores[0], (1.0, 0.0, 1.0))
        self.assertEqual(attention.scores[1], (0.0, 1.0, 1.0))
        self.assertEqual(attention.best_indices, (0, 1))
        self.assertEqual(attention.best_scores, (1.0, 1.0))
        self.assertAlmostEqual(sum(attention.best_scores), maxsim(self.query, self.document))
        flat = [value for row in attention.normalized for value in row]
        self.assertEqual(min(flat), 0.0)
        self.assertEqual(max(flat), 1.0)

    def test_payload_is_json_serializable(self):
        attention = maxsim_attention(self.query, self.document)
        payload = heatmap_payload(["q0", "q1"], ["p0", "p1", "p2"], attention)
        self.assertEqual(payload["rows"][0]["best_patch"], 0)
        self.assertEqual(payload["rows"][1]["scores"], [0.0, 1.0, 1.0])
        json.dumps(payload)
        with self.assertRaises(ValueError):
            heatmap_payload(["q0"], ["p0", "p1", "p2"], attention)
        with self.assertRaises(ValueError):
            heatmap_payload(["q0", "q1"], ["p0"], attention)

    def test_rejects_mismatched_dimensions(self):
        with self.assertRaises(ValueError):
            maxsim_attention([[1.0, 2.0]], [[1.0]])


if __name__ == "__main__":
    unittest.main()
