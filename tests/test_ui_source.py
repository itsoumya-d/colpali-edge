import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class HeatmapUiSourceTests(unittest.TestCase):
    def setUp(self):
        self.html = (ROOT / "demo" / "heatmap.html").read_text()

    def test_baseline_constraints(self):
        html = self.html
        self.assertIn("100dvh", html)
        self.assertNotIn("h-screen", html)
        self.assertNotIn("tracking-", html)
        self.assertNotIn("letter-spacing", html)
        self.assertNotIn("backdrop-filter", html)
        self.assertNotIn("blur(", html)
        self.assertNotIn("will-change", html)
        self.assertIn("env(safe-area-inset", html)
        self.assertIn("prefers-reduced-motion", html)
        self.assertIn("text-wrap: balance", html)
        self.assertIn("text-wrap: pretty", html)
        self.assertIn("font-variant-numeric: tabular-nums", html)
        self.assertIn("min-block-size: 44px", html)
        self.assertIn('role="status"', html)
        self.assertIn('aria-live="polite"', html)
        self.assertIn("aria-pressed=", html)
        self.assertIn('aria-label="Document patch heatmap"', html)
        self.assertIn('data-df-color-scale="detailed"', html)
        self.assertIn('data-df-density="compact"', html)

    def test_fixture_is_embedded_and_labeled(self):
        self.assertIn("synthetic fixture", self.html)
        self.assertIn("Fixture only", self.html)
        self.assertNotIn("__HEATMAP_DATA__", self.html)
        self.assertIn('"rows": [', self.html)


if __name__ == "__main__":
    unittest.main()
