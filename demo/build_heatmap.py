"""Generate a synthetic MaxSim attention fixture and inject it into the UI."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from colpali_edge.heatmap import heatmap_payload, maxsim_attention

HERE = Path(__file__).resolve().parent
ROWS, COLS = 6, 8
PATCHES = ROWS * COLS
QUERY_LABELS = ["chart title", "y-axis label", "legend", "Q3 bar", "footnote"]
PATCH_LABELS = [f"P{index + 1:02d}" for index in range(PATCHES)]
TARGETS = [3, 17, 25, 36, 47]
SIGMA = 2.2


def build_scores() -> list[list[float]]:
    rng = random.Random(19)
    rows = []
    for target in TARGETS:
        row = []
        for patch in range(PATCHES):
            distance = abs(patch - target)
            primary = math.exp(-(distance * distance) / (2 * SIGMA * SIGMA))
            secondary_target = (target + 17) % PATCHES
            secondary = 0.35 * math.exp(
                -((patch - secondary_target) ** 2) / (2 * (SIGMA * 1.6) ** 2)
            )
            noise = rng.uniform(0.0, 0.04)
            row.append(round(primary + secondary + noise, 6))
        rows.append(row)
    return rows


def main() -> None:
    scores = build_scores()
    # Query token j is a one-hot vector e_j; patch p is its score column, so
    # the dot-product matrix equals the desired score matrix exactly.
    query = [[1.0 if j == k else 0.0 for k in range(len(QUERY_LABELS))]
             for j in range(len(QUERY_LABELS))]
    document = [[scores[j][p] for j in range(len(QUERY_LABELS))]
                for p in range(PATCHES)]

    attention = maxsim_attention(query, document)
    payload = heatmap_payload(QUERY_LABELS, PATCH_LABELS, attention)
    payload["page"] = {
        "rows": ROWS,
        "cols": COLS,
        "title": "Synthetic quarterly revenue page",
        "note": "Fixture only. Patch coordinates do not come from a real PDF.",
    }

    data_path = HERE / "heatmap-data.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")

    template = (HERE / "heatmap.template.html").read_text()
    if "__HEATMAP_DATA__" not in template:
        raise SystemExit("template is missing __HEATMAP_DATA__")
    (HERE / "heatmap.html").write_text(
        template.replace("__HEATMAP_DATA__", data_path.read_text().strip())
    )
    print(json.dumps({
        "queries": len(QUERY_LABELS),
        "patches": PATCHES,
        "wrote": str(HERE / "heatmap.html"),
    }))


if __name__ == "__main__":
    main()
