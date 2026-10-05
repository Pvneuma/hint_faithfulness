from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.summarize_attention_condition_difference import (
    automatic_output_paths,
    compute_condition_difference,
)


class ConditionDifferenceTest(unittest.TestCase):
    @staticmethod
    def _write_sample(
        root: Path,
        condition: str,
        sample_id: int,
        enrichment: np.ndarray,
    ) -> None:
        path = root / condition / f"{sample_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"sample_id": sample_id, "condition": condition}
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                enrichment=enrichment,
                mass=enrichment,
                density=enrichment,
                key_regions=np.asarray(["hint_answer", "hint_source"]),
                metadata_json=np.asarray(json.dumps(metadata)),
            )

    def test_pair_then_average_heads_layers_and_samples(self) -> None:
        # Shape: [2 bins, 2 layers, 2 heads, 2 regions].
        # Useful values are 1 and 3 for the two samples; Harmful values are
        # respectively +2 and +4 higher. The final paired mean must be +3.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sample_id, useful_value, difference in ((0, 1.0, 2.0), (1, 3.0, 4.0)):
                useful = np.full((2, 2, 2, 2), useful_value, dtype=np.float32)
                harmful = useful + difference
                self._write_sample(root, "useful", sample_id, useful)
                self._write_sample(root, "harmful", sample_id, harmful)

            summary = compute_condition_difference(
                root,
                metric="enrichment",
                region="hint_answer",
            )

            np.testing.assert_allclose(summary.useful_by_bin, [2.0, 2.0])
            np.testing.assert_allclose(summary.harmful_by_bin, [5.0, 5.0])
            np.testing.assert_allclose(summary.difference_by_bin, [3.0, 3.0])
            np.testing.assert_allclose(summary.difference_layer_bin, 3.0)

    def test_default_names_include_metric_and_region(self) -> None:
        csv_path, npz_path = automatic_output_paths(
            Path("output"), metric="enrichment", region="hint_source"
        )

        self.assertEqual(
            csv_path,
            Path("output/attention_enrichment_harmful_minus_useful_hint_source.csv"),
        )
        self.assertEqual(
            npz_path,
            Path(
                "output/attention_enrichment_harmful_minus_useful_hint_source_layer_bin.npz"
            ),
        )


if __name__ == "__main__":
    unittest.main()
