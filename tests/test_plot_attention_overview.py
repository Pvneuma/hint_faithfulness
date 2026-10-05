from __future__ import annotations

import unittest

import numpy as np

from scripts.plot_attention_overview import paired_bootstrap_curves


class PairedBootstrapTest(unittest.TestCase):
    def test_means_and_ci_shapes(self) -> None:
        useful = np.arange(5 * 10 * 2, dtype=np.float64).reshape(5, 10, 2)
        harmful = useful + 2.0

        result = paired_bootstrap_curves(
            useful,
            harmful,
            bootstrap_samples=100,
            seed=7,
        )

        np.testing.assert_allclose(result["difference_mean"], 2.0)
        self.assertEqual(result["useful_ci"].shape, (2, 10, 2))
        self.assertEqual(result["harmful_ci"].shape, (2, 10, 2))
        self.assertEqual(result["difference_ci"].shape, (2, 10, 2))
        np.testing.assert_allclose(result["difference_ci"], 2.0)


if __name__ == "__main__":
    unittest.main()
