from __future__ import annotations

import math
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from scripts.qwen3_attention_metrics import (
    KEY_REGION_ORDER,
    PreparedReplay,
    mass_density_enrichment,
    proportional_bins,
    replay_and_aggregate,
    token_indices_overlapping,
)


class ProportionalBinsTest(unittest.TestCase):
    def test_every_token_is_used_exactly_once(self) -> None:
        positions = list(range(100, 123))
        bins = proportional_bins(positions, 10)

        self.assertEqual([item for group in bins for item in group], positions)
        self.assertEqual(len(bins), 10)
        self.assertLessEqual(max(map(len, bins)) - min(map(len, bins)), 1)

    def test_short_cot_retains_all_tokens(self) -> None:
        positions = [7, 8, 9]
        bins = proportional_bins(positions, 10)

        self.assertEqual([item for group in bins for item in group], positions)
        self.assertEqual(sum(bool(group) for group in bins), 3)


class TokenOverlapTest(unittest.TestCase):
    def test_nonempty_overlaps_only(self) -> None:
        offsets = [(0, 0), (0, 3), (3, 5), (5, 9)]
        self.assertEqual(token_indices_overlapping(offsets, [(2, 6)]), [1, 2, 3])


class AttentionMetricTest(unittest.TestCase):
    def test_uniform_attention_has_zero_enrichment(self) -> None:
        # Query 0 is at global position 3 and sees four keys. Query 1 is at
        # position 5 and sees six. Both rows are uniform over visible keys.
        attention = torch.tensor(
            [
                [
                    [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],
                    [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                ]
            ],
            dtype=torch.float32,
        )
        query_positions = torch.tensor([3, 5])

        mass, density, enrichment = mass_density_enrichment(
            attention,
            key_positions=[0, 1],
            global_query_positions=query_positions,
            epsilon=1e-12,
        )

        torch.testing.assert_close(mass, torch.tensor([[0.5, 1 / 3]]))
        torch.testing.assert_close(density, torch.tensor([[0.25, 1 / 6]]))
        torch.testing.assert_close(enrichment, torch.zeros_like(enrichment), atol=1e-6, rtol=0)

    def test_enrichment_is_log2_multiple_of_uniform_mass(self) -> None:
        attention = torch.tensor(
            [[[0.5, 0.0, 0.25, 0.25]]],
            dtype=torch.float32,
        )
        mass, density, enrichment = mass_density_enrichment(
            attention,
            key_positions=[0],
            global_query_positions=torch.tensor([3]),
            epsilon=1e-12,
        )

        self.assertTrue(math.isclose(float(mass.item()), 0.5))
        self.assertTrue(math.isclose(float(density.item()), 0.5))
        # Uniform mass is 1/4; actual mass is 1/2, i.e. 2x uniform.
        self.assertTrue(math.isclose(float(enrichment.item()), 1.0, abs_tol=1e-6))


class ReplayIntegrationTest(unittest.TestCase):
    def test_tiny_qwen3_replay_returns_expected_shapes(self) -> None:
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
        )
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config).eval()

        cot_positions = list(range(10, 30))
        cot_bins = proportional_bins(cot_positions, 10)
        regions = {
            "hint_answer": [1],
            "hint_source": [2, 3],
            "hint_frame": [4, 5],
            "hint_all": [1, 2, 3, 4, 5],
            "question": [6, 7, 8, 9],
        }
        prepared = PreparedReplay(
            sample_id=0,
            condition="useful",
            prompt_text="",
            response_text="",
            input_ids=[index % 127 for index in range(30)],
            prompt_token_count=10,
            cot_token_positions=cot_positions,
            cot_bins=cot_bins,
            key_regions=regions,
            key_region_tokens={name: [] for name in KEY_REGION_ORDER},
            cot_closed=True,
            hint_label="A",
            reported=False,
            extracted_answer="A",
        )

        mass, density, enrichment, qa = replay_and_aggregate(
            model,
            prepared,
            chunk_size=4,
            epsilon=1e-12,
        )

        self.assertEqual(mass.shape, (10, 2, 4, len(KEY_REGION_ORDER)))
        self.assertEqual(density.shape, mass.shape)
        self.assertEqual(enrichment.shape, mass.shape)
        self.assertTrue(torch.from_numpy(mass).isfinite().all())
        self.assertTrue(torch.from_numpy(enrichment).isfinite().all())
        self.assertEqual(qa["bin_query_counts"], [2] * 10)
        self.assertLess(qa["max_attention_row_sum_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
