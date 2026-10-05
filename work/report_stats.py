from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_attention_overview import (
    collect_overview_data,
    paired_bootstrap_curves,
)


METRICS_DIR = Path("output/qwen3_attention_metrics")


def print_curve_table(metric: str) -> None:
    data = collect_overview_data(METRICS_DIR, metric=metric)
    bootstrap = paired_bootstrap_curves(
        data.useful_curves,
        data.harmful_curves,
        bootstrap_samples=2_000,
        seed=42,
    )
    print(f"\n## {metric} curves")
    print("region,bin,useful,harmful,H-U,CI_low,CI_high")
    for region_index, region in enumerate(data.regions):
        for bin_index in range(data.useful_curves.shape[1]):
            print(
                f"{region},{bin_index + 1},"
                f"{bootstrap['useful_mean'][bin_index, region_index]:.8f},"
                f"{bootstrap['harmful_mean'][bin_index, region_index]:.8f},"
                f"{bootstrap['difference_mean'][bin_index, region_index]:.8f},"
                f"{bootstrap['difference_ci'][0, bin_index, region_index]:.8f},"
                f"{bootstrap['difference_ci'][1, bin_index, region_index]:.8f}"
            )


def print_heatmap_blocks() -> None:
    data = collect_overview_data(METRICS_DIR, metric="enrichment")
    difference = data.harmful_layer_bin - data.useful_layer_bin
    bin_bands = ((0, 3, "bins1-3"), (3, 7, "bins4-7"), (7, 10, "bins8-10"))
    layer_bands = ((0, 12, "layers0-11"), (12, 24, "layers12-23"), (24, 36, "layers24-35"))
    print("\n## enrichment H-U heatmap block means")
    print("region,layer_band,bin_band,mean")
    for region_index, region in enumerate(data.regions):
        for layer_start, layer_end, layer_name in layer_bands:
            for bin_start, bin_end, bin_name in bin_bands:
                value = difference[
                    bin_start:bin_end, layer_start:layer_end, region_index
                ].mean()
                print(f"{region},{layer_name},{bin_name},{value:.6f}")

    report_difference = (
        data.useful_reported_bin1 - data.useful_unreported_bin1
    )
    print("\n## Useful Reported-Unreported at bin1")
    print("region,layer_band,mean,positive_cell_fraction")
    for region_index, region in enumerate(data.regions):
        panel = report_difference[..., region_index]
        for layer_start, layer_end, layer_name in layer_bands:
            block = panel[layer_start:layer_end]
            print(
                f"{region},{layer_name},{block.mean():.6f},"
                f"{np.mean(block > 0):.4f}"
            )
        flat_indices = np.argsort(np.abs(panel), axis=None)[-5:][::-1]
        print(f"top_abs_cells {region}")
        for flat_index in flat_indices:
            layer, head = np.unravel_index(flat_index, panel.shape)
            print(f"layer={layer},head={head},difference={panel[layer, head]:.6f}")


def print_metadata_summary() -> None:
    print("\n## metadata")
    for condition in ("useful", "harmful"):
        cot_lengths = []
        reported = []
        closed = []
        for path in sorted((METRICS_DIR / condition).glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(data["metadata_json"].item())
            cot_lengths.append(metadata["cot_token_count"])
            reported.append(metadata["reported"])
            closed.append(metadata["cot_closed"])
        values = np.asarray(cot_lengths)
        print(
            condition,
            "n=", len(values),
            "cot_mean=", f"{values.mean():.2f}",
            "cot_median=", f"{np.median(values):.2f}",
            "cot_q1=", f"{np.quantile(values, .25):.2f}",
            "cot_q3=", f"{np.quantile(values, .75):.2f}",
            "reported=", sum(value is True for value in reported),
            "closed=", sum(bool(value) for value in closed),
        )


if __name__ == "__main__":
    print_curve_table("enrichment")
    print_curve_table("mass")
    print_heatmap_blocks()
    print_metadata_summary()
