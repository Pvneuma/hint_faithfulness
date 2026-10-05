"""Summarize the paired Harmful - Useful attention-metric difference.

For every sample id, the calculation order is:

1. select one key region, e.g. ``hint_answer``;
2. calculate the paired condition difference ``Harmful - Useful``;
3. average the 32 query heads within every layer;
4. average the 36 layers within every CoT bin;
5. average the paired per-item values across all questions.

The resulting CSV has one row for each of the ten proportional CoT bins.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_METRICS_DIR = Path("output/qwen3_attention_metrics")
DEFAULT_OUTPUT_DIR = Path("output")


@dataclass(frozen=True)
class ConditionDifferenceSummary:
    sample_ids: list[str]
    region: str
    metric: str
    useful_by_bin: np.ndarray
    harmful_by_bin: np.ndarray
    difference_by_bin: np.ndarray
    useful_layer_bin: np.ndarray
    harmful_layer_bin: np.ndarray
    difference_layer_bin: np.ndarray


def paired_paths(metrics_dir: Path) -> list[tuple[str, Path, Path]]:
    useful_dir = metrics_dir / "useful"
    harmful_dir = metrics_dir / "harmful"
    useful = {path.stem: path for path in useful_dir.glob("*.npz")}
    harmful = {path.stem: path for path in harmful_dir.glob("*.npz")}

    if not useful:
        raise ValueError(f"No Useful NPZ files found under {useful_dir}")
    if not harmful:
        raise ValueError(f"No Harmful NPZ files found under {harmful_dir}")

    missing_harmful = sorted(set(useful) - set(harmful))
    missing_useful = sorted(set(harmful) - set(useful))
    if missing_harmful or missing_useful:
        raise ValueError(
            "Useful/Harmful sample ids are not paired. "
            f"Missing Harmful: {missing_harmful[:10]}; "
            f"missing Useful: {missing_useful[:10]}"
        )

    def sort_key(sample_id: str) -> tuple[int, int | str]:
        try:
            return (0, int(sample_id))
        except ValueError:
            return (1, sample_id)

    return [
        (sample_id, useful[sample_id], harmful[sample_id])
        for sample_id in sorted(useful, key=sort_key)
    ]


def load_region_metric(
    path: Path,
    *,
    metric: str,
    region: str,
    expected_condition: str,
    expected_sample_id: str,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if metric not in data.files:
            raise KeyError(f"{path} has no metric array {metric!r}")
        region_names = [str(name) for name in data["key_regions"].tolist()]
        if region not in region_names:
            raise KeyError(
                f"{path} has no region {region!r}; available regions: {region_names}"
            )
        metadata: dict[str, Any] = json.loads(data["metadata_json"].item())
        if metadata.get("condition") != expected_condition:
            raise ValueError(
                f"{path}: metadata condition is {metadata.get('condition')!r}, "
                f"expected {expected_condition!r}"
            )
        if str(metadata.get("sample_id")) != expected_sample_id:
            raise ValueError(
                f"{path}: metadata sample_id is {metadata.get('sample_id')!r}, "
                f"expected {expected_sample_id!r}"
            )

        values = np.asarray(data[metric], dtype=np.float64)
        if values.ndim != 4:
            raise ValueError(
                f"{path}: expected [bin, layer, head, region], got {values.shape}"
            )
        return values[..., region_names.index(region)]


def compute_condition_difference(
    metrics_dir: Path,
    *,
    metric: str = "enrichment",
    region: str = "hint_answer",
) -> ConditionDifferenceSummary:
    pairs = paired_paths(metrics_dir)
    sum_useful_layer_bin: np.ndarray | None = None
    sum_harmful_layer_bin: np.ndarray | None = None

    for sample_id, useful_path, harmful_path in pairs:
        useful = load_region_metric(
            useful_path,
            metric=metric,
            region=region,
            expected_condition="useful",
            expected_sample_id=sample_id,
        )
        harmful = load_region_metric(
            harmful_path,
            metric=metric,
            region=region,
            expected_condition="harmful",
            expected_sample_id=sample_id,
        )
        if useful.shape != harmful.shape:
            raise ValueError(
                f"Sample {sample_id}: Useful shape {useful.shape} != "
                f"Harmful shape {harmful.shape}"
            )

        # [CoT bin, layer, head] -> [CoT bin, layer]
        # The head average happens inside each paired sample, before samples
        # are pooled.  This keeps every question equally weighted.
        useful_head_mean = useful.mean(axis=2)
        harmful_head_mean = harmful.mean(axis=2)

        if sum_useful_layer_bin is None:
            sum_useful_layer_bin = np.zeros_like(useful_head_mean, dtype=np.float64)
            sum_harmful_layer_bin = np.zeros_like(harmful_head_mean, dtype=np.float64)
        sum_useful_layer_bin += useful_head_mean
        assert sum_harmful_layer_bin is not None
        sum_harmful_layer_bin += harmful_head_mean

    sample_count = len(pairs)
    assert sum_useful_layer_bin is not None
    assert sum_harmful_layer_bin is not None
    useful_layer_bin = sum_useful_layer_bin / sample_count
    harmful_layer_bin = sum_harmful_layer_bin / sample_count
    difference_layer_bin = harmful_layer_bin - useful_layer_bin

    # [CoT bin, layer] -> [CoT bin]
    useful_by_bin = useful_layer_bin.mean(axis=1)
    harmful_by_bin = harmful_layer_bin.mean(axis=1)
    difference_by_bin = difference_layer_bin.mean(axis=1)

    return ConditionDifferenceSummary(
        sample_ids=[sample_id for sample_id, _, _ in pairs],
        region=region,
        metric=metric,
        useful_by_bin=useful_by_bin,
        harmful_by_bin=harmful_by_bin,
        difference_by_bin=difference_by_bin,
        useful_layer_bin=useful_layer_bin,
        harmful_layer_bin=harmful_layer_bin,
        difference_layer_bin=difference_layer_bin,
    )


def write_csv(path: Path, summary: ConditionDifferenceSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cot_bin",
                f"useful_mean_{summary.metric}",
                f"harmful_mean_{summary.metric}",
                f"harmful_minus_useful_{summary.metric}",
                "region",
                "paired_sample_count",
            ]
        )
        for bin_index, (useful, harmful, difference) in enumerate(
            zip(
                summary.useful_by_bin,
                summary.harmful_by_bin,
                summary.difference_by_bin,
            ),
            start=1,
        ):
            writer.writerow(
                [
                    bin_index,
                    float(useful),
                    float(harmful),
                    float(difference),
                    summary.region,
                    len(summary.sample_ids),
                ]
            )


def write_layer_npz(path: Path, summary: ConditionDifferenceSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            useful=summary.useful_layer_bin.astype(np.float32),
            harmful=summary.harmful_layer_bin.astype(np.float32),
            harmful_minus_useful=summary.difference_layer_bin.astype(np.float32),
            region=np.asarray(summary.region),
            metric=np.asarray(summary.metric),
            sample_count=np.asarray(len(summary.sample_ids)),
        )


def automatic_output_paths(
    output_dir: Path,
    *,
    metric: str,
    region: str,
) -> tuple[Path, Path]:
    """Build collision-resistant default names from metric and region."""

    safe_metric = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric).strip("_")
    safe_region = re.sub(r"[^A-Za-z0-9_.-]+", "_", region).strip("_")
    if not safe_metric or not safe_region:
        raise ValueError("metric and region must contain filename-safe characters")
    stem = f"attention_{safe_metric}_harmful_minus_useful_{safe_region}"
    return output_dir / f"{stem}.csv", output_dir / f"{stem}_layer_bin.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument(
        "--metric", choices=("mass", "density", "enrichment"), default="enrichment"
    )
    parser.add_argument("--region", default="hint_answer")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used for automatically named output files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional explicit CSV path; otherwise metric and region determine the name.",
    )
    parser.add_argument(
        "--output-layer-npz",
        type=Path,
        default=None,
        help="Optional explicit layer-bin NPZ path; otherwise metric and region determine the name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    automatic_csv, automatic_layer_npz = automatic_output_paths(
        args.output_dir,
        metric=args.metric,
        region=args.region,
    )
    output_csv = args.output_csv or automatic_csv
    output_layer_npz = args.output_layer_npz or automatic_layer_npz
    summary = compute_condition_difference(
        args.metrics_dir,
        metric=args.metric,
        region=args.region,
    )
    write_csv(output_csv, summary)
    write_layer_npz(output_layer_npz, summary)

    print(
        f"Paired samples: {len(summary.sample_ids)}; metric={summary.metric}; "
        f"region={summary.region}"
    )
    print("bin\tuseful\tharmful\tharmful-useful")
    for bin_index, (useful, harmful, difference) in enumerate(
        zip(
            summary.useful_by_bin,
            summary.harmful_by_bin,
            summary.difference_by_bin,
        ),
        start=1,
    ):
        print(f"{bin_index}\t{useful:.6f}\t{harmful:.6f}\t{difference:.6f}")
    print(f"Saved bin summary to {output_csv}")
    print(f"Saved head-averaged layer x bin arrays to {output_layer_npz}")


if __name__ == "__main__":
    main()
