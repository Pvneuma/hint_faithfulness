"""Create the three pre-specified overview figures for attention analysis.

Figures
-------
1. Useful/Harmful curves across ten CoT bins with paired-bootstrap 95% CIs,
   combining ``hint_answer`` and ``hint_source`` in one plot.
2. Harmful - Useful Layer x CoT-bin heatmaps, averaged over samples and heads.
3. Useful Reported - Unreported Layer x Head heatmaps at CoT bin 1.

All statistics are calculated from the original per-sample NPZ files under
``output/qwen3_attention_metrics``.  This preserves item pairing and report
labels; the previously generated summary NPZ files are not used.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__:
    from .summarize_attention_condition_difference import paired_paths
else:
    from summarize_attention_condition_difference import paired_paths


DEFAULT_METRICS_DIR = Path("output/qwen3_attention_metrics")
DEFAULT_OUTPUT_DIR = Path("output/plots/attention_overview")
REGIONS = ("hint_answer", "hint_source")
REGION_TITLES = {
    "hint_answer": "Hint answer",
    "hint_source": "Hint source",
}


@dataclass(frozen=True)
class OverviewData:
    sample_ids: list[str]
    metric: str
    regions: tuple[str, ...]
    useful_curves: np.ndarray
    harmful_curves: np.ndarray
    useful_layer_bin: np.ndarray
    harmful_layer_bin: np.ndarray
    useful_reported_bin1: np.ndarray
    useful_unreported_bin1: np.ndarray
    reported_count: int
    unreported_count: int


def configure_matplotlib() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "hint_faithfulness_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def load_regions(
    path: Path,
    *,
    metric: str,
    regions: Sequence[str],
    expected_condition: str,
    expected_sample_id: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load selected regions as [bin, layer, head, selected_region]."""

    with np.load(path, allow_pickle=False) as data:
        if metric not in data.files:
            raise KeyError(f"{path} has no metric array {metric!r}")
        available_regions = [str(name) for name in data["key_regions"].tolist()]
        missing = [region for region in regions if region not in available_regions]
        if missing:
            raise KeyError(
                f"{path} lacks regions {missing}; available: {available_regions}"
            )
        metadata: dict[str, Any] = json.loads(data["metadata_json"].item())
        if metadata.get("condition") != expected_condition:
            raise ValueError(
                f"{path}: condition {metadata.get('condition')!r} != "
                f"{expected_condition!r}"
            )
        if str(metadata.get("sample_id")) != expected_sample_id:
            raise ValueError(
                f"{path}: sample id {metadata.get('sample_id')!r} != "
                f"{expected_sample_id!r}"
            )
        values = np.asarray(data[metric], dtype=np.float64)
        if values.ndim != 4:
            raise ValueError(
                f"{path}: expected [bin, layer, head, region], got {values.shape}"
            )
        selected = np.stack(
            [values[..., available_regions.index(region)] for region in regions],
            axis=-1,
        )
        return selected, metadata


def collect_overview_data(
    metrics_dir: Path,
    *,
    metric: str,
    regions: Sequence[str] = REGIONS,
) -> OverviewData:
    pairs = paired_paths(metrics_dir)
    useful_curves: list[np.ndarray] = []
    harmful_curves: list[np.ndarray] = []
    useful_layer_bin_sum: np.ndarray | None = None
    harmful_layer_bin_sum: np.ndarray | None = None
    reported_bin1_sum: np.ndarray | None = None
    unreported_bin1_sum: np.ndarray | None = None
    reported_count = 0
    unreported_count = 0

    for sample_id, useful_path, harmful_path in pairs:
        useful, useful_metadata = load_regions(
            useful_path,
            metric=metric,
            regions=regions,
            expected_condition="useful",
            expected_sample_id=sample_id,
        )
        harmful, _ = load_regions(
            harmful_path,
            metric=metric,
            regions=regions,
            expected_condition="harmful",
            expected_sample_id=sample_id,
        )
        if useful.shape != harmful.shape:
            raise ValueError(
                f"Sample {sample_id}: Useful shape {useful.shape} != "
                f"Harmful shape {harmful.shape}"
            )

        # Per-item curves: average all layers and all query heads while
        # retaining CoT bin and semantic key region.
        useful_curves.append(useful.mean(axis=(1, 2)))
        harmful_curves.append(harmful.mean(axis=(1, 2)))

        # Layer x bin data: average heads, then pool samples below.
        useful_head_mean = useful.mean(axis=2)
        harmful_head_mean = harmful.mean(axis=2)
        if useful_layer_bin_sum is None:
            useful_layer_bin_sum = np.zeros_like(useful_head_mean)
            harmful_layer_bin_sum = np.zeros_like(harmful_head_mean)
        useful_layer_bin_sum += useful_head_mean
        assert harmful_layer_bin_sum is not None
        harmful_layer_bin_sum += harmful_head_mean

        # Bin 1 retains every layer and head. This is a descriptive contrast;
        # CoT length adjustment belongs in the later inferential analysis.
        reported = useful_metadata.get("reported")
        if reported is True:
            if reported_bin1_sum is None:
                reported_bin1_sum = np.zeros_like(useful[0])
            reported_bin1_sum += useful[0]
            reported_count += 1
        elif reported is False:
            if unreported_bin1_sum is None:
                unreported_bin1_sum = np.zeros_like(useful[0])
            unreported_bin1_sum += useful[0]
            unreported_count += 1
        else:
            raise ValueError(
                f"Sample {sample_id}: Useful metadata has no boolean report label"
            )

    if (
        useful_layer_bin_sum is None
        or harmful_layer_bin_sum is None
        or reported_bin1_sum is None
        or unreported_bin1_sum is None
    ):
        raise ValueError("The dataset did not contain all required groups")

    sample_count = len(pairs)
    return OverviewData(
        sample_ids=[sample_id for sample_id, _, _ in pairs],
        metric=metric,
        regions=tuple(regions),
        useful_curves=np.stack(useful_curves),
        harmful_curves=np.stack(harmful_curves),
        useful_layer_bin=useful_layer_bin_sum / sample_count,
        harmful_layer_bin=harmful_layer_bin_sum / sample_count,
        useful_reported_bin1=reported_bin1_sum / reported_count,
        useful_unreported_bin1=unreported_bin1_sum / unreported_count,
        reported_count=reported_count,
        unreported_count=unreported_count,
    )


def paired_bootstrap_curves(
    useful: np.ndarray,
    harmful: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Paired item bootstrap for condition curves.

    ``useful`` and ``harmful`` are [sample, bin, region]. The same resampled
    item indices are applied to both conditions on each bootstrap draw.
    """

    if useful.shape != harmful.shape:
        raise ValueError("Useful and Harmful bootstrap arrays must have equal shape")
    if useful.ndim != 3:
        raise ValueError(f"Expected [sample, bin, region], got {useful.shape}")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    rng = np.random.default_rng(seed)
    sample_count = useful.shape[0]
    indices = rng.integers(
        0, sample_count, size=(bootstrap_samples, sample_count), endpoint=False
    )
    useful_bootstrap = useful[indices].mean(axis=1)
    harmful_bootstrap = harmful[indices].mean(axis=1)
    difference_bootstrap = (harmful - useful)[indices].mean(axis=1)
    return {
        "useful_mean": useful.mean(axis=0),
        "harmful_mean": harmful.mean(axis=0),
        "difference_mean": (harmful - useful).mean(axis=0),
        "useful_ci": np.percentile(useful_bootstrap, [2.5, 97.5], axis=0),
        "harmful_ci": np.percentile(harmful_bootstrap, [2.5, 97.5], axis=0),
        "difference_ci": np.percentile(difference_bootstrap, [2.5, 97.5], axis=0),
    }


def symmetric_limit(values: Sequence[np.ndarray]) -> float:
    maximum = max(float(np.nanmax(np.abs(value))) for value in values)
    return maximum if maximum > 0 else 1.0


def plot_condition_curves(
    data: OverviewData,
    *,
    output_path: Path,
    bootstrap_samples: int,
    seed: int,
    dpi: int,
) -> None:
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bootstrap = paired_bootstrap_curves(
        data.useful_curves,
        data.harmful_curves,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    x = np.arange(1, data.useful_curves.shape[1] + 1)
    fig, axis = plt.subplots(1, 1, figsize=(9, 5.5))
    styles = {
        ("hint_answer", "useful"): ("#1f77b4", "o"),
        ("hint_answer", "harmful"): ("#d62728", "s"),
        ("hint_source", "useful"): ("#2ca02c", "^"),
        ("hint_source", "harmful"): ("#9467bd", "D"),
    }
    region_labels = {
        "hint_answer": "answer",
        "hint_source": "source",
    }

    for region_index, region in enumerate(data.regions):
        useful_mean = bootstrap["useful_mean"][:, region_index]
        harmful_mean = bootstrap["harmful_mean"][:, region_index]
        useful_ci = bootstrap["useful_ci"][:, :, region_index]
        harmful_ci = bootstrap["harmful_ci"][:, :, region_index]
        region_label = region_labels.get(region, region)
        useful_color, useful_marker = styles.get(
            (region, "useful"), ("#1f77b4", "o")
        )
        harmful_color, harmful_marker = styles.get(
            (region, "harmful"), ("#d62728", "s")
        )

        axis.plot(
            x,
            useful_mean,
            color=useful_color,
            marker=useful_marker,
            label=f"useful hint / {region_label}",
        )
        axis.fill_between(
            x, useful_ci[0], useful_ci[1], color=useful_color, alpha=0.2
        )
        axis.plot(
            x,
            harmful_mean,
            color=harmful_color,
            marker=harmful_marker,
            label=f"harmful hint / {region_label}",
        )
        axis.fill_between(
            x, harmful_ci[0], harmful_ci[1], color=harmful_color, alpha=0.2
        )

    metric_label = data.metric.title()
    title = f"Attention {metric_label} for Hint Answer and Source Across CoT Bins"
    axis.set_xlabel("CoT bin")
    axis.set_ylabel(f"Mean {metric_label}")
    axis.set_xticks(x)
    axis.margins(x=0.01)
    axis.grid(True, linestyle="--", alpha=0.3)
    handles, labels = axis.get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        prop={"size": 14},
        handletextpad=0.5,
        columnspacing=0.9,
        borderpad=0.45,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#666666",
    )
    legend.get_frame().set_linewidth(0.8)
    fig.suptitle(title, fontsize=14, y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_layer_bin_heatmaps(
    data: OverviewData,
    *,
    output_path: Path,
    dpi: int,
) -> None:
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    difference = data.harmful_layer_bin - data.useful_layer_bin
    panels = [difference[..., index].T for index in range(len(data.regions))]
    limit = symmetric_limit(panels)

    fig, axes = plt.subplots(1, len(panels), figsize=(13, 8), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    image = None
    for axis, panel, region in zip(axes, panels, data.regions):
        image = axis.imshow(
            panel,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(REGION_TITLES.get(region, region))
        axis.set_xlabel("CoT bin")
        axis.set_xticks(np.arange(10), np.arange(1, 11))
        axis.set_yticks(np.arange(0, panel.shape[0], 5))
    axes[0].set_ylabel("Layer")
    assert image is not None
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.09, top=0.88, wspace=0.12)
    colorbar_axis = fig.add_axes((0.89, 0.18, 0.018, 0.64))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(f"Harmful - Useful {data.metric}")
    fig.suptitle(
        f"Condition difference by layer and CoT phase\n"
        f"Averaged over samples and 32 query heads, n={len(data.sample_ids)}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_useful_report_heatmaps(
    data: OverviewData,
    *,
    output_path: Path,
    dpi: int,
) -> None:
    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    difference = data.useful_reported_bin1 - data.useful_unreported_bin1
    panels = [difference[..., index] for index in range(len(data.regions))]
    limit = symmetric_limit(panels)

    fig, axes = plt.subplots(1, len(panels), figsize=(14, 8), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    image = None
    for axis, panel, region in zip(axes, panels, data.regions):
        image = axis.imshow(
            panel,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(REGION_TITLES.get(region, region))
        axis.set_xlabel("Attention Head")
        axis.set_xticks(np.arange(0, panel.shape[1], 4))
        axis.set_yticks(np.arange(0, panel.shape[0], 5))
    axes[0].set_ylabel("Layer")
    assert image is not None
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.09, top=0.88, wspace=0.12)
    colorbar_axis = fig.add_axes((0.89, 0.18, 0.018, 0.64))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(f"Useful Reported - Unreported {data.metric}")
    fig.suptitle(
        "Useful hint reporting contrast at CoT bin 1\n"
        f"Reported n={data.reported_count}, Unreported n={data.unreported_count}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metric", choices=("mass", "density", "enrichment"), default="enrichment"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")

    data = collect_overview_data(args.metrics_dir, metric=args.metric)
    prefix = f"attention_{args.metric}"
    paths = {
        "condition_curves": args.output_dir / f"{prefix}_condition_curves.png",
        "layer_bin": args.output_dir / f"{prefix}_harmful_minus_useful_layer_bin.png",
        "useful_report": args.output_dir / f"{prefix}_useful_reported_minus_unreported_bin1.png",
    }
    plot_condition_curves(
        data,
        output_path=paths["condition_curves"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        dpi=args.dpi,
    )
    plot_layer_bin_heatmaps(data, output_path=paths["layer_bin"], dpi=args.dpi)
    plot_useful_report_heatmaps(data, output_path=paths["useful_report"], dpi=args.dpi)

    print(
        f"Loaded {len(data.sample_ids)} paired samples; Useful reported="
        f"{data.reported_count}, unreported={data.unreported_count}"
    )
    for name, path in paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
