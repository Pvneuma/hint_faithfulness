"""Plot bootstrap results for logit-lens metrics.

Reads the JSON outputs produced by `bootstrap_analysis.py` and generates
per-metric plots with helpful/harmful curves (with 95% CI shading) for
both `attn_top` and `resid_post_top`.

Outputs PNG files into a configurable directory (default: output/plots).
Each metric plot contains four curves:
- attn_top helpful / harmful
- resid_post_top helpful / harmful
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


LAYERS = 36
METRICS = ["occurrence", "mean_logit", "has"]


def _to_array(values: List) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _load_result(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def _plot_four_curves(ax, x, curves, title, ylabel):
    colors = {
        "attn_help": "#1f77b4",
        "attn_harm": "#d62728",
        "resid_help": "#2ca02c",
        "resid_harm": "#9467bd",
    }
    markers = {
        "attn_help": "o",
        "attn_harm": "s",
        "resid_help": "^",
        "resid_harm": "D",
    }
    labels = {
        "attn_help": "attn helpful",
        "attn_harm": "attn harmful",
        "resid_help": "resid helpful",
        "resid_harm": "resid harmful",
    }

    for key, (base, ci) in curves.items():
        c = colors[key]
        ax.plot(x, base, label=labels[key], color=c, marker=markers[key], markevery=1, linewidth=1.5, markersize=4)
        ax.fill_between(x, ci[0], ci[1], color=c, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(x), max(x))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_combined(attn_result: Dict, resid_result: Dict, out_dir: Path):
    import matplotlib.pyplot as plt

    x = np.arange(LAYERS)

    for metric in METRICS:
        curves = {
            "attn_help": (
                _to_array(attn_result["baseline"]["helpful"][metric]),
                np.vstack([
                    _to_array(attn_result["ci"]["helpful"][metric][0]),
                    _to_array(attn_result["ci"]["helpful"][metric][1]),
                ]),
            ),
            "attn_harm": (
                _to_array(attn_result["baseline"]["harmful"][metric]),
                np.vstack([
                    _to_array(attn_result["ci"]["harmful"][metric][0]),
                    _to_array(attn_result["ci"]["harmful"][metric][1]),
                ]),
            ),
            "resid_help": (
                _to_array(resid_result["baseline"]["helpful"][metric]),
                np.vstack([
                    _to_array(resid_result["ci"]["helpful"][metric][0]),
                    _to_array(resid_result["ci"]["helpful"][metric][1]),
                ]),
            ),
            "resid_harm": (
                _to_array(resid_result["baseline"]["harmful"][metric]),
                np.vstack([
                    _to_array(resid_result["ci"]["harmful"][metric][0]),
                    _to_array(resid_result["ci"]["harmful"][metric][1]),
                ]),
            ),
        }

        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        _plot_four_curves(
            ax,
            x,
            curves,
            title=f"{metric} (attn/resid, helpful/harmful)",
            ylabel=metric,
        )

        fig.tight_layout()
        out_path = out_dir / f"plot_{metric}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("output"), help="Directory containing bootstrap JSON files.")
    parser.add_argument("--out-dir", type=Path, default=Path("output/plots"), help="Directory to save plots.")
    args = parser.parse_args()

    attn_path = args.input_dir / "bootstrap_attn_top.json"
    resid_path = args.input_dir / "bootstrap_resid_post_top.json"
    missing = [str(p) for p in (attn_path, resid_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing bootstrap file(s): {missing}")

    attn_result = _load_result(attn_path)
    resid_result = _load_result(resid_path)
    plot_combined(attn_result, resid_result, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
