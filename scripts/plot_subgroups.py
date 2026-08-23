"""Plot helpful subgroup bootstrap results (reported vs unreported) for attn/resid.

Expects JSON outputs from `bootstrap_helpful_subgroups.py`:
  - bootstrap_helpful_reported_attn_top.json
  - bootstrap_helpful_unreported_attn_top.json
  - bootstrap_helpful_reported_resid_post_top.json
  - bootstrap_helpful_unreported_resid_post_top.json

For each metric (occurrence, mrr), generates one PNG containing
four curves: attn/resid × reported/unreported, with 95% CI shading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


LAYERS = 36
METRICS = ["occurrence", "mrr"]


def _to_array(values: List) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _load(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def _plot(ax, x, curves, title, ylabel):
    colors = {
        "attn_reported": "#1f77b4",
        "attn_unreported": "#d62728",
        "resid_reported": "#2ca02c",
        "resid_unreported": "#9467bd",
    }
    markers = {
        "attn_reported": "o",
        "attn_unreported": "s",
        "resid_reported": "^",
        "resid_unreported": "D",
    }
    labels = {
        "attn_reported": "MHA / helpful-reported",
        "attn_unreported": "MHA / helpful-unreported",
        "resid_reported": "MLP / helpful-reported",
        "resid_unreported": "MLP / helpful-unreported",
    }

    for key, (base, ci) in curves.items():
        c = colors[key]
        ax.plot(
            x,
            base,
            label=labels[key],
            color=c,
            marker=markers[key],
            markevery=1,
            linewidth=1.5,
            markersize=5,
        )
        ax.fill_between(x, ci[0], ci[1], color=c, alpha=0.2)

    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(x), max(x))
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_all(attn_rep, attn_unrep, resid_rep, resid_unrep, out_dir: Path):
    import matplotlib.pyplot as plt

    x = np.arange(LAYERS)

    for metric in METRICS:
        curves = {
            "attn_reported": (
                _to_array(attn_rep["baseline"][metric]),
                np.vstack([
                    _to_array(attn_rep["ci"][metric][0]),
                    _to_array(attn_rep["ci"][metric][1]),
                ]),
            ),
            "attn_unreported": (
                _to_array(attn_unrep["baseline"][metric]),
                np.vstack([
                    _to_array(attn_unrep["ci"][metric][0]),
                    _to_array(attn_unrep["ci"][metric][1]),
                ]),
            ),
            "resid_reported": (
                _to_array(resid_rep["baseline"][metric]),
                np.vstack([
                    _to_array(resid_rep["ci"][metric][0]),
                    _to_array(resid_rep["ci"][metric][1]),
                ]),
            ),
            "resid_unreported": (
                _to_array(resid_unrep["baseline"][metric]),
                np.vstack([
                    _to_array(resid_unrep["ci"][metric][0]),
                    _to_array(resid_unrep["ci"][metric][1]),
                ]),
            ),
        }

        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        _plot(ax, x, curves, title=f"Mean Reciprocal Rank (MHA/MLP, helpful-reported/helpful-unreported)", ylabel=metric)

        fig.tight_layout()
        out_path = out_dir / f"plot_helpful_subgroups_{metric}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("output"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/plots"))
    args = parser.parse_args()

    files = {
        "attn_rep": args.input_dir / "bootstrap_helpful_reported_attn_top.json",
        "attn_unrep": args.input_dir / "bootstrap_helpful_unreported_attn_top.json",
        "resid_rep": args.input_dir / "bootstrap_helpful_reported_resid_post_top.json",
        "resid_unrep": args.input_dir / "bootstrap_helpful_unreported_resid_post_top.json",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing bootstrap file(s): {missing}")

    attn_rep = _load(files["attn_rep"])
    attn_unrep = _load(files["attn_unrep"])
    resid_rep = _load(files["resid_rep"])
    resid_unrep = _load(files["resid_unrep"])

    plot_all(attn_rep, attn_unrep, resid_rep, resid_unrep, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
