"""Analyze logit lens outputs for hint-related tokens.

This script streams the provided jsonl logit lens files and computes two
metrics per layer for both `attn_top` and `resid_post_top` separately:

1. Occurrence Frequency: fraction of positions where any hint token appears
   in the Top-5 decoded logits for that layer.
2. Conditional Mean Logit: mean logit of hint-token entries (conditional on
   appearance) for that layer.

Defaults match the repository file layout but can be overridden via CLI.
Hint tokens are configured by editing the `HINT_TOKENS` list below.
The script is designed to be memory efficient and can optionally limit the
number of examples processed for quick smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple


HintTokens = Set[str]
LayerStats = Tuple[Dict[int, int], Dict[int, int], Dict[int, float], Dict[int, int]]

# Configure hint-related tokens here (case-insensitive)
HINT_TOKENS: List[str] = ["stanford", "professor","b","user","given"]


def _normalize_tokens(raw: Sequence[str]) -> HintTokens:
    return {t.strip().lower() for t in raw if t.strip()}


def process_logitlens_file(
    path: Path, top_field: str, hint_tokens: HintTokens, limit: int | None = None
) -> LayerStats:
    """Stream a logit lens jsonl file and accumulate statistics.

    Returns four dicts keyed by layer id:
    - total_positions: total number of positions observed for the layer
    - hint_hits: positions where any hint token is in Top-5
    - logit_sum: sum of logits for hint-token entries
    - logit_count: count of hint-token entries
    """

    total_positions: Dict[int, int] = defaultdict(int)
    hint_hits: Dict[int, int] = defaultdict(int)
    logit_sum: Dict[int, float] = defaultdict(float)
    logit_count: Dict[int, int] = defaultdict(int)

    with path.open() as f:
        for line_idx, line in enumerate(f):
            if limit is not None and line_idx >= limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)

            for position in record.get("positions", []):
                for layer_entry in position.get("layers", []):
                    layer_id = layer_entry.get("layer")
                    if layer_id is None:
                        continue

                    total_positions[layer_id] += 1

                    found_hint = False
                    for item in layer_entry.get(top_field, []):
                        token = str(item.get("token", "")).strip().lower()
                        if token in hint_tokens:
                            found_hint = True
                            logit_sum[layer_id] += float(item.get("logit", 0.0))
                            logit_count[layer_id] += 1

                    if found_hint:
                        hint_hits[layer_id] += 1

    return total_positions, hint_hits, logit_sum, logit_count


def _dicts_to_rows(
    dataset_name: str,
    top_field: str,
    total_positions: Mapping[int, int],
    hint_hits: Mapping[int, int],
    logit_sum: Mapping[int, float],
    logit_count: Mapping[int, int],
):
    rows = []
    for layer in sorted(total_positions):
        total = total_positions[layer]
        hits = hint_hits.get(layer, 0)
        occur_ratio = (hits / total) if total else 0.0
        occur_pct = occur_ratio * 100
        mean_logit = (
            logit_sum.get(layer, 0.0) / logit_count[layer]
            if logit_count.get(layer, 0)
            else float("nan")
        )
        if total and logit_count.get(layer, 0) and hits:
            activation = (logit_sum[layer] * hits) / (logit_count[layer] * total)
        else:
            activation = float("nan")

        rows.append(
            {
                "dataset": dataset_name,
                "top_field": top_field,
                "layer": layer,
                "total_positions": total,
                "hint_hits": hits,
                "occurrence_frequency_pct": occur_pct,
                "mean_logit": mean_logit,
                "hint_activation_score": activation,
            }
        )
    return rows


def _format_table(
    label: str,
    total_positions: Mapping[int, int],
    hint_hits: Mapping[int, int],
    logit_sum: Mapping[int, float],
    logit_count: Mapping[int, int],
) -> str:
    headers = "Layer\tTotal\tHit\tOccur%\tMeanLogit\tActivation"
    lines: List[str] = [f"=== {label} ===", headers]

    for layer in sorted(total_positions):
        total = total_positions[layer]
        hits = hint_hits.get(layer, 0)
        occur_ratio = (hits / total) if total else 0.0
        occur_pct = occur_ratio * 100
        mean_logit = (logit_sum.get(layer, 0.0) / logit_count[layer]) if logit_count.get(layer, 0) else float("nan")
        if total and logit_count.get(layer, 0) and hits:
            activation = (logit_sum[layer] * hits) / (logit_count[layer] * total)
        else:
            activation = float("nan")
        lines.append(
            f"{layer}\t{total}\t{hits}\t{occur_pct:.2f}%\t{mean_logit:.4f}\t{activation:.4f}"
        )

    return "\n".join(lines)


def analyze_one_condition(
    dataset_path: Path,
    hint_tokens: HintTokens,
    limit: int | None = None,
) -> Tuple[List[str], List[Dict[str, object]]]:
    outputs: List[str] = []
    rows: List[Dict[str, object]] = []

    for top_field in ("attn_top", "resid_post_top"):
        totals, hits, log_sum, log_cnt = process_logitlens_file(
            dataset_path, top_field=top_field, hint_tokens=hint_tokens, limit=limit
        )

        rows.extend(
            _dicts_to_rows(
                dataset_name=dataset_path.name,
                top_field=top_field,
                total_positions=totals,
                hint_hits=hits,
                logit_sum=log_sum,
                logit_count=log_cnt,
            )
        )

        outputs.append(
            _format_table(
                label=f"{dataset_path.name} | {top_field}",
                total_positions=totals,
                hint_hits=hits,
                logit_sum=log_sum,
                logit_count=log_cnt,
            )
        )

    return outputs, rows


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--useful",
        type=Path,
        default=Path("output/qwen3_useful_logitlens.jsonl"),
        help="Path to logit lens jsonl for useful hints.",
    )
    parser.add_argument(
        "--harmful",
        type=Path,
        default=Path("output/qwen3_harmful_logitlens.jsonl"),
        help="Path to logit lens jsonl for harmful hints.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of lines to process for quick tests.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Directory to write per-dataset CSV outputs.",
    )

    args = parser.parse_args(argv)
    hint_tokens = _normalize_tokens(HINT_TOKENS)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [("Useful", args.useful), ("Harmful", args.harmful)]

    for label, path in datasets:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        tables, rows = analyze_one_condition(path, hint_tokens=hint_tokens, limit=args.limit)
        print(f"\n##### {label} ({path}) #####")
        for table in tables:
            print(table)
            print()

        # Write CSV per dataset path (separately for attn/resid already combined)
        csv_path = out_dir / f"{path.stem}_metrics.csv"
        with csv_path.open("w", newline="") as csvfile:
            fieldnames = [
                "dataset",
                "top_field",
                "layer",
                "total_positions",
                "hint_hits",
                "occurrence_frequency_pct",
                "mean_logit",
                "hint_activation_score",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
