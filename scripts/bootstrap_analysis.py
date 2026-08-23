"""Paired bootstrap analysis for hint-related logit-lens metrics.

Computes Occurrence Rate and Mean Rank Score
for helpful vs harmful conditions with 95% percentile
bootstrap CIs (paired by question id).

Requirements
------------
- Bootstrap unit: question (not token/position/layer)
- Paired bootstrap: helpful/harmful sampled together per question
- Bootstrap iterations: 10,000, seed=42
- Vectorized numpy implementation; minimal Python loops (chunked for memory)

Outputs
-------
For each top_field (attn_top, resid_post_top), saves JSON with:
- baseline metrics (no sampling) for Helpful and Harmful
- 95% percentile CI for each metric per layer

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import math


# Configure hint-related tokens (case-insensitive, leading whitespace ignored)
HINT_TOKENS: List[str] = ["stanford", "professor"]

LAYERS = 36
BOOTSTRAP_SEED = 42
BOOTSTRAP_ITER = 10_000
CHUNK = 1000  # process bootstrap samples in batches to reduce peak memory


def _normalize_token(tok: str) -> str:
    return tok.strip().lower()


def _accumulate_record(record: dict, top_field: str, hint_set: set[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = np.zeros(LAYERS, dtype=np.int64)
    hits = np.zeros(LAYERS, dtype=np.int64)
    score_sum = np.zeros(LAYERS, dtype=np.float64)
    score_cnt = np.zeros(LAYERS, dtype=np.int64)

    for pos in record.get("positions", []):
        for layer_entry in pos.get("layers", []):
            layer = layer_entry.get("layer")
            if layer is None or layer < 0 or layer >= LAYERS:
                continue

            total[layer] += 1

            found = False
            for rank, item in enumerate(layer_entry.get(top_field, [])):
                token = _normalize_token(str(item.get("token", "")))
                if token in hint_set:
                    found = True
                    score = 1.0 / (rank + 1)
                    score_sum[layer] += score
                    score_cnt[layer] += 1

            if found:
                hits[layer] += 1

    return total, hits, score_sum, score_cnt


def _load_per_question(path: Path, top_field: str, hint_set: set[str]) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    data: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            qid = int(record["id"])
            data[qid] = _accumulate_record(record, top_field=top_field, hint_set=hint_set)
    return data


def _align_pairs(helpful: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], harmful: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    ids = sorted(set(helpful) & set(harmful))
    if not ids:
        raise ValueError("No overlapping question ids between helpful and harmful datasets")

    def stack_component(idx: int) -> np.ndarray:
        return np.stack([helpful[i][idx] for i in ids], axis=0)

    h_total, h_hits, h_scoresum, h_scorecnt = (stack_component(j) for j in range(4))

    def stack_component_harm(idx: int) -> np.ndarray:
        return np.stack([harmful[i][idx] for i in ids], axis=0)

    hf_total, hf_hits, hf_scoresum, hf_scorecnt = (stack_component_harm(j) for j in range(4))
    return ids, (h_total, h_hits, h_scoresum, h_scorecnt), (hf_total, hf_hits, hf_scoresum, hf_scorecnt)


def _aggregate_metrics(total: np.ndarray, hits: np.ndarray, score_sum: np.ndarray, score_cnt: np.ndarray):
    total_s = total.sum(axis=0)
    hits_s = hits.sum(axis=0)
    score_sum_s = score_sum.sum(axis=0)
    score_cnt_s = score_cnt.sum(axis=0)

    occur = np.divide(hits_s, total_s, out=np.zeros_like(score_sum_s, dtype=np.float64), where=total_s > 0)
    mrr = np.divide(score_sum_s, total_s, out=np.full_like(score_sum_s, np.nan, dtype=np.float64), where=total_s > 0)

    return occur, mrr


def _bootstrap_metrics(
    total: np.ndarray,
    hits: np.ndarray,
    score_sum: np.ndarray,
    score_cnt: np.ndarray,
    rng: np.random.Generator,
):
    n_questions = total.shape[0]
    idx = rng.integers(0, n_questions, size=(BOOTSTRAP_ITER, n_questions), dtype=np.int32)

    def agg_from_indices(indices: np.ndarray):
        # indices shape (m, n_questions)
        total_s = np.add.reduce(total[indices], axis=1)
        hits_s = np.add.reduce(hits[indices], axis=1)
        score_sum_s = np.add.reduce(score_sum[indices], axis=1)
        score_cnt_s = np.add.reduce(score_cnt[indices], axis=1)

        occur = np.divide(hits_s, total_s, out=np.zeros_like(score_sum_s, dtype=np.float64), where=total_s > 0)
        mrr = np.divide(score_sum_s, total_s, out=np.full_like(score_sum_s, np.nan, dtype=np.float64), where=total_s > 0)
        return occur, mrr

    occur_boot = np.empty((BOOTSTRAP_ITER, LAYERS), dtype=np.float64)
    mrr_boot = np.empty_like(occur_boot)

    for start in range(0, BOOTSTRAP_ITER, CHUNK):
        end = min(start + CHUNK, BOOTSTRAP_ITER)
        occur_c, mrr_c = agg_from_indices(idx[start:end])
        occur_boot[start:end] = occur_c
        mrr_boot[start:end] = mrr_c

    return occur_boot, mrr_boot


def _percentile_ci(samples: np.ndarray):
    return np.percentile(samples, [2.5, 97.5], axis=0)


def _clean_nans(obj):
    """Recursively convert NaN/inf to None so JSON stays standard-compliant."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, list):
        return [_clean_nans(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_clean_nans(list(obj)))
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    return obj


def run_analysis(useful_path: Path, harmful_path: Path, top_field: str, out_path: Path):
    hint_set = {_normalize_token(t) for t in HINT_TOKENS}

    helpful = _load_per_question(useful_path, top_field=top_field, hint_set=hint_set)
    harmful = _load_per_question(harmful_path, top_field=top_field, hint_set=hint_set)

    ids, h_arrays, hf_arrays = _align_pairs(helpful, harmful)
    (h_total, h_hits, h_scoresum, h_scorecnt) = h_arrays
    (hf_total, hf_hits, hf_scoresum, hf_scorecnt) = hf_arrays

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # Baseline (full data)
    h_occ, h_mrr = _aggregate_metrics(h_total, h_hits, h_scoresum, h_scorecnt)
    hf_occ, hf_mrr = _aggregate_metrics(hf_total, hf_hits, hf_scoresum, hf_scorecnt)

    # Bootstrap samples
    h_occ_b, h_mrr_b = _bootstrap_metrics(h_total, h_hits, h_scoresum, h_scorecnt, rng)
    hf_occ_b, hf_mrr_b = _bootstrap_metrics(hf_total, hf_hits, hf_scoresum, hf_scorecnt, rng)

    result = {
        "ids": ids,
        "top_field": top_field,
        "baseline": {
            "helpful": {
                "occurrence": h_occ.tolist(),
                "mrr": h_mrr.tolist(),
            },
            "harmful": {
                "occurrence": hf_occ.tolist(),
                "mrr": hf_mrr.tolist(),
            },
        },
        "ci": {
            "helpful": {
                "occurrence": _percentile_ci(h_occ_b).tolist(),
                "mrr": _percentile_ci(h_mrr_b).tolist(),
            },
            "harmful": {
                "occurrence": _percentile_ci(hf_occ_b).tolist(),
                "mrr": _percentile_ci(hf_mrr_b).tolist(),
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_clean_nans(result), allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--useful", type=Path, default=Path("output/qwen3_useful_logitlens.jsonl"))
    parser.add_argument("--harmful", type=Path, default=Path("output/qwen3_harmful_logitlens.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    for top_field in ("attn_top", "resid_post_top"):
        out_path = args.out_dir / f"bootstrap_{top_field}.json"
        run_analysis(args.useful, args.harmful, top_field=top_field, out_path=out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
