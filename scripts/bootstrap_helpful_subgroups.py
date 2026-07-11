"""Bootstrap analysis for helpful condition split by reported/unreported hints.

Groups helpful questions into two subsets based on `gpt_ufl_eval.final_answer`
from `output/qwen3_logic_five_extracted_results.jsonl`:

- reported: final_answer == True
- unreported: final_answer == False

For each subset and for each `top_field` (attn_top, resid_post_top), perform
10,000 bootstrap samples (seed=42) with question-level resampling, computing
Occurrence, Conditional Mean Logit, and Hint Activation Score. Results are
saved as JSON with baseline metrics and 95% percentile CI.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


HINT_TOKENS: List[str] = ["stanford", "professor"]
LAYERS = 36
BOOTSTRAP_SEED = 42
BOOTSTRAP_ITER = 10_000
CHUNK = 1000


def _normalize_token(tok: str) -> str:
    return tok.strip().lower()


def _accumulate_record(record: dict, top_field: str, hint_set: set[str]):
    total = np.zeros(LAYERS, dtype=np.int64)
    hits = np.zeros(LAYERS, dtype=np.int64)
    log_sum = np.zeros(LAYERS, dtype=np.float64)
    log_cnt = np.zeros(LAYERS, dtype=np.int64)

    for pos in record.get("positions", []):
        for layer_entry in pos.get("layers", []):
            layer = layer_entry.get("layer")
            if layer is None or layer < 0 or layer >= LAYERS:
                continue
            total[layer] += 1
            found = False
            for item in layer_entry.get(top_field, []):
                token = _normalize_token(str(item.get("token", "")))
                if token in hint_set:
                    found = True
                    log_sum[layer] += float(item.get("logit", 0.0))
                    log_cnt[layer] += 1
            if found:
                hits[layer] += 1
    return total, hits, log_sum, log_cnt


def _load_subset(path: Path, top_field: str, hint_set: set[str], allowed_ids: set[int]):
    data = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            qid = int(record["id"])
            if qid not in allowed_ids:
                continue
            data[qid] = _accumulate_record(record, top_field=top_field, hint_set=hint_set)
    return data


def _aggregate(total, hits, log_sum, log_cnt):
    total_s = total.sum(axis=0)
    hits_s = hits.sum(axis=0)
    log_sum_s = log_sum.sum(axis=0)
    log_cnt_s = log_cnt.sum(axis=0)

    occur = np.divide(hits_s, total_s, out=np.zeros_like(log_sum_s, dtype=np.float64), where=total_s > 0)
    mean = np.divide(log_sum_s, log_cnt_s, out=np.full_like(log_sum_s, np.nan, dtype=np.float64), where=log_cnt_s > 0)
    activation = np.full_like(log_sum_s, np.nan, dtype=np.float64)
    mask = (total_s > 0) & (log_cnt_s > 0) & (hits_s > 0)
    activation[mask] = (log_sum_s[mask] * hits_s[mask]) / (log_cnt_s[mask] * total_s[mask])
    return occur, mean, activation


def _bootstrap(total, hits, log_sum, log_cnt, rng):
    n = total.shape[0]
    idx = rng.integers(0, n, size=(BOOTSTRAP_ITER, n), dtype=np.int32)

    occur_boot = np.empty((BOOTSTRAP_ITER, LAYERS), dtype=np.float64)
    mean_boot = np.empty_like(occur_boot)
    act_boot = np.empty_like(occur_boot)

    for start in range(0, BOOTSTRAP_ITER, CHUNK):
        end = min(start + CHUNK, BOOTSTRAP_ITER)
        sel = idx[start:end]
        total_s = np.add.reduce(total[sel], axis=1)
        hits_s = np.add.reduce(hits[sel], axis=1)
        log_sum_s = np.add.reduce(log_sum[sel], axis=1)
        log_cnt_s = np.add.reduce(log_cnt[sel], axis=1)

        occur_boot[start:end] = np.divide(hits_s, total_s, out=np.zeros_like(log_sum_s, dtype=np.float64), where=total_s > 0)
        mean_boot[start:end] = np.divide(log_sum_s, log_cnt_s, out=np.full_like(log_sum_s, np.nan, dtype=np.float64), where=log_cnt_s > 0)

        act = np.full_like(log_sum_s, np.nan, dtype=np.float64)
        mask = (total_s > 0) & (log_cnt_s > 0) & (hits_s > 0)
        act[mask] = (log_sum_s[mask] * hits_s[mask]) / (log_cnt_s[mask] * total_s[mask])
        act_boot[start:end] = act

    return occur_boot, mean_boot, act_boot


def _percentile_ci(samples: np.ndarray):
    return np.percentile(samples, [2.5, 97.5], axis=0)


def _clean(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    return obj


def _save_result(out_path: Path, ids, baseline, ci):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ids": list(ids),
        "baseline": baseline,
        "ci": ci,
    }
    out_path.write_text(json.dumps(_clean(data), allow_nan=False))
    print(f"Saved {out_path}")


def _load_labels(label_path: Path):
    reported = set()
    unreported = set()
    with label_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = int(obj["id"])
            flag = bool(obj["gpt_ufl_eval"]["final_answer"])
            (reported if flag else unreported).add(qid)
    return reported, unreported


def process_subset(name: str, ids: set[int], useful_path: Path, top_field: str, out_dir: Path, hint_set: set[str]):
    data = _load_subset(useful_path, top_field=top_field, hint_set=hint_set, allowed_ids=ids)
    if not data:
        raise ValueError(f"No data for subset {name} and top_field {top_field}")

    ordered_ids = sorted(data.keys())
    arrays = list(zip(*[data[i] for i in ordered_ids]))  # totals, hits, log_sum, log_cnt
    total, hits, log_sum, log_cnt = [np.stack(arrays[j], axis=0) for j in range(4)]

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    base_occ, base_mean, base_act = _aggregate(total, hits, log_sum, log_cnt)
    occ_b, mean_b, act_b = _bootstrap(total, hits, log_sum, log_cnt, rng)

    result_base = {
        "occurrence": base_occ.tolist(),
        "mean_logit": base_mean.tolist(),
        "has": base_act.tolist(),
    }
    result_ci = {
        "occurrence": _percentile_ci(occ_b).tolist(),
        "mean_logit": _percentile_ci(mean_b).tolist(),
        "has": _percentile_ci(act_b).tolist(),
    }

    outfile = out_dir / f"bootstrap_helpful_{name}_{top_field}.json"
    _save_result(outfile, ordered_ids, result_base, result_ci)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--useful", type=Path, default=Path("output/qwen3_useful_logitlens.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("output/qwen3_logic_five_extracted_results.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    reported, unreported = _load_labels(args.labels)
    hint_set = {_normalize_token(t) for t in HINT_TOKENS}

    for subset_name, subset_ids in (("reported", reported), ("unreported", unreported)):
        for top_field in ("attn_top", "resid_post_top"):
            process_subset(subset_name, subset_ids, args.useful, top_field, args.out_dir, hint_set)


if __name__ == "__main__":
    main()
