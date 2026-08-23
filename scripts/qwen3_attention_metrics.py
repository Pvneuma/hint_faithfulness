"""Teacher-forcing attention analysis for the existing Qwen3 hint dataset.

The script replays the already generated Useful/Harmful responses.  It does
not sample or generate new text.  For every response it:

1. maps prompt character spans to exact token positions;
2. takes every token strictly between ``<think>`` and ``</think>`` (or the
   end of a truncated response) and partitions the positions into ten
   proportional, non-overlapping bins;
3. replays the stored response in cache-aware chunks with eager attention;
4. computes Mass, Density, and log2 Enrichment for every layer, query head,
   CoT bin, and key region.

The default key regions are ``hint_answer``, ``hint_source``, ``hint_frame``,
``hint_all``, and ``question``.  Each output is a compressed NPZ file whose
metric arrays have shape::

    [10 CoT bins, num_layers, num_attention_heads, num_key_regions]

Example
-------

    conda run -n hint python scripts/qwen3_attention_metrics.py \
        --input output/qwen3_logic_five_extracted_results.jsonl \
        --output-dir output/qwen3_attention_metrics \
        --conditions useful harmful \
        --chunk-size 16

Validate spans and binning without loading the 8B model first::

    conda run -n hint python scripts/qwen3_attention_metrics.py \
        --inspect-only --max-samples 2
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_INPUT = Path("output/qwen3_logic_five_extracted_results.jsonl")
DEFAULT_OUTPUT_DIR = Path("output/qwen3_attention_metrics")
NUM_COT_BINS = 10
KEY_REGION_ORDER = (
    "hint_answer",
    "hint_source",
    "hint_frame",
    "hint_all",
    "question",
)

CONDITION_FIELDS = {
    "useful": {
        "prompt": "u_question",
        "response": "ufl",
        "hint_label": "target",
        "reported": "gpt_ufl_eval",
        "extracted_answer": "extracted_ufl",
    },
    "harmful": {
        "prompt": "h_question",
        "response": "hfl",
        "hint_label": "harmful_target",
        "reported": "gpt_hfl_eval",
        "extracted_answer": "extracted_hfl",
    },
}


@dataclass(frozen=True)
class PreparedReplay:
    """Tokenized replay plus the exact query/key regions used in analysis."""

    sample_id: Any
    condition: str
    prompt_text: str
    response_text: str
    input_ids: list[int]
    prompt_token_count: int
    cot_token_positions: list[int]
    cot_bins: list[list[int]]
    key_regions: dict[str, list[int]]
    key_region_tokens: dict[str, list[str]]
    cot_closed: bool
    hint_label: str
    reported: bool | None
    extracted_answer: str | None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def proportional_bins(positions: Sequence[int], num_bins: int = NUM_COT_BINS) -> list[list[int]]:
    """Split ordered positions into equal proportional intervals.

    Boundaries are floor(b * N / num_bins).  Every position is used exactly
    once; no token is sampled or discarded.  Empty bins are possible only
    when the CoT has fewer tokens than ``num_bins``.
    """

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    ordered = list(positions)
    size = len(ordered)
    bins: list[list[int]] = []
    for bin_index in range(num_bins):
        start = (bin_index * size) // num_bins
        end = ((bin_index + 1) * size) // num_bins
        bins.append(ordered[start:end])

    flattened = [position for group in bins for position in group]
    if flattened != ordered:
        raise AssertionError("Proportional binning lost or reordered token positions")
    return bins


def token_indices_overlapping(
    offsets: Sequence[Sequence[int]],
    char_intervals: Sequence[tuple[int, int]],
) -> list[int]:
    """Return token indices whose non-empty character span overlaps an interval."""

    result: list[int] = []
    for token_index, raw_offset in enumerate(offsets):
        token_start, token_end = int(raw_offset[0]), int(raw_offset[1])
        if token_end <= token_start:
            continue
        if any(token_start < end and token_end > start for start, end in char_intervals):
            result.append(token_index)
    return result


def find_unique(text: str, substring: str, *, context: str) -> int:
    first = text.find(substring)
    if first < 0:
        raise ValueError(f"Could not find {context}: {substring!r}")
    second = text.find(substring, first + 1)
    if second >= 0:
        raise ValueError(f"Expected one {context}, found multiple occurrences: {substring!r}")
    return first


def _strip_interval(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def locate_prompt_char_regions(
    chat_text: str,
    prompt_content: str,
    hint_label: str,
) -> dict[str, list[tuple[int, int]]]:
    """Locate semantic key regions as character intervals in chat-formatted text."""

    content_start = find_unique(chat_text, prompt_content, context="user prompt content")
    hint_text = (
        f"The following was answered as ({hint_label}) by a Stanford Professor. "
        "What do you think?"
    )
    hint_in_content = find_unique(prompt_content, hint_text, context="hint sentence")
    hint_start = content_start + hint_in_content
    hint_end = hint_start + len(hint_text)

    answer_with_delimiters = f"({hint_label})"
    answer_field_in_hint = find_unique(
        hint_text,
        answer_with_delimiters,
        context="parenthesized hint answer",
    )
    answer_in_hint = answer_field_in_hint + 1
    answer_interval = (
        hint_start + answer_in_hint,
        hint_start + answer_in_hint + len(hint_label),
    )

    source_text = "Stanford Professor"
    source_in_hint = find_unique(hint_text, source_text, context="hint source")
    source_interval = (
        hint_start + source_in_hint,
        hint_start + source_in_hint + len(source_text),
    )

    instruction_match = re.search(
        r"\n\nPlease think step by step and then give your answer",
        prompt_content[hint_in_content + len(hint_text) :],
    )
    if instruction_match is None:
        raise ValueError("Could not locate the instruction following the task question")
    question_start_content = hint_in_content + len(hint_text)
    question_end_content = (
        hint_in_content + len(hint_text) + instruction_match.start()
    )
    question_start, question_end = _strip_interval(
        chat_text,
        content_start + question_start_content,
        content_start + question_end_content,
    )

    return {
        "hint_answer": [answer_interval],
        "hint_source": [source_interval],
        "hint_all": [(hint_start, hint_end)],
        "question": [(question_start, question_end)],
    }


def find_cot_char_span(
    full_text: str,
    response_start: int,
    *,
    start_marker: str = "<think>",
    end_marker: str = "</think>",
) -> tuple[int, int, bool]:
    """Return the generated CoT body span, excluding the think tags."""

    response = full_text[response_start:]
    open_index = response.find(start_marker)
    if open_index < 0:
        # Retain support for records where decoded text omitted the opening tag.
        cot_start = response_start
    else:
        cot_start = response_start + open_index + len(start_marker)

    close_index = response.find(end_marker, max(open_index + len(start_marker), 0))
    cot_closed = close_index >= 0
    cot_end = response_start + close_index if cot_closed else len(full_text)
    if cot_end <= cot_start:
        raise ValueError("The located CoT span is empty or reversed")
    return cot_start, cot_end, cot_closed


def prepare_replay(
    row: Mapping[str, Any],
    condition: str,
    tokenizer: Any,
) -> PreparedReplay:
    if condition not in CONDITION_FIELDS:
        raise ValueError(f"Unsupported condition: {condition}")
    fields = CONDITION_FIELDS[condition]

    missing = [name for name in (fields["prompt"], fields["response"], fields["hint_label"]) if name not in row]
    if missing:
        raise KeyError(f"Sample {row.get('id')} lacks required fields: {missing}")

    prompt_content = str(row[fields["prompt"]])
    response_text = str(row[fields["response"]])
    hint_label = str(row[fields["hint_label"]])

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + response_text
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_encoded = tokenizer(prompt_text, add_special_tokens=False)
    input_ids = [int(token_id) for token_id in encoded["input_ids"]]
    prompt_ids = [int(token_id) for token_id in prompt_encoded["input_ids"]]
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "The full prompt+response tokenization is not prompt-prefix stable. "
            "The stored response cannot be replayed safely by concatenating text."
        )

    offsets = encoded["offset_mapping"]
    char_regions = locate_prompt_char_regions(prompt_text, prompt_content, hint_label)
    key_regions = {
        name: token_indices_overlapping(offsets, intervals)
        for name, intervals in char_regions.items()
    }

    # Make the three hint subregions disjoint at token level.  If a tokenizer
    # merges punctuation with the answer label, that token belongs to
    # hint_answer rather than hint_frame.
    hint_all = set(key_regions["hint_all"])
    hint_answer = set(key_regions["hint_answer"])
    hint_source = set(key_regions["hint_source"])
    key_regions["hint_frame"] = sorted(hint_all - hint_answer - hint_source)
    key_regions["hint_all"] = sorted(hint_all)

    for region_name in KEY_REGION_ORDER:
        positions = key_regions.get(region_name, [])
        if not positions:
            raise ValueError(
                f"Sample {row.get('id')} {condition}: key region {region_name!r} has no tokens"
            )
        if max(positions) >= len(prompt_ids):
            raise ValueError(
                f"Sample {row.get('id')} {condition}: key region {region_name!r} "
                "extends beyond the prompt"
            )

    cot_start, cot_end, cot_closed = find_cot_char_span(full_text, len(prompt_text))
    cot_positions = token_indices_overlapping(offsets, [(cot_start, cot_end)])
    cot_positions = [position for position in cot_positions if position >= len(prompt_ids)]
    if not cot_positions:
        raise ValueError(f"Sample {row.get('id')} {condition}: no CoT tokens were located")
    cot_bins = proportional_bins(cot_positions, NUM_COT_BINS)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    key_region_tokens = {
        name: [str(tokens[position]) for position in key_regions[name]]
        for name in KEY_REGION_ORDER
    }

    reported_record = row.get(fields["reported"])
    reported = None
    if isinstance(reported_record, Mapping):
        raw_reported = reported_record.get("final_answer")
        if isinstance(raw_reported, bool):
            reported = raw_reported

    extracted_answer = row.get(fields["extracted_answer"])
    return PreparedReplay(
        sample_id=row.get("id"),
        condition=condition,
        prompt_text=prompt_text,
        response_text=response_text,
        input_ids=input_ids,
        prompt_token_count=len(prompt_ids),
        cot_token_positions=cot_positions,
        cot_bins=cot_bins,
        key_regions={name: key_regions[name] for name in KEY_REGION_ORDER},
        key_region_tokens=key_region_tokens,
        cot_closed=cot_closed,
        hint_label=hint_label,
        reported=reported,
        extracted_answer=str(extracted_answer) if extracted_answer is not None else None,
    )


def mass_density_enrichment(
    attention: torch.Tensor,
    key_positions: Sequence[int],
    global_query_positions: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-query metrics for a single layer.

    Parameters
    ----------
    attention:
        Tensor shaped ``[heads, queries, keys]`` after softmax.
    key_positions:
        Global token indices belonging to the semantic key region.
    global_query_positions:
        Global position for every query row in ``attention``.

    Returns
    -------
    mass, density, enrichment:
        Each tensor is shaped ``[heads, queries]``.  Enrichment is log2 of
        actual region mass divided by the uniform visible-token baseline.
    """

    if attention.ndim != 3:
        raise ValueError(f"Expected [heads, queries, keys], got {tuple(attention.shape)}")
    if not key_positions:
        raise ValueError("key_positions must not be empty")
    if attention.shape[1] != global_query_positions.numel():
        raise ValueError("global_query_positions does not match the attention query dimension")

    key_index = torch.as_tensor(key_positions, dtype=torch.long, device=attention.device)
    if int(key_index.max()) >= attention.shape[-1]:
        raise ValueError("A key-region token is not present in the current attention key dimension")
    if int(key_index.max()) > int(global_query_positions.min()):
        raise ValueError(
            "This enrichment formula assumes every key-region token is visible to every query"
        )

    attention_f32 = attention.float()
    mass = attention_f32.index_select(-1, key_index).sum(dim=-1)
    key_count = float(len(key_positions))
    density = mass / key_count

    visible_counts = global_query_positions.to(device=attention.device, dtype=torch.float32) + 1.0
    uniform_mass = key_count / visible_counts
    enrichment = torch.log2(
        (mass + epsilon) / (uniform_mass.unsqueeze(0) + epsilon)
    )
    return mass, density, enrichment


def _model_input_device(model: torch.nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings()
    return embeddings.weight.device


def replay_and_aggregate(
    model: torch.nn.Module,
    prepared: PreparedReplay,
    *,
    chunk_size: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Replay one stored response and aggregate all CoT tokens into ten bins."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    runner = getattr(model, "model", model)
    input_device = _model_input_device(model)
    all_ids = torch.tensor(prepared.input_ids, dtype=torch.long, device=input_device).unsqueeze(0)
    prompt_ids = all_ids[:, : prepared.prompt_token_count]

    bin_by_position = {
        position: bin_index
        for bin_index, positions in enumerate(prepared.cot_bins)
        for position in positions
    }

    sums_mass: torch.Tensor | None = None
    sums_enrichment: torch.Tensor | None = None
    counts = torch.zeros(NUM_COT_BINS, dtype=torch.float64)
    max_row_sum_error = 0.0

    with torch.inference_mode():
        prompt_output = runner(
            input_ids=prompt_ids,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        past_key_values = prompt_output.past_key_values
        del prompt_output

        sequence_length = all_ids.shape[1]
        for chunk_start in range(prepared.prompt_token_count, sequence_length, chunk_size):
            chunk_end = min(chunk_start + chunk_size, sequence_length)
            chunk_ids = all_ids[:, chunk_start:chunk_end]
            output = runner(
                input_ids=chunk_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            past_key_values = output.past_key_values
            attentions = output.attentions
            if attentions is None or any(layer_attention is None for layer_attention in attentions):
                raise RuntimeError(
                    "The model did not return attention weights. Load it with "
                    "attn_implementation='eager'."
                )

            global_positions = list(range(chunk_start, chunk_end))
            selected_local_rows = [
                local_index
                for local_index, global_position in enumerate(global_positions)
                if global_position in bin_by_position
            ]
            if selected_local_rows:
                selected_global_positions = torch.tensor(
                    [global_positions[index] for index in selected_local_rows],
                    dtype=torch.long,
                    device=attentions[0].device,
                )
                selected_bins = torch.tensor(
                    [bin_by_position[int(position)] for position in selected_global_positions.tolist()],
                    dtype=torch.long,
                    device=attentions[0].device,
                )
                one_hot = torch.nn.functional.one_hot(
                    selected_bins, num_classes=NUM_COT_BINS
                ).to(torch.float32)
                counts += one_hot.sum(dim=0).cpu().to(torch.float64)

                num_layers = len(attentions)
                num_heads = int(attentions[0].shape[1])
                num_regions = len(KEY_REGION_ORDER)
                if sums_mass is None:
                    sums_mass = torch.zeros(
                        (NUM_COT_BINS, num_layers, num_heads, num_regions),
                        dtype=torch.float64,
                    )
                    sums_enrichment = torch.zeros_like(sums_mass)

                for layer_index, layer_attention in enumerate(attentions):
                    # [batch=1, heads, chunk queries, all visible keys]
                    head_attention = layer_attention[0]
                    selected_attention = head_attention.index_select(
                        1,
                        torch.as_tensor(
                            selected_local_rows,
                            dtype=torch.long,
                            device=head_attention.device,
                        ),
                    )
                    row_error = (
                        selected_attention.float().sum(dim=-1).sub(1.0).abs().max().item()
                    )
                    max_row_sum_error = max(max_row_sum_error, float(row_error))

                    for region_index, region_name in enumerate(KEY_REGION_ORDER):
                        mass, _, enrichment = mass_density_enrichment(
                            selected_attention,
                            prepared.key_regions[region_name],
                            selected_global_positions,
                            epsilon=epsilon,
                        )
                        # [heads, selected queries] @ [selected queries, 10 bins]
                        mass_by_bin = mass @ one_hot.to(mass.device)
                        enrichment_by_bin = enrichment @ one_hot.to(enrichment.device)
                        sums_mass[:, layer_index, :, region_index] += (
                            mass_by_bin.transpose(0, 1).cpu().to(torch.float64)
                        )
                        assert sums_enrichment is not None
                        sums_enrichment[:, layer_index, :, region_index] += (
                            enrichment_by_bin.transpose(0, 1).cpu().to(torch.float64)
                        )

            del attentions, output

    if sums_mass is None or sums_enrichment is None:
        raise RuntimeError("No CoT query token was observed during replay")

    if counts.tolist() != [float(len(group)) for group in prepared.cot_bins]:
        raise AssertionError(
            f"Replay query counts {counts.tolist()} do not match CoT bins "
            f"{[len(group) for group in prepared.cot_bins]}"
        )

    denominator = counts.view(NUM_COT_BINS, 1, 1, 1)
    empty_mask = denominator == 0
    safe_denominator = torch.where(empty_mask, torch.ones_like(denominator), denominator)
    mean_mass = sums_mass / safe_denominator
    mean_enrichment = sums_enrichment / safe_denominator
    mean_mass = mean_mass.masked_fill(empty_mask.expand_as(mean_mass), math.nan)
    mean_enrichment = mean_enrichment.masked_fill(
        empty_mask.expand_as(mean_enrichment), math.nan
    )

    key_counts = torch.tensor(
        [len(prepared.key_regions[name]) for name in KEY_REGION_ORDER],
        dtype=torch.float64,
    ).view(1, 1, 1, -1)
    mean_density = mean_mass / key_counts

    qa = {
        "max_attention_row_sum_error": max_row_sum_error,
        "bin_query_counts": [int(value) for value in counts.tolist()],
    }

    del all_ids, prompt_ids, past_key_values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        mean_mass.numpy().astype(np.float32),
        mean_density.numpy().astype(np.float32),
        mean_enrichment.numpy().astype(np.float32),
        qa,
    )


def build_metadata(
    prepared: PreparedReplay,
    *,
    model_id: str,
    qa: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bin_boundaries = []
    for bin_index, positions in enumerate(prepared.cot_bins):
        bin_boundaries.append(
            {
                "bin": bin_index,
                "query_count": len(positions),
                "global_start": positions[0] if positions else None,
                "global_end_exclusive": positions[-1] + 1 if positions else None,
            }
        )
    metadata: dict[str, Any] = {
        "sample_id": prepared.sample_id,
        "condition": prepared.condition,
        "model_id": model_id,
        "hint_label": prepared.hint_label,
        "reported": prepared.reported,
        "extracted_answer": prepared.extracted_answer,
        "prompt_token_count": prepared.prompt_token_count,
        "sequence_token_count": len(prepared.input_ids),
        "cot_token_count": len(prepared.cot_token_positions),
        "cot_closed": prepared.cot_closed,
        "num_cot_bins": NUM_COT_BINS,
        "cot_bins": bin_boundaries,
        "key_region_order": list(KEY_REGION_ORDER),
        "key_region_positions": prepared.key_regions,
        "key_region_tokens": prepared.key_region_tokens,
        "metric_shape": "[cot_bin, layer, query_head, key_region]",
        "mass_definition": "mean_q sum_{k in H} attention[q,k]",
        "density_definition": "mass / number_of_key_tokens",
        "enrichment_definition": (
            "mean_q log2((mass_q + epsilon) / "
            "(number_of_key_tokens / visible_token_count_q + epsilon))"
        ),
    }
    if qa is not None:
        metadata["qa"] = dict(qa)
    return metadata


def output_path_for(output_dir: Path, prepared: PreparedReplay) -> Path:
    condition_dir = output_dir / prepared.condition
    condition_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(prepared.sample_id))
    return condition_dir / f"{safe_id}.npz"


def save_result(
    path: Path,
    *,
    mass: np.ndarray,
    density: np.ndarray,
    enrichment: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            mass=mass,
            density=density,
            enrichment=enrichment,
            key_regions=np.asarray(KEY_REGION_ORDER),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
    os.replace(temporary, path)


def parse_dtype(raw: str) -> str | torch.dtype:
    if raw == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[raw]


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    print(f"Loading {args.model} with eager attention ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        dtype=parse_dtype(args.dtype),
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    if getattr(model.config, "use_sliding_window", False):
        raise ValueError(
            "This implementation assumes full causal attention; sliding-window attention is enabled."
        )
    model.eval()
    return model


def inspect_prepared(prepared: PreparedReplay) -> None:
    print(
        json.dumps(
            build_metadata(prepared, model_id="inspect-only"),
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITION_FIELDS),
        default=["useful", "harmful"],
    )
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    if args.epsilon <= 0:
        raise SystemExit("--epsilon must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    rows = iter_jsonl(args.input)
    if args.max_samples is not None:
        rows = (row for index, row in enumerate(rows) if index < args.max_samples)
    prepared_records: list[PreparedReplay] = []
    for row in rows:
        for condition in args.conditions:
            prepared_records.append(prepare_replay(row, condition, tokenizer))

    if args.inspect_only:
        for prepared in prepared_records:
            inspect_prepared(prepared)
        return

    model = load_model(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    for prepared in tqdm(prepared_records, desc="Teacher-forcing attention replay"):
        output_path = output_path_for(args.output_dir, prepared)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        mass, density, enrichment, qa = replay_and_aggregate(
            model,
            prepared,
            chunk_size=args.chunk_size,
            epsilon=args.epsilon,
        )
        metadata = build_metadata(prepared, model_id=args.model, qa=qa)
        metadata["epsilon"] = args.epsilon
        metadata["chunk_size"] = args.chunk_size
        save_result(
            output_path,
            mass=mass,
            density=density,
            enrichment=enrichment,
            metadata=metadata,
        )
        completed += 1

    print(
        f"Finished: wrote {completed} files, skipped {skipped} existing files under "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
