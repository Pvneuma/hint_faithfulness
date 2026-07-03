"""Logit-lens with TransformerLens on Qwen3-8B.

Features
--------
- Loads any HF causal LM that TransformerLens can wrap (tested with Qwen/Qwen2-7B-Instruct; Qwen3-8B should work once weights are accessible).
- Reads a jsonl dataset (specify text field) and, for each sample, records per-layer top-K next-token predictions using the model's unembedding matrix.
- Captures logit-lens outputs for **generated tokens only** (prompt logits are omitted), grouping by token (outer) then layer (inner).
- Greedy decoding (`do_sample=False`) is used when generating tokens for analysis.
- Outputs jsonl where each line stores the prompt and per-layer top tokens + logits.

Usage
-----
Edit the CONFIG block below, then run:

    python scripts/qwen3_tlens_logitlens.py

Dependencies
------------
pip install transformer-lens datasets
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer


# ------------------ CONFIG ------------------
DATA_PATH = "output/qwen3_logic_five_extracted_results.jsonl"          # jsonl file path
TEXT_FIELD = "u_question"                     # column containing prompt text
MODEL_ID = "Qwen/Qwen3-8b"               # HF model id or local path
OUTPUT_PATH = "output/qwen3_useful_logitlens.jsonl"
TOKENS_DUMP_PATH = "output/qwen3_useful_all_tokens.jsonl"
MAX_SAMPLES = None                     # int or None
TOP_K = 5                                 # top-k tokens per layer
DEVICE = None                             # None -> auto; or "cuda", "cuda:0", "mps", "cpu"
MAX_NEW_TOKENS = 4096                       # >0: greedy-generate this many tokens; only these tokens are logged
# To reduce multiprocessing semaphore warnings from tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# -------------------------------------------


def infer_device(user_choice: str | None) -> str:
    if user_choice:
        return user_choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, device: str) -> HookedTransformer:
    print(f"Loading {model_id} via TransformerLens on {device}")
    model = HookedTransformer.from_pretrained(
        model_id,
        device=device,
        fold_ln=False,  # keep original layer norms
        center_writing_weights=False,
        center_unembed=False,
        trust_remote_code=True,
    )
    model.eval()
    return model


def layer_topk_all_positions(
    model: HookedTransformer,
    prompt: str,
    top_k: int,
    max_new_tokens: int = 0,
) -> List[Dict[str, Any]]:
    """Return per-generated-token top-k logits, with per-layer detail nested inside each token.

    Only generated tokens are included (prompt positions are skipped).
    """

    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)  # (1, prompt_len)

    with torch.no_grad():
        generated = model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_type="tokens",
        )

    # Separate prompt and generated spans
    prompt_len = prompt_tokens.shape[1]
    # Free prompt tokens tensor if on GPU to save memory
    del prompt_tokens
    gen_tokens = generated[:, prompt_len:]  # (1, max_new_tokens)
    all_tokens = generated  # full for caching

    # Cache hook_resid_mid and resid_post to save memory
    names_filter = lambda name: ("resid_post" in name) or ("hook_resid_mid" in name)
    with torch.no_grad():
        _, cache = model.run_with_cache(
            all_tokens,
            remove_batch_dim=False,
            return_type=None,
            names_filter=names_filter,
        )

    # We no longer need all_tokens
    del all_tokens

    token_strings = [
        model.tokenizer.decode([int(t)], skip_special_tokens=False) for t in gen_tokens[0].tolist()
    ]
    gen_len = gen_tokens.shape[1]

    positions: List[Dict[str, Any]] = []
    # Precompute lm_head weight for efficiency
    unembed_w = model.unembed.W_U if hasattr(model.unembed, "W_U") else model.unembed.weight

    for layer in range(model.cfg.n_layers):
        # Resid mid matrix (gen_len, d_model)
        attn_key = f"blocks.{layer}.hook_resid_mid"
        if attn_key not in cache:
            raise KeyError(
                f"Resid mid hook not found: {attn_key}; available keys: {[k for k in cache.keys() if k.startswith(f'blocks.{layer}.')]}")
        attn_mat = cache[attn_key][0, prompt_len:, :].detach()  # (gen_len, d)

        # Resid post matrix (gen_len, d_model)
        resid_key = f"blocks.{layer}.hook_resid_post"
        if resid_key not in cache:
            raise KeyError(f"Resid post hook not found: {resid_key}; available keys: {[k for k in cache.keys() if k.startswith(f'blocks.{layer}.')]}")
        resid_mat = cache[resid_key][0, prompt_len:, :].detach()
        if hasattr(model, "ln_final") and model.ln_final is not None:
            resid_mat = model.ln_final(resid_mat)

        # Apply final LN to resid_mid as well for fair comparison
        if hasattr(model, "ln_final") and model.ln_final is not None:
            attn_mat = model.ln_final(attn_mat)

        # Batched logits; handle weight orientation (vocab x d_model) vs (d_model x vocab)
        if unembed_w.shape[0] == attn_mat.shape[1]:
            attn_logits = attn_mat @ unembed_w
            resid_logits = resid_mat @ unembed_w
        elif unembed_w.shape[1] == attn_mat.shape[1]:
            attn_logits = attn_mat @ unembed_w.T
            resid_logits = resid_mat @ unembed_w.T
        else:
            raise ValueError(
                f"Unexpected unembed weight shape {unembed_w.shape} for hidden size {attn_mat.shape[1]}"
            )

        attn_values, attn_idx = torch.topk(attn_logits, k=top_k, dim=-1)
        resid_values, resid_idx = torch.topk(resid_logits, k=top_k, dim=-1)

        # Free large tensors
        del attn_logits, resid_logits, attn_mat, resid_mat

        for pos_idx in range(gen_len):
            if len(positions) <= pos_idx:
                positions.append({"idx": pos_idx, "token": token_strings[pos_idx], "layers": []})

            attn_tokens = [
                model.tokenizer.decode([int(t)], skip_special_tokens=False)
                for t in attn_idx[pos_idx].tolist()
            ]
            resid_tokens = [
                model.tokenizer.decode([int(t)], skip_special_tokens=False)
                for t in resid_idx[pos_idx].tolist()
            ]

            positions[pos_idx]["layers"].append(
                {
                    "layer": layer,
                    "attn_top": [
                        {"token": tok, "logit": float(val)}
                        for tok, val in zip(attn_tokens, attn_values[pos_idx].tolist())
                    ],
                    "resid_post_top": [
                        {"token": tok, "logit": float(val)}
                        for tok, val in zip(resid_tokens, resid_values[pos_idx].tolist())
                    ],
                }
            )

        del attn_values, attn_idx, resid_values, resid_idx

    # Help GC release cache tensors early
    cache = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return positions, all_tokens.tolist()


def main():
    device = infer_device(DEVICE)

    model = load_model(MODEL_ID, device)

    dataset = load_dataset("json", data_files=str(DATA_PATH), streaming=False)["train"]
    if MAX_SAMPLES is not None:
        dataset = dataset.select(range(min(len(dataset), MAX_SAMPLES)))

    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f, tokens_dump_path.open("w", encoding="utf-8") as tf:
        total = len(dataset)
        for idx, row in enumerate(dataset):
            if idx % 10 == 0:
                print(f"Processing sample {idx}/{total}")
            text = row.get(TEXT_FIELD)
            if text is None:
                raise KeyError(
                    f"Row {idx} missing text field '{TEXT_FIELD}'. Available keys: {list(row.keys())}"
                )

            positions, all_tokens = layer_topk_all_positions(
                model, text, TOP_K, max_new_tokens=MAX_NEW_TOKENS
            )

            input_id = row.get("id", idx)

            rec = {
                "id": input_id,
                "text": text,
                "model_id": MODEL_ID,
                "top_k": TOP_K,
                "positions": positions,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # dump tokens for reuse
            tf.write(json.dumps({"id": input_id, "all_tokens": all_tokens}, ensure_ascii=False) + "\n")

    print(f"Saved logit-lens results to {out_path}")


if __name__ == "__main__":
    main()
