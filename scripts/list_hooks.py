"""List available hook names for a HookedTransformer model.

Usage: edit CONFIG then run:

    python scripts/list_hooks.py

It prints all hook names (keys in `model.hook_dict`), sorted.
"""

from __future__ import annotations

import os
import sys
import warnings
from pprint import pprint

from transformer_lens import HookedTransformer


# ------------------ CONFIG ------------------
MODEL_ID = "Qwen/Qwen3-0.6B"   # model repo or local path
DEVICE = None                # None->auto, or "cuda", "cuda:0", "mps", "cpu"
# -------------------------------------------

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")


def infer_device(user_choice: str | None) -> str:
    import torch

    if user_choice:
        return user_choice
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = infer_device(DEVICE)
    try:
        print(f"Loading {MODEL_ID} on {device} ...", flush=True)
        model = HookedTransformer.from_pretrained(
            MODEL_ID,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            trust_remote_code=True,
        )
        names = sorted(model.hook_dict.keys())
        print(f"Total hooks: {len(names)}", flush=True)
        pprint(names, stream=sys.stdout)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
