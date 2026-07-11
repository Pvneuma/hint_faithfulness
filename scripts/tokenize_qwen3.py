"""Tokenize text with the Qwen3-8B tokenizer.

This script loads the tokenizer, breaks the provided text into tokens, and
prints a simple table showing index, token string, token id, and the original
substring span.
"""

import argparse
import sys
from typing import List, Tuple

from transformers import AutoTokenizer


def format_rows(tokens: List[str], ids: List[int], spans: List[Tuple[int, int]], text: str) -> str:
    lines = ["idx\ttoken\tid\tspan"]
    for idx, (tok, tid, span) in enumerate(zip(tokens, ids, spans)):
        start, end = span
        original = text[start:end]
        lines.append(f"{idx}\t{tok}\t{tid}\t{original}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize text with the Qwen3-8B tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to tokenize. If omitted, the script reads from stdin.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Tokenizer model id from Hugging Face",
    )
    args = parser.parse_args()

    text = args.text if args.text is not None else sys.stdin.read().strip()
    if not text:
        raise SystemExit("No text provided. Pass text as an argument or via stdin.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = encoded["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    spans = encoded.get("offset_mapping", [(0, 0)] * len(tokens))

    print(format_rows(tokens, input_ids, spans, text))


if __name__ == "__main__":
    main()
