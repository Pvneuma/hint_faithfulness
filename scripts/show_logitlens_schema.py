"""Print a compact schema summary for a JSONL file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def describe(value: Any, depth: int = 0) -> list[str]:
    """Return a list of indented lines describing *value*.

    To keep output manageable, only the first element of each list is shown,
    but nesting is not depth-limited.
    """

    indent = "  " * depth
    lines: list[str] = []

    if isinstance(value, dict):
        lines.append(f"{indent}object with {len(value)} keys")
        for key in sorted(value.keys()):
            lines.append(f"{indent}  {key}:")
            lines.extend(describe(value[key], depth + 2))
    elif isinstance(value, list):
        lines.append(f"{indent}list (len={len(value)})")
        if value:
            lines.append(f"{indent}  first element:")
            lines.extend(describe(value[0], depth + 2))
    elif isinstance(value, str):
        preview = value[:60].replace("\n", "\\n")
        more = "…" if len(value) > 60 else ""
        lines.append(f"{indent}string (len={len(value)}): '{preview}{more}'")
    elif value is None:
        lines.append(f"{indent}null")
    else:
        lines.append(f"{indent}{type(value).__name__}: {value}")

    return lines


def summarize_record(record: Any, index: int) -> str:
    header = f"--- Record {index} ---"
    body = "\n".join(describe(record))
    return f"{header}\n{body}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        default="output/qwen3_useful_logitlens.jsonl",
        type=Path,
        help="Path to the JSONL file (default: output/qwen3_useful_logitlens.jsonl)",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=1,
        help="Number of lines (records) to sample from the start of the file.",
    )
    args = parser.parse_args()

    if args.lines < 1:
        raise SystemExit("--lines must be at least 1")

    path = args.file
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    with path.open() as f:
        for i in range(args.lines):
            line = f.readline()
            if not line:
                break
            record = json.loads(line)
            print(summarize_record(record, i))


if __name__ == "__main__":
    main()
