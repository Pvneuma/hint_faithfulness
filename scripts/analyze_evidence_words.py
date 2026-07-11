import argparse
import json
import re
from collections import Counter
from pathlib import Path


def tokenize(text: str):
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def accumulate_counts(path: Path) -> Counter:
    counter = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for key in ("gpt_ufl_eval", "gpt_hfl_eval"):
                section = record.get(key) or {}
                if section.get("final_answer") is True:
                    for evidence in section.get("evidence", []):
                        counter.update(tokenize(evidence))
    return counter


def main():
    parser = argparse.ArgumentParser(description="Count word frequency in evidence fields")
    parser.add_argument(
        "--input",
        default="output/qwen3_logic_five_extracted_results.jsonl",
        type=Path,
        help="Path to input JSONL file",
    )
    args = parser.parse_args()
    counts = accumulate_counts(args.input)
    for word, freq in counts.most_common(20):
        print(f"{word}\t{freq}")


if __name__ == "__main__":
    main()
