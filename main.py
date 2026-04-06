#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def shift_feature_numbers(text: str, offset: int) -> str:
    # Replaces tokens like f1, f23, etc. with f(number + offset).
    # This will update ids and function names like:
    # "id": "f1" -> "id": "f61"
    # def f1(s, a): -> def f61(s, a):
    pattern = re.compile(r"\bf(\d+)\b")

    def repl(match: re.Match[str]) -> str:
        old_num = int(match.group(1))
        return f"f{old_num + offset}"

    return pattern.sub(repl, text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shift feature ids/functions in a JSON feature library (fN -> fN+offset)."
    )
    parser.add_argument("input_file", help="Path to input JSON file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file path (default: overwrite input file)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=60,
        help="Amount to add to feature number (default: 60)",
    )
    args = parser.parse_args()

    in_path = Path(args.input_file)
    out_path = Path(args.output) if args.output else in_path

    text = in_path.read_text(encoding="utf-8")
    updated = shift_feature_numbers(text, args.offset)
    out_path.write_text(updated, encoding="utf-8")

    print(f"Done. Wrote updated file to: {out_path}")


if __name__ == "__main__":
    main()