#!/usr/bin/env python3
"""Create a fixed-denominator paired Shopping benchmark report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from groundedvision.agentic_rl.evaluation import compare_paired_shopping_results


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    expected = [row["task_id"] for row in read_jsonl(args.tasks)]
    report = compare_paired_shopping_results(
        read_jsonl(args.baseline),
        read_jsonl(args.candidate),
        expected,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
