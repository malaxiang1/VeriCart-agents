#!/usr/bin/env python3
"""Report exact Qwen3.5 token coverage for public Shopping SFT trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def percentile(sorted_values: list[int], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--caps", nargs="+", type=int, default=[8192, 12288, 16384, 20480, 24576])
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line]
    lengths = []
    for row in rows:
        messages = json.loads(json.dumps(row["messages"], ensure_ascii=False))
        for message in messages:
            for call in message.get("tool_calls") or []:
                arguments = call.get("function", {}).get("arguments")
                if isinstance(arguments, str):
                    call["function"]["arguments"] = json.loads(arguments)
        rendered = processor.apply_chat_template(
            messages,
            tools=row.get("tools") or [],
            tokenize=False,
            add_generation_prompt=False,
        )
        lengths.append(len(tokenizer(rendered, add_special_tokens=False)["input_ids"]))
    lengths.sort()
    report = {
        "rows": len(lengths),
        "tokens": {
            key: percentile(lengths, q)
            for key, q in {"min": 0, "p50": 0.5, "p90": 0.9, "p95": 0.95, "p99": 0.99, "max": 1}.items()
        },
        "coverage": {
            str(cap): {
                "kept": sum(length <= cap for length in lengths),
                "rate": sum(length <= cap for length in lengths) / len(lengths),
            }
            for cap in args.caps
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
