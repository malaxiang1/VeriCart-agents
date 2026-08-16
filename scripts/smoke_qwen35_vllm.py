#!/usr/bin/env python3
"""Load a local Qwen3.5 actor through the pinned 3090 vLLM port."""

import os

from vllm import LLM, SamplingParams


def main() -> None:
    model = os.environ.get("MODEL")
    if not model:
        raise SystemExit("set MODEL to a local Qwen3.5-2B checkpoint")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
        max_num_seqs=1,
    )
    output = llm.generate(
        ["你好，请只回复 OK。"],
        SamplingParams(temperature=0, max_tokens=8),
    )
    print("VLLM_QWEN35_SMOKE", repr(output[0].outputs[0].text))


if __name__ == "__main__":
    main()
