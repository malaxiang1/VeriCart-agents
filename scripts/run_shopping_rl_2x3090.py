#!/usr/bin/env python3
"""Launch bounded Shopping RL experiments on the pinned environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/shopping-grpo-longhorizon"
CONFIG = ROOT / "configs/shopping_grpo_2x3090.yaml"
AGENT_CONFIG = ROOT / "configs/shopping_agent_loop_2x3090.yaml"
DEFAULT_MODEL = UPSTREAM / "outputs/models/sft-merged-qwen35-2b-20k"
EXPECTED = {
    "verl": "0.8.0",
    "vllm": "0.18.0",
    "torch": "2.10.0+cu128",
    "transformers": "5.15.0.dev0",
    "flashinfer-python": "0.6.6",
    "flashinfer-cubin": "0.6.6",
}
DATA_HASHES = {
    "data/grpo/train.parquet": "e4b4765b67efcc064ba4e656db625a812a34cbcff00da0e23d6a3df8aac5fdd4",
    "data/grpo/validation.parquet": "9aa370f00d7ead942e47cf9aed4ab0c55cd7426220f4603f4fed5dc949ea2788",
    "data/evaluation/tasks.jsonl": "2c4ff070e13ddc30796d38e85170210e7d3c211992425a62090f2419fe8e0208",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_has_weights(path: Path) -> bool:
    names = {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
    return path.is_dir() and (path / "config.json").is_file() and any(
        (path / name).is_file() for name in names
    )


def load_ids(path: Path) -> set[str]:
    return {
        str(json.loads(line)["task_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    }


def dynamic_sampling_override(method: str) -> str:
    """Keep resampling opt-in and isolated from matched comparison methods."""
    return "shopping_dynamic_sampling.enable=" + (
        "true"
        if method in {"dapo", "valid-group", "partial-valid-group"}
        else "false"
    )


def dynamic_selection_mode_override(method: str) -> str:
    """Use partial-valid packing for DAPO's noisy tool environment."""
    return "shopping_dynamic_sampling.selection_mode=" + (
        "partial_valid"
        if method in {"dapo", "partial-valid-group"}
        else "whole"
    )


def dataloader_workers_override(method: str) -> str | None:
    """Keep matched GRPO/HGPO runs free of worker teardown races."""
    if method in {"grpo", "dapo", "valid-group", "partial-valid-group"}:
        return "data.dataloader_num_workers=0"
    return None


def single_group_overrides(method: str, enabled: bool) -> list[str]:
    """Keep a one-prompt GRPO update internally batch-consistent on two GPUs."""
    if not enabled:
        return []
    if method != "valid-group":
        raise SystemExit("--single-group is only supported by --method valid-group")
    return [
        "data.train_batch_size=1",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
    ]


def partial_valid_group_overrides(method: str) -> list[str]:
    """Seed adaptive packing with a one-prompt-equivalent actor minibatch."""
    if method not in {"dapo", "partial-valid-group"}:
        return []
    return [
        "data.train_batch_size=2",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
    ]


def dapo_overrides(method: str) -> list[str]:
    """Apply the DAPO clip-higher contract on top of dynamic sampling."""
    if method != "dapo":
        return []
    return [
        "actor_rollout_ref.actor.clip_ratio_low=0.20",
        "actor_rollout_ref.actor.clip_ratio_high=0.28",
    ]


def preflight(model: Path, method: str) -> dict[str, object]:
    import torch

    if not model_has_weights(model):
        raise SystemExit(f"model is missing config or weights: {model}")
    installed = {name: version(name) for name in EXPECTED}
    mismatches = {
        name: {"expected": expected, "actual": installed[name]}
        for name, expected in EXPECTED.items()
        if installed[name] != expected
    }
    if mismatches:
        raise SystemExit(f"3090 runtime version mismatch: {json.dumps(mismatches)}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise SystemExit("matched Shopping RL requires exactly two visible CUDA GPUs")
    if any(torch.cuda.get_device_properties(index).total_memory < 23 * 1024**3 for index in range(2)):
        raise SystemExit("each visible GPU must provide at least 23 GiB")

    import verl

    trainer = Path(verl.__file__).resolve().parent / "trainer/ppo/ray_trainer.py"
    trainer_text = trainer.read_text(encoding="utf-8")
    required_markers = ["SHOPPING_GRPO_DYNAMIC_SAMPLING_PATCH_V3"]
    if method == "partial-valid-group":
        required_markers.append("GROUNDINGVISION_PARTIAL_VALID_GROUP_V1")
    if method == "pad":
        required_markers.append("GROUNDINGVISION_PAD_GRPO_HOOK_V1")
    if method == "event":
        required_markers.append("GROUNDINGVISION_SHOPPING_EVENT_CREDIT_V1")
    missing = [marker for marker in required_markers if marker not in trainer_text]
    if missing:
        raise SystemExit(f"veRL trainer is missing patches: {missing}")

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["trainer"]["n_gpus_per_node"] != 2:
        raise SystemExit("3090 config must use two GPUs")
    if config["actor_rollout_ref"]["rollout"]["max_model_len"] != 8192:
        raise SystemExit("matched 3090 rollout context must use the audited 8K contract")
    if not config["actor_rollout_ref"]["model"]["use_remove_padding"]:
        raise SystemExit("the 16K actor path requires remove-padding on 24 GiB GPUs")
    if not config["actor_rollout_ref"]["actor"]["use_remove_padding"]:
        raise SystemExit("the 16K actor engine requires remove-padding on 24 GiB GPUs")
    dynamic_sampling_enabled = bool(config["shopping_dynamic_sampling"]["enable"])
    if dynamic_sampling_enabled:
        raise SystemExit(
            "the base Shopping config must keep dynamic sampling disabled; "
            "only the valid-group launcher enables it explicitly"
        )
    agent_config = yaml.safe_load(AGENT_CONFIG.read_text(encoding="utf-8"))[0]
    if agent_config["context_window_tokens"] != 8192:
        raise SystemExit("AgentLoop context must match the 8K rollout context")
    if not agent_config["context_compaction_enable"]:
        raise SystemExit("the 16K port requires complete-turn context compaction")

    actual_hashes = {}
    for relative, expected in DATA_HASHES.items():
        actual = sha256(UPSTREAM / relative)
        if actual != expected:
            raise SystemExit(f"data hash mismatch for {relative}: {actual}")
        actual_hashes[relative] = actual
    train_ids = load_ids(UPSTREAM / "data/grpo/train.jsonl")
    val_ids = load_ids(UPSTREAM / "data/grpo/validation.jsonl")
    final_ids = load_ids(UPSTREAM / "data/evaluation/tasks.jsonl")
    if train_ids & val_ids or train_ids & final_ids or val_ids & final_ids:
        raise SystemExit("GRPO train, validation, and Final-200 task IDs overlap")

    return {
        "method": method,
        "model": str(model),
        "config": str(CONFIG),
        "config_sha256": sha256(CONFIG),
        "runtime": installed,
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(2)],
        "data_sha256": actual_hashes,
        "final_200_touched_for_training": False,
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=("grpo", "dapo", "pad", "event", "valid-group", "partial-valid-group"),
        required=True,
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--skip-initial-validation",
        action="store_true",
        help="Skip Validation-50 only for bounded smoke runs (steps <= 2).",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Optional smoke-only prompt batch override.",
    )
    parser.add_argument(
        "--single-group",
        action="store_true",
        help="Use one complete four-rollout GRPO group per valid-group update.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.skip_initial_validation and args.steps > 2:
        raise SystemExit("--skip-initial-validation is restricted to <=2-step smoke runs")
    if args.train_batch_size is not None:
        if args.steps > 2:
            raise SystemExit("--train-batch-size override is restricted to <=2-step smoke runs")
        if args.train_batch_size < 2:
            raise SystemExit("--train-batch-size must be at least 2")
    if args.single_group and args.train_batch_size is not None:
        raise SystemExit("--single-group cannot be combined with --train-batch-size")
    single_group_overrides(args.method, args.single_group)
    model = args.model.expanduser().resolve()
    output = (args.output or UPSTREAM / f"outputs/models/{args.method}-2x3090").resolve()
    audit = preflight(model, args.method)
    audit.update(
        {
            "steps": args.steps,
            "output": str(output),
            "single_group": args.single_group,
        }
    )

    environment = dict(os.environ)
    environment.update(
        {
            # veRL's fused FSDP workers need both GPUs visible so each Ray
            # actor can select its assigned physical device. If Ray masks each
            # actor to a single logical cuda:0, NCCL sees both ranks on the
            # same PCI device and aborts with "Duplicate GPU detected".
            "CUDA_VISIBLE_DEVICES": "0,1",
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "PYTHONPATH": f"{ROOT}:{UPSTREAM / 'src'}",
            "SHOPPING_GRPO_ROOT": str(UPSTREAM),
            "SHOPPING_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
            "SHOPPING_ENV_MANIFEST": str(UPSTREAM / "data/environment.json"),
            "SHOPSIM_BASE_URL": "http://127.0.0.1:5700",
            "SHOPPING_AGENT_LOOP_CONFIG": str(AGENT_CONFIG),
            "SHOPPING_TOOL_CONFIG": str(UPSTREAM / "configs/tools.json"),
            "GRPO_MODEL_PATH": str(model),
            "GRPO_TRAIN_FILE": str(UPSTREAM / "data/grpo/train.parquet"),
            "GRPO_VAL_FILE": str(UPSTREAM / "data/grpo/validation.parquet"),
            "GRPO_OUTPUT_DIR": str(output),
        }
    )
    audit["device_mapping"] = {
        "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": environment[
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"
        ],
    }
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={CONFIG.parent}",
        f"--config-name={CONFIG.stem}",
        f"trainer.total_training_steps={args.steps}",
        f"trainer.experiment_name=shopping-{args.method}-2x3090",
        f"pad_grpo.enable={'true' if args.method == 'pad' else 'false'}",
        f"event_credit.enable={'true' if args.method == 'event' else 'false'}",
        "shopping_hgpo.enable=false",
        dynamic_sampling_override(args.method),
        dynamic_selection_mode_override(args.method),
    ]
    workers_override = dataloader_workers_override(args.method)
    if workers_override is not None:
        command.append(workers_override)
    command.extend(single_group_overrides(args.method, args.single_group))
    command.extend(partial_valid_group_overrides(args.method))
    command.extend(dapo_overrides(args.method))
    if args.method == "event":
        command.append("algorithm.adv_estimator=grpo_event_rtg")
    if args.skip_initial_validation:
        command.append("trainer.val_before_train=false")
    if args.train_batch_size is not None:
        command.append(f"data.train_batch_size={args.train_batch_size}")
    audit["command"] = command
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    raise SystemExit(subprocess.call(command, cwd=UPSTREAM, env=environment))


if __name__ == "__main__":
    main()
