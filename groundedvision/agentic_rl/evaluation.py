"""Fixed-denominator paired evaluation for shopping-agent checkpoints."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence


def _task_id(row: Mapping[str, object]) -> str:
    if "task_id" not in row:
        raise ValueError("benchmark row is missing task_id")
    return str(row["task_id"])


def _index(rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    indexed = {}
    for row in rows:
        task_id = _task_id(row)
        if task_id in indexed:
            raise ValueError(f"duplicate benchmark task_id: {task_id}")
        indexed[task_id] = row
    return indexed


def _metric(row: Mapping[str, object] | None, name: str) -> float:
    if row is None:
        return 0.0
    aliases = {
        "strict_success": ("strict_success", "strict_gold_success"),
        "purchase_success": ("purchase_success",),
        "final_reward": ("final_reward", "terminal_utility"),
    }
    for key in aliases[name]:
        if key in row:
            return float(row[key])
    reward = row.get("reward")
    if isinstance(reward, Mapping):
        for key in aliases[name]:
            if key in reward:
                return float(reward[key])
    terminal = row.get("terminal_result")
    if isinstance(terminal, Mapping):
        detail = terminal.get("reward_detail")
        if isinstance(detail, Mapping):
            if name == "purchase_success":
                return float(detail.get("purchase_success") is True)
            if name == "strict_success":
                return float(
                    row.get("status") == "done"
                    and row.get("done") is True
                    and terminal.get("done") is True
                    and terminal.get("over") is True
                    and detail.get("reward_type") == "gold_purchase"
                    and detail.get("reward_valid") is True
                    and detail.get("purchase_success") is True
                    and detail.get("termination_reason") == "gold_purchase"
                )
    return 0.0


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_mean_interval(
    differences: Sequence[float], *, samples: int, seed: int
) -> tuple[float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    size = len(differences)
    estimates = [
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def compare_paired_shopping_results(
    baseline_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    expected_task_ids: Sequence[object],
    *,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    """Compare two policies on one immutable set, retaining missing-task failures."""

    expected = [str(task_id) for task_id in expected_task_ids]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected task IDs must be non-empty and unique")
    baseline = _index(baseline_rows)
    candidate = _index(candidate_rows)
    unexpected = (set(baseline) | set(candidate)) - set(expected)
    if unexpected:
        raise ValueError(f"results contain unexpected task IDs: {sorted(unexpected)[:5]}")

    report: dict[str, object] = {
        "expected_tasks": len(expected),
        "baseline_completed": sum(task_id in baseline for task_id in expected),
        "candidate_completed": sum(task_id in candidate for task_id in expected),
        "metrics": {},
    }
    for metric in ("strict_success", "purchase_success", "final_reward"):
        base_values = [_metric(baseline.get(task_id), metric) for task_id in expected]
        candidate_values = [_metric(candidate.get(task_id), metric) for task_id in expected]
        differences = [right - left for left, right in zip(base_values, candidate_values, strict=True)]
        base_mean = sum(base_values) / len(expected)
        candidate_mean = sum(candidate_values) / len(expected)
        interval = _bootstrap_mean_interval(
            differences, samples=bootstrap_samples, seed=seed
        )
        report["metrics"][metric] = {
            "baseline": base_mean,
            "candidate": candidate_mean,
            "absolute_delta": candidate_mean - base_mean,
            "relative_delta": (
                (candidate_mean - base_mean) / abs(base_mean)
                if base_mean != 0
                else None
            ),
            "paired_95ci": list(interval),
        }

    strict_base = [_metric(baseline.get(task_id), "strict_success") > 0 for task_id in expected]
    strict_candidate = [_metric(candidate.get(task_id), "strict_success") > 0 for task_id in expected]
    report["strict_transitions"] = {
        "loss": sum(left and not right for left, right in zip(strict_base, strict_candidate, strict=True)),
        "win": sum(not left and right for left, right in zip(strict_base, strict_candidate, strict=True)),
        "both_success": sum(left and right for left, right in zip(strict_base, strict_candidate, strict=True)),
        "both_failure": sum(not left and not right for left, right in zip(strict_base, strict_candidate, strict=True)),
    }
    return report


def summarize_useful_update_metrics(metric_rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Aggregate valid-group counters without using policy reward as a proxy."""

    if not metric_rows:
        raise ValueError("at least one metric row is required")
    total_groups = sum(float(row.get("pad/groups_total", 0.0)) for row in metric_rows)
    terminal_groups = sum(float(row.get("pad/groups_terminal", 0.0)) for row in metric_rows)
    recovered_groups = sum(float(row.get("pad/groups_recovered", 0.0)) for row in metric_rows)
    useful_groups = terminal_groups + recovered_groups
    generated_rollouts = sum(float(row.get("rollout/generated_total", 0.0)) for row in metric_rows)
    return {
        "updates": float(len(metric_rows)),
        "groups_total": total_groups,
        "groups_terminal": terminal_groups,
        "groups_recovered": recovered_groups,
        "useful_group_rate": useful_groups / total_groups if total_groups else 0.0,
        "rollouts_per_useful_group": generated_rollouts / useful_groups if useful_groups else 0.0,
    }
