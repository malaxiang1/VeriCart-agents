import pytest

from groundedvision.agentic_rl.evaluation import (
    compare_paired_shopping_results,
    summarize_useful_update_metrics,
)


def test_paired_comparison_keeps_missing_tasks_in_denominator():
    report = compare_paired_shopping_results(
        [
            {"task_id": 1, "strict_success": True, "purchase_success": True, "final_reward": 1},
            {"task_id": 2, "strict_success": False, "purchase_success": False, "final_reward": -1},
        ],
        [
            {"task_id": 1, "strict_success": True, "purchase_success": True, "final_reward": 1},
            {"task_id": 2, "strict_success": True, "purchase_success": True, "final_reward": 1},
        ],
        [1, 2, 3],
        bootstrap_samples=100,
    )
    assert report["expected_tasks"] == 3
    assert report["baseline_completed"] == 2
    assert report["candidate_completed"] == 2
    assert report["metrics"]["strict_success"]["baseline"] == pytest.approx(1 / 3)
    assert report["metrics"]["strict_success"]["candidate"] == pytest.approx(2 / 3)
    assert report["strict_transitions"] == {
        "loss": 0,
        "win": 1,
        "both_success": 1,
        "both_failure": 1,
    }


def test_zero_baseline_relative_delta_is_explicitly_undefined():
    report = compare_paired_shopping_results(
        [],
        [{"task_id": "a", "strict_success": True}],
        ["a"],
        bootstrap_samples=10,
    )
    assert report["metrics"]["strict_success"]["relative_delta"] is None


def test_paired_comparison_reads_native_shopsimulator_reward_detail():
    trajectory = {
        "task_id": 1,
        "status": "done",
        "done": True,
        "final_reward": 1.0,
        "terminal_result": {
            "done": True,
            "over": True,
            "reward_detail": {
                "reward_type": "gold_purchase",
                "reward_valid": True,
                "purchase_success": True,
                "termination_reason": "gold_purchase",
            },
        },
    }
    report = compare_paired_shopping_results([], [trajectory], [1], bootstrap_samples=10)
    assert report["metrics"]["strict_success"]["candidate"] == 1.0
    assert report["metrics"]["purchase_success"]["candidate"] == 1.0


def test_unexpected_or_duplicate_task_ids_fail_closed():
    with pytest.raises(ValueError, match="unexpected"):
        compare_paired_shopping_results([], [{"task_id": 2}], [1], bootstrap_samples=10)
    with pytest.raises(ValueError, match="duplicate"):
        compare_paired_shopping_results(
            [{"task_id": 1}, {"task_id": 1}], [], [1], bootstrap_samples=10
        )


def test_useful_update_summary_counts_recovery_not_reward():
    summary = summarize_useful_update_metrics(
        [
            {
                "pad/groups_total": 2,
                "pad/groups_terminal": 1,
                "pad/groups_recovered": 1,
                "rollout/generated_total": 8,
            },
            {
                "pad/groups_total": 2,
                "pad/groups_terminal": 0,
                "pad/groups_recovered": 1,
                "rollout/generated_total": 8,
            },
        ]
    )
    assert summary["useful_group_rate"] == pytest.approx(0.75)
    assert summary["rollouts_per_useful_group"] == pytest.approx(16 / 3)
