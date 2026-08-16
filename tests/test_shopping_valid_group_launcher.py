"""Focused launch-contract tests for valid-group GRPO."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_shopping_rl_2x3090 as launcher


class ValidGroupLauncherTest(unittest.TestCase):
    def test_dapo_command_uses_dynamic_sampling_and_clip_higher(self):
        with patch.object(
            sys,
            "argv",
            [
                "run_shopping_rl_2x3090.py",
                "--method",
                "dapo",
                "--steps",
                "2",
                "--skip-initial-validation",
                "--dry-run",
            ],
        ), patch.object(launcher, "preflight", return_value={}):
            with patch("builtins.print") as printed:
                launcher.main()

        command = printed.call_args.args[0]
        self.assertIn("shopping_dynamic_sampling.enable=true", command)
        self.assertIn("shopping_dynamic_sampling.selection_mode=partial_valid", command)
        self.assertIn("actor_rollout_ref.actor.clip_ratio_low=0.20", command)
        self.assertIn("actor_rollout_ref.actor.clip_ratio_high=0.28", command)

    def test_valid_group_command_enables_only_dynamic_sampling(self):
        with patch.object(
            sys,
            "argv",
            [
                "run_shopping_rl_2x3090.py",
                "--method",
                "valid-group",
                "--steps",
                "2",
                "--skip-initial-validation",
                "--dry-run",
            ],
        ), patch.object(launcher, "preflight", return_value={}):
            with patch("builtins.print") as printed:
                launcher.main()

        command = printed.call_args.args[0]
        self.assertIn("shopping_dynamic_sampling.enable=true", command)
        self.assertIn("shopping_dynamic_sampling.selection_mode=whole", command)
        self.assertIn("data.dataloader_num_workers=0", command)
        self.assertIn("pad_grpo.enable=false", command)
        self.assertIn("event_credit.enable=false", command)

    def test_standard_grpo_command_keeps_dynamic_sampling_disabled(self):
        with patch.object(
            sys,
            "argv",
            [
                "run_shopping_rl_2x3090.py",
                "--method",
                "grpo",
                "--steps",
                "2",
                "--skip-initial-validation",
                "--dry-run",
            ],
        ), patch.object(launcher, "preflight", return_value={}):
            with patch("builtins.print") as printed:
                launcher.main()

        command = printed.call_args.args[0]
        self.assertIn("shopping_dynamic_sampling.enable=false", command)

    def test_only_valid_group_requests_resampling(self):
        self.assertEqual(
            launcher.dynamic_sampling_override("dapo"),
            "shopping_dynamic_sampling.enable=true",
        )
        self.assertEqual(
            launcher.dynamic_selection_mode_override("dapo"),
            "shopping_dynamic_sampling.selection_mode=partial_valid",
        )
        self.assertEqual(
            launcher.dynamic_sampling_override("valid-group"),
            "shopping_dynamic_sampling.enable=true",
        )
        self.assertEqual(
            launcher.dynamic_sampling_override("partial-valid-group"),
            "shopping_dynamic_sampling.enable=true",
        )
        for method in ("grpo", "hgpo", "pad", "event"):
            self.assertEqual(
                launcher.dynamic_sampling_override(method),
                "shopping_dynamic_sampling.enable=false",
            )

    def test_matched_group_methods_disable_dataloader_workers(self):
        for method in ("grpo", "dapo", "valid-group", "partial-valid-group"):
            self.assertEqual(
                launcher.dataloader_workers_override(method),
                "data.dataloader_num_workers=0",
            )
        for method in ("pad", "event"):
            self.assertIsNone(launcher.dataloader_workers_override(method))

    def test_partial_valid_group_uses_explicit_selection_and_batch_contract(self):
        self.assertEqual(
            launcher.dynamic_selection_mode_override("partial-valid-group"),
            "shopping_dynamic_sampling.selection_mode=partial_valid",
        )
        self.assertEqual(
            launcher.partial_valid_group_overrides("dapo"),
            [
                "data.train_batch_size=2",
                "actor_rollout_ref.actor.ppo_mini_batch_size=1",
            ],
        )
        self.assertEqual(
            launcher.partial_valid_group_overrides("partial-valid-group"),
            [
                "data.train_batch_size=2",
                "actor_rollout_ref.actor.ppo_mini_batch_size=1",
            ],
        )
        self.assertEqual(
            launcher.dataloader_workers_override("partial-valid-group"),
            "data.dataloader_num_workers=0",
        )

    def test_single_group_command_keeps_one_complete_group_consistent(self):
        with patch.object(
            sys,
            "argv",
            [
                "run_shopping_rl_2x3090.py",
                "--method",
                "valid-group",
                "--single-group",
                "--steps",
                "2",
                "--skip-initial-validation",
                "--dry-run",
            ],
        ), patch.object(launcher, "preflight", return_value={}):
            with patch("builtins.print") as printed:
                launcher.main()

        command = printed.call_args.args[0]
        self.assertIn("data.train_batch_size=1", command)
        self.assertIn("actor_rollout_ref.actor.ppo_mini_batch_size=1", command)

    def test_single_group_rejects_other_methods(self):
        with self.assertRaisesRegex(SystemExit, "only supported"):
            launcher.single_group_overrides("grpo", True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
