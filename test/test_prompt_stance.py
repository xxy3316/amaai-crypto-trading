"""The prompt stance is an experimental variable, so it must behave like one.

Background: the decision prompt ended with "Use 0 when the agent inputs add
nothing" and "Be conservative: the rules are a reasonable baseline". Measured on
15 identical decisions (same bars, same model, temperature 0, only this
paragraph differing):

    conservative : adjustments {0: 15}                non-zero  0/15
    neutral      : adjustments {-1: 6, 0: 3, +1: 6}   non-zero 12/15

Under `conservative` the model returned a BEARISH stance alongside an adjustment
of zero -- it held a view and was instructed not to act on it. So a null result
for "the LLM changed nothing" would really have been a result about that
paragraph, not about the model.

What these tests protect:

  * the default must stay `conservative`, byte-identical to the historical
    prompt, or every number produced before today becomes unattributable;
  * an unknown stance must never silently become a different experiment;
  * the stance must reach the run metadata, because a reported result is not
    interpretable without knowing which variant produced it.

Run directly (the package name `test` shadows the stdlib `test` module):

    python test/test_prompt_stance.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llm import (  # noqa: E402
    PROMPT_STANCE_ASSERTIVE,
    PROMPT_STANCE_CONSERVATIVE,
    PROMPT_STANCE_NEUTRAL,
    PROMPT_STANCES,
    prompt_stance_text,
    resolved_prompt_stance,
)

#: The exact text every result before 2026-09-13 was produced under. Pinned so
#: that editing the conservative stance to chase a better number fails loudly
#: instead of silently rewriting the baseline of every earlier run.
HISTORICAL_TAIL = """- Use 0 when the agent inputs add nothing beyond what the rules already capture.
- Set veto=true ONLY to block a trade on clear risk grounds.
- Every entry in key_factors must cite a specific input above, not generic advice.
Be conservative: the rules are a reasonable baseline, so only move the score when
the qualitative agent inputs genuinely justify it."""


class TestStanceRegistry(unittest.TestCase):

    def test_the_three_variants_exist(self):
        self.assertEqual(set(PROMPT_STANCES),
                         {"conservative", "neutral", "assertive"})

    def test_conservative_is_byte_identical_to_the_historical_prompt(self):
        """Changing this silently re-bases every number produced before today."""
        self.assertEqual(PROMPT_STANCE_CONSERVATIVE, HISTORICAL_TAIL)

    def test_default_is_conservative(self):
        """The default must preserve existing behaviour, not the new option."""
        self.assertEqual(prompt_stance_text(), PROMPT_STANCE_CONSERVATIVE)

    def test_the_variants_are_actually_different(self):
        texts = [PROMPT_STANCE_CONSERVATIVE, PROMPT_STANCE_NEUTRAL,
                 PROMPT_STANCE_ASSERTIVE]
        self.assertEqual(len(set(texts)), 3)

    def test_every_variant_keeps_the_non_negotiable_instructions(self):
        """Latitude over the ADJUSTMENT only; the guardrails do not vary.

        If a stance dropped the veto restriction or the citation requirement,
        it would differ in more than one way and the comparison would no longer
        isolate conservatism.
        """
        for name, text in PROMPT_STANCES.items():
            with self.subTest(stance=name):
                self.assertIn("veto=true ONLY", text)
                self.assertIn("key_factors", text)

    def test_no_variant_mentions_a_specific_adjustment_value(self):
        """A stance must not smuggle in the answer, e.g. 'prefer +1'."""
        for name, text in PROMPT_STANCES.items():
            with self.subTest(stance=name):
                for token in ("+1", "+2", "+3", "-1", "-2", "-3"):
                    self.assertNotIn(token, text)


class TestStanceResolution(unittest.TestCase):

    def test_named_stance_is_returned(self):
        for name in PROMPT_STANCES:
            with self.subTest(stance=name):
                self.assertEqual(prompt_stance_text(name), PROMPT_STANCES[name])
                self.assertEqual(resolved_prompt_stance(name), name)

    def test_unknown_stance_falls_back_without_raising(self):
        """A typo must not kill a long backtest..."""
        self.assertEqual(prompt_stance_text("nonsense"),
                         PROMPT_STANCE_CONSERVATIVE)

    def test_unknown_stance_reports_what_actually_ran(self):
        """...but the metadata must say what was really used, not the typo."""
        self.assertEqual(resolved_prompt_stance("nonsense"), "conservative")

    def test_case_and_whitespace_are_tolerated(self):
        for spelling in ("NEUTRAL", " neutral ", "Neutral"):
            with self.subTest(spelling=spelling):
                self.assertEqual(resolved_prompt_stance(spelling), "neutral")

    def test_empty_means_default(self):
        for empty in (None, "", "   "):
            with self.subTest(value=empty):
                self.assertEqual(resolved_prompt_stance(empty), "conservative")


class TestStanceIsRecorded(unittest.TestCase):

    def test_configuration_snapshot_carries_the_stance(self):
        """A result is not interpretable without knowing which variant ran."""
        from core.config import describe_configuration
        snapshot = describe_configuration()
        self.assertIn("llm_prompt_stance", snapshot["phase1"])

    def test_snapshot_reflects_the_environment(self):
        import importlib
        import core.config as cfg
        previous = os.environ.get("LLM_PROMPT_STANCE")
        try:
            os.environ["LLM_PROMPT_STANCE"] = "assertive"
            importlib.reload(cfg)
            self.assertEqual(
                cfg.describe_configuration()["phase1"]["llm_prompt_stance"],
                "assertive")
        finally:
            if previous is None:
                os.environ.pop("LLM_PROMPT_STANCE", None)
            else:
                os.environ["LLM_PROMPT_STANCE"] = previous
            importlib.reload(cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
