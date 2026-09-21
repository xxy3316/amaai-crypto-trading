"""`core` must stay importable without Streamlit.

This is the property the 2026-09-21 extraction bought, and it is worth a test
because it decays silently. Nothing fails loudly when someone adds `import
streamlit` or an `st.spinner` to a core module -- the app keeps working, the
tests keep passing, and the breakage only shows up later as an experiment that
cannot run headlessly.

Before the extraction, `TradingDecisionAgent` -- the component every claim in
the paper rests on -- lived in a 3,900-line Streamlit script. The experiment
harness reached it by faking the whole Streamlit API, so the experiments
depended on that stub staying faithful to a UI library nobody was tracking.

Each core module is imported in a SUBPROCESS with `streamlit` poisoned, so an
accidental import fails here rather than in a backtest three weeks later.

Run directly (the package name `test` shadows the stdlib `test` module):

    python test/test_core_is_headless.py
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: Every module under core/. The engine lives here; the UI does not.
CORE_MODULES = [
    "core.config",
    "core.data",
    "core.execution",
    "core.llm",
    "core.metrics",
    "core.agents",
]

#: Poisons `import streamlit` so any core module that reaches for it raises.
PROBE = """
import sys

class _Poison:
    def __getattr__(self, name):
        raise AssertionError(
            "core module imported streamlit (attribute %r). core must stay "
            "headless so experiments can run without faking a UI." % name)

class _Blocker:
    def find_module(self, fullname, path=None):
        return self if fullname == "streamlit" or fullname.startswith("streamlit.") else None
    def load_module(self, fullname):
        raise ImportError(
            "core must not import streamlit (blocked: %s)" % fullname)

sys.meta_path.insert(0, _Blocker())
import {module}
print("OK")
"""


class TestCoreIsHeadless(unittest.TestCase):

    def _import_without_streamlit(self, module: str):
        result = subprocess.run(
            [sys.executable, "-c", PROBE.format(module=module)],
            cwd=str(REPO), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180,
        )
        return result

    def test_every_core_module_imports_without_streamlit(self):
        for module in CORE_MODULES:
            with self.subTest(module=module):
                result = self._import_without_streamlit(module)
                self.assertIn("OK", result.stdout or "",
                              f"{module} failed to import without streamlit:\n"
                              f"{result.stderr[-1500:]}")

    def test_signals_are_headless_too(self):
        """The exogenous channels are engine, not UI, for the same reason."""
        for module in ("signals.binance_positioning", "signals.text_sentiment"):
            with self.subTest(module=module):
                result = self._import_without_streamlit(module)
                self.assertIn("OK", result.stdout or "",
                              f"{module} failed:\n{result.stderr[-1500:]}")

    def test_no_core_source_file_mentions_streamlit(self):
        """A cheap second net: catches a deferred import inside a function.

        The import probe above only exercises module scope, so an `import
        streamlit` hidden inside a rarely-called branch would slip past it.
        """
        offenders = []
        for path in sorted((REPO / "core").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue          # prose about the rule is not a violation
                if "import streamlit" in stripped:
                    offenders.append(f"{path.name}:{lineno}")
        self.assertEqual(offenders, [],
                         "core must not import streamlit anywhere: "
                         + ", ".join(offenders))

    def test_the_decision_agent_lives_in_core(self):
        """The component the paper rests on must be reachable headlessly."""
        from core.agents import TradingDecisionAgent
        self.assertTrue(callable(TradingDecisionAgent))


if __name__ == "__main__":
    unittest.main(verbosity=2)
