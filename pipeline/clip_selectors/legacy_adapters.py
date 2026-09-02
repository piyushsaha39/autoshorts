"""
Thin adapters that let your EXISTING pipeline/llm_selector.py and
pipeline/action_selector.py plug into the BaseSelector interface without
changing a single line inside those two files.

I don't have your actual source for llm_selector.py / action_selector.py
(only the architecture description you shared), so the two spots marked
TODO below need a one-line adjustment to match your real function/attribute
names. Everything else is usable as-is.

Until selection.mode is switched to "ensemble" in config/pipeline.yaml,
these adapters are not called at all - main.py's existing binary
LLM-or-YAMNet branch keeps running unmodified.
"""
from .base import BaseSelector, Candidate


class LegacyLLMSelectorAdapter(BaseSelector):
    name = "llm_narrative"

    def is_available(self) -> bool:
        try:
            from pipeline import llm_selector  # noqa: F401
            return True
        except ImportError:
            return False

    def score(self, candidate: Candidate, context: dict):
        # TODO: point this at whatever numeric confidence/quality value
        # your llm_selector.py already produces per segment.
        #
        # If llm_selector.py currently only returns a final list of
        # *accepted* segments with no per-segment score, the simplest
        # non-breaking bridge is: mark accepted segments with
        # candidate.meta["llm_score"] = 1.0 when you build the Candidate
        # list, and leave it unset (None) for everything else. See
        # IMPLEMENTATION_GUIDE.md Section 1.3 for the one-line prompt
        # change that gets you a REAL 0-1 score instead of this binary
        # stand-in.
        return candidate.meta.get("llm_score")


class LegacyActionSelectorAdapter(BaseSelector):
    name = "action_energy"

    def is_available(self) -> bool:
        try:
            from pipeline import action_selector  # noqa: F401
            return True
        except ImportError:
            return False

    def score(self, candidate: Candidate, context: dict):
        # TODO: point this at the normalized rolling-average energy value
        # action_selector.py computes for this window. Normalize it to
        # 0..1 before returning (e.g. value / running_max_energy).
        return candidate.meta.get("energy_score")
