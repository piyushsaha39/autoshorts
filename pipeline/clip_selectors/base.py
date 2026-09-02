"""
Shared interface every clip-candidate selector implements — the "existing"
LLM/YAMNet selectors (via thin adapters, see legacy_adapters.py) and every
new signal source (diarization, emotion, hook-strength) alike.

This is what lets main.py's core loop stay exactly as-is for
selection.mode: "legacy", while selection.mode: "ensemble" can loop over
`selectors` and combine scores without any selector needing to know about
the others.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Candidate:
    """A candidate clip window under consideration."""
    start: float
    end: float
    text: Optional[str] = None
    source: str = "unknown"          # e.g. "llm", "action", "manual"
    scores: dict = field(default_factory=dict)   # selector_name -> 0..1 score
    meta: dict = field(default_factory=dict)     # free-form extra data


class BaseSelector(ABC):
    """
    Contract:
      - `name` is a stable string used as the key in
        selection.ensemble.weights in config/pipeline.yaml.
      - `is_available()` must NEVER raise. If a dependency is missing or a
        model fails to load, return False so the ensemble silently
        excludes and renormalizes around this selector.
      - `score()` returns a float in [0, 1], or None if this selector has
        no opinion about the candidate (e.g. no faces in frame for the
        emotion selector). None is excluded from the weighted average,
        it is NOT treated as 0.
    """
    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def score(self, candidate: Candidate, context: dict) -> Optional[float]:
        ...
