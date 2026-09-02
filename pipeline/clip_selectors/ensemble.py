"""
Combines every available BaseSelector's opinion into one ranked list of
Candidates, using the weights from config/pipeline.yaml's
selection.ensemble.weights.

Unavailable selectors (missing dependency, model failed to load) and
selectors with no opinion on a given candidate (score() returned None)
are excluded and the remaining weights are renormalized - a selector
being down never zeroes-out a candidate, it just falls back to whatever
signals ARE available.
"""
from typing import List
from .base import BaseSelector, Candidate


class EnsembleSelector:
    def __init__(self, selectors: List[BaseSelector], weights: dict, min_selectors_available: int = 1):
        self.selectors = selectors
        self.weights = weights
        self.min_selectors_available = min_selectors_available

    def rank(self, candidates: List[Candidate], context: dict) -> List[Candidate]:
        available = [s for s in self.selectors if s.is_available()]

        for candidate in candidates:
            weighted_sum = 0.0
            total_weight = 0.0
            contributing = 0

            for selector in available:
                s = selector.score(candidate, context)
                if s is None:
                    continue
                w = self.weights.get(selector.name, 0.0)
                if w <= 0:
                    continue
                weighted_sum += s * w
                total_weight += w
                contributing += 1
                candidate.scores[selector.name] = s

            if contributing < self.min_selectors_available or total_weight == 0:
                candidate.scores["ensemble_total"] = 0.0
                candidate.meta["ensemble_fallback"] = True
            else:
                candidate.scores["ensemble_total"] = weighted_sum / total_weight

        return sorted(candidates, key=lambda c: c.scores.get("ensemble_total", 0.0), reverse=True)
