"""
Hook-strength scoring: rates ONLY the opening ~2-3 seconds of a candidate
as a scroll-stopping hook, separately from the LLM's overall
narrative-quality judgment of the full clip.

Two ways to wire this in, in increasing order of efficiency and risk -
pick based on how comfortable you are editing llm_selector.py's prompt:

  A) SEPARATE CALL (this file, as-is): one extra small Ollama call per
     candidate. Zero changes to your existing llm_selector.py prompt or
     JSON schema. Slower (2x LLM calls) but the safest possible integration.

  B) MERGED SCHEMA (recommended once A is validated): add a
     "hook_score": <0-10> field to the JSON schema you already ask the
     LLM for per-segment in llm_selector.py, and read
     candidate.meta["hook_score"] directly in score() below instead of
     making a second call. See IMPLEMENTATION_GUIDE.md Section 1.3 for
     the exact one-line prompt diff.
"""
import json
from typing import Optional
from .base import BaseSelector, Candidate


def score_hook_strength(ollama_client, model_name: str, first_seconds_text: str, full_candidate_text: str) -> Optional[float]:
    prompt = f"""You are scoring short-form video hooks for viewer retention.

Opening line/moment (first 2-3 seconds of the clip):
\"\"\"{first_seconds_text}\"\"\"

Full clip context (for reference only - do not score this part):
\"\"\"{full_candidate_text[:500]}\"\"\"

Rate ONLY the opening's strength as a scroll-stopping hook, from 0 to 10.
Respond with ONLY a JSON object, no other text:
{{"hook_score": <int 0-10>, "reason": "<one short sentence>"}}
"""
    response = ollama_client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    raw = response["message"]["content"].strip().replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(raw)
        return max(0.0, min(1.0, float(data.get("hook_score", 0)) / 10.0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


class HookStrengthSelector(BaseSelector):
    name = "hook_strength"

    def __init__(self, ollama_client=None, model_name: str = "qwen2.5:7b-instruct"):
        self.ollama_client = ollama_client
        self.model_name = model_name

    def is_available(self) -> bool:
        return self.ollama_client is not None

    def score(self, candidate: Candidate, context: dict) -> Optional[float]:
        # Path B (merged schema) short-circuit - use it if present
        if "hook_score" in candidate.meta:
            return candidate.meta["hook_score"]

        if not candidate.text:
            return None

        # crude first-3-seconds text approximation: first ~12 words
        # (replace with a real word-timestamp cutoff if you have per-word
        # timing on the candidate already, for a cleaner boundary)
        words = candidate.text.split()
        first_seconds_text = " ".join(words[:12])

        return score_hook_strength(self.ollama_client, self.model_name, first_seconds_text, candidate.text)
