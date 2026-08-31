"""Pick highlight-worthy clip candidates from the transcript using an LLM.

Uses LiteLLM so the same code works with any provider -- just change the
model string and set the matching API key in .env:

    "groq/llama-3.3-70b-versatile"    -> free tier, very fast   (needs GROQ_API_KEY)
    "gemini/gemini-2.5-flash-lite"    -> free tier, huge context (needs GEMINI_API_KEY)
    "ollama/qwen2.5:7b-instruct"      -> fully local/offline    (needs Ollama running)

Because the LLM is only asked to pick segment INDICES (not to re-type
timestamps), clip boundaries always land exactly on a real sentence
boundary from the transcript -- no awkward mid-word cuts.
"""
import json
import re
from dataclasses import dataclass

import litellm

from .transcribe import Segment


@dataclass
class ClipCandidate:
    start: float
    end: float
    title: str
    hook: str
    score: float


PROMPT_TEMPLATE = """You are editing a video into short vertical clips. Below is a transcript, \
split into numbered segments with timestamps in seconds.

Pick the {num_clips} best highlights. Rules:
- You MUST pick candidates, even if the dialogue jumps around (like a trailer). Just find the most interesting parts.
- Aim for durations between {min_len} and {max_len} seconds, but it does not have to be perfect.
- Always start/end exactly on a segment boundary from the list below.
- Do not pick overlapping ranges.

Return ONLY a JSON array, no prose, no markdown fences:
[{{"start_segment": int, "end_segment": int, "title": "short punchy title", "hook": "why this clip works, one sentence", "score": 0-10}}]

Transcript segments:
{transcript}
"""


def _format_transcript(segments: list[Segment]) -> str:
    return "\n".join(f"[{i}] ({s.start:.1f}-{s.end:.1f}s) {s.text}" for i, s in enumerate(segments))


def select_clips(
    segments: list[Segment],
    model: str = "groq/llama-3.3-70b-versatile",
    num_clips: int = 6,
    min_len: int = 10,
    max_len: int = 60,
) -> list[ClipCandidate]:
    prompt = PROMPT_TEMPLATE.format(
        num_clips=num_clips,
        min_len=min_len,
        max_len=max_len,
        transcript=_format_transcript(segments),
    )

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        picks = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON. Raw output:\n{raw}") from e

    candidates: list[ClipCandidate] = []
    for p in picks:
        i, j = p.get("start_segment"), p.get("end_segment")
        if i is None or j is None or not (0 <= i <= j < len(segments)):
            continue
        start, end = segments[i].start, segments[j].end
        
        # REMOVED: Strict duration check. We keep everything the LLM selects.

        candidates.append(
            ClipCandidate(
                start=start,
                end=end,
                title=p.get("title", "Untitled clip"),
                hook=p.get("hook", ""),
                score=float(p.get("score", 0)),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates