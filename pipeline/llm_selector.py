"""Pick highlight-worthy clip candidates from the transcript using an LLM."""

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


PROMPT_TEMPLATE = """You are a video editing API. Your ONLY job is to extract the {num_clips} most entertaining highlights from the transcript provided.

You must output your response EXACTLY as a valid JSON object with a single key "clips", containing an array of objects. 
DO NOT output any conversational text, explanations, or summaries.

Example of EXACT required output:
{{
  "clips": [
    {{
      "start_segment": 0,
      "end_segment": 5,
      "title": "Hilarious opening joke",
      "hook": "Wait, did you really just say that?",
      "score": 9.5
    }}
  ]
}}

Rules:
- Clip duration should be roughly between {min_len} and {max_len} seconds.
- start_segment and end_segment must be integer IDs from the transcript below.
- Do not pick overlapping ranges.

Transcript to analyze:
{transcript}
"""


def _format_transcript(segments: list[Segment]) -> str:
    return "\n".join(f"[{i}] ({s.start:.1f}-{s.end:.1f}s) {s.text}" for i, s in enumerate(segments))


def select_clips(
    segments: list[Segment],
    model: str = "ollama/qwen2.5:7b-instruct",
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

    # Force Ollama into native JSON mode and drop temperature to absolute zero
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a machine that outputs only valid JSON objects. Never generate conversational text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
    except Exception:
        # Fallback if your specific LiteLLM version rejects response_format for local Ollama
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a machine that outputs only valid JSON objects. Never generate conversational text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

    raw = response.choices[0].message.content.strip()
    
    # Strip markdown code blocks
    raw_cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed_data = json.loads(raw_cleaned)
    except json.JSONDecodeError:
        # Aggressive Regex: Hunt for anything that looks like the "clips" JSON object
        match = re.search(r"\{.*\"clips\"\s*:\s*\[.*\]\s*\}", raw, re.DOTALL)
        if match:
            try:
                parsed_data = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                raise ValueError(f"LLM extraction failed completely. Raw output:\n{raw}") from e
        else:
            raise ValueError(f"LLM stubbornly refused to output JSON. Raw output:\n{raw}")

    # Handle cases where the model returns just the array instead of the requested object
    if isinstance(parsed_data, list):
        picks = parsed_data
    else:
        picks = parsed_data.get("clips", [])

    candidates: list[ClipCandidate] = []
    for p in picks:
        i, j = p.get("start_segment"), p.get("end_segment")
        if i is None or j is None or not (0 <= i <= j < len(segments)):
            continue
        start, end = segments[i].start, segments[j].end

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