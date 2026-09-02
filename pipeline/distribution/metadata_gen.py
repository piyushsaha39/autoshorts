"""
Per-clip metadata generation (title, description, hashtags) using the same
local Ollama model already used for highlight selection. Purely additive:
appends new fields to metadata.json, never renames or removes existing ones.
"""
import json


def generate_clip_metadata(ollama_client, model_name: str, transcript_text: str,
                            platform: str = "youtube_shorts") -> dict:
    prompt = f"""Based on this short video clip's transcript, write metadata for {platform}.

Transcript:
\"\"\"{transcript_text[:2000]}\"\"\"

Respond with ONLY a JSON object in this exact shape, no other text:
{{
  "title": "<under 60 characters, punchy>",
  "description": "<1-2 sentences>",
  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]
}}
"""
    response = ollama_client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    raw = response["message"]["content"].strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"title": None, "description": None, "hashtags": [], "generation_error": True}
