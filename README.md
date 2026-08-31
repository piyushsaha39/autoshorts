# AutoShorts

Turn one long YouTube video into several ready-to-post vertical clips —
completely free, running on your own machine.

```
python main.py "https://youtube.com/watch?v=XXXXXXXXXXX" --num-clips 6
```

Output: `outputs/clip_01.mp4` … `clip_0N.mp4`, each 9:16, captioned, plus a
`metadata.json` with the suggested title/hook for each clip so you're not
staring at a blank caption box when you upload.

## How it works

1. **Download** the source video with **yt-dlp**.
2. **Transcribe** it locally with **faster-whisper**, getting word-level
   timestamps (needed for both clip boundaries and captions).
3. **Select highlights** by handing the timestamped transcript to an LLM
   and asking it to pick the N most "stop-the-scroll" self-contained
   moments, each 10–60s. The LLM only picks *segment indices*, so every
   cut lands on a real sentence boundary — never mid-word.
4. **Reframe to 9:16**: a free on-device face detector (**mediapipe**)
   checks where the speaker is. If it finds them reliably, it crops the
   frame to keep them centered. If not (slides, gameplay, wide shots), it
   falls back to a blurred-background pad instead of cropping something
   important out.
5. **Burn in captions**: word-level timestamps become a karaoke-style
   `.ass` subtitle (current word highlighted), rendered by ffmpeg's
   built-in subtitle support — the same "bouncing word" look you're used
   to seeing on Shorts/Reels.
6. **Encode** the final clips with ffmpeg.

Every tool in that list is free and open-source, except step 3, which
needs *some* LLM call — see below for three ways to keep that free too.

## Setup

```bash
git clone <this folder, or just unzip it>
cd autoshorts
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

You also need **ffmpeg** on your PATH (`ffmpeg -version` to check). It's
already free — install via your package manager (`sudo apt install ffmpeg`,
`brew install ffmpeg`, or the static builds at ffmpeg.org for Windows).

### Pick a free LLM provider for step 3

Copy `.env.example` to `.env` and fill in **one** of these:

| Provider | Cost | Notes |
|---|---|---|
| **Groq** (`groq/llama-3.3-70b-versatile`) | Free tier | Fastest option, generous daily limits. Get a key at console.groq.com/keys. |
| **Gemini** (`gemini/gemini-2.5-flash-lite`) | Free tier | Very high daily request quota, huge context window if your video is long. Get a key at aistudio.google.com/apikey. |
| **Ollama** (`ollama/qwen2.5:7b-instruct`) | $0, fully offline | No API key, no rate limits, nothing leaves your machine. Install Ollama, then `ollama pull qwen2.5:7b-instruct`. |

Pass whichever you pick with `--llm-model`, e.g.:
```bash
python main.py "<url>" --llm-model "ollama/qwen2.5:7b-instruct"
```

## Tuning for your hardware

An RTX 4050 with 6GB VRAM comfortably runs:
- `faster-whisper` at `--whisper-model medium --device cuda --compute-type float16`
  (drop to `small` if you ever see an out-of-memory error).
- Rendering is CPU-bound by default (`libx264`). Add `--nvenc` to use the
  4050's hardware encoder instead — noticeably faster for a batch of clips,
  and it frees the CPU to keep transcribing/encoding queued clips in
  parallel if you extend the script to do so.
- No GPU at all? Use `--device cpu --compute-type int8` — slower, but still
  $0 and it'll get there.

## Customizing

- **Clip count/length**: `--num-clips`, `--min-len`, `--max-len`.
- **Caption look**: edit `pipeline/captions.py` — `fontname`, `fontsize`,
  `HIGHLIGHT_COLOR` (ASS hex is `&HBBGGRR&`, not RGB), `MAX_WORDS_PER_LINE`.
  If `DejaVu Sans Bold` isn't installed on your system, swap it for any
  bold font you have (`fc-list | grep -i bold` to see what's available).
- **Highlight-picking prompt**: `pipeline/llm_selector.py` → `PROMPT_TEMPLATE`.
  This is the single biggest lever for clip quality — tune it toward your
  content (e.g. "prefer moments with a concrete number or statistic" for
  data-heavy videos, "prefer emotional beats" for interviews/vlogs).
- **Reframing**: `pipeline/reframe.py` uses a single static crop position
  per clip (robust, and plenty for typical talking-head footage). If your
  source has the speaker walking around a lot, you could extend
  `analyze_crop` to return a moving position and build a time-varying
  `crop` expression in `render.py` — flagged here as the natural next
  upgrade, not implemented by default to keep the ffmpeg command simple
  and reliable.

## Limitations & natural next steps

- The LLM only sees text, not audio/video — it can't detect laughter,
  applause, or visual comedy on its own. If that matters for your content,
  add simple audio-energy peak detection (e.g. via `librosa`) as a second
  signal alongside the transcript.
- No auto-upload step. YouTube (Shorts), Instagram (Reels), and TikTok all
  have upload APIs, but each needs its own app review/approval process —
  worth adding once you're happy with clip quality, not before.
- Batch mode (a list of URLs processed overnight) is a small change to
  `main.py`'s loop — everything else already supports it.

## A quick note on rights

This is a personal repurposing tool. Only run it on videos you own, have
explicit permission to clip, or that are otherwise licensed for reuse —
the same rule any clipping/editing tool operates under.
