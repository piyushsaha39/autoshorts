"""Build a karaoke-style .ass subtitle file (current word highlighted) from
Whisper's word-level timestamps -- the "bouncing word" caption look every
Shorts/Reels editor uses. Rendered for free by ffmpeg's built-in libass
support, no external caption tool needed.
"""
import pysubs2

from .transcribe import Segment

DEFAULT_COLOR = "&HFFFFFF&"     # white   (ASS format: &HBBGGRR&)
HIGHLIGHT_COLOR = "&H00D7FF&"   # gold/amber for the active word
MAX_WORDS_PER_LINE = 5


def build_captions(segments: list[Segment], clip_start: float, clip_end: float, out_path: str) -> None:
    subs = pysubs2.SSAFile()
    
    # 1. READABLE SIZE: Define the video resolution explicitly so fonts scale correctly.
    # Without this, FFmpeg assumes a tiny 384x288 canvas, making all font sizes massive.
    subs.info["PlayResX"] = "1080"
    subs.info["PlayResY"] = "1920"

    style = pysubs2.SSAStyle(
        # 3. TRENDY FONT: Using Arial Black for a bold, modern short-form look. 
        # (Can also be swapped to "Impact" or "Montserrat Black" if installed)
        fontname="Arial Black",
        
        # Now that the canvas is 1920px tall, size 75 is perfectly scaled and readable.
        fontsize=75,
        bold=True,
        primarycolor=pysubs2.Color(255, 255, 255),
        outlinecolor=pysubs2.Color(0, 0, 0),
        outline=6,
        shadow=2,
        
        # 2. POSITION: Bottom-Center, pushed up by 480 pixels (exactly 1/4 of 1920).
        alignment=pysubs2.Alignment.BOTTOM_CENTER,
        marginv=480,
    )
    subs.styles["Highlight"] = style

    words = [
        w
        for seg in segments
        for w in seg.words
        if w.text and w.end > clip_start and w.start < clip_end
    ]
    if not words:
        subs.save(out_path)
        return

    # group consecutive words into short on-screen lines
    lines: list[list] = []
    current: list = []
    for w in words:
        current.append(w)
        if len(current) >= MAX_WORDS_PER_LINE or w.text.endswith((".", "?", "!")):
            lines.append(current)
            current = []
    if current:
        lines.append(current)

    for line in lines:
        for active_idx, active_word in enumerate(line):
            text = " ".join(
                "{\\c" + (HIGHLIGHT_COLOR if j == active_idx else DEFAULT_COLOR) + "}" + w.text
                for j, w in enumerate(line)
            )
            start_ms = int(max(0, active_word.start - clip_start) * 1000)
            end_ms = int(max(0, active_word.end - clip_start) * 1000)
            if end_ms <= start_ms:
                continue
            subs.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=text, style="Highlight"))

    subs.save(out_path)