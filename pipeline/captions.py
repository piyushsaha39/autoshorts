"""Generate bouncing/highlight ASS subtitles using PySubs2 and configuration presets."""

import pysubs2
from .transcribe import Segment

try:
    from pipeline.caption_styles import get_style
except ImportError:
    get_style = None


def build_captions(
    segments: list[Segment],
    clip_start: float,
    clip_end: float,
    out_path: str,
    style_name: str = "default",
) -> None:
    subs = pysubs2.SSAFile()
    subs.info["PlayResX"] = "1080"
    subs.info["PlayResY"] = "1920"

    style_cfg = get_style(style_name) if get_style is not None else {}

    font_name = style_cfg.get("font", "Arial Black")
    font_size = style_cfg.get("fontsize", 75)
    bold = style_cfg.get("bold", True)
    margin_v = style_cfg.get("margin_v", 480)
    highlight_color = style_cfg.get("highlight_color", "&H00D7FF&")
    default_color = style_cfg.get("default_color", "&HFFFFFF&")
    max_words_per_line = style_cfg.get("max_words_per_line", 5)

    style = pysubs2.SSAStyle(
        fontname=font_name,
        fontsize=font_size,
        bold=bold,
        primarycolor=pysubs2.Color(255, 255, 255),
        outlinecolor=pysubs2.Color(0, 0, 0),
        outline=style_cfg.get("outline", 6),
        shadow=style_cfg.get("shadow", 2),
        alignment=pysubs2.Alignment.BOTTOM_CENTER,
        marginv=margin_v,
    )
    subs.styles["Highlight"] = style

    words = [
        w for seg in segments for w in seg.words
        if w.text and w.end > clip_start and w.start < clip_end
    ]

    if not words:
        subs.save(out_path)
        return

    lines: list[list] = []
    current: list = []
    for w in words:
        current.append(w)
        if len(current) >= max_words_per_line or w.text.endswith((".", "?", "!")):
            lines.append(current)
            current = []
    if current:
        lines.append(current)

    for line in lines:
        for active_idx, active_word in enumerate(line):
            text = " ".join(
                "{\\c" + (highlight_color if j == active_idx else default_color) + "}" + w.text.strip().upper()
                for j, w in enumerate(line)
            )
            start_ms = int(max(0.0, active_word.start - clip_start) * 1000)
            end_ms = int(max(0.0, active_word.end - clip_start) * 1000)
            if end_ms <= start_ms:
                continue
            subs.append(pysubs2.SSAEvent(start=start_ms, end=end_ms, text=text, style="Highlight"))

    subs.save(out_path)