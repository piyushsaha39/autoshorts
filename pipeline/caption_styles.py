"""
Caption style loader for config/caption_styles.yaml.

`_DEFAULT_STYLE` below is a hardcoded fallback matching the CURRENT
hardcoded look in captions.py. If the YAML file is ever missing or
malformed, get_style("default") still returns this exact dict, so
captions never silently change appearance due to a config-loading
problem.
"""
import os
import yaml

_DEFAULT_STYLE = {
    "font": "Arial Black",
    "base_color": "&H00FFFFFF",
    "highlight_color": "&H0000BFFF",
    "uppercase": True,
    "bold": True,
    "margin_v": 480,
    "font_size": 72,
    "outline": 3,
}

_style_cache = None


def load_styles(config_path: str = "config/caption_styles.yaml") -> dict:
    global _style_cache
    if _style_cache is not None:
        return _style_cache

    if not os.path.exists(config_path):
        _style_cache = {"default": _DEFAULT_STYLE}
        return _style_cache

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("default", _DEFAULT_STYLE)
    _style_cache = data
    return _style_cache


def get_style(name: str = "default") -> dict:
    styles = load_styles()
    return styles.get(name, styles["default"])
