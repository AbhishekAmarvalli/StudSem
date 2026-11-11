"""Colour-contrast smoke test for the local web UI themes.

Reads the real tokens from webapp/app.css so a palette change that hides text
fails the suite instead of shipping.
"""
import re
from pathlib import Path

CSS = (Path(__file__).resolve().parents[1] / "webapp" / "app.css").read_text(encoding="utf-8")


def _block(selector: str) -> str:
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", CSS)
    assert match, f"missing CSS block {selector}"
    return match.group(1)


def _token(block: str, name: str) -> str:
    match = re.search(rf"--{re.escape(name)}:\s*(#[0-9a-fA-F]{{6}})", block)
    assert match, f"missing --{name} token"
    return match.group(1)


def _luminance(color: str) -> float:
    rgb = [int(color.lstrip("#")[index:index + 2], 16) / 255 for index in (0, 2, 4)]
    linear = [value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
              for value in rgb]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground: str, background: str) -> float:
    light, dark = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def _assert_readable(pairs: dict, theme: str) -> None:
    for label, (foreground, background) in pairs.items():
        value = round(_contrast(foreground, background), 2)
        assert value >= 4.5, f"{theme} {label} contrast is only {value}:1"


def test_light_theme_text_is_readable():
    root = _block(":root")
    _assert_readable({
        "body": (_token(root, "ink"), _token(root, "paper")),
        "muted": (_token(root, "muted"), _token(root, "paper")),
        "engine ready": (_token(root, "blue"), _token(root, "blue-soft")),
        "engine in use": (_token(root, "green"), _token(root, "green-soft")),
        "engine paused": (_token(root, "red"), _token(root, "red-soft")),
        "placeholder": ("#676767", "#ffffff"),
    }, "light")


def test_dark_theme_text_is_readable():
    dark = _block('html[data-theme="dark"]')
    _assert_readable({
        "body": (_token(dark, "ink"), _token(dark, "paper")),
        "muted": (_token(dark, "muted"), _token(dark, "paper")),
        "engine ready": (_token(dark, "blue"), _token(dark, "blue-soft")),
        "engine in use": (_token(dark, "green"), _token(dark, "green-soft")),
        "engine paused": (_token(dark, "red"), _token(dark, "red-soft")),
        "placeholder": ("#9a9488", "#1b1b1b"),
        "link hover on yellow": ("#0a0a0a", "#ffe600"),
    }, "dark")


def test_engines_have_a_colour_class_per_state():
    for state in ("ready", "preferred", "paused", "off"):
        assert re.search(rf"\.engine\.{state}\s*\{{", CSS), f"missing .engine.{state}"
