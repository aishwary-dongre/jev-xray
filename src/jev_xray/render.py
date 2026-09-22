"""Turning an attribution into something a person can read at a glance.

The terminal heatmap is the point of the whole exercise. A ranked table tells
you which segment mattered; painting the state itself shows you *where* the
decision lives, in the original wording, which is what makes a surprising answer
explicable in a couple of seconds.

Green means the segment was holding the answer up: removing it moved the tracked
scalar down. Red means it was pushing against the answer. Intensity is relative
to the strongest segment in this explanation, so a heatmap is readable on its own
but two heatmaps are not comparable by colour alone.
"""

from __future__ import annotations

import os
import sys

from .attribution import Attribution, SegmentEffect

__all__ = ["heatmap", "table", "report", "supports_color"]

_RESET = "\x1b[0m"
_FG_BLACK = "\x1b[38;5;16m"
_DIM = "\x1b[2m"

# Light to saturated, five bins each.
_GREENS = (194, 157, 120, 84, 46)
_REDS = (224, 217, 210, 203, 196)

_EPSILON = 1e-4


def supports_color(stream: object | None = None) -> bool:
    """Colour only when a human on a terminal is going to see it."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream if stream is not None else sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def _bin(magnitude: float, strongest: float) -> int:
    if strongest <= _EPSILON:
        return -1
    share = magnitude / strongest
    if share < 0.05:
        return -1
    return min(4, int(share * 5))


def _paint(text: str, effect: SegmentEffect, strongest: float, color: bool) -> str:
    if not color:
        return text
    index = _bin(effect.magnitude, strongest)
    if index < 0:
        return f"{_DIM}{text}{_RESET}"
    palette = _GREENS if effect.delta > 0 else _REDS
    return f"\x1b[48;5;{palette[index]}m{_FG_BLACK}{text}{_RESET}"


def heatmap(attribution: Attribution, *, color: bool | None = None) -> str:
    """Paint the original state, segment by segment.

    Only meaningful for a text state, where segments are spans of the original
    string. For a structured state the segments are paths with no linear layout,
    so this falls back to the ranked table.
    """
    if not isinstance(attribution.state, str):
        return table(attribution)

    color = supports_color() if color is None else color
    text = attribution.state
    effects = {e.segment.id: e for e in attribution.effects}
    strongest = max((e.magnitude for e in attribution.effects), default=0.0)

    pieces: list[str] = []
    cursor = 0
    for segment in sorted(attribution.segments, key=lambda s: s.start or 0):
        start, end = segment.start or 0, segment.end or 0
        if start > cursor:
            pieces.append(text[cursor:start])
        effect = effects.get(segment.id)
        body = text[start:end]
        pieces.append(_paint(body, effect, strongest, color) if effect else body)
        cursor = end
    if cursor < len(text):
        pieces.append(text[cursor:])

    legend = (
        f"  {_dimmed('green = held the answer up, red = pushed against it, ', color)}"
        f"{_dimmed('intensity relative to the strongest segment', color)}"
    )
    return "".join(pieces) + "\n" + legend


def table(attribution: Attribution, *, limit: int = 10, color: bool | None = None) -> str:
    """Ranked segments, strongest influence first."""
    color = supports_color() if color is None else color
    rows = attribution.ranked()[:limit]
    if not rows:
        return "  (no segment effects recorded)"

    width = max(len(e.segment.label) for e in rows)
    lines = [
        f"  {'segment'.ljust(width)}  {'delta':>8}  {'ablated':>8}   evidence",
        f"  {'-' * width}  {'-' * 8}  {'-' * 8}   {'-' * 40}",
    ]
    for effect in rows:
        arrow = "+" if effect.delta > 0 else "-" if effect.delta < 0 else " "
        marker = _dimmed(arrow, color)
        lines.append(
            f"  {effect.segment.label.ljust(width)}  {effect.delta:+8.4f}  "
            f"{effect.ablated_value:8.4f} {marker} {effect.segment.preview(56)}"
        )
    return "\n".join(lines)


def report(attribution: Attribution, *, color: bool | None = None, limit: int = 10) -> str:
    """Everything worth printing for one explanation."""
    color = supports_color() if color is None else color
    blocks = [
        attribution.summary(),
        "",
        _dimmed("state", color),
        heatmap(attribution, color=color),
        "",
        _dimmed("ranked evidence", color),
        table(attribution, limit=limit, color=color),
    ]

    residual = attribution.interaction_residual
    if residual is not None and abs(residual) > 0.05:
        blocks += [
            "",
            _dimmed("note", color),
            f"  interaction residual {residual:+.4f}: these segments do not act "
            f"independently,\n  so treat the ranking as indicative and the "
            f"magnitudes as not additive.",
        ]

    if attribution.failures:
        blocks += ["", _dimmed("failed ablations", color)]
        blocks += [f"  {seg.label}: {why}" for seg, why in attribution.failures]

    return "\n".join(blocks)


def _dimmed(text: str, color: bool) -> str:
    return f"{_DIM}{text}{_RESET}" if color else text
