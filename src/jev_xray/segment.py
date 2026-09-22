"""Cutting a state into the units an explanation is written in.

Segmentation is the most consequential choice in occlusion attribution. The
segments are the vocabulary of the explanation: nothing finer can ever be
attributed, and segments that straddle two independent pieces of evidence blur
them together. Sentence-level is a good default for prose, field-level for
records, turn-level for conversations.

Two ablation modes, because they fail differently and the disagreement between
them is itself a signal:

``delete``
    Remove the span entirely. Faithful to "what if this had not been said", but
    it shortens the state and shifts everything after it, which is a
    perturbation of its own.
``mask``
    Replace the span with a short neutral placeholder. Preserves structure and
    position at the cost of introducing text that was never there.

Where the two disagree, the reported effect is partly an artifact of ablation
rather than the content. Phase 2 uses that comparison as a control; phase 1 at
least makes it cheap to run both.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence, runtime_checkable

from .types import State

__all__ = [
    "Segment",
    "Segmenter",
    "AblationMode",
    "SentenceSegmenter",
    "LineSegmenter",
    "TurnSegmenter",
    "JsonFieldSegmenter",
    "DEFAULT_MASK",
    "get_segmenter",
]

AblationMode = Literal["delete", "mask"]

DEFAULT_MASK = "[...]"


@dataclass(frozen=True, slots=True)
class Segment:
    """One ablatable unit of a state."""

    id: int
    text: str
    kind: str
    start: int | None = None
    end: int | None = None
    path: str | None = None

    @property
    def label(self) -> str:
        """Short human-readable handle, used in reports."""
        if self.path is not None:
            return self.path
        return f"{self.kind} {self.id}"

    def preview(self, width: int = 72) -> str:
        flat = " ".join(self.text.split())
        if len(flat) <= width:
            return flat
        return flat[: width - 1] + "\u2026"


@runtime_checkable
class Segmenter(Protocol):
    kind: str

    def split(self, state: State) -> list[Segment]: ...

    def ablate(
        self, state: State, drop: Iterable[int], *, mode: AblationMode = "delete"
    ) -> State: ...


# --------------------------------------------------------------------------
# String states: segments are spans, ablation rewrites the string
# --------------------------------------------------------------------------


class _SpanSegmenter:
    """Shared machinery for segmenters over a text state."""

    kind = "span"

    def __init__(self, *, mask: str = DEFAULT_MASK) -> None:
        self.mask = mask

    def _spans(self, text: str) -> list[tuple[int, int]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def split(self, state: State) -> list[Segment]:
        text = _require_text(state, self.kind)
        return [
            Segment(id=i, text=text[start:end], kind=self.kind, start=start, end=end)
            for i, (start, end) in enumerate(self._spans(text))
        ]

    def ablate(
        self, state: State, drop: Iterable[int], *, mode: AblationMode = "delete"
    ) -> State:
        text = _require_text(state, self.kind)
        drop_ids = set(drop)
        if not drop_ids:
            return text

        segments = self.split(text)
        known = {s.id for s in segments}
        unknown = drop_ids - known
        if unknown:
            raise KeyError(f"no such segment ids: {sorted(unknown)}")

        # Rewrite from the end so earlier offsets stay valid.
        out = text
        for segment in sorted(segments, key=lambda s: s.start or 0, reverse=True):
            if segment.id not in drop_ids:
                continue
            start, end = segment.start or 0, segment.end or 0
            if mode == "mask":
                out = out[:start] + self.mask + out[end:]
            else:
                # Absorb the whitespace that followed the span so deleting a
                # sentence does not leave a double space behind. Odd spacing is
                # itself a perturbation of a literal reader.
                tail = end
                while tail < len(out) and out[tail] in " \t":
                    tail += 1
                out = out[:start] + out[tail:]
        return out


class SentenceSegmenter(_SpanSegmenter):
    """Split prose into sentences.

    A regex, not a parser. It breaks after ``.``, ``!`` or ``?`` followed by
    whitespace, and at blank lines. Known limitation: abbreviations, decimals
    and ellipses can produce a spurious break. Where that matters, pass an
    explicit ``pattern`` or use :class:`LineSegmenter` over pre-split text.
    """

    kind = "sentence"

    _BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n{2,}")

    def __init__(self, *, mask: str = DEFAULT_MASK, pattern: re.Pattern[str] | None = None) -> None:
        super().__init__(mask=mask)
        self._boundary = pattern or self._BOUNDARY

    def _spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        for match in self._boundary.finditer(text):
            end = match.start()
            if text[cursor:end].strip():
                spans.append((cursor, end))
            cursor = match.end()
        if text[cursor:].strip():
            spans.append((cursor, len(text)))
        return spans


class LineSegmenter(_SpanSegmenter):
    """One segment per non-blank line. The right default for logs, diffs and code."""

    kind = "line"

    def _spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        cursor = 0
        for line in text.splitlines(keepends=True):
            stripped = line.rstrip("\r\n")
            if stripped.strip():
                spans.append((cursor, cursor + len(stripped)))
            cursor += len(line)
        return spans


class TurnSegmenter(_SpanSegmenter):
    """One segment per conversation turn.

    For a text state, a turn starts at a line carrying a short ``Speaker:``
    prefix and runs until the next one, so a multi-line message stays whole.
    For a list state, use :class:`JsonFieldSegmenter`, where each element is
    already its own addressable unit.
    """

    kind = "turn"

    _SPEAKER = re.compile(r"^[ \t]*([A-Za-z][\w .'\-]{0,38}):[ \t]", re.MULTILINE)

    def _spans(self, text: str) -> list[tuple[int, int]]:
        starts = [m.start() for m in self._SPEAKER.finditer(text)]
        if not starts:
            return LineSegmenter()._spans(text)
        if starts[0] > 0 and text[: starts[0]].strip():
            starts.insert(0, 0)

        spans: list[tuple[int, int]] = []
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            chunk = text[start:end].rstrip()
            if chunk.strip():
                spans.append((start, start + len(chunk)))
        return spans


# --------------------------------------------------------------------------
# JSON states: segments are paths, ablation rewrites the structure
# --------------------------------------------------------------------------


class JsonFieldSegmenter:
    """One segment per leaf of a structured state.

    Paths use the dot-and-index form the docs use when pointing a question at
    part of the state, so a segment label can be pasted straight into an
    instruction: ``conversation.messages.0.text``.
    """

    kind = "field"

    def __init__(
        self,
        *,
        mask: str = DEFAULT_MASK,
        include_scalars: bool = False,
        min_length: int = 1,
    ) -> None:
        self.mask = mask
        self.include_scalars = include_scalars
        self.min_length = min_length

    def _leaves(self, node: Any, prefix: str = "") -> list[tuple[str, Any]]:
        if isinstance(node, Mapping):
            found: list[tuple[str, Any]] = []
            for key, value in node.items():
                found.extend(self._leaves(value, f"{prefix}.{key}" if prefix else str(key)))
            return found
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            found = []
            for index, value in enumerate(node):
                found.extend(self._leaves(value, f"{prefix}.{index}" if prefix else str(index)))
            return found
        return [(prefix, node)]

    def split(self, state: State) -> list[Segment]:
        if isinstance(state, str):
            raise TypeError(
                "JsonFieldSegmenter needs an object or array state; "
                "use SentenceSegmenter or LineSegmenter for text"
            )

        segments: list[Segment] = []
        for path, value in self._leaves(state):
            if isinstance(value, str):
                if len(value.strip()) < self.min_length:
                    continue
                text = value
            elif self.include_scalars and value is not None:
                text = str(value)
            else:
                continue
            segments.append(
                Segment(id=len(segments), text=text, kind=self.kind, path=path)
            )
        return segments

    def ablate(
        self, state: State, drop: Iterable[int], *, mode: AblationMode = "delete"
    ) -> State:
        drop_ids = set(drop)
        if not drop_ids:
            return state

        segments = {s.id: s for s in self.split(state)}
        unknown = drop_ids - set(segments)
        if unknown:
            raise KeyError(f"no such segment ids: {sorted(unknown)}")

        out = copy.deepcopy(state)
        # Deepest paths first: removing a list element shifts the indices of its
        # siblings, so a shallower edit must not invalidate a pending deeper one.
        targets = sorted(
            (segments[i].path or "" for i in drop_ids),
            key=lambda p: (p.count("."), p),
            reverse=True,
        )
        for path in targets:
            _edit_path(out, path, mask=self.mask if mode == "mask" else None)
        return out


def _edit_path(root: Any, path: str, *, mask: str | None) -> None:
    """Mask or remove the leaf at ``path``. Missing paths are ignored."""
    parts = path.split(".") if path else []
    if not parts:
        return

    node = root
    for part in parts[:-1]:
        node = _descend(node, part)
        if node is None:
            return

    leaf = parts[-1]
    if isinstance(node, Mapping):
        if leaf not in node:
            return
        if mask is None:
            del node[leaf]  # type: ignore[union-attr]
        else:
            node[leaf] = mask  # type: ignore[index]
        return

    if isinstance(node, list):
        try:
            index = int(leaf)
        except ValueError:
            return
        if not 0 <= index < len(node):
            return
        if mask is None:
            del node[index]
        else:
            node[index] = mask


def _descend(node: Any, part: str) -> Any:
    if isinstance(node, Mapping):
        return node.get(part)
    if isinstance(node, list):
        try:
            index = int(part)
        except ValueError:
            return None
        return node[index] if 0 <= index < len(node) else None
    return None


# --------------------------------------------------------------------------


_REGISTRY: dict[str, type] = {
    "sentence": SentenceSegmenter,
    "sentences": SentenceSegmenter,
    "line": LineSegmenter,
    "lines": LineSegmenter,
    "turn": TurnSegmenter,
    "turns": TurnSegmenter,
    "field": JsonFieldSegmenter,
    "fields": JsonFieldSegmenter,
    "json": JsonFieldSegmenter,
}


def get_segmenter(spec: str | Segmenter, **kwargs: Any) -> Segmenter:
    """Resolve a segmenter by name, or pass one through unchanged."""
    if not isinstance(spec, str):
        return spec
    try:
        factory = _REGISTRY[spec.lower()]
    except KeyError:
        raise ValueError(
            f"unknown segmenter {spec!r}; choose from {sorted(set(_REGISTRY))}"
        ) from None
    return factory(**kwargs)  # type: ignore[return-value]


def auto_segmenter(state: State, **kwargs: Any) -> Segmenter:
    """Pick a reasonable segmenter from the shape of the state."""
    if not isinstance(state, str):
        return JsonFieldSegmenter(**kwargs)
    if TurnSegmenter._SPEAKER.search(state):
        return TurnSegmenter(**kwargs)
    # Many short lines look like a log or a diff; flowing text does not.
    lines = [line for line in state.splitlines() if line.strip()]
    if len(lines) >= 3 and sum(len(line) for line in lines) / len(lines) < 80:
        return LineSegmenter(**kwargs)
    return SentenceSegmenter(**kwargs)


def _require_text(state: State, kind: str) -> str:
    if not isinstance(state, str):
        raise TypeError(
            f"{kind} segmentation needs a text state; "
            "use JsonFieldSegmenter for objects and arrays"
        )
    return state
