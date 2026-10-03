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
    "auto_segmenter",
    "reassemble",
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

    # How reassembled segments are joined when a probe reorders them. Sentences
    # run together on one line; lines and turns need their newline back or the
    # state changes shape as well as order, which would confound the measurement.
    joiner = " "

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


# Tokens that end in a full stop without ending a sentence. Splitting after one
# of these produces a fragment, and a fragment is worse than a coarse segment:
# ablating "Dr" on its own measures nothing anybody wrote.
_ABBREVIATIONS = frozenset(
    """
    mr mrs ms mx dr prof sr jr st rev hon capt sgt lt col gen
    inc ltd llc llp co corp dept est fig no vs etc al
    jan feb mar apr jun jul aug sept sep oct nov dec
    mon tue tues wed thu thur thurs fri sat sun
    approx avg max min dept ref acct amt apt
    """.split()
)

# "e.g." and "i.e." end in a stop after a single letter, which the initials rule
# below would also catch, but naming them is clearer than relying on that.
_DOTTED = frozenset({"e.g", "i.e", "a.m", "p.m", "u.s", "u.k"})


class SentenceSegmenter(_SpanSegmenter):
    """Split prose into sentences.

    A regex with a guard, not a parser. It breaks after ``.``, ``!`` or ``?``
    followed by whitespace, and at blank lines, except where the stop belongs to
    something other than the end of a sentence:

    * a known abbreviation — ``Dr.``, ``Inc.``, ``etc.``, ``e.g.``
    * an initial — the ``J.`` in ``J. Smith``
    * a number — the stop in ``No. 5``

    The guard only ever *removes* a candidate boundary, so a false negative
    yields a coarser segment rather than a fragment. That is the right way round:
    a fragment attributes a score to text nobody wrote as a unit.

    Still not a parser. Pass an explicit ``pattern`` or use
    :class:`LineSegmenter` over pre-split text where correctness matters more
    than convenience.
    """

    kind = "sentence"

    # The closing punctuation is captured rather than consumed. Left in the
    # separator it was silently dropped from the segment, so `He said "fine."`
    # became `He said "fine.` — which corrupts both the quoted evidence in a
    # minimal-evidence result and the text restored when that segment is kept.
    _BOUNDARY = re.compile(r"(?<=[.!?])(?P<close>[\"')\]]*)\s+|\n{2,}")
    _TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.[\"')\]]*$")

    def __init__(
        self,
        *,
        mask: str = DEFAULT_MASK,
        pattern: re.Pattern[str] | None = None,
        guard_abbreviations: bool = True,
    ) -> None:
        super().__init__(mask=mask)
        self._boundary = pattern or self._BOUNDARY
        self._guard = guard_abbreviations

    def _is_boundary(self, text: str, position: int) -> bool:
        """Whether the stop ending at ``position`` really ends a sentence."""
        if not self._guard:
            return True

        head = text[:position]
        if not head.endswith((".", '."', ".'", ".)", ".]")):
            return True  # ! or ? never abbreviate

        match = self._TRAILING_WORD.search(head)
        if match is None:
            # No word before the stop, so there is no abbreviation to protect.
            # Decimals need no special case: "3.5" has no whitespace after the
            # stop, so the boundary pattern never proposes it in the first place.
            return True

        word = match.group(1)
        if len(word) == 1 and word.isupper():
            return False  # an initial, as in "J. Smith"
        lowered = word.lower().rstrip(".")
        return lowered not in _ABBREVIATIONS and lowered not in _DOTTED

    def _spans(self, text: str) -> list[tuple[int, int]]:
        has_close = "close" in self._boundary.groupindex
        spans: list[tuple[int, int]] = []
        cursor = 0
        for match in self._boundary.finditer(text):
            closers = (match.group("close") or "") if has_close else ""
            end = match.start() + len(closers)
            if not self._is_boundary(text, end):
                continue
            if text[cursor:end].strip():
                spans.append((cursor, end))
            cursor = match.end()
        if text[cursor:].strip():
            spans.append((cursor, len(text)))
        return spans


class LineSegmenter(_SpanSegmenter):
    """One segment per non-blank line. The right default for logs, diffs and code."""

    kind = "line"
    joiner = "\n"

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
    joiner = "\n"

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
        prune_empty: bool = True,
    ) -> None:
        self.mask = mask
        self.include_scalars = include_scalars
        self.min_length = min_length
        self.prune_empty = prune_empty

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

        if mode == "delete" and self.prune_empty:
            # Removing the only leaf under a key leaves `"customer": {}` behind.
            # That is an artifact of how the removal was done, not content the
            # model was ever meant to read, and it still costs tokens. "delete"
            # should mean gone.
            out = _drop_empty_containers(out)
        return out


def _drop_empty_containers(node: Any) -> Any:
    """Recursively remove containers left empty by a deletion.

    Bottom-up, so a branch that becomes empty only because its children were
    pruned is removed too. Scalars, empty strings and ``None`` are left alone:
    they are content, however uninformative, and only containers emptied by our
    own edit are artifacts.
    """
    if isinstance(node, Mapping):
        cleaned = {}
        for key, value in node.items():
            pruned = _drop_empty_containers(value)
            if isinstance(pruned, (Mapping, list)) and len(pruned) == 0:
                continue
            cleaned[key] = pruned
        return cleaned

    if isinstance(node, list):
        cleaned_list = []
        for value in node:
            pruned = _drop_empty_containers(value)
            if isinstance(pruned, (Mapping, list)) and len(pruned) == 0:
                continue
            cleaned_list.append(pruned)
        return cleaned_list

    return node


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


def reassemble(
    segmenter: Segmenter, segments: Sequence[Segment], order: Sequence[int]
) -> State:
    """Rebuild a text state with its segments in a different order.

    Used by the ordering probe: if the same evidence in a different sequence
    produces a different answer, position is acting as evidence, which it should
    not be.

    Structured states have no linear order to permute, so they are not supported.
    """
    joiner = getattr(segmenter, "joiner", None)
    if joiner is None:
        raise TypeError(
            f"{type(segmenter).__name__} has no linear ordering to permute; "
            "reordering only applies to text states"
        )

    by_id = {segment.id: segment for segment in segments}
    missing = [i for i in order if i not in by_id]
    if missing:
        raise KeyError(f"no such segment ids: {missing}")

    return joiner.join(by_id[i].text.strip() for i in order)


def _require_text(state: State, kind: str) -> str:
    if not isinstance(state, str):
        raise TypeError(
            f"{kind} segmentation needs a text state; "
            "use JsonFieldSegmenter for objects and arrays"
        )
    return state
