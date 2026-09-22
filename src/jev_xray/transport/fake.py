"""A deterministic stand-in for Jev.

This is not an attempt to imitate Jev's judgment. It is a fixture with *known
causal structure*: you declare which patterns in the state push which answers
which way, and the fake answers accordingly. That makes the attribution engine
falsifiable. If you plant evidence in segment 4 and leave-one-out does not rank
segment 4 first, the engine is wrong — and you can prove that without an API
key, which matters while hosted signups are closed.

It is deterministic by design. No RNG, no sampling, so a test that passes twice
passes for a reason.

The confidence statistic here is one minus normalised entropy. The real model's
exact statistic is not published beyond "collapses the distribution's shape into
a single number, flatter means less confident", so do not calibrate anything
against this number.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..budget import estimate_tokens
from ..types import Choice, Noul, Score, SystemOneRequest
from .base import TransportError

__all__ = ["FakeTransport", "Signal"]


@dataclass(frozen=True, slots=True)
class Signal:
    """A planted piece of evidence.

    ``pattern``
        Case-insensitive regex searched in the flattened state.
    ``weight``
        Logit contribution when present. Negative weights model evidence that
        argues *against* an answer, which is what makes opposing-segment
        detection testable.
    ``label``
        For a Choice, the option this evidence favours. For a Score, the level
        index as a string. For a Noul, ``"true"`` (the default) or ``"false"``.
    ``question_id``
        Restrict the signal to one question. ``None`` applies it to all of them.
    """

    pattern: str
    weight: float = 1.0
    label: str | None = None
    question_id: str | None = None

    def matches(self, text: str) -> bool:
        return re.search(self.pattern, text, re.IGNORECASE) is not None


@dataclass(slots=True)
class FakeTransport:
    """In-process transport driven by declared signals."""

    signals: Sequence[Signal] = field(default_factory=tuple)
    bias: float = 0.0
    latency: float = 0.0
    fail_first: int = 0
    model: str | None = None

    calls: int = field(default=0, init=False)
    seen_states: list[str] = field(default_factory=list, init=False)

    async def send(self, request: SystemOneRequest) -> Mapping[str, Any]:
        # Latency is applied before the injected failure so that the two compose
        # the way a real request does: time passes, then it fails. That ordering
        # also gives concurrent callers a window to coalesce onto a call that is
        # going to fail, which is the interesting case to test.
        if self.latency:
            import asyncio

            await asyncio.sleep(self.latency)

        if self.fail_first > 0:
            self.fail_first -= 1
            raise TransportError("injected failure", status=503)

        text = _flatten(request.state)
        self.calls += 1
        self.seen_states.append(text)

        active = [s for s in self.signals if s.matches(text)]

        answers: dict[str, Any] = {}
        for qid, question in request.questions.items():
            relevant = [s for s in active if s.question_id in (None, qid)]
            answers[qid] = self._answer(question, relevant)

        return {
            "model": self.model or request.model,
            "answers": answers,
            "usage": {
                "input_tokens": estimate_tokens(request.wire()),
                "output_tokens": 0,
            },
        }

    def _answer(self, question: Any, signals: Sequence[Signal]) -> dict[str, Any]:
        if isinstance(question, Noul):
            logit = self.bias + sum(
                (-s.weight if s.label == "false" else s.weight) for s in signals
            )
            return {"noul": _round(_sigmoid(logit))}

        if isinstance(question, Choice):
            labels = list(question.options)
        elif isinstance(question, Score):
            labels = [str(i) for i in range(question.levels)]
        else:  # pragma: no cover - guarded by the type union
            raise TypeError(f"unsupported question type {type(question).__name__}")

        logits = {label: self.bias for label in labels}
        for s in signals:
            target = s.label if s.label in logits else labels[0]
            logits[target] += s.weight

        probabilities = _softmax(logits)

        if isinstance(question, Choice):
            best = max(probabilities, key=lambda k: probabilities[k])
            return {
                "choice": best,
                "probabilities": probabilities,
                "confidence": _round(_confidence(probabilities)),
            }

        expectation = sum(float(label) * p for label, p in probabilities.items())
        return {
            "score": _round(expectation),
            "legend": {str(i): text for i, text in enumerate(question.criteria)},
            "probabilities": probabilities,
            "confidence": _round(_confidence(probabilities)),
        }

    async def aclose(self) -> None:
        return None


def _flatten(state: Any) -> str:
    """Render any accepted state shape as the text the model would read."""
    if isinstance(state, str):
        return state
    try:
        return json.dumps(state, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(state)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _softmax(logits: Mapping[str, float]) -> dict[str, float]:
    peak = max(logits.values())
    exps = {k: math.exp(v - peak) for k, v in logits.items()}
    total = sum(exps.values())
    raw = {k: v / total for k, v in exps.items()}

    # Round, then push the rounding residue onto the largest entry so the
    # distribution still sums to 1.0 within the parser's tolerance.
    rounded = {k: _round(v) for k, v in raw.items()}
    residue = 1.0 - sum(rounded.values())
    if abs(residue) > 0:
        largest = max(rounded, key=lambda k: rounded[k])
        rounded[largest] = _round(rounded[largest] + residue)
    return rounded


def _confidence(probabilities: Mapping[str, float]) -> float:
    """One minus normalised entropy: 1.0 for a point mass, 0.0 for uniform."""
    n = len(probabilities)
    if n <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probabilities.values() if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(n)))


def _round(value: float) -> float:
    return round(value + 0.0, 6)
