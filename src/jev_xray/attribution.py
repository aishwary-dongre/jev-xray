"""Occlusion attribution: which part of the state produced the answer.

The method is old and simple. Ask the question against the full state to get a
baseline. Then, for each segment, ask the *identical* question against a state
with that segment removed, and measure how far the tracked scalar moved. A
positive delta means the segment was holding the answer up.

What is new is that this is affordable. Post-hoc attribution has always needed
tens to hundreds of forward passes per explanation, which is why nobody runs it
against a frontier model in production. Against a model priced at $0.042 per
million input tokens with free output, a forty-segment explanation costs a
fraction of a cent, and the fixed answer space means every ablation is read on
the same axis with no parsing and no format drift.

Leave-one-out is the cheap estimator and it has a known blind spot: it cannot
see evidence that only counts jointly, and it reads redundant evidence as
worthless because removing either copy leaves the other carrying the decision.
``Attribution.interaction_residual`` measures how badly that blind spot applies
to a given explanation, which is the signal for reaching for the more expensive
estimator.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Sequence

from .budget import (
    PRICE_USD_PER_INPUT_TOKEN,
    Budget,
    BudgetExceeded,
    Ledger,
    estimate_tokens,
)
from .client import Client
from .segment import AblationMode, DEFAULT_MASK, Segment, Segmenter, auto_segmenter, get_segmenter
from .types import Answer, Question, State, SystemOneRequest, Target

__all__ = ["SegmentEffect", "Attribution", "leave_one_out"]


@dataclass(frozen=True, slots=True)
class SegmentEffect:
    """What happened to the answer when one segment was taken away."""

    segment: Segment
    ablated_value: float
    delta: float

    @property
    def supports(self) -> bool:
        """True when removing this segment pushed the answer away from the baseline."""
        return self.delta > 0

    @property
    def magnitude(self) -> float:
        return abs(self.delta)


@dataclass(slots=True)
class Attribution:
    """A completed evidence map for one question against one state."""

    question_id: str
    question: Question
    target: Target
    model: str
    mode: AblationMode
    state: State
    segments: list[Segment]
    baseline_answer: Answer
    baseline_value: float
    effects: list[SegmentEffect]
    ledger: Ledger
    empty_value: float | None = None
    failures: list[tuple[Segment, str]] = field(default_factory=list)

    # -- ordering helpers ---------------------------------------------------

    def ranked(self) -> list[SegmentEffect]:
        """Effects ordered by absolute influence, strongest first."""
        return sorted(self.effects, key=lambda e: e.magnitude, reverse=True)

    def top(self, n: int = 5) -> list[SegmentEffect]:
        return self.ranked()[:n]

    def supporting(self, min_delta: float = 0.0) -> list[SegmentEffect]:
        return [e for e in sorted(self.effects, key=lambda e: e.delta, reverse=True) if e.delta > min_delta]

    def opposing(self, min_delta: float = 0.0) -> list[SegmentEffect]:
        return [e for e in sorted(self.effects, key=lambda e: e.delta) if e.delta < -min_delta]

    def by_id(self, segment_id: int) -> SegmentEffect:
        for effect in self.effects:
            if effect.segment.id == segment_id:
                return effect
        raise KeyError(f"no effect recorded for segment {segment_id}")

    # -- diagnostics --------------------------------------------------------

    @property
    def total_effect(self) -> float:
        """Sum of the individual deltas."""
        return sum(e.delta for e in self.effects)

    @property
    def total_swing(self) -> float | None:
        """How far the answer moves between the full state and an empty one.

        The denominator for "how much of this decision did we account for".
        ``None`` when the empty-state reference was not requested.
        """
        if self.empty_value is None:
            return None
        return self.baseline_value - self.empty_value

    @property
    def interaction_residual(self) -> float | None:
        """Total swing minus the sum of individual effects.

        Near zero means the segments act independently and leave-one-out is an
        adequate account of this decision. Large in magnitude means they do not:

        * strongly positive residual — evidence is *redundant*. Removing any one
          segment changes little because others still carry the decision, so
          leave-one-out under-reports every one of them.
        * strongly negative residual — the individual deltas over-count, which
          happens when segments reinforce each other and each looks pivotal on
          its own.

        Either way the ranking is still informative but the magnitudes are not
        additive, and a Shapley-style estimator over random subsets is the
        honest next step.
        """
        swing = self.total_swing
        if swing is None:
            return None
        return swing - self.total_effect

    def is_additive(self, tolerance: float = 0.05) -> bool:
        """Whether leave-one-out is an adequate account of this decision."""
        residual = self.interaction_residual
        return residual is not None and abs(residual) <= tolerance

    def summary(self) -> str:
        lines = [
            f"question {self.question_id!r} on {self.model}",
            f"  target        {self.target.describe()} = {self.baseline_value:.4f}",
            f"  segments      {len(self.segments)} ({self.mode} ablation)",
        ]
        if self.empty_value is not None:
            lines.append(
                f"  empty state   {self.empty_value:.4f} "
                f"(total swing {self.total_swing:+.4f})"
            )
        residual = self.interaction_residual
        if residual is not None:
            verdict = "additive" if abs(residual) <= 0.05 else "interacting"
            lines.append(f"  residual      {residual:+.4f} ({verdict})")
        if self.failures:
            lines.append(f"  failed        {len(self.failures)} ablations")
        lines.append(f"  spent         {self.ledger.summary()}")
        return "\n".join(lines)


async def leave_one_out(
    client: Client,
    state: State,
    question: Question,
    *,
    question_id: str = "q",
    segmenter: str | Segmenter = "auto",
    target: Target | None = None,
    mode: AblationMode = "delete",
    budget: Budget | None = None,
    concurrency: int = 8,
    include_empty: bool = True,
    ledger: Ledger | None = None,
) -> Attribution:
    """Attribute one answer across the segments of its state.

    Issues ``len(segments) + 1`` requests, plus one more when ``include_empty``.
    They run concurrently, bounded by ``concurrency`` and by the client's rate
    limiter, so wall time is close to a single call's latency for a small state.

    A failed ablation is recorded in ``Attribution.failures`` rather than
    aborting the run: losing one segment's delta degrades the map, while losing
    the whole explanation to one transient 503 wastes everything already spent.
    """
    budget = budget if budget is not None else Budget()
    ledger = ledger if ledger is not None else Ledger()

    chosen = auto_segmenter(state) if segmenter == "auto" else get_segmenter(segmenter)
    segments = chosen.split(state)
    if len(segments) < 2:
        raise ValueError(
            f"{type(chosen).__name__} found {len(segments)} segment(s); attribution "
            "needs at least two. Try a finer segmenter."
        )

    questions = {question_id: question}
    planned = len(segments) + 1 + (1 if include_empty else 0)
    _preflight(state, questions, planned, budget, len(segments))

    baseline = await client.ask(state, questions, ledger=ledger, budget=budget)
    baseline_answer = baseline.answers[question_id]
    target = target if target is not None else Target.baseline(baseline_answer)
    baseline_value = target.read(baseline_answer)

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def evaluate(drop: Sequence[int]) -> float:
        async with semaphore:
            ablated = chosen.ablate(state, drop, mode=mode)
            if isinstance(ablated, str) and not ablated.strip():
                # An empty state is not a meaningful request and may be rejected
                # outright; the neutral placeholder keeps the axis comparable.
                ablated = DEFAULT_MASK
            response = await client.ask(ablated, questions, ledger=ledger, budget=budget)
            return target.read(response.answers[question_id])

    results = await asyncio.gather(
        *(evaluate([segment.id]) for segment in segments),
        return_exceptions=True,
    )

    effects: list[SegmentEffect] = []
    failures: list[tuple[Segment, str]] = []
    for segment, result in zip(segments, results):
        if isinstance(result, BaseException):
            if isinstance(result, (BudgetExceeded, asyncio.CancelledError)):
                failures.append((segment, type(result).__name__))
                continue
            failures.append((segment, f"{type(result).__name__}: {result}"))
            continue
        effects.append(
            SegmentEffect(
                segment=segment,
                ablated_value=result,
                delta=baseline_value - result,
            )
        )

    empty_value: float | None = None
    if include_empty:
        try:
            empty_value = await evaluate([s.id for s in segments])
        except Exception:  # noqa: BLE001 - a diagnostic reference, never fatal
            empty_value = None

    ledger.finish()

    return Attribution(
        question_id=question_id,
        question=question,
        target=target,
        model=client.model,
        mode=mode,
        state=state,
        segments=segments,
        baseline_answer=baseline_answer,
        baseline_value=baseline_value,
        effects=effects,
        ledger=ledger,
        empty_value=empty_value,
        failures=failures,
    )


def _preflight(
    state: State,
    questions: dict,
    planned_requests: int,
    budget: Budget,
    segment_count: int,
) -> None:
    """Refuse a plan that obviously cannot fit the budget, before spending anything.

    Uses an estimated token count, so it is a guard rail rather than an exact
    forecast. The error names the knob that fixes it.
    """
    if budget.max_requests is not None and planned_requests > budget.max_requests:
        raise BudgetExceeded(
            f"{segment_count} segments need ~{planned_requests} requests but the "
            f"budget allows {budget.max_requests}. Raise max_requests, or use a "
            f"coarser segmenter and drill into the segments that matter."
        )

    per_request = estimate_tokens(
        SystemOneRequest(model="x", state=state, questions=questions).wire()
    )
    projected = per_request * planned_requests

    if budget.max_input_tokens is not None and projected > budget.max_input_tokens:
        raise BudgetExceeded(
            f"~{projected:,} estimated input tokens exceeds the "
            f"{budget.max_input_tokens:,} token budget."
        )

    projected_usd = projected * PRICE_USD_PER_INPUT_TOKEN
    if budget.max_usd is not None and projected_usd > budget.max_usd:
        raise BudgetExceeded(
            f"~${projected_usd:.6f} estimated for {planned_requests} requests over a "
            f"{per_request:,}-token state exceeds the ${budget.max_usd:.6f} budget."
        )
