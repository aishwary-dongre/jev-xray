"""Is this question stable enough to hang a threshold on?

Attribution explains one answer. It cannot tell you whether the *question* is
sound. Those are different failures and they need different probes.

A question can be unsound in ways that never show up in any single answer:

* the answer barely depends on the input, so every threshold fires on everything
* appending irrelevant text moves it, so the answer depends on what else happens
  to be in the state
* reordering the same evidence moves it, so position is acting as evidence
* rewording the question moves it, so you measured your phrasing, not the world
* a Choice answer depends on the order you listed the options in

Every probe here is **label-free**. It needs no ground truth, no annotated set,
no idea of the right answer. It only needs perturbations that *should not* change
the answer, and it measures whether they did. That is what makes it runnable on
day one against your own data, which a labelled evaluation is not.

Each probe returns a verdict alongside its number. The thresholds behind those
verdicts are heuristics, documented on each probe and overridable — they are a
starting point for judgement, not a standard.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Sequence

from .budget import Budget, BudgetExceeded, Ledger
from .client import Client
from .minimal import Decision
from .segment import (
    DEFAULT_MASK,
    AblationMode,
    Segment,
    Segmenter,
    auto_segmenter,
    get_segmenter,
    reassemble,
)
from .types import Answer, Question, State, Target

__all__ = [
    "Verdict",
    "ProbeResult",
    "ProbeContext",
    "DISTRACTOR",
    "prior_saturation",
    "distractor_drift",
    "order_sensitivity",
]

Verdict = Literal["ok", "warn", "fail"]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """One stability measurement and what to make of it."""

    name: str
    verdict: Verdict
    detail: str
    measurement: float | None = None
    requests: int = 0
    skipped_reason: str | None = None

    @property
    def skipped(self) -> bool:
        return self.skipped_reason is not None

    @property
    def symbol(self) -> str:
        if self.skipped:
            return "-"
        return {"ok": "ok", "warn": "!!", "fail": "XX"}[self.verdict]

    def line(self) -> str:
        if self.skipped:
            return f"  --  {self.name}: skipped, {self.skipped_reason}"
        measured = "" if self.measurement is None else f" [{self.measurement:+.4f}]"
        return f"  {self.symbol}  {self.name}{measured}\n      {self.detail}"


@dataclass(slots=True)
class ProbeContext:
    """Everything a probe needs to perturb one question and read the result.

    Probes go through :meth:`ask`, which keeps the target fixed. Reading every
    perturbation off the same axis as the baseline is the whole basis for
    comparing them.
    """

    client: Client
    state: State
    question: Question
    question_id: str
    target: Target
    baseline_answer: Answer
    baseline_value: float
    segmenter: Segmenter
    segments: list[Segment]
    budget: Budget
    ledger: Ledger
    decision: Decision
    mode: AblationMode = "delete"
    concurrency: int = 8
    seed: int = 0

    _requests_at_start: int = field(default=0, init=False)

    async def ask(
        self, *, state: State | None = None, question: Question | None = None
    ) -> float:
        """Evaluate a perturbation and read the tracked scalar."""
        response = await self.client.ask(
            self.state if state is None else state,
            {self.question_id: self.question if question is None else question},
            ledger=self.ledger,
            budget=self.budget,
        )
        return self.target.read(response.answers[self.question_id])

    def mark(self) -> None:
        self._requests_at_start = self.ledger.requests

    def spent(self) -> int:
        return self.ledger.requests - self._requests_at_start

    def rng(self, salt: int = 0) -> random.Random:
        """Seeded randomness, so a stability report is reproducible."""
        return random.Random(self.seed + salt)


Probe = Callable[[ProbeContext], Awaitable[ProbeResult]]


# --------------------------------------------------------------------------
# state-level probes: none of these require generating text
# --------------------------------------------------------------------------


async def prior_saturation(ctx: ProbeContext) -> ProbeResult:
    """Does the answer depend on the input at all?

    Asks the question against an empty state. If that lands on the same side of
    the decision boundary as the full state, the input never changes the outcome
    and the threshold is decoration. This is the single most valuable check in the
    file, and it costs one request.

    fail
        the empty state decides the same way as the full state
    warn
        the decision does change, but the whole swing is under 0.10, so there is
        very little room between "no evidence" and "all the evidence"
    """
    ctx.mark()
    stripped = ctx.segmenter.ablate(
        ctx.state, [s.id for s in ctx.segments], mode=ctx.mode
    )
    if isinstance(stripped, str) and not stripped.strip():
        # An empty string is not a meaningful request and some hosts reject it.
        stripped = DEFAULT_MASK
    empty = await ctx.ask(state=stripped)

    swing = ctx.baseline_value - empty
    same_side = ctx.decision.holds(empty) == ctx.decision.holds(ctx.baseline_value)

    if same_side:
        verdict: Verdict = "fail"
        detail = (
            f"an empty state answers {empty:.4f}, the same side of "
            f"{ctx.decision.describe()} as the full state ({ctx.baseline_value:.4f}). "
            f"the input cannot change this decision, so the threshold is decoration"
        )
    elif abs(swing) < 0.10:
        verdict = "warn"
        detail = (
            f"an empty state answers {empty:.4f} against {ctx.baseline_value:.4f}. "
            f"the decision does flip, but the entire usable range is {abs(swing):.4f}"
        )
    else:
        verdict = "ok"
        detail = (
            f"an empty state answers {empty:.4f} against {ctx.baseline_value:.4f}, "
            f"a usable range of {abs(swing):.4f}"
        )

    return ProbeResult(
        name="prior saturation",
        verdict=verdict,
        detail=detail,
        measurement=swing,
        requests=ctx.spent(),
    )


# Deliberately bland, on-topic-sounding but decision-irrelevant filler. Short, so
# the probe stays cheap, and free of anything that reads as evidence either way.
DISTRACTOR = (
    "For reference, our office hours are Monday to Friday. "
    "Tickets are logged automatically on receipt. "
    "This account is on the standard plan and the billing cycle renews monthly."
)


async def distractor_drift(
    ctx: ProbeContext, *, distractor: str = DISTRACTOR
) -> ProbeResult:
    """Does appending irrelevant text move the answer?

    TypeSafe documents that accuracy falls as a state grows with content unrelated
    to the decision. This measures it for your question rather than in general:
    append filler that carries no bearing on the judgment and see what moves.

    Two requests, because the filler is tried at both ends. Position matters
    independently of content, and a probe that only appends would miss it.

    fail
        either placement moves the answer by more than 0.10
    warn
        more than 0.05
    """
    ctx.mark()
    if not isinstance(ctx.state, str):
        return ProbeResult(
            name="distractor drift",
            verdict="ok",
            detail="",
            skipped_reason="structured state; append a field instead",
            requests=0,
        )

    appended = await ctx.ask(state=f"{ctx.state}\n{distractor}")
    prepended = await ctx.ask(state=f"{distractor}\n{ctx.state}")

    shifts = {"appended": appended - ctx.baseline_value, "prepended": prepended - ctx.baseline_value}
    worst_name = max(shifts, key=lambda k: abs(shifts[k]))
    worst = shifts[worst_name]

    if abs(worst) > 0.10:
        verdict: Verdict = "fail"
    elif abs(worst) > 0.05:
        verdict = "warn"
    else:
        verdict = "ok"

    detail = (
        f"filler {worst_name} to the state moved the answer by {worst:+.4f} "
        f"(appended {shifts['appended']:+.4f}, prepended {shifts['prepended']:+.4f}). "
        "text with no bearing on the judgment should move it by nothing"
    )
    if verdict != "ok":
        detail += "; filter the state in code before sending it"

    return ProbeResult(
        name="distractor drift",
        verdict=verdict,
        detail=detail,
        measurement=worst,
        requests=ctx.spent(),
    )


async def order_sensitivity(ctx: ProbeContext, *, shuffles: int = 4) -> ProbeResult:
    """Does the same evidence in a different order give a different answer?

    Position should not be evidence. If it is, then two customers who said the
    same things in a different sequence get different decisions, which is
    difficult to defend and impossible to debug from the answer alone.

    fail
        the spread across orderings exceeds 0.10
    warn
        exceeds 0.05
    """
    ctx.mark()
    if getattr(ctx.segmenter, "joiner", None) is None:
        return ProbeResult(
            name="order sensitivity",
            verdict="ok",
            detail="",
            skipped_reason="structured state has no linear order to permute",
            requests=0,
        )
    if len(ctx.segments) < 3:
        return ProbeResult(
            name="order sensitivity",
            verdict="ok",
            detail="",
            skipped_reason="needs at least three segments to reorder meaningfully",
            requests=0,
        )

    ids = [s.id for s in ctx.segments]
    rng = ctx.rng(salt=17)

    # The identity ordering, reassembled the same way as the shuffles, so the
    # comparison isolates order from the reassembly itself. Joining segments
    # normalises whitespace, and that alone can move an answer.
    identity = await ctx.ask(state=reassemble(ctx.segmenter, ctx.segments, ids))

    values = [identity]
    seen = {tuple(ids)}
    for _ in range(shuffles):
        order = list(ids)
        for _attempt in range(8):
            rng.shuffle(order)
            if tuple(order) not in seen:
                break
        seen.add(tuple(order))
        values.append(await ctx.ask(state=reassemble(ctx.segmenter, ctx.segments, order)))

    spread = max(values) - min(values)

    if spread > 0.10:
        verdict: Verdict = "fail"
    elif spread > 0.05:
        verdict = "warn"
    else:
        verdict = "ok"

    detail = (
        f"across {len(values)} orderings of the same evidence the answer ranged "
        f"{min(values):.4f} to {max(values):.4f}, a spread of {spread:.4f}"
    )
    if verdict != "ok":
        detail += "; position is acting as evidence"

    return ProbeResult(
        name="order sensitivity",
        verdict=verdict,
        detail=detail,
        measurement=spread,
        requests=ctx.spent(),
    )


# --------------------------------------------------------------------------


async def build_context(
    client: Client,
    state: State,
    question: Question,
    *,
    question_id: str = "q",
    segmenter: str | Segmenter = "auto",
    target: Target | None = None,
    mode: AblationMode = "delete",
    budget: Budget | None = None,
    decision: Decision | None = None,
    concurrency: int = 8,
    seed: int = 0,
    ledger: Ledger | None = None,
) -> ProbeContext:
    """Establish the baseline and the axis every probe measures against."""
    budget = budget if budget is not None else Budget()
    ledger = ledger if ledger is not None else Ledger()
    chosen = auto_segmenter(state) if segmenter == "auto" else get_segmenter(segmenter)
    segments = chosen.split(state)

    baseline = await client.ask(
        state, {question_id: question}, ledger=ledger, budget=budget
    )
    answer = baseline.answers[question_id]
    resolved = target if target is not None else Target.baseline(answer)

    return ProbeContext(
        client=client,
        state=state,
        question=question,
        question_id=question_id,
        target=resolved,
        baseline_answer=answer,
        baseline_value=resolved.read(answer),
        segmenter=chosen,
        segments=segments,
        budget=budget,
        ledger=ledger,
        decision=decision if decision is not None else Decision(),
        mode=mode,
        concurrency=concurrency,
        seed=seed,
    )


async def run_probes(
    ctx: ProbeContext, probes: Sequence[Probe]
) -> list[ProbeResult]:
    """Run probes in sequence, surviving individual failures.

    Sequential on purpose: probes are independent but a stability run is already
    the most request-hungry thing in the package, and hammering an endpoint to
    measure its consistency is a poor way to measure its consistency.
    """
    results: list[ProbeResult] = []
    for probe in probes:
        try:
            results.append(await probe(ctx))
        except BudgetExceeded as exc:
            results.append(
                ProbeResult(
                    name=getattr(probe, "__name__", "probe"),
                    verdict="warn",
                    detail="",
                    skipped_reason=f"budget exhausted: {exc}",
                )
            )
            break
        except Exception as exc:  # noqa: BLE001 - one probe failing is not fatal
            results.append(
                ProbeResult(
                    name=getattr(probe, "__name__", "probe"),
                    verdict="warn",
                    detail="",
                    skipped_reason=f"{type(exc).__name__}: {exc}",
                )
            )
    return results
