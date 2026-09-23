"""Evaluating the answer for an arbitrary subset of the evidence.

Every estimator in this package reduces to the same primitive question: *what
does the model answer when only these segments are present?* Leave-one-out asks
it n+2 times. Shapley asks it for many subsets. The minimal-evidence searches
walk it greedily. So it lives in one place.

Coalitions are identified by the set of segments to **keep**, not to drop. That
makes the two anchors read naturally — the full state is every id, the empty
state is the empty set — and it means a subset's identity does not change when
the segmenter finds a different number of segments.

Memoisation matters more here than it looks. Exact Shapley over six segments
touches 64 distinct subsets, but the permutations that generate them revisit the
same subset many times over. The client caches on the request body too; this
layer just avoids building the request at all.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .budget import Budget, BudgetExceeded, Ledger
from .client import Client
from .segment import AblationMode, DEFAULT_MASK, Segment, Segmenter
from .types import Answer, Question, State, Target

__all__ = ["CoalitionEvaluator", "CoalitionFailure"]


class CoalitionFailure(RuntimeError):
    """One coalition could not be evaluated."""


@dataclass(slots=True)
class CoalitionEvaluator:
    """Reads the tracked scalar for subsets of the evidence.

    Construct via :meth:`prepare`, which establishes the baseline and derives the
    target from it, since a target that is not anchored to the baseline answer
    cannot be compared across subsets.
    """

    client: Client
    state: State
    question: Question
    question_id: str
    segmenter: Segmenter
    segments: Sequence[Segment]
    target: Target
    mode: AblationMode
    budget: Budget
    ledger: Ledger
    baseline_answer: Answer
    baseline_value: float
    concurrency: int = 8

    _cache: dict[frozenset[int], float] = field(default_factory=dict, init=False)
    _failures: dict[frozenset[int], str] = field(default_factory=dict, init=False)
    _semaphore: asyncio.Semaphore | None = field(default=None, init=False)

    # -- construction -------------------------------------------------------

    @classmethod
    async def prepare(
        cls,
        client: Client,
        state: State,
        question: Question,
        *,
        question_id: str,
        segmenter: Segmenter,
        segments: Sequence[Segment],
        target: Target | None,
        mode: AblationMode,
        budget: Budget,
        ledger: Ledger,
        concurrency: int = 8,
    ) -> "CoalitionEvaluator":
        questions = {question_id: question}
        baseline = await client.ask(state, questions, ledger=ledger, budget=budget)
        answer = baseline.answers[question_id]
        resolved = target if target is not None else Target.baseline(answer)

        evaluator = cls(
            client=client,
            state=state,
            question=question,
            question_id=question_id,
            segmenter=segmenter,
            segments=list(segments),
            target=resolved,
            mode=mode,
            budget=budget,
            ledger=ledger,
            baseline_answer=answer,
            baseline_value=resolved.read(answer),
            concurrency=concurrency,
        )
        # The full coalition is the baseline; seed it so nothing re-requests it.
        evaluator._cache[evaluator.everything] = evaluator.baseline_value
        return evaluator

    # -- anchors ------------------------------------------------------------

    @property
    def everything(self) -> frozenset[int]:
        return frozenset(s.id for s in self.segments)

    @property
    def nothing(self) -> frozenset[int]:
        return frozenset()

    @property
    def evaluations(self) -> int:
        """Distinct coalitions whose value is known."""
        return len(self._cache)

    @property
    def failures(self) -> Mapping[frozenset[int], str]:
        return dict(self._failures)

    def known(self, keep: frozenset[int]) -> float | None:
        return self._cache.get(keep)

    # -- evaluation ---------------------------------------------------------

    async def value(self, keep: frozenset[int]) -> float:
        cached = self._cache.get(keep)
        if cached is not None:
            return cached
        if keep in self._failures:
            raise CoalitionFailure(self._failures[keep])

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self.concurrency))

        async with self._semaphore:
            # Re-check: a concurrent caller may have filled it while we waited.
            cached = self._cache.get(keep)
            if cached is not None:
                return cached

            drop = [s.id for s in self.segments if s.id not in keep]
            ablated = self.segmenter.ablate(self.state, drop, mode=self.mode)
            if isinstance(ablated, str) and not ablated.strip():
                # An empty state is not a meaningful request and some hosts reject
                # it outright. The neutral placeholder keeps the axis comparable.
                ablated = DEFAULT_MASK

            response = await self.client.ask(
                ablated,
                {self.question_id: self.question},
                ledger=self.ledger,
                budget=self.budget,
            )
            result = self.target.read(response.answers[self.question_id])

        self._cache[keep] = result
        return result

    async def values(
        self, coalitions: Iterable[frozenset[int]], *, strict: bool = False
    ) -> dict[frozenset[int], float]:
        """Evaluate many coalitions concurrently, deduplicated.

        Failures are recorded and omitted from the result rather than aborting the
        batch, unless ``strict``. A partial map degrades an estimate; a raised
        exception throws away everything already paid for. The exception is a
        budget breach, which is always fatal: continuing would mean spending past
        a ceiling the caller set deliberately.
        """
        wanted = {c for c in coalitions if c not in self._cache}
        if not wanted:
            return dict(self._cache)

        ordered = list(wanted)
        results = await asyncio.gather(
            *(self.value(c) for c in ordered), return_exceptions=True
        )

        for coalition, result in zip(ordered, results):
            if not isinstance(result, BaseException):
                continue
            if isinstance(result, BudgetExceeded):
                raise result
            if isinstance(result, asyncio.CancelledError):
                raise result
            self._failures[coalition] = f"{type(result).__name__}: {result}"
            if strict:
                raise CoalitionFailure(self._failures[coalition]) from result

        return dict(self._cache)

    def segment_by_id(self, segment_id: int) -> Segment:
        for segment in self.segments:
            if segment.id == segment_id:
                return segment
        raise KeyError(f"no segment {segment_id}")
