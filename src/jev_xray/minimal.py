"""The smallest answers to "why" and "why not".

An attribution map ranks every segment. That is the right output for diagnosing a
question, and the wrong one for explaining a decision to a person. Nobody wants
six numbers; they want the sentence that did it.

Two searches, both on top of an attribution ordering:

**Minimal sufficient evidence** — the smallest set of segments that reproduces
the decision on its own. This is the closest thing to a quotable *because*: three
sentences out of forty, and the answer holds without the rest.

**Minimal flipping set** — the smallest set whose removal changes the decision.
This is the counterfactual, and it is usually what an auditor, a support agent or
an angry customer actually asks for: what would have had to be different?

Both are greedy, guided by the attribution order, then pruned. Greedy because the
exact versions are subset-search problems and the attribution ordering is a good
enough guide to make greedy land close. Pruned because greedy overshoots: it adds
in importance order, and a later addition can make an earlier one redundant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .coalition import CoalitionEvaluator
from .segment import Segment

__all__ = [
    "Decision",
    "SufficientEvidence",
    "FlippingSet",
    "minimal_sufficient",
    "minimal_flip",
]


@dataclass(frozen=True, slots=True)
class Decision:
    """What counts as "the same decision".

    The tracked scalar is a probability in every default target, so a threshold is
    the natural predicate. Set it to the threshold your application actually acts
    on — the explanation is only meaningful relative to the decision you make.
    """

    threshold: float = 0.5

    def holds(self, value: float) -> bool:
        return value >= self.threshold

    def describe(self) -> str:
        return f"value >= {self.threshold:g}"


@dataclass(slots=True)
class SufficientEvidence:
    """The smallest subset found that reproduces the baseline answer."""

    segments: list[Segment]
    value: float
    baseline_value: float
    epsilon: float
    evaluations: int
    total: int = 0
    found: bool = True

    @property
    def size(self) -> int:
        return len(self.segments)

    @property
    def gap(self) -> float:
        return abs(self.baseline_value - self.value)

    def quote(self, separator: str = " ") -> str:
        """The evidence as text, in the order it appeared in the state."""
        ordered = sorted(self.segments, key=lambda s: (s.start or 0, s.id))
        return separator.join(s.text.strip() for s in ordered)

    def summary(self) -> str:
        if not self.found:
            return (
                f"  no sufficient subset within {self.epsilon:g} of "
                f"{self.baseline_value:.4f}; the answer needs most of the state"
            )
        return (
            f"  {self.size} of {self.total} segment(s) reproduce the answer: "
            f"{self.value:.4f} against {self.baseline_value:.4f} "
            f"(gap {self.gap:.4f}, within {self.epsilon:g})"
        )


@dataclass(slots=True)
class FlippingSet:
    """The smallest subset found whose removal changes the decision."""

    segments: list[Segment]
    value: float
    baseline_value: float
    decision: Decision
    evaluations: int
    found: bool = True
    exhaustive: bool = False

    @property
    def size(self) -> int:
        return len(self.segments)

    def quote(self, separator: str = " ") -> str:
        ordered = sorted(self.segments, key=lambda s: (s.start or 0, s.id))
        return separator.join(s.text.strip() for s in ordered)

    def summary(self) -> str:
        if not self.found:
            scope = "no subset" if self.exhaustive else "no subset along the greedy path"
            return (
                f"  {scope} flips {self.decision.describe()}; "
                f"the decision is robust to removing evidence"
            )
        return (
            f"  removing {self.size} segment(s) moves the answer to {self.value:.4f}, "
            f"crossing {self.decision.describe()}"
        )


# --------------------------------------------------------------------------


async def minimal_sufficient(
    evaluator: CoalitionEvaluator,
    order: Sequence[int],
    *,
    epsilon: float = 0.05,
    prune: bool = True,
) -> SufficientEvidence:
    """Smallest subset of segments whose answer is within ``epsilon`` of the baseline.

    ``order`` is segment ids by descending importance, from either estimator. Adds
    in that order until the answer is close enough, then tries dropping each
    member back out, cheapest-looking first.
    """
    baseline = evaluator.baseline_value
    start = evaluator.evaluations

    keep: set[int] = set()
    value = await evaluator.value(frozenset())

    for segment_id in order:
        if abs(baseline - value) <= epsilon:
            break
        keep.add(segment_id)
        value = await evaluator.value(frozenset(keep))

    if abs(baseline - value) > epsilon:
        return SufficientEvidence(
            segments=[evaluator.segment_by_id(i) for i in keep],
            value=value,
            baseline_value=baseline,
            epsilon=epsilon,
            evaluations=evaluator.evaluations - start,
            total=len(evaluator.segments),
            found=False,
        )

    if prune:
        # Reverse importance: the least important member is the likeliest to have
        # been made redundant by something added after it.
        for segment_id in reversed([i for i in order if i in keep]):
            candidate = keep - {segment_id}
            if not candidate:
                continue
            trial = await evaluator.value(frozenset(candidate))
            if abs(baseline - trial) <= epsilon:
                keep = candidate
                value = trial

    return SufficientEvidence(
        segments=[evaluator.segment_by_id(i) for i in keep],
        value=value,
        baseline_value=baseline,
        epsilon=epsilon,
        evaluations=evaluator.evaluations - start,
        total=len(evaluator.segments),
    )


async def minimal_flip(
    evaluator: CoalitionEvaluator,
    order: Sequence[int],
    *,
    decision: Decision | None = None,
    prune: bool = True,
    singles_first: bool = True,
) -> FlippingSet:
    """Smallest set of segments whose removal changes the decision.

    Tries every single segment first when ``singles_first``, because a one-segment
    counterfactual is the most useful result and the attribution order does not
    always surface it: a segment can be individually decisive while ranking below
    one that only matters in combination.
    """
    decision = decision if decision is not None else Decision()
    baseline = evaluator.baseline_value
    everything = evaluator.everything
    start = evaluator.evaluations
    originally = decision.holds(baseline)

    if singles_first:
        # Batched, and usually free: an attribution run has already evaluated
        # every leave-one-out coalition, and exact Shapley has evaluated all of
        # them.
        await evaluator.values([everything - {i} for i in order])
        singles = [
            (i, value)
            for i in order
            if (value := evaluator.known(everything - {i})) is not None
        ]
        flipped = [(i, v) for i, v in singles if decision.holds(v) != originally]
        if flipped:
            # The one that clears the threshold by the widest margin is the most
            # convincing counterfactual, not merely the first one found.
            segment_id, value = max(
                flipped, key=lambda pair: abs(pair[1] - decision.threshold)
            )
            return FlippingSet(
                segments=[evaluator.segment_by_id(segment_id)],
                value=value,
                baseline_value=baseline,
                decision=decision,
                evaluations=evaluator.evaluations - start,
                exhaustive=True,
            )

    removed: set[int] = set()
    value = baseline
    for segment_id in order:
        removed.add(segment_id)
        value = await evaluator.value(everything - removed)
        if decision.holds(value) != originally:
            break

    if decision.holds(value) == originally:
        return FlippingSet(
            segments=[evaluator.segment_by_id(i) for i in removed],
            value=value,
            baseline_value=baseline,
            decision=decision,
            evaluations=evaluator.evaluations - start,
            found=False,
        )

    if prune:
        for segment_id in reversed([i for i in order if i in removed]):
            candidate = removed - {segment_id}
            if not candidate:
                continue
            trial = await evaluator.value(everything - candidate)
            if decision.holds(trial) != originally:
                removed = candidate
                value = trial

    return FlippingSet(
        segments=[evaluator.segment_by_id(i) for i in removed],
        value=value,
        baseline_value=baseline,
        decision=decision,
        evaluations=evaluator.evaluations - start,
    )
