"""Shapley attribution: the fix for evidence that does not act alone.

Leave-one-out asks one question per segment: what happens without you? That is
cheap and it is wrong whenever segments interact.

* **Redundant evidence.** The same signal appears twice. Remove either copy and
  the other still carries the decision, so leave-one-out scores both at zero
  while the pair is clearly doing all the work.
* **Complementary evidence.** Two segments matter only together. Individually
  neither moves the answer, so both look inert.

Shapley asks instead: across every possible context this segment could arrive
in, how much does it add on average? That is the only attribution satisfying a
property we actually want here — **efficiency**, meaning the values sum exactly
to the full-state answer minus the empty-state answer. The interaction residual
that leave-one-out reports as a warning is, for exact Shapley, zero by
construction. So the gap becomes a correctness check on the computation rather
than a caveat on the result.

The cost is coalitions. Exact Shapley needs every one of the 2^n subsets, which
is 64 requests for six segments and 1,024 for ten. At $0.042 per million input
tokens that is a fraction of a cent, and the request rate, not the money, is
what eventually bites. So: exact when it fits, permutation sampling when it does
not, and the report always says which it used.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

from .budget import (
    PRICE_USD_PER_INPUT_TOKEN,
    Budget,
    BudgetExceeded,
    Ledger,
    estimate_tokens,
)
from .client import Client
from .coalition import CoalitionEvaluator
from .segment import AblationMode, Segment, Segmenter, auto_segmenter, get_segmenter
from .types import Answer, Question, State, SystemOneRequest, Target

__all__ = [
    "ShapleyEffect",
    "ShapleyAttribution",
    "shapley",
    "MAX_EXACT_SEGMENTS",
    "DEFAULT_SAMPLES",
]

# 2**9 = 512 coalitions. Beyond this, exact enumeration starts costing real wall
# time against a 1,200 requests-per-minute ceiling for little accuracy gained
# over a few dozen permutations.
MAX_EXACT_SEGMENTS = 9

DEFAULT_SAMPLES = 32


@dataclass(frozen=True, slots=True)
class ShapleyEffect:
    """One segment's average marginal contribution."""

    segment: Segment
    value: float
    samples: int
    std_error: float | None = None

    @property
    def supports(self) -> bool:
        return self.value > 0

    @property
    def magnitude(self) -> float:
        return abs(self.value)

    @property
    def signed(self) -> float:
        """Contribution on a common axis, so renderers work on either estimator."""
        return self.value

    @property
    def ablated_value(self) -> float | None:
        """No single ablated value exists: this is an average over many."""
        return None

    @property
    def is_significant(self) -> bool:
        """Whether the estimate is distinguishable from zero.

        Exact values carry no sampling error, so they are significant whenever
        they are non-trivial. Sampled values need to clear twice their own
        standard error, which is roughly a 95% interval excluding zero.
        """
        if self.std_error is None:
            return self.magnitude > 1e-4
        return self.magnitude > 2 * self.std_error


@dataclass(slots=True)
class ShapleyAttribution:
    """A completed Shapley evidence map for one question against one state."""

    question_id: str
    question: Question
    target: Target
    model: str
    mode: AblationMode
    state: State
    segments: list[Segment]
    baseline_answer: Answer
    baseline_value: float
    empty_value: float
    effects: list[ShapleyEffect]
    exact: bool
    coalitions_evaluated: int
    ledger: Ledger
    permutations: int = 0
    failures: list[tuple[frozenset[int], str]] = field(default_factory=list)

    # -- ordering -----------------------------------------------------------

    def ranked(self) -> list[ShapleyEffect]:
        return sorted(self.effects, key=lambda e: e.magnitude, reverse=True)

    def top(self, n: int = 5) -> list[ShapleyEffect]:
        return self.ranked()[:n]

    def supporting(self) -> list[ShapleyEffect]:
        return [e for e in sorted(self.effects, key=lambda e: e.value, reverse=True) if e.value > 0]

    def opposing(self) -> list[ShapleyEffect]:
        return [e for e in sorted(self.effects, key=lambda e: e.value) if e.value < 0]

    def by_id(self, segment_id: int) -> ShapleyEffect:
        for effect in self.effects:
            if effect.segment.id == segment_id:
                return effect
        raise KeyError(f"no effect recorded for segment {segment_id}")

    # -- diagnostics --------------------------------------------------------

    @property
    def total_effect(self) -> float:
        return sum(e.value for e in self.effects)

    @property
    def total_swing(self) -> float:
        """How far the answer moves between the full state and an empty one."""
        return self.baseline_value - self.empty_value

    @property
    def efficiency_gap(self) -> float:
        """Total swing minus the sum of attributed values.

        Shapley values are *defined* to sum to the total swing, so for an exact
        computation this is zero up to floating point. It is therefore a check on
        the arithmetic, not a property of the model. For a sampled estimate it
        shows how far from convergence the run is.
        """
        return self.total_swing - self.total_effect

    @property
    def interaction_residual(self) -> float:
        """Same quantity leave-one-out reports, under its efficiency name.

        Lets one renderer handle both estimators. For exact Shapley this is zero
        by construction, which is the whole point of using it.
        """
        return self.efficiency_gap

    def is_additive(self, tolerance: float = 0.05) -> bool:
        return abs(self.efficiency_gap) <= tolerance

    @property
    def unexplained_prior(self) -> float:
        """The answer with no evidence at all.

        Attribution can only account for the *swing* a state produces. If the
        empty-state answer is already near the decision threshold, then most of
        the answer is a prior the input never touches, and a threshold on that
        question cannot discriminate however good the evidence map looks.
        """
        return self.empty_value

    def summary(self) -> str:
        method = (
            f"exact over {self.coalitions_evaluated} coalitions"
            if self.exact
            else f"{self.permutations} permutations, {self.coalitions_evaluated} coalitions"
        )
        lines = [
            f"question {self.question_id!r} on {self.model}  [shapley]",
            f"  target        {self.target.describe()} = {self.baseline_value:.4f}",
            f"  empty state   {self.empty_value:.4f} (total swing {self.total_swing:+.4f})",
            f"  segments      {len(self.segments)} ({self.mode} ablation)",
            f"  method        {method}",
            f"  efficiency    {self.efficiency_gap:+.4f} "
            f"({'exact, arithmetic check' if self.exact else 'sampling error'})",
        ]
        if self.failures:
            lines.append(f"  failed        {len(self.failures)} coalitions")
        lines.append(f"  spent         {self.ledger.summary()}")
        return "\n".join(lines)


async def shapley(
    client: Client,
    state: State,
    question: Question,
    *,
    question_id: str = "q",
    segmenter: str | Segmenter = "auto",
    target: Target | None = None,
    mode: AblationMode = "delete",
    budget: Budget | None = None,
    exact: bool | None = None,
    samples: int | None = None,
    seed: int = 0,
    concurrency: int = 8,
    ledger: Ledger | None = None,
    evaluator: CoalitionEvaluator | None = None,
) -> ShapleyAttribution:
    """Attribute one answer across its state by average marginal contribution.

    ``exact=None`` decides for you: exact enumeration when the segment count and
    the budget both allow it, permutation sampling otherwise.

    ``seed`` fixes the permutations, so a sampled run is reproducible. A
    debugging tool that gives a different answer each time is not much use for
    debugging.
    """
    budget = budget if budget is not None else Budget()
    ledger = ledger if ledger is not None else Ledger()

    if evaluator is not None:
        # Reusing an evaluator means reusing its cache. The minimal-evidence
        # searches and a Shapley run overlap heavily, so sharing one is often the
        # difference between paying twice and paying once.
        segments = list(evaluator.segments)
        state = evaluator.state
        question = evaluator.question
        question_id = evaluator.question_id
        mode = evaluator.mode
        ledger = evaluator.ledger
    else:
        chosen = auto_segmenter(state) if segmenter == "auto" else get_segmenter(segmenter)
        segments = chosen.split(state)

    n = len(segments)
    if n < 2:
        raise ValueError(
            f"found {n} segment(s); attribution needs at least two. "
            "Try a finer segmenter."
        )

    use_exact, permutations = _plan(
        n=n,
        exact=exact,
        samples=samples,
        state=state,
        questions={question_id: question},
        budget=budget,
    )

    if evaluator is None:
        evaluator = await CoalitionEvaluator.prepare(
            client,
            state,
            question,
            question_id=question_id,
            segmenter=chosen,
            segments=segments,
            target=target,
            mode=mode,
            budget=budget,
            ledger=ledger,
            concurrency=concurrency,
        )

    ids = [s.id for s in segments]

    if use_exact:
        coalitions = _all_subsets(ids)
        orderings: list[list[int]] = []
    else:
        rng = random.Random(seed)
        orderings = []
        for _ in range(permutations):
            order = list(ids)
            rng.shuffle(order)
            orderings.append(order)
        coalitions = _prefix_coalitions(orderings)

    await evaluator.values(coalitions)

    known = {c: evaluator.known(c) for c in coalitions}
    known = {c: v for c, v in known.items() if v is not None}

    empty_value = known.get(frozenset())
    if empty_value is None:
        empty_value = await evaluator.value(evaluator.nothing)
        known[frozenset()] = empty_value

    if use_exact:
        effects = _exact_values(segments, ids, known)
    else:
        effects = _sampled_values(segments, orderings, known)

    ledger.finish()

    return ShapleyAttribution(
        question_id=question_id,
        question=question,
        target=evaluator.target,
        model=client.model,
        mode=mode,
        state=state,
        segments=segments,
        baseline_answer=evaluator.baseline_answer,
        baseline_value=evaluator.baseline_value,
        empty_value=empty_value,
        effects=effects,
        exact=use_exact,
        coalitions_evaluated=evaluator.evaluations,
        ledger=ledger,
        permutations=0 if use_exact else len(orderings),
        failures=[(c, why) for c, why in evaluator.failures.items()],
    )


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def _plan(
    *,
    n: int,
    exact: bool | None,
    samples: int | None,
    state: State,
    questions: dict,
    budget: Budget,
) -> tuple[bool, int]:
    """Choose exact or sampled, and refuse a plan that cannot fit the budget."""
    requested = samples if samples is not None else DEFAULT_SAMPLES
    exact_requests = 2**n
    sampled_requests = min(requested * (n + 1), exact_requests)

    if exact is None:
        affordable = n <= MAX_EXACT_SEGMENTS and _fits(
            exact_requests, state, questions, budget
        )
        use_exact = affordable
    else:
        use_exact = exact

    if use_exact and n > 20:
        raise ValueError(
            f"exact Shapley over {n} segments would need 2^{n} evaluations. "
            "Use exact=False, or a coarser segmenter."
        )

    planned = exact_requests if use_exact else sampled_requests
    _preflight(planned, state, questions, budget, n, use_exact)
    return use_exact, requested


def _fits(requests: int, state: State, questions: dict, budget: Budget) -> bool:
    if budget.max_requests is not None and requests > budget.max_requests:
        return False
    per_request = estimate_tokens(
        SystemOneRequest(model="x", state=state, questions=questions).wire()
    )
    projected = per_request * requests
    if budget.max_input_tokens is not None and projected > budget.max_input_tokens:
        return False
    if budget.max_usd is not None:
        if projected * PRICE_USD_PER_INPUT_TOKEN > budget.max_usd:
            return False
    return True


def _preflight(
    planned: int,
    state: State,
    questions: dict,
    budget: Budget,
    n: int,
    use_exact: bool,
) -> None:
    method = "exact" if use_exact else "sampled"
    if budget.max_requests is not None and planned > budget.max_requests:
        raise BudgetExceeded(
            f"{method} Shapley over {n} segments needs ~{planned} evaluations but "
            f"the budget allows {budget.max_requests}. Raise max_requests, lower "
            f"samples, or use a coarser segmenter."
        )

    per_request = estimate_tokens(
        SystemOneRequest(model="x", state=state, questions=questions).wire()
    )
    projected = per_request * planned

    if budget.max_input_tokens is not None and projected > budget.max_input_tokens:
        raise BudgetExceeded(
            f"~{projected:,} estimated input tokens exceeds the "
            f"{budget.max_input_tokens:,} token budget."
        )

    projected_usd = projected * PRICE_USD_PER_INPUT_TOKEN
    if budget.max_usd is not None and projected_usd > budget.max_usd:
        raise BudgetExceeded(
            f"~${projected_usd:.6f} estimated for {planned} {method} evaluations "
            f"exceeds the ${budget.max_usd:.6f} budget."
        )


# --------------------------------------------------------------------------
# coalition generation
# --------------------------------------------------------------------------


def _all_subsets(ids: Sequence[int]) -> list[frozenset[int]]:
    subsets: list[frozenset[int]] = []
    for size in range(len(ids) + 1):
        subsets.extend(frozenset(combo) for combo in combinations(ids, size))
    return subsets


def _prefix_coalitions(orderings: Sequence[Sequence[int]]) -> list[frozenset[int]]:
    """Every prefix of every permutation, deduplicated.

    One permutation of n segments yields n marginal contributions from n+1
    evaluations, which is why permutation sampling beats drawing independent
    random subsets per segment.
    """
    seen: set[frozenset[int]] = set()
    for order in orderings:
        current: set[int] = set()
        seen.add(frozenset())
        for member in order:
            current.add(member)
            seen.add(frozenset(current))
    return list(seen)


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------


def _exact_values(
    segments: Sequence[Segment],
    ids: Sequence[int],
    known: dict[frozenset[int], float],
) -> list[ShapleyEffect]:
    n = len(ids)
    factorial = math.factorial
    total = factorial(n)

    effects: list[ShapleyEffect] = []
    for segment in segments:
        others = [i for i in ids if i != segment.id]
        accumulated = 0.0
        counted = 0
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / total
            for combo in combinations(others, size):
                without = frozenset(combo)
                with_it = without | {segment.id}
                a, b = known.get(without), known.get(with_it)
                if a is None or b is None:
                    continue  # a failed coalition; skip its term
                accumulated += weight * (b - a)
                counted += 1
        effects.append(
            ShapleyEffect(segment=segment, value=accumulated, samples=counted)
        )
    return effects


def _sampled_values(
    segments: Sequence[Segment],
    orderings: Sequence[Sequence[int]],
    known: dict[frozenset[int], float],
) -> list[ShapleyEffect]:
    marginals: dict[int, list[float]] = {s.id: [] for s in segments}

    for order in orderings:
        current: set[int] = set()
        for member in order:
            before = frozenset(current)
            current.add(member)
            after = frozenset(current)
            a, b = known.get(before), known.get(after)
            if a is None or b is None:
                continue
            marginals[member].append(b - a)

    effects: list[ShapleyEffect] = []
    for segment in segments:
        draws = marginals[segment.id]
        if not draws:
            effects.append(ShapleyEffect(segment=segment, value=0.0, samples=0))
            continue
        mean = statistics.fmean(draws)
        error: float | None = None
        if len(draws) > 1:
            error = statistics.stdev(draws) / math.sqrt(len(draws))
        effects.append(
            ShapleyEffect(
                segment=segment, value=mean, samples=len(draws), std_error=error
            )
        )
    return effects
