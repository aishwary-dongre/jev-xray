"""One probe, three answers.

A decision is worth three different explanations, and they answer different
questions:

* the **map** — how much did each part of the input contribute?
* the **because** — what is the smallest quote that produces this answer?
* the **counterfactual** — what would have had to be different?

All three ride on the same coalition cache, so running them together costs very
little more than running the map alone. With exact Shapley they cost nothing
extra at all, because every subset the searches need has already been evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass

from .budget import Budget, Ledger
from .client import Client
from .coalition import CoalitionEvaluator
from .minimal import (
    Decision,
    FlippingSet,
    SufficientEvidence,
    minimal_flip,
    minimal_sufficient,
)
from .segment import AblationMode, Segmenter, auto_segmenter, get_segmenter
from .shapley import ShapleyAttribution, shapley
from .types import Question, State, Target

__all__ = ["DeepExplanation", "deep_explain"]


@dataclass(slots=True)
class DeepExplanation:
    """A Shapley map plus the two minimal-evidence searches."""

    attribution: ShapleyAttribution
    sufficient: SufficientEvidence
    flipping: FlippingSet
    ledger: Ledger

    @property
    def baseline_value(self) -> float:
        return self.attribution.baseline_value

    def summary(self) -> str:
        parts = [
            self.attribution.summary(),
            "",
            "smallest evidence that reproduces the answer",
            self.sufficient.summary(),
        ]
        if self.sufficient.found:
            parts.append(f'    "{self.sufficient.quote()}"')

        parts += ["", "smallest change that flips the decision", self.flipping.summary()]
        if self.flipping.found:
            parts.append(f'    remove: "{self.flipping.quote()}"')

        prior = self.attribution.unexplained_prior
        if self.flipping.decision.holds(prior) == self.flipping.decision.holds(
            self.baseline_value
        ):
            parts += [
                "",
                "warning",
                f"  an empty state already answers {prior:.4f}, which lands on the "
                f"same side of {self.flipping.decision.describe()} as the full "
                f"state.\n  this question cannot discriminate: the answer is mostly "
                f"a prior the input never touches.",
            ]

        return "\n".join(parts)


async def deep_explain(
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
    epsilon: float = 0.05,
    exact: bool | None = None,
    samples: int | None = None,
    seed: int = 0,
    concurrency: int = 8,
    ledger: Ledger | None = None,
) -> DeepExplanation:
    """Run the Shapley map and both minimal-evidence searches over one cache."""
    budget = budget if budget is not None else Budget()
    ledger = ledger if ledger is not None else Ledger()

    chosen = auto_segmenter(state) if segmenter == "auto" else get_segmenter(segmenter)
    segments = chosen.split(state)
    if len(segments) < 2:
        raise ValueError(
            f"{type(chosen).__name__} found {len(segments)} segment(s); attribution "
            "needs at least two. Try a finer segmenter."
        )

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

    attribution = await shapley(
        client,
        state,
        question,
        budget=budget,
        exact=exact,
        samples=samples,
        seed=seed,
        evaluator=evaluator,
    )

    order = [e.segment.id for e in attribution.ranked()]

    sufficient = await minimal_sufficient(evaluator, order, epsilon=epsilon)
    flipping = await minimal_flip(evaluator, order, decision=decision)

    ledger.finish()
    return DeepExplanation(
        attribution=attribution,
        sufficient=sufficient,
        flipping=flipping,
        ledger=ledger,
    )
