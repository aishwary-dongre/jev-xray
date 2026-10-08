"""Coarse-to-fine attribution, for states too long to attribute exactly.

Exact Shapley evaluates every one of 2^n coalitions. That is 64 requests for six
segments and perfectly affordable; it is 2^40 for a forty-sentence document and
not a plan. Sampling is the usual answer and it costs precision exactly where you
want it most, on the handful of spans that turn out to matter.

So: attribute coarsely and exactly, then attribute again inside only the coarse
segments that carried weight. A forty-sentence document split into six paragraphs
costs 2^6 for the overview plus 2^5 for each paragraph drilled into. Sixty-four
plus sixty-four, not a trillion.

The fine pass holds the rest of the state present in every coalition. That is a
correctness property, not an optimisation: a sentence's contribution measured
inside an isolated paragraph is a different quantity from its contribution in the
document it actually appeared in, and the second one is what you asked about.

What this does not give you is a single additive map over all the fine segments,
and the reason is worth stating precisely because it is easy to assume otherwise.

The coarse values are exact Shapley values and sum to the total swing. Each fine
map is exact within its own region, and its values sum to the swing from removing
that whole region *with every other region present* — which is the region's
leave-one-out effect, not its Shapley value. Those two coincide only when the
regions do not interact, so a region's fine values generally will not add up to
its coarse value. The gap is reported rather than smoothed over: it is the same
interaction signal the estimators report elsewhere, measured one level up.

Flattening the two levels into one ranking would therefore be a figure nobody
computed, and the result keeps them separate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .budget import Budget, Ledger
from .client import Client
from .segment import (
    AblationMode,
    LineSegmenter,
    Segment,
    Segmenter,
    SentenceSegmenter,
    TurnSegmenter,
    WithinSpan,
    get_segmenter,
)
from .shapley import ShapleyAttribution, shapley
from .types import Question, State, Target

__all__ = ["Region", "DrillDown", "drill_down"]


@dataclass(slots=True)
class Region:
    """One coarse segment and the fine attribution inside it."""

    segment: Segment
    coarse_value: float
    fine: ShapleyAttribution | None = None
    skipped_reason: str | None = None

    @property
    def label(self) -> str:
        return self.segment.label

    @property
    def fine_total(self) -> float | None:
        """Sum of the fine values inside this region.

        Equals the swing from removing the whole region with everything else
        present. That is the region's leave-one-out effect, which matches its
        coarse Shapley value only when the regions act independently.
        """
        if self.fine is None:
            return None
        return self.fine.total_effect

    @property
    def gap(self) -> float | None:
        """Coarse value minus the fine total: how much the regions interact."""
        total = self.fine_total
        return None if total is None else self.coarse_value - total

    def summary(self) -> str:
        head = f"  [{self.label}] contributed {self.coarse_value:+.4f}"
        if self.fine is None:
            return f"{head}\n      not drilled into: {self.skipped_reason}"

        lines = [head]
        for effect in self.fine.ranked():
            lines.append(
                f"      {effect.signed:+.4f}  {effect.segment.preview(56)}"
            )

        total, gap = self.fine_total, self.gap
        detail = f"      inside sums to {total:+.4f}"
        if gap is not None and abs(gap) > 0.02:
            detail += (
                f", against a coarse value of {self.coarse_value:+.4f}. "
                f"the {gap:+.4f} gap is\n      this region interacting with the "
                f"others, not an error"
            )
        lines.append(detail)
        return "\n".join(lines)


@dataclass(slots=True)
class DrillDown:
    """A coarse attribution plus fine attributions inside the parts that mattered."""

    coarse: ShapleyAttribution
    regions: list[Region]
    ledger: Ledger
    coarse_kind: str = ""
    fine_kind: str = ""

    @property
    def baseline_value(self) -> float:
        return self.coarse.baseline_value

    @property
    def drilled(self) -> list[Region]:
        return [r for r in self.regions if r.fine is not None]

    def summary(self) -> str:
        lines = [
            self.coarse.summary(),
            "",
            f"coarse level ({self.coarse_kind})",
        ]
        for effect in self.coarse.ranked():
            lines.append(
                f"  {effect.signed:+.4f}  {effect.segment.preview(60)}"
            )

        if self.drilled:
            lines += ["", f"drilled into the top {len(self.drilled)} ({self.fine_kind})"]
            lines += [r.summary() for r in self.regions if r.fine is not None]

        skipped = [r for r in self.regions if r.fine is None]
        if skipped:
            # Worth saying what was not examined. A reader comparing the coarse
            # list against the fine section would otherwise have to work out the
            # difference themselves, and might assume it was examined and empty.
            grouped: dict[str, int] = {}
            for region in skipped:
                reason = region.skipped_reason or "skipped"
                grouped[reason] = grouped.get(reason, 0) + 1
            lines += ["", f"not drilled into ({len(skipped)} region(s))"]
            lines += [f"  {count}x {reason}" for reason, count in grouped.items()]

        lines += [
            "",
            "note",
            "  the coarse values are exact Shapley values and sum to the total swing.",
            "  a fine map is exact within its region and sums to the swing from",
            "  removing that whole region with the others present, which is not the",
            "  same quantity as its coarse value unless the regions act independently.",
            "  the two levels are not one additive ranking and are not shown as one.",
        ]
        return "\n".join(lines)


_COARSER = {
    "sentence": TurnSegmenter,
    "line": TurnSegmenter,
    "turn": TurnSegmenter,
}


def _pick_coarse(fine: Segmenter) -> Segmenter:
    """A segmenter one level coarser than the fine one."""
    factory = _COARSER.get(getattr(fine, "kind", ""), TurnSegmenter)
    return factory()


async def drill_down(
    client: Client,
    state: State,
    question: Question,
    *,
    question_id: str = "q",
    coarse: str | Segmenter | None = None,
    fine: str | Segmenter = "sentence",
    top_k: int = 2,
    min_contribution: float = 0.02,
    target: Target | None = None,
    mode: AblationMode = "delete",
    budget: Budget | None = None,
    concurrency: int = 8,
    ledger: Ledger | None = None,
) -> DrillDown:
    """Attribute coarsely and exactly, then drill into the parts that mattered.

    ``top_k`` caps how many coarse regions are examined in detail, and
    ``min_contribution`` skips any whose coarse effect is too small to be worth
    the requests. A region that contributed nothing has no interesting internal
    structure, and paying 2^f to confirm that is waste.

    Only text states are supported: a structured state's fields are already the
    natural coarse level, and :class:`~jev_xray.segment.JsonFieldSegmenter` with
    exact Shapley covers it.
    """
    if not isinstance(state, str):
        raise TypeError(
            "drill_down needs a text state; for a structured state the fields are "
            "already the coarse level, so use segmenter='field' directly"
        )

    budget = budget if budget is not None else Budget()
    ledger = ledger if ledger is not None else Ledger()

    fine_segmenter = get_segmenter(fine) if isinstance(fine, str) else fine
    coarse_segmenter = (
        _pick_coarse(fine_segmenter)
        if coarse is None
        else (get_segmenter(coarse) if isinstance(coarse, str) else coarse)
    )

    coarse_map = await shapley(
        client,
        state,
        question,
        question_id=question_id,
        segmenter=coarse_segmenter,
        target=target,
        mode=mode,
        budget=budget,
        ledger=ledger,
        concurrency=concurrency,
    )

    # Keep the axis the coarse pass established, so a fine value is comparable to
    # the coarse value it decomposes.
    axis = coarse_map.target

    regions: list[Region] = []
    ranked = coarse_map.ranked()
    for position, effect in enumerate(ranked):
        segment = effect.segment
        region = Region(segment=segment, coarse_value=effect.signed)

        if position >= top_k:
            region.skipped_reason = f"outside the top {top_k}"
            regions.append(region)
            continue
        if abs(effect.signed) < min_contribution:
            region.skipped_reason = (
                f"contributed less than {min_contribution:g}; nothing inside it to find"
            )
            regions.append(region)
            continue
        if segment.start is None or segment.end is None:
            region.skipped_reason = "no span to restrict to"
            regions.append(region)
            continue

        restricted = WithinSpan(fine_segmenter, segment.start, segment.end)
        if len(restricted.split(state)) < 2:
            region.skipped_reason = "does not divide further"
            regions.append(region)
            continue

        region.fine = await shapley(
            client,
            state,
            question,
            question_id=question_id,
            segmenter=restricted,
            target=axis,
            mode=mode,
            budget=budget,
            ledger=ledger,
            concurrency=concurrency,
        )
        regions.append(region)

    ledger.finish()
    return DrillDown(
        coarse=coarse_map,
        regions=regions,
        ledger=ledger,
        coarse_kind=getattr(coarse_segmenter, "kind", "coarse"),
        fine_kind=getattr(fine_segmenter, "kind", "fine"),
    )
