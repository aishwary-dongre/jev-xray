"""Coarse-to-fine attribution.

Exact Shapley is 2^n, so a forty-sentence document is not attributable exactly.
Attributing paragraphs exactly and then re-attributing inside only the ones that
mattered costs 2^6 + 2^5 + 2^5 instead.

The property worth testing hardest is not the saving, it is the honesty of the
two levels. A fine map's values sum to the swing from removing its whole region
with the others present, which is that region's leave-one-out effect. That equals
its coarse Shapley value only when the regions do not interact, and the gap has to
be reported rather than quietly presented as a decomposition.
"""

from __future__ import annotations

import asyncio

import pytest

from jev_xray import (
    Budget,
    Client,
    FakeTransport,
    Noul,
    RateLimiter,
    SentenceSegmenter,
    Signal,
)
from jev_xray.drill import DrillDown, drill_down
from jev_xray.segment import TurnSegmenter, WithinSpan

REFUND = Noul(instructions="The customer is asking for a refund.")

STATE = (
    "Customer: My order A-104 arrived yesterday. The outer box was crushed on one "
    "corner. The jacket itself looks fine honestly.\n"
    "Agent: Thanks for letting us know. I can see the order on your account.\n"
    "Customer: I have been shopping with you since 2019. I am not sure whether to "
    "send it back or keep it at a discount. Your policy page says damaged items "
    "qualify for a full refund. Let me know what my options are."
)

SIGNALS = (
    Signal(r"qualify for a full refund", 2.4),
    Signal(r"looks fine", 0.9, label="false"),
)


def run(coro):
    return asyncio.run(coro)


def client(signals=SIGNALS, bias=-1.0):
    return Client(
        FakeTransport(signals=tuple(signals), bias=bias),
        model="fake-1",
        limiter=RateLimiter.unlimited(),
    )


def drill(state=STATE, **kwargs):
    kwargs.setdefault("coarse", "turn")
    kwargs.setdefault("fine", "sentence")
    kwargs.setdefault("top_k", 2)

    async def go():
        return await drill_down(
            client(),
            state,
            REFUND,
            question_id="refund",
            budget=Budget.unlimited(),
            **kwargs,
        )

    return run(go())


class TestWithinSpan:
    TEXT = "Alpha one. Alpha two.\nBeta one. Beta two."

    def test_segments_only_inside_the_region(self):
        turns = TurnSegmenter().split(self.TEXT)
        region = turns[0]
        restricted = WithinSpan(SentenceSegmenter(), region.start, region.end)
        assert [s.text for s in restricted.split(self.TEXT)] == [
            "Alpha one.",
            "Alpha two.",
        ]

    def test_spans_are_offset_into_the_full_text(self):
        turns = TurnSegmenter().split(self.TEXT)
        restricted = WithinSpan(SentenceSegmenter(), turns[1].start, turns[1].end)
        for segment in restricted.split(self.TEXT):
            assert self.TEXT[segment.start : segment.end] == segment.text

    def test_ablation_keeps_the_rest_of_the_state(self):
        # The point of the restriction: every coalition still carries the other
        # regions, so a fine value is measured in real context.
        turns = TurnSegmenter().split(self.TEXT)
        restricted = WithinSpan(SentenceSegmenter(), turns[0].start, turns[0].end)
        result = restricted.ablate(self.TEXT, [0])
        assert "Alpha one." not in result
        assert "Beta one. Beta two." in result

    def test_a_region_beyond_the_text_is_clamped(self):
        restricted = WithinSpan(SentenceSegmenter(), 0, 10_000)
        assert len(restricted.split(self.TEXT)) == 4

    def test_the_kind_records_the_inner_segmenter(self):
        restricted = WithinSpan(SentenceSegmenter(), 0, 5)
        assert restricted.kind == "within-sentence"


class TestDrillDown:
    def test_the_coarse_pass_is_exact(self):
        result = drill()
        assert result.coarse.exact
        assert result.coarse.efficiency_gap == pytest.approx(0.0, abs=1e-9)

    def test_it_drills_into_the_top_regions_only(self):
        result = drill(top_k=2)
        assert len(result.drilled) == 2
        skipped = [r for r in result.regions if r.fine is None]
        assert any("outside the top 2" in (r.skipped_reason or "") for r in skipped)

    def test_the_fine_map_finds_the_decisive_sentence(self):
        result = drill()
        strongest = max(result.drilled, key=lambda r: abs(r.coarse_value))
        top = strongest.fine.ranked()[0]
        assert "qualify for a full refund" in top.segment.text

    def test_the_fine_map_finds_an_opposing_sentence(self):
        result = drill()
        opposing = [r for r in result.drilled if r.coarse_value < 0]
        assert opposing
        top = opposing[0].fine.ranked()[0]
        assert "looks fine" in top.segment.text
        assert top.signed < 0

    def test_it_costs_far_less_than_exact_over_every_sentence(self):
        result = drill()
        sentences = len(SentenceSegmenter().split(STATE))
        assert sentences >= 8
        assert result.ledger.requests < 2**sentences / 4

    def test_a_region_contributing_nothing_is_not_drilled_into(self):
        # Paying 2^f to confirm an inert region has no internal structure is waste.
        result = drill(top_k=3, min_contribution=0.02)
        inert = [
            r
            for r in result.regions
            if abs(r.coarse_value) < 0.02 and r.fine is None
        ]
        assert inert
        assert any("contributed less than" in (r.skipped_reason or "") for r in inert)

    def test_the_axis_is_shared_between_levels(self):
        result = drill()
        for region in result.drilled:
            assert region.fine.target.kind == result.coarse.target.kind
            assert region.fine.target.label == result.coarse.target.label

    def test_an_undividable_region_is_reported_as_such(self):
        # The region has to actually contribute, or it is skipped for being
        # inert before the divisibility check is ever reached.
        state = (
            "Alpha: one sentence only.\n"
            "Beta: damaged items qualify for a full refund.\n"
            "Gamma: and one more."
        )
        result = drill(state=state, top_k=3)
        carrier = max(result.regions, key=lambda r: abs(r.coarse_value))
        assert abs(carrier.coarse_value) > 0.02
        assert carrier.skipped_reason == "does not divide further"

    def test_a_structured_state_is_rejected_with_a_pointer(self):
        async def go():
            return await drill_down(
                client(), {"a": "one", "b": "two"}, REFUND, question_id="refund"
            )

        with pytest.raises(TypeError, match="segmenter='field'"):
            run(go())

    def test_result_type(self):
        assert isinstance(drill(), DrillDown)


class TestHonestyOfTheTwoLevels:
    def test_the_gap_between_levels_is_computed(self):
        result = drill()
        for region in result.drilled:
            assert region.fine_total is not None
            assert region.gap == pytest.approx(
                region.coarse_value - region.fine_total
            )

    def test_a_meaningful_gap_is_explained_rather_than_hidden(self):
        result = drill()
        wide = [r for r in result.drilled if r.gap is not None and abs(r.gap) > 0.02]
        assert wide, "expected at least one interacting region in this fixture"
        text = wide[0].summary()
        assert "interacting with the" in text
        assert "not an error" in text

    def test_the_closing_note_does_not_claim_the_levels_decompose(self):
        summary = drill().summary()
        assert "not the" in summary and "same quantity" in summary
        assert "not one additive ranking" in summary

    def test_a_skipped_region_has_no_totals(self):
        result = drill(top_k=1)
        skipped = [r for r in result.regions if r.fine is None]
        assert skipped
        assert skipped[0].fine_total is None
        assert skipped[0].gap is None


class TestReporting:
    def test_summary_names_both_levels(self):
        summary = drill().summary()
        assert "coarse level (turn)" in summary
        assert "sentence" in summary

    def test_summary_includes_the_coarse_attribution_header(self):
        assert "[shapley]" in drill().summary()

    def test_a_skipped_region_says_why(self):
        summary = drill(top_k=1).summary()
        assert "not drilled into" in summary
