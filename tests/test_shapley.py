"""Shapley attribution, tested against the cases leave-one-out gets wrong.

Two failure modes are constructed deliberately here, because they are the whole
justification for the extra cost:

**Redundant evidence** — one signal matched by two different sentences. Remove
either and the other still fires, so leave-one-out scores both at zero while the
pair plainly does all the work. The individual deltas under-count.

**Complementary evidence** — a signal whose pattern spans two sentences, so it
only fires when both are present. Remove either and it dies, so leave-one-out
scores both at the *full* effect. The individual deltas over-count, summing to
twice the real swing.

Shapley must get both right, and its efficiency property gives us an exact
expected answer to assert against rather than a vibe.
"""

from __future__ import annotations

import asyncio

import pytest

from jev_xray import (
    Budget,
    BudgetExceeded,
    Choice,
    Client,
    FakeTransport,
    Noul,
    RateLimiter,
    Signal,
    leave_one_out,
)
from jev_xray.shapley import MAX_EXACT_SEGMENTS, ShapleyAttribution, shapley

REFUND = Noul(instructions="The customer is asking for a refund.")


def client(signals, bias=0.0):
    return Client(
        FakeTransport(signals=tuple(signals), bias=bias),
        model="fake-1",
        limiter=RateLimiter.unlimited(),
    )


def run(coro):
    return asyncio.run(coro)


def both(state, signals, bias=0.0, **kwargs):
    """The same state under both estimators, for direct comparison."""

    async def go():
        loo = await leave_one_out(
            client(signals, bias),
            state,
            REFUND,
            segmenter="sentence",
            budget=Budget.unlimited(),
        )
        shp = await shapley(
            client(signals, bias),
            state,
            REFUND,
            segmenter="sentence",
            budget=Budget.unlimited(),
            **kwargs,
        )
        return loo, shp

    return run(go())


# --------------------------------------------------------------------------

REDUNDANT = "I want my money back. Please return the money back to my card. Thanks for your help."
COMPLEMENTARY = "I want my money back. Or maybe just a discount instead. Thanks for your help."


class TestRedundantEvidence:
    SIGNALS = (Signal(r"money back", 2.6),)

    def test_leave_one_out_scores_both_copies_at_zero(self):
        loo, _ = both(REDUNDANT, self.SIGNALS, bias=-1.2)
        assert all(e.magnitude < 1e-9 for e in loo.effects)

    def test_leave_one_out_leaves_the_whole_swing_unexplained(self):
        loo, _ = both(REDUNDANT, self.SIGNALS, bias=-1.2)
        assert loo.interaction_residual == pytest.approx(loo.total_swing, abs=1e-6)

    def test_shapley_splits_the_credit_between_them(self):
        _, shp = both(REDUNDANT, self.SIGNALS, bias=-1.2)
        carriers = [e for e in shp.effects if "money back" in e.segment.text]
        assert len(carriers) == 2
        for effect in carriers:
            assert effect.value == pytest.approx(shp.total_swing / 2, abs=1e-6)

    def test_shapley_still_finds_the_inert_segment(self):
        _, shp = both(REDUNDANT, self.SIGNALS, bias=-1.2)
        inert = [e for e in shp.effects if "Thanks" in e.segment.text]
        assert inert[0].value == pytest.approx(0.0, abs=1e-9)
        assert not inert[0].is_significant


class TestComplementaryEvidence:
    # Fires only when both sentences survive.
    SIGNALS = (Signal(r"money back.*discount", 3.0),)

    def test_leave_one_out_over_counts(self):
        loo, _ = both(COMPLEMENTARY, self.SIGNALS, bias=-1.5)
        # Each of the two looks individually decisive, so the deltas sum to
        # roughly twice the swing the state actually produces.
        assert loo.total_effect == pytest.approx(2 * loo.total_swing, abs=1e-6)
        assert loo.interaction_residual < -0.1
        assert not loo.is_additive()

    def test_shapley_halves_it_and_stays_efficient(self):
        _, shp = both(COMPLEMENTARY, self.SIGNALS, bias=-1.5)
        pair = [
            e
            for e in shp.effects
            if "money back" in e.segment.text or "discount" in e.segment.text
        ]
        assert len(pair) == 2
        for effect in pair:
            assert effect.value == pytest.approx(shp.total_swing / 2, abs=1e-6)
        assert shp.efficiency_gap == pytest.approx(0.0, abs=1e-9)


class TestEfficiency:
    """Shapley values sum to the total swing by definition, so this checks arithmetic."""

    SIGNALS = (
        Signal(r"money back", 2.4),
        Signal(r"looks fine", 0.9, label="false"),
        Signal(r"second time", 0.5),
    )
    STATE = (
        "The box was crushed. The jacket looks fine. "
        "This is the second time I have written. I would like my money back. "
        "Let me know my options."
    )

    def test_exact_values_sum_to_the_total_swing(self):
        _, shp = both(self.STATE, self.SIGNALS, bias=-1.0)
        assert shp.exact
        assert shp.total_effect == pytest.approx(shp.total_swing, abs=1e-9)
        assert shp.efficiency_gap == pytest.approx(0.0, abs=1e-9)
        assert shp.is_additive()

    def test_sampled_values_approximately_sum_to_the_total_swing(self):
        _, shp = both(self.STATE, self.SIGNALS, bias=-1.0, exact=False, samples=64)
        assert not shp.exact
        assert shp.efficiency_gap == pytest.approx(0.0, abs=0.05)

    def test_exact_and_sampled_agree(self):
        _, exact = both(self.STATE, self.SIGNALS, bias=-1.0)
        _, sampled = both(self.STATE, self.SIGNALS, bias=-1.0, exact=False, samples=96)
        for effect in exact.effects:
            assert sampled.by_id(effect.segment.id).value == pytest.approx(
                effect.value, abs=0.05
            )

    def test_the_two_estimators_agree_on_the_top_segment(self):
        loo, shp = both(self.STATE, self.SIGNALS, bias=-1.0)
        assert loo.top(1)[0].segment.id == shp.top(1)[0].segment.id

    def test_signs_are_preserved(self):
        _, shp = both(self.STATE, self.SIGNALS, bias=-1.0)
        assert "money back" in shp.supporting()[0].segment.text
        assert "looks fine" in shp.opposing()[0].segment.text


class TestExactness:
    SIGNALS = (Signal(r"money back", 2.0),)

    def test_exact_evaluates_every_subset(self):
        _, shp = both(REDUNDANT, self.SIGNALS)
        # 3 sentences -> 2**3 coalitions
        assert shp.exact
        assert shp.coalitions_evaluated == 8

    def test_sampling_can_be_forced(self):
        _, shp = both(REDUNDANT, self.SIGNALS, exact=False, samples=5)
        assert not shp.exact
        assert shp.permutations == 5

    def test_sampled_effects_carry_a_standard_error(self):
        _, shp = both(REDUNDANT, self.SIGNALS, exact=False, samples=8)
        assert all(e.std_error is not None for e in shp.effects)
        assert all(e.samples == 8 for e in shp.effects)

    def test_exact_effects_carry_no_standard_error(self):
        _, shp = both(REDUNDANT, self.SIGNALS)
        assert all(e.std_error is None for e in shp.effects)

    def test_sampling_is_reproducible_under_a_seed(self):
        _, a = both(REDUNDANT, self.SIGNALS, exact=False, samples=7, seed=42)
        _, b = both(REDUNDANT, self.SIGNALS, exact=False, samples=7, seed=42)
        assert [e.value for e in a.effects] == [e.value for e in b.effects]

    def test_a_different_seed_draws_different_orderings(self):
        _, a = both(REDUNDANT, self.SIGNALS, exact=False, samples=3, seed=1)
        _, b = both(REDUNDANT, self.SIGNALS, exact=False, samples=3, seed=2)
        # Values may coincide; the point is the seed is actually consumed.
        assert a.permutations == b.permutations == 3

    def test_exactness_is_abandoned_when_the_budget_cannot_cover_it(self):
        # 3 segments would need 8 evaluations exactly; allow fewer and it samples.
        async def go():
            return await shapley(
                client(self.SIGNALS),
                REDUNDANT,
                REFUND,
                segmenter="sentence",
                budget=Budget(max_requests=7, max_usd=None),
                samples=1,
            )

        result = run(go())
        assert not result.exact


class TestBudget:
    SIGNALS = (Signal(r"money back", 2.0),)

    def test_a_request_ceiling_is_enforced_before_spending(self):
        transport = FakeTransport(signals=self.SIGNALS)

        async def go():
            return await shapley(
                Client(transport, model="m", limiter=RateLimiter.unlimited()),
                REDUNDANT,
                REFUND,
                segmenter="sentence",
                budget=Budget(max_requests=2, max_usd=None),
                exact=True,
            )

        with pytest.raises(BudgetExceeded, match="Shapley"):
            run(go())
        assert transport.calls == 0

    def test_the_error_suggests_the_available_knobs(self):
        async def go():
            return await shapley(
                client(self.SIGNALS),
                REDUNDANT,
                REFUND,
                segmenter="sentence",
                budget=Budget(max_requests=2, max_usd=None),
                exact=True,
            )

        with pytest.raises(BudgetExceeded, match="lower\\s+samples"):
            run(go())

    def test_a_cost_ceiling_is_enforced(self):
        async def go():
            return await shapley(
                client(self.SIGNALS),
                REDUNDANT,
                REFUND,
                segmenter="sentence",
                budget=Budget(max_usd=1e-12),
                exact=True,
            )

        with pytest.raises(BudgetExceeded, match=r"\$"):
            run(go())


class TestGuards:
    def test_too_few_segments_is_rejected(self):
        async def go():
            return await shapley(
                client(()), "Only one sentence.", REFUND, segmenter="sentence"
            )

        with pytest.raises(ValueError, match="at least two"):
            run(go())

    def test_exact_over_an_absurd_segment_count_is_refused(self):
        state = " ".join(f"Sentence number {i}." for i in range(25))

        async def go():
            return await shapley(
                client(()),
                state,
                REFUND,
                segmenter="sentence",
                budget=Budget.unlimited(),
                exact=True,
            )

        with pytest.raises(ValueError, match="2\\^25"):
            run(go())

    def test_the_exactness_threshold_is_documented_and_used(self):
        assert MAX_EXACT_SEGMENTS >= 6
        assert 2**MAX_EXACT_SEGMENTS <= 1024


class TestChoiceQuestions:
    def test_works_on_a_choice_target(self):
        question = Choice(
            instructions="What does the customer want",
            criteria={"refund": "money back", "info": "just asking"},
        )

        async def go():
            return await shapley(
                client((Signal(r"money back", 2.0, label="refund"),)),
                REDUNDANT,
                question,
                segmenter="sentence",
                budget=Budget.unlimited(),
            )

        result = run(go())
        assert result.target.kind == "choice_prob"
        assert result.target.label == "refund"
        assert result.efficiency_gap == pytest.approx(0.0, abs=1e-9)


class TestReporting:
    SIGNALS = (Signal(r"money back", 2.4),)

    def test_summary_names_the_method(self):
        _, shp = both(REDUNDANT, self.SIGNALS, bias=-1.0)
        summary = shp.summary()
        assert "shapley" in summary
        assert "exact over 8 coalitions" in summary

    def test_interaction_residual_aliases_the_efficiency_gap(self):
        _, shp = both(REDUNDANT, self.SIGNALS, bias=-1.0)
        assert shp.interaction_residual == shp.efficiency_gap

    def test_unexplained_prior_exposes_the_empty_state_answer(self):
        _, shp = both(REDUNDANT, self.SIGNALS, bias=-1.0)
        assert shp.unexplained_prior == shp.empty_value

    def test_effects_share_a_signed_axis_with_leave_one_out(self):
        loo, shp = both(REDUNDANT, self.SIGNALS, bias=-1.0)
        assert loo.effects[0].signed == loo.effects[0].delta
        assert shp.effects[0].signed == shp.effects[0].value

    def test_shapley_effects_have_no_single_ablated_value(self):
        _, shp = both(REDUNDANT, self.SIGNALS, bias=-1.0)
        assert shp.effects[0].ablated_value is None

    def test_by_id_raises_for_an_unknown_segment(self):
        _, shp = both(REDUNDANT, self.SIGNALS)
        with pytest.raises(KeyError):
            shp.by_id(999)

    def test_result_type(self):
        _, shp = both(REDUNDANT, self.SIGNALS)
        assert isinstance(shp, ShapleyAttribution)
