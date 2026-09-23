"""The smallest sufficient evidence, and the smallest change that flips a decision.

These are the outputs a person reads. A ranked list of six numbers diagnoses a
question; a quoted sentence explains a decision. So the assertions here are about
*which sentences come back*, not about magnitudes.

One result deserves special attention: a sufficient set of size zero. That means
the empty state already produces the answer, so the question does not need its
input at all. It looks like a degenerate case and it is actually the most
important thing the tool can tell you.
"""

from __future__ import annotations

import asyncio

import pytest

from jev_xray import Budget, Client, FakeTransport, Noul, RateLimiter, Signal
from jev_xray.coalition import CoalitionEvaluator
from jev_xray.minimal import Decision, minimal_flip, minimal_sufficient
from jev_xray.probe import DeepExplanation, deep_explain
from jev_xray.segment import SentenceSegmenter

REFUND = Noul(instructions="The customer is asking for a refund.")

STATE = (
    "The box was crushed on one corner. "
    "The jacket looks fine. "
    "I have been a customer since 2019. "
    "I would like my money back. "
    "Let me know my options."
)

SIGNALS = (
    Signal(r"money back", 2.6),
    Signal(r"looks fine", 0.8, label="false"),
)


def run(coro):
    return asyncio.run(coro)


def make_client(signals=SIGNALS, bias=-1.0):
    return Client(
        FakeTransport(signals=tuple(signals), bias=bias),
        model="fake-1",
        limiter=RateLimiter.unlimited(),
    )


async def prepare(state=STATE, signals=SIGNALS, bias=-1.0):
    segmenter = SentenceSegmenter()
    segments = segmenter.split(state)
    return await CoalitionEvaluator.prepare(
        make_client(signals, bias),
        state,
        REFUND,
        question_id="refund",
        segmenter=segmenter,
        segments=segments,
        target=None,
        mode="delete",
        budget=Budget.unlimited(),
        ledger=__import__("jev_xray").Ledger(),
    )


def order_by_shapley(evaluator):
    """Importance order without a full Shapley run, for search-only tests."""
    everything = evaluator.everything

    async def go():
        await evaluator.values([everything - {s.id} for s in evaluator.segments])
        scored = []
        for segment in evaluator.segments:
            value = evaluator.known(everything - {segment.id})
            scored.append((abs(evaluator.baseline_value - (value or 0.0)), segment.id))
        scored.sort(reverse=True)
        return [sid for _, sid in scored]

    return go()


class TestMinimalSufficient:
    def test_finds_a_subset_that_reproduces_the_answer(self):
        async def go():
            ev = await prepare()
            order = await order_by_shapley(ev)
            return await minimal_sufficient(ev, order, epsilon=0.05)

        result = run(go())
        assert result.found
        assert result.gap <= 0.05
        assert result.size < result.total

    def test_the_decisive_sentence_is_in_it(self):
        async def go():
            ev = await prepare()
            order = await order_by_shapley(ev)
            return await minimal_sufficient(ev, order, epsilon=0.05)

        result = run(go())
        assert any("money back" in s.text for s in result.segments)

    def test_quote_returns_text_in_original_order(self):
        async def go():
            ev = await prepare()
            order = await order_by_shapley(ev)
            return await minimal_sufficient(ev, order, epsilon=0.3)

        result = run(go())
        quote = result.quote()
        if len(result.segments) > 1:
            positions = [quote.find(s.text.strip()) for s in result.segments]
            assert positions == sorted(positions)

    def test_a_tight_epsilon_can_be_unsatisfiable(self):
        async def go():
            ev = await prepare()
            order = await order_by_shapley(ev)
            return await minimal_sufficient(ev, order, epsilon=0.0)

        result = run(go())
        # With an exact-match requirement the greedy path may fail; either way the
        # result must report honestly rather than pretend.
        assert result.found == (result.gap <= 0.0)

    def test_a_loose_epsilon_needs_no_evidence_at_all(self):
        # The gap between full and empty is smaller than epsilon, so zero segments
        # suffice. That is the saturated-prior signal.
        async def go():
            ev = await prepare()
            order = await order_by_shapley(ev)
            return await minimal_sufficient(ev, order, epsilon=1.0)

        result = run(go())
        assert result.found
        assert result.size == 0
        assert result.quote() == ""

    def test_pruning_shrinks_an_overshooting_greedy_set(self):
        async def go():
            ev = await prepare()
            order = await order_by_shapley(ev)
            pruned = await minimal_sufficient(ev, order, epsilon=0.05, prune=True)
            raw = await minimal_sufficient(ev, order, epsilon=0.05, prune=False)
            return pruned, raw

        pruned, raw = run(go())
        assert pruned.size <= raw.size

    def test_reports_how_many_evaluations_it_spent(self):
        async def go():
            ev = await prepare()
            order = await order_by_shapley(ev)
            return await minimal_sufficient(ev, order)

        assert run(go()).evaluations >= 0


class TestMinimalFlip:
    def test_finds_the_single_decisive_sentence(self):
        async def go():
            ev = await prepare()
            order = await order_by_shapley(ev)
            return await minimal_flip(ev, order, decision=Decision(0.5))

        result = run(go())
        assert result.found
        assert result.size == 1
        assert "money back" in result.segments[0].text

    def test_the_flipped_value_is_on_the_other_side_of_the_threshold(self):
        async def go():
            ev = await prepare()
            order = await order_by_shapley(ev)
            return ev.baseline_value, await minimal_flip(ev, order, decision=Decision(0.5))

        baseline, result = run(go())
        assert baseline >= 0.5
        assert result.value < 0.5

    def test_a_single_segment_search_is_marked_exhaustive(self):
        async def go():
            ev = await prepare()
            order = await order_by_shapley(ev)
            return await minimal_flip(ev, order)

        assert run(go()).exhaustive

    def test_reports_honestly_when_nothing_flips(self):
        # A large positive bias keeps every subset above the threshold.
        async def go():
            ev = await prepare(bias=4.0)
            order = await order_by_shapley(ev)
            return await minimal_flip(ev, order, decision=Decision(0.5))

        result = run(go())
        assert not result.found
        assert "robust" in result.summary()

    def test_the_threshold_is_respected(self):
        async def go():
            ev = await prepare()
            order = await order_by_shapley(ev)
            strict = await minimal_flip(ev, order, decision=Decision(0.99))
            return strict

        # At a 0.99 threshold the baseline already fails, so "flipping" means
        # crossing upward, which removing evidence cannot achieve.
        result = run(go())
        assert not result.found

    def test_redundant_evidence_needs_both_copies_removed(self):
        state = "I want my money back. Refund the money back please. Thanks."

        async def go():
            ev = await prepare(state=state, signals=(Signal(r"money back", 3.0),), bias=-1.5)
            order = await order_by_shapley(ev)
            return await minimal_flip(ev, order, decision=Decision(0.5))

        result = run(go())
        assert result.found
        # No single removal can flip it; the greedy path has to take both.
        assert result.size == 2
        assert not result.exhaustive


class TestDecision:
    def test_holds_at_and_above_the_threshold(self):
        decision = Decision(0.5)
        assert decision.holds(0.5)
        assert decision.holds(0.51)
        assert not decision.holds(0.49)

    def test_describes_itself(self):
        assert Decision(0.8).describe() == "value >= 0.8"


class TestDeepExplain:
    def test_returns_all_three_results(self):
        async def go():
            return await deep_explain(
                make_client(),
                STATE,
                REFUND,
                question_id="refund",
                segmenter="sentence",
                budget=Budget.unlimited(),
            )

        result = run(go())
        assert isinstance(result, DeepExplanation)
        assert result.attribution.exact
        assert result.sufficient.found
        assert result.flipping.found

    def test_the_searches_reuse_the_shapley_cache(self):
        # Exact Shapley evaluates every subset, so both searches should be free.
        async def go():
            transport = FakeTransport(signals=SIGNALS, bias=-1.0)
            client = Client(transport, model="m", limiter=RateLimiter.unlimited())
            result = await deep_explain(
                client,
                STATE,
                REFUND,
                question_id="refund",
                segmenter="sentence",
                budget=Budget.unlimited(),
            )
            return transport.calls, result

        calls, result = run(go())
        assert result.sufficient.evaluations == 0
        assert result.flipping.evaluations == 0
        assert calls == 2 ** len(result.attribution.segments)

    def test_summary_includes_the_quote_and_the_counterfactual(self):
        async def go():
            return await deep_explain(
                make_client(),
                STATE,
                REFUND,
                question_id="refund",
                segmenter="sentence",
                budget=Budget.unlimited(),
            )

        summary = run(go()).summary()
        assert "smallest evidence" in summary
        assert "flips the decision" in summary
        assert "money back" in summary

    def test_warns_when_the_question_cannot_discriminate(self):
        # Both the full and empty state land above the threshold, so the answer
        # barely depends on the input.
        async def go():
            return await deep_explain(
                make_client(bias=4.0),
                STATE,
                REFUND,
                question_id="refund",
                segmenter="sentence",
                budget=Budget.unlimited(),
            )

        summary = run(go()).summary()
        assert "cannot discriminate" in summary

    def test_no_warning_when_the_question_does_discriminate(self):
        async def go():
            return await deep_explain(
                make_client(),
                STATE,
                REFUND,
                question_id="refund",
                segmenter="sentence",
                budget=Budget.unlimited(),
            )

        assert "cannot discriminate" not in run(go()).summary()

    def test_too_few_segments_is_rejected(self):
        async def go():
            return await deep_explain(
                make_client(), "Only one.", REFUND, segmenter="sentence"
            )

        with pytest.raises(ValueError, match="at least two"):
            run(go())
