"""Question-level stability probes.

These perturb the *question* rather than the state, so the fixture has to be able
to notice. Two capabilities make that possible: signals scoped to the question
text, and a position bonus on the first-listed option. Both mirror real
phenomena, and without them a passing test would only prove the probe runs.
"""

from __future__ import annotations

import asyncio

import pytest

from jev_xray import Budget, Choice, Client, FakeTransport, Noul, RateLimiter, Signal
from jev_xray.stability import (
    build_context,
    framings,
    negation_coherence,
    option_order_flip,
    paraphrase_spread,
)

STATE = "I was charged twice for order A-104 and the second charge is still pending."

DEPARTMENT = Choice(
    instructions="Which team should handle this?",
    criteria={
        "billing": "Payments, charges and refunds",
        "technical": "Bugs and integration faults",
        "sales": "Pricing and new accounts",
    },
)

REFUND = Noul(
    instructions="The customer is asking for their money back.",
    criteria={
        "true": "The customer wants a refund, stated or clearly implied",
        "false": "The customer wants something else",
    },
)


def run(coro):
    return asyncio.run(coro)


async def context(question, *, signals=(), bias=0.0, transport=None, **kwargs):
    client = Client(
        transport or FakeTransport(signals=tuple(signals), bias=bias),
        model="fake-1",
        limiter=RateLimiter.unlimited(),
    )
    return await build_context(
        client,
        STATE,
        question,
        question_id="q",
        segmenter="sentence",
        budget=Budget.unlimited(),
        **kwargs,
    )


class TestOptionOrderFlip:
    def test_passes_when_ordering_is_ignored(self):
        async def go():
            ctx = await context(
                DEPARTMENT, signals=(Signal(r"charged twice", 3.0, label="billing"),)
            )
            return await option_order_flip(ctx)

        result = run(go())
        assert result.verdict == "ok"
        assert result.measurement == pytest.approx(0.0, abs=1e-9)

    def test_fails_when_the_first_listed_option_gets_a_bonus(self):
        # Position bias strong enough to overturn the evidence, which is exactly
        # the case that makes a Choice answer depend on dictionary ordering.
        async def go():
            transport = FakeTransport(
                signals=(Signal(r"charged twice", 1.0, label="billing"),),
                first_option_bonus=3.0,
            )
            ctx = await context(DEPARTMENT, transport=transport)
            return await option_order_flip(ctx)

        result = run(go())
        assert result.verdict == "fail"
        assert "the selection changed" in result.detail
        assert "depends on the order" in result.detail

    def test_warns_when_only_the_probability_moves(self):
        # A bonus too small to change the winner but large enough to shift mass.
        async def go():
            transport = FakeTransport(
                signals=(Signal(r"charged twice", 4.0, label="billing"),),
                first_option_bonus=1.2,
            )
            ctx = await context(DEPARTMENT, transport=transport)
            return await option_order_flip(ctx)

        result = run(go())
        assert result.verdict == "warn"
        assert "the selection held" in result.detail

    def test_skipped_for_a_noul(self):
        async def go():
            ctx = await context(REFUND)
            return await option_order_flip(ctx)

        result = run(go())
        assert result.skipped
        assert "Choice" in result.skipped_reason

    def test_skipped_when_there_are_too_few_options(self):
        async def go():
            ctx = await context(
                Choice(instructions="Which?", criteria={"a": "A", "b": "B"})
            )
            return await option_order_flip(ctx)

        result = run(go())
        assert result.skipped
        assert "three options" in result.skipped_reason

    def test_is_reproducible_under_a_seed(self):
        async def go(seed):
            transport = FakeTransport(
                signals=(Signal(r"charged twice", 4.0, label="billing"),),
                first_option_bonus=1.2,
            )
            ctx = await context(DEPARTMENT, transport=transport, seed=seed)
            return (await option_order_flip(ctx)).measurement

        assert run(go(3)) == run(go(3))


class _CoherentNoul:
    """A fixture that actually reads the criteria, so coherence can pass.

    The default fake ignores criteria entirely, which is a legitimate failure for
    the probe to catch but leaves no way to test the passing branch.
    """

    def __init__(self, probability: float, positive: str) -> None:
        self.probability = probability
        self.positive = positive
        self.calls = 0

    async def send(self, request):
        self.calls += 1
        answers = {}
        for qid, question in request.questions.items():
            criteria = question.criteria or {}
            aligned = criteria.get("true", "") == self.positive
            p = self.probability if aligned else 1.0 - self.probability
            answers[qid] = {"type": "noul", "noul": round(p, 6)}
        return {"model": "coherent", "answers": answers, "usage": {"input_tokens": 1}}

    async def aclose(self):
        return None


class TestNegationCoherence:
    def test_passes_when_the_criteria_are_respected(self):
        async def go():
            transport = _CoherentNoul(0.82, REFUND.criteria["true"])
            ctx = await context(REFUND, transport=transport)
            return await negation_coherence(ctx)

        result = run(go())
        assert result.verdict == "ok"
        assert result.measurement == pytest.approx(0.0, abs=1e-6)
        assert "close to the 1.0" in result.detail

    def test_fails_when_the_criteria_are_ignored(self):
        # The default fake keys on the state only, so swapping true and false
        # changes nothing and the two answers sum to well over 1.
        async def go():
            ctx = await context(REFUND, signals=(Signal(r"charged twice", 2.5),))
            return await negation_coherence(ctx)

        result = run(go())
        assert result.verdict == "fail"
        assert "criteria are being ignored" in result.detail
        assert "doing nothing" in result.detail

    def test_costs_one_request(self):
        async def go():
            transport = _CoherentNoul(0.82, REFUND.criteria["true"])
            ctx = await context(REFUND, transport=transport)
            return await negation_coherence(ctx)

        assert run(go()).requests == 1

    def test_skipped_for_a_choice(self):
        async def go():
            ctx = await context(DEPARTMENT)
            return await negation_coherence(ctx)

        result = run(go())
        assert result.skipped
        assert "Noul" in result.skipped_reason

    def test_skipped_without_explicit_criteria(self):
        async def go():
            ctx = await context(Noul(instructions="Is it urgent?"))
            return await negation_coherence(ctx)

        result = run(go())
        assert result.skipped
        assert "true and false criteria" in result.skipped_reason

    def test_the_measurement_is_the_signed_error_from_one(self):
        async def go():
            ctx = await context(REFUND, signals=(Signal(r"charged twice", 2.5),))
            return ctx.baseline_value, await negation_coherence(ctx)

        baseline, result = run(go())
        assert result.measurement == pytest.approx(2 * baseline - 1.0, abs=1e-6)


class TestFramings:
    def test_produces_several_wordings(self):
        assert len(framings("The customer wants a refund.")) == 3

    def test_preserves_the_original_wording_inside_each(self):
        for variant in framings("The customer wants a refund."):
            assert "customer wants a refund" in variant

    def test_does_not_double_up_the_full_stop(self):
        for variant in framings("The customer wants a refund."):
            assert ".." not in variant

    def test_tolerates_instructions_without_a_full_stop(self):
        assert framings("Is this urgent")


class TestParaphraseSpread:
    def test_passes_when_wording_does_not_matter(self):
        async def go():
            ctx = await context(REFUND, signals=(Signal(r"charged twice", 2.0),))
            return await paraphrase_spread(ctx)

        result = run(go())
        assert result.verdict == "ok"
        assert result.measurement == pytest.approx(0.0, abs=1e-9)

    def test_fails_when_a_reworded_question_answers_differently(self):
        # A question-scoped signal: the answer now depends on a phrase that only
        # appears in some framings, which is framing sensitivity by construction.
        async def go():
            ctx = await context(
                REFUND,
                signals=(
                    Signal(r"charged twice", 1.0),
                    Signal(r"only what is stated", 3.0, scope="question"),
                ),
                bias=-1.5,
            )
            return await paraphrase_spread(ctx)

        result = run(go())
        assert result.verdict == "fail"
        assert "measuring your phrasing" in result.detail

    def test_notes_when_only_mechanical_framings_were_used(self):
        async def go():
            ctx = await context(REFUND, signals=(Signal(r"charged twice", 2.0),))
            return await paraphrase_spread(ctx)

        assert "mechanical framings only" in run(go()).detail

    def test_supplied_paraphrases_are_used_verbatim(self):
        async def go():
            ctx = await context(
                REFUND,
                signals=(Signal(r"wants money returned", 3.0, scope="question"),),
                bias=-1.5,
            )
            return await paraphrase_spread(
                ctx, paraphrases=["The customer wants money returned."]
            )

        result = run(go())
        assert result.verdict == "fail"
        assert "mechanical framings only" not in result.detail
        assert result.requests == 1

    def test_skipped_for_structured_instructions(self):
        async def go():
            ctx = await context(
                Noul(instructions={"question": "refund?", "note": "x"})
            )
            return await paraphrase_spread(ctx)

        result = run(go())
        assert result.skipped
        assert "structured instructions" in result.skipped_reason

    def test_an_empty_paraphrase_list_is_skipped(self):
        async def go():
            ctx = await context(REFUND, signals=(Signal(r"charged twice", 2.0),))
            return await paraphrase_spread(ctx, paraphrases=[])

        assert run(go()).skipped is False  # falls back to mechanical framings


class TestSignalScope:
    def test_state_scope_is_the_default(self):
        assert Signal(r"x").scope == "state"

    def test_question_scope_searches_the_question_only(self):
        signal = Signal(r"needle", scope="question")
        assert signal.haystack("state text", "needle in question") == "needle in question"

    def test_both_scope_searches_the_pair(self):
        signal = Signal(r"needle", scope="both")
        haystack = signal.haystack("state", "question")
        assert "state" in haystack and "question" in haystack
