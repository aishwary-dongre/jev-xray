"""State-level stability probes.

Each probe applies a perturbation that *should not* change the answer, then
measures whether it did. So the tests come in pairs: a question the fake answers
stably, which must pass, and one rigged to be unstable in exactly the way the
probe looks for, which must fail. A probe that only ever returns "ok" is
worthless, and this is how we know these do not.
"""

from __future__ import annotations

import asyncio

import pytest

from jev_xray import Budget, Client, FakeTransport, Noul, RateLimiter, Signal
from jev_xray.minimal import Decision
from jev_xray.segment import LineSegmenter, SentenceSegmenter, reassemble
from jev_xray.stability import (
    DISTRACTOR,
    ProbeResult,
    build_context,
    distractor_drift,
    order_sensitivity,
    prior_saturation,
    run_probes,
)

REFUND = Noul(instructions="The customer is asking for a refund.")

STATE = (
    "The box was crushed on one corner. "
    "The jacket looks fine. "
    "I would like my money back. "
    "Let me know my options."
)


def run(coro):
    return asyncio.run(coro)


async def context(signals=(), bias=0.0, state=STATE, segmenter="sentence", **kwargs):
    client = Client(
        FakeTransport(signals=tuple(signals), bias=bias),
        model="fake-1",
        limiter=RateLimiter.unlimited(),
    )
    return await build_context(
        client,
        state,
        REFUND,
        question_id="refund",
        segmenter=segmenter,
        budget=Budget.unlimited(),
        **kwargs,
    )


class TestPriorSaturation:
    def test_passes_when_the_input_decides_the_outcome(self):
        async def go():
            ctx = await context((Signal(r"money back", 3.0),), bias=-2.0)
            return await prior_saturation(ctx)

        result = run(go())
        assert result.verdict == "ok"
        assert "usable range" in result.detail

    def test_fails_when_an_empty_state_decides_the_same_way(self):
        # A large positive bias keeps the answer above the threshold whatever the
        # input says, which is the saturated-prior failure.
        async def go():
            ctx = await context((Signal(r"money back", 0.4),), bias=3.0)
            return await prior_saturation(ctx)

        result = run(go())
        assert result.verdict == "fail"
        assert "cannot change this decision" in result.detail
        assert "decoration" in result.detail

    def test_warns_when_the_decision_flips_but_barely(self):
        # Straddles 0.5 with a very narrow total range.
        async def go():
            ctx = await context((Signal(r"money back", 0.25),), bias=-0.12)
            return await prior_saturation(ctx)

        result = run(go())
        assert result.verdict == "warn"
        assert "usable range" in result.detail

    def test_costs_one_request(self):
        async def go():
            ctx = await context((Signal(r"money back", 3.0),), bias=-2.0)
            return await prior_saturation(ctx)

        assert run(go()).requests == 1

    def test_the_measurement_is_the_swing(self):
        async def go():
            ctx = await context((Signal(r"money back", 3.0),), bias=-2.0)
            return ctx.baseline_value, await prior_saturation(ctx)

        baseline, result = run(go())
        assert result.measurement == pytest.approx(baseline - 0.1192, abs=0.01)

    def test_respects_a_custom_decision_threshold(self):
        async def go():
            ctx = await context(
                (Signal(r"money back", 3.0),), bias=-2.0, decision=Decision(0.99)
            )
            return await prior_saturation(ctx)

        # At a 0.99 boundary both the full and empty state fall below it.
        assert run(go()).verdict == "fail"


class TestDistractorDrift:
    def test_passes_when_irrelevant_text_is_ignored(self):
        async def go():
            ctx = await context((Signal(r"money back", 3.0),), bias=-1.0)
            return await distractor_drift(ctx)

        result = run(go())
        assert result.verdict == "ok"
        assert result.measurement == pytest.approx(0.0, abs=1e-9)

    def test_fails_when_the_filler_itself_moves_the_answer(self):
        # A signal that the distractor text matches: the answer now depends on
        # text that has no bearing on the judgment.
        async def go():
            ctx = await context(
                (Signal(r"money back", 1.0), Signal(r"billing cycle", 2.5)), bias=-1.0
            )
            return await distractor_drift(ctx)

        result = run(go())
        assert result.verdict == "fail"
        assert "no bearing" in result.detail
        assert "filter the state in code" in result.detail

    def test_tries_both_placements(self):
        async def go():
            ctx = await context((Signal(r"money back", 3.0),), bias=-1.0)
            return await distractor_drift(ctx)

        result = run(go())
        assert result.requests == 2
        assert "appended" in result.detail
        assert "prepended" in result.detail

    def test_a_custom_distractor_can_be_supplied(self):
        async def go():
            ctx = await context((Signal(r"zebra", 3.0),), bias=-1.0)
            return await distractor_drift(ctx, distractor="A zebra walked past.")

        assert run(go()).verdict == "fail"

    def test_skipped_for_a_structured_state(self):
        async def go():
            ctx = await context(
                state={"ticket": "I want my money back.", "note": "standard plan"},
                signals=(Signal(r"money back", 2.0),),
                segmenter="field",
            )
            return await distractor_drift(ctx)

        result = run(go())
        assert result.skipped
        assert "structured state" in result.skipped_reason
        assert result.requests == 0

    def test_the_default_distractor_is_decision_neutral_prose(self):
        assert "office hours" in DISTRACTOR
        assert "refund" not in DISTRACTOR.lower()


class TestOrderSensitivity:
    def test_passes_when_order_does_not_matter(self):
        async def go():
            ctx = await context((Signal(r"money back", 3.0),), bias=-1.0)
            return await order_sensitivity(ctx)

        result = run(go())
        assert result.verdict == "ok"
        assert result.measurement == pytest.approx(0.0, abs=1e-9)

    def test_fails_when_the_signal_depends_on_adjacency(self):
        # The pattern only matches when two sentences sit next to each other, so
        # shuffling breaks it. That is position acting as evidence.
        async def go():
            ctx = await context(
                (Signal(r"looks fine\.\s*I would like my money back", 3.0),), bias=-1.0
            )
            return await order_sensitivity(ctx)

        result = run(go())
        assert result.verdict == "fail"
        assert "position is acting as evidence" in result.detail

    def test_compares_against_a_reassembled_identity_ordering(self):
        # Joining segments normalises whitespace, which can move an answer on its
        # own, so the identity ordering goes through the same reassembly rather
        # than reusing the baseline value directly.
        async def go():
            ctx = await context((Signal(r"money back", 3.0),), bias=-1.0)
            return await order_sensitivity(ctx, shuffles=2)

        result = run(go())
        assert "across 3 orderings" in result.detail
        # For this state the reassembly is byte-identical to the original, so the
        # identity evaluation is served from cache and only the shuffles are paid
        # for. That is the cache doing its job, not a missing measurement.
        assert result.requests == 2

    def test_skipped_when_there_are_too_few_segments(self):
        async def go():
            ctx = await context(
                (Signal(r"money back", 2.0),), state="I want my money back. Thanks."
            )
            return await order_sensitivity(ctx)

        result = run(go())
        assert result.skipped
        assert "three segments" in result.skipped_reason

    def test_is_reproducible_under_a_seed(self):
        async def go(seed):
            ctx = await context(
                (Signal(r"looks fine\.\s*I would like", 3.0),), bias=-1.0, seed=seed
            )
            return (await order_sensitivity(ctx)).measurement

        assert run(go(5)) == run(go(5))


class TestReassemble:
    def test_reorders_a_text_state(self):
        segmenter = SentenceSegmenter()
        segments = segmenter.split("One. Two. Three.")
        assert reassemble(segmenter, segments, [2, 0, 1]) == "Three. One. Two."

    def test_joins_lines_with_newlines(self):
        segmenter = LineSegmenter()
        segments = segmenter.split("alpha\nbeta\ngamma")
        assert reassemble(segmenter, segments, [1, 0, 2]) == "beta\nalpha\ngamma"

    def test_rejects_a_structured_state(self):
        from jev_xray import JsonFieldSegmenter

        segmenter = JsonFieldSegmenter()
        segments = segmenter.split({"a": "one", "b": "two"})
        with pytest.raises(TypeError, match="no linear ordering"):
            reassemble(segmenter, segments, [0, 1])

    def test_rejects_unknown_ids(self):
        segmenter = SentenceSegmenter()
        segments = segmenter.split("One. Two.")
        with pytest.raises(KeyError):
            reassemble(segmenter, segments, [0, 99])


class TestProbeResult:
    def test_renders_a_line_with_a_verdict_symbol(self):
        result = ProbeResult(name="x", verdict="fail", detail="because", measurement=0.5)
        assert "XX" in result.line()
        assert "+0.5000" in result.line()
        assert "because" in result.line()

    def test_a_skipped_probe_says_why(self):
        result = ProbeResult(name="x", verdict="ok", detail="", skipped_reason="no state")
        assert result.skipped
        assert "skipped, no state" in result.line()


class TestRunProbes:
    def test_runs_every_probe(self):
        async def go():
            ctx = await context((Signal(r"money back", 3.0),), bias=-1.0)
            return await run_probes(
                ctx, [prior_saturation, distractor_drift, order_sensitivity]
            )

        results = run(go())
        assert len(results) == 3
        assert {r.name for r in results} == {
            "prior saturation",
            "distractor drift",
            "order sensitivity",
        }

    def test_one_probe_failing_does_not_lose_the_others(self):
        async def boom(ctx):
            raise RuntimeError("probe exploded")

        async def go():
            ctx = await context((Signal(r"money back", 3.0),), bias=-1.0)
            return await run_probes(ctx, [boom, prior_saturation])

        results = run(go())
        assert results[0].skipped
        assert "probe exploded" in results[0].skipped_reason
        assert results[1].verdict == "ok"

    def test_a_budget_breach_stops_the_run(self):
        async def go():
            ctx = await context((Signal(r"money back", 3.0),), bias=-1.0)
            ctx.budget = Budget(max_requests=1, max_usd=None)
            return await run_probes(
                ctx, [prior_saturation, distractor_drift, order_sensitivity]
            )

        results = run(go())
        assert results[-1].skipped
        assert "budget" in results[-1].skipped_reason
        assert len(results) < 3
