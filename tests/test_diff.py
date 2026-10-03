"""Comparing two model versions on the same corpus.

The distinction the whole module rests on: a **flip** is a decision that changed
side, and a **shift** is a probability that moved. One flip is worth more
attention than a large average shift that crosses nothing, because a flip is a
refund that would now be declined or an action that would now be allowed.

Two fakes configured differently stand in for two model versions, which is enough
to test the comparison logic exactly — the fixture's answers are known, so the
expected flips and shifts are known too.
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
)
from jev_xray.diff import QuestionDrift, version_diff
from jev_xray.minimal import Decision

REFUND = Noul(instructions="The customer is asking for a refund.")
URGENT = Noul(instructions="This message is time sensitive.")

STATES = [
    "I would like my money back for order A-104.",
    "Just checking where my order is.",
    "The box arrived crushed. Please refund me.",
]


def run(coro):
    return asyncio.run(coro)


def client(signals=(), bias=0.0, model="v1"):
    return Client(
        FakeTransport(signals=tuple(signals), bias=bias),
        model=model,
        limiter=RateLimiter.unlimited(),
    )


def compare(left, right, states=STATES, questions=None, **kwargs):
    questions = questions or {"refund": REFUND}

    async def go():
        return await version_diff(
            left, right, states, questions, budget=Budget.unlimited(), **kwargs
        )

    return run(go())


class TestNoChange:
    def test_identical_versions_show_nothing(self):
        signals = (Signal(r"money back|refund", 2.5),)
        result = compare(
            client(signals, bias=-1.0, model="v1"),
            client(signals, bias=-1.0, model="v2"),
        )
        drift = result.by_id("refund")
        assert drift.flips == 0
        assert drift.max_abs_shift == pytest.approx(0.0)
        assert result.safe_to_migrate
        assert result.worst_verdict == "ok"

    def test_the_verdict_says_so(self):
        signals = (Signal(r"money back|refund", 2.5),)
        result = compare(
            client(signals, bias=-1.0), client(signals, bias=-1.0, model="v2")
        )
        assert "no decision flipped" in result.verdict_line()


class TestFlips:
    def test_a_decision_crossing_the_boundary_is_a_flip(self):
        # The candidate's bias pushes every answer below 0.5.
        signals = (Signal(r"money back|refund", 2.5),)
        result = compare(
            client(signals, bias=-1.0, model="v1"),
            client(signals, bias=-4.0, model="v2"),
        )
        drift = result.by_id("refund")
        assert drift.flips > 0
        assert not result.safe_to_migrate
        assert drift.verdict == "fail"

    def test_the_verdict_names_the_affected_question(self):
        signals = (Signal(r"money back|refund", 2.5),)
        result = compare(
            client(signals, bias=-1.0), client(signals, bias=-4.0, model="v2")
        )
        line = result.verdict_line()
        assert "DO NOT MIGRATE BLIND" in line
        assert "refund" in line
        assert "behaviour changes" in line

    def test_the_widest_flip_is_recorded_with_its_state(self):
        signals = (Signal(r"money back|refund", 2.5),)
        result = compare(
            client(signals, bias=-1.0), client(signals, bias=-4.0, model="v2")
        )
        drift = result.by_id("refund")
        assert drift.worst_flip is not None
        index, before, after = drift.worst_flip
        assert 0 <= index < len(STATES)
        assert before >= 0.5 > after

    def test_flip_rate_is_over_the_samples_seen(self):
        signals = (Signal(r"money back|refund", 2.5),)
        result = compare(
            client(signals, bias=-1.0), client(signals, bias=-4.0, model="v2")
        )
        drift = result.by_id("refund")
        assert drift.samples == len(STATES)
        assert drift.flip_rate == pytest.approx(drift.flips / len(STATES))

    def test_a_custom_threshold_changes_what_counts_as_a_flip(self):
        signals = (Signal(r"money back|refund", 2.5),)
        # A boundary both versions sit above means nothing flips.
        lenient = compare(
            client(signals, bias=-1.0),
            client(signals, bias=-1.2, model="v2"),
            decision=Decision(0.01),
        )
        assert lenient.total_flips == 0


class TestShift:
    def test_a_move_that_crosses_nothing_is_a_warning_not_a_failure(self):
        signals = (Signal(r"money back|refund", 2.5),)
        result = compare(
            client(signals, bias=1.0, model="v1"),
            client(signals, bias=2.2, model="v2"),
        )
        drift = result.by_id("refund")
        assert drift.flips == 0
        assert drift.max_abs_shift > 0.10
        assert drift.verdict == "warn"
        assert "no decision flipped, but" in result.verdict_line()
        assert "no longer the threshold you tuned" in result.verdict_line()

    def test_mean_shift_is_signed_so_bias_is_visible(self):
        signals = (Signal(r"money back|refund", 2.5),)
        result = compare(
            client(signals, bias=-1.0), client(signals, bias=-2.0, model="v2")
        )
        drift = result.by_id("refund")
        assert drift.mean_shift < 0
        assert drift.mean_abs_shift > 0

    def test_mean_abs_shift_ignores_direction(self):
        drift = QuestionDrift(question_id="q", samples=2, shifts=[0.2, -0.2])
        assert drift.mean_shift == pytest.approx(0.0)
        assert drift.mean_abs_shift == pytest.approx(0.2)


class TestMultipleQuestions:
    QUESTIONS = {"refund": REFUND, "urgent": URGENT}

    def test_both_questions_are_tracked(self):
        signals = (Signal(r"money back|refund", 2.5),)
        result = compare(
            client(signals, bias=-1.0),
            client(signals, bias=-1.0, model="v2"),
            questions=self.QUESTIONS,
        )
        assert {d.question_id for d in result.drifts} == {"refund", "urgent"}

    def test_all_questions_for_one_state_share_a_request(self):
        # The point of the model: asking more questions costs tokens, not calls.
        left = FakeTransport(signals=(Signal(r"refund", 2.0),))
        right = FakeTransport(signals=(Signal(r"refund", 2.0),))

        async def go():
            return await version_diff(
                Client(left, model="v1", limiter=RateLimiter.unlimited()),
                Client(right, model="v2", limiter=RateLimiter.unlimited()),
                STATES,
                self.QUESTIONS,
                budget=Budget.unlimited(),
            )

        run(go())
        assert left.calls == len(STATES)
        assert right.calls == len(STATES)

    def test_the_worst_verdict_aggregates(self):
        signals = (Signal(r"money back|refund", 2.5),)
        result = compare(
            client(signals, bias=-1.0),
            client(signals, bias=-4.0, model="v2"),
            questions=self.QUESTIONS,
        )
        assert result.worst_verdict == "fail"


class TestTargetStability:
    def test_the_axis_is_fixed_from_the_baseline(self):
        """Both versions must be read off the same axis.

        For a Choice, deriving the target from each answer separately would
        compare one option's probability against a different option's and report
        the difference as drift. The baseline's selection fixes the axis.
        """
        question = Choice(
            instructions="What does the customer want?",
            criteria={"refund": "money back", "info": "just asking"},
        )
        # The candidate prefers the other option outright.
        left = client((Signal(r"refund|money back", 3.0, label="refund"),), model="v1")
        right = client((Signal(r"refund|money back", 3.0, label="info"),), model="v2")

        result = compare(left, right, questions={"want": question})
        drift = result.by_id("want")
        # Read on the baseline's axis this is a large negative shift, and a flip.
        assert drift.mean_shift < -0.5
        assert drift.flips > 0


class TestRobustness:
    def test_a_failing_state_is_counted_and_skipped(self):
        left = FakeTransport(signals=(Signal(r"refund", 2.0),), fail_first=1)
        right = FakeTransport(signals=(Signal(r"refund", 2.0),))

        async def go():
            return await version_diff(
                Client(left, model="v1", limiter=RateLimiter.unlimited()),
                Client(right, model="v2", limiter=RateLimiter.unlimited()),
                STATES,
                {"refund": REFUND},
                budget=Budget.unlimited(),
            )

        result = run(go())
        drift = result.by_id("refund")
        assert drift.errors == 1
        assert drift.samples == len(STATES) - 1

    def test_a_budget_breach_is_fatal(self):
        signals = (Signal(r"refund", 2.0),)

        async def go():
            return await version_diff(
                client(signals, model="v1"),
                client(signals, model="v2"),
                STATES,
                {"refund": REFUND},
                budget=Budget(max_requests=2, max_usd=None),
            )

        with pytest.raises(BudgetExceeded):
            run(go())

    def test_an_empty_corpus_is_rejected(self):
        signals = (Signal(r"refund", 2.0),)

        async def go():
            return await version_diff(
                client(signals), client(signals, model="v2"), [], {"refund": REFUND}
            )

        with pytest.raises(ValueError, match="at least one state"):
            run(go())

    def test_no_questions_is_rejected(self):
        signals = (Signal(r"refund", 2.0),)

        async def go():
            return await version_diff(
                client(signals), client(signals, model="v2"), STATES, {}
            )

        with pytest.raises(ValueError, match="at least one question"):
            run(go())

    def test_an_unknown_question_id_raises(self):
        signals = (Signal(r"refund", 2.0),)
        result = compare(client(signals), client(signals, model="v2"))
        with pytest.raises(KeyError):
            result.by_id("nope")


class TestReporting:
    def test_summary_names_both_versions(self):
        signals = (Signal(r"refund", 2.0),)
        result = compare(
            client(signals, model="jev-1.13.0"),
            client(signals, model="jev-1.14.0"),
        )
        summary = result.summary()
        assert "jev-1.13.0" in summary
        assert "jev-1.14.0" in summary
        assert "per question" in summary
        assert "verdict" in summary

    def test_a_flip_shows_the_before_and_after(self):
        signals = (Signal(r"money back|refund", 2.5),)
        result = compare(
            client(signals, bias=-1.0), client(signals, bias=-4.0, model="v2")
        )
        assert "widest flip at state" in result.by_id("refund").line()

    def test_reports_what_it_spent(self):
        signals = (Signal(r"refund", 2.0),)
        result = compare(client(signals), client(signals, model="v2"))
        assert result.ledger.requests == 2 * len(STATES)
