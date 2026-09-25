"""The stability verdict.

The probes are diagnostics. This is the part that answers the actual question a
developer has: can I put a threshold on this and have it mean anything?

It reduces to two quantities the probes already measured. **Usable range** is how
far the input can move the answer. **Noise band** is the largest movement caused
by something that should have moved nothing. If the band is as wide as the range,
the question is measuring its own packaging.
"""

from __future__ import annotations

import asyncio

import pytest

from jev_xray import Budget, Choice, Client, FakeTransport, Noul, RateLimiter, Signal
from jev_xray.minimal import Decision
from jev_xray.stability import (
    DEFAULT_PROBES,
    ProbeResult,
    StabilityReport,
    prior_saturation,
    stability,
)

STATE = (
    "The box was crushed on one corner. "
    "The jacket looks fine. "
    "I would like my money back. "
    "Let me know my options."
)

REFUND = Noul(
    instructions="The customer is asking for their money back.",
    criteria={"true": "Wants a refund", "false": "Wants something else"},
)


def run(coro):
    return asyncio.run(coro)


def report(signals=(), bias=0.0, *, transport=None, **kwargs):
    client = Client(
        transport or FakeTransport(signals=tuple(signals), bias=bias),
        model="fake-1",
        limiter=RateLimiter.unlimited(),
    )

    async def go():
        return await stability(
            client,
            STATE,
            REFUND,
            question_id="refund",
            segmenter="sentence",
            budget=Budget.unlimited(),
            **kwargs,
        )

    return run(go())


def synthetic(**kwargs) -> StabilityReport:
    """A report built directly, to test the aggregates in isolation."""
    from jev_xray import Ledger
    from jev_xray.types import Target

    defaults = dict(
        question_id="q",
        question=REFUND,
        model="m",
        target=Target("noul"),
        baseline_value=0.9,
        decision=Decision(0.5),
        results=[],
        ledger=Ledger(),
    )
    defaults.update(kwargs)
    return StabilityReport(**defaults)


class TestAggregates:
    def test_noise_band_takes_the_worst_noise_measurement(self):
        rep = synthetic(
            results=[
                ProbeResult("a", "ok", "", measurement=0.02, noise=True),
                ProbeResult("b", "warn", "", measurement=-0.09, noise=True),
                ProbeResult("c", "ok", "", measurement=0.50, noise=False),
            ]
        )
        assert rep.noise_band == pytest.approx(0.09)

    def test_signal_measurements_are_excluded_from_the_band(self):
        rep = synthetic(
            results=[
                ProbeResult("prior saturation", "ok", "", measurement=0.8, noise=False),
                ProbeResult("b", "ok", "", measurement=0.01, noise=True),
            ]
        )
        assert rep.noise_band == pytest.approx(0.01)
        assert rep.usable_range == pytest.approx(0.8)

    def test_skipped_probes_do_not_contribute(self):
        rep = synthetic(
            results=[
                ProbeResult("a", "ok", "", measurement=0.9, noise=True, skipped_reason="x"),
                ProbeResult("b", "ok", "", measurement=0.02, noise=True),
            ]
        )
        assert rep.noise_band == pytest.approx(0.02)

    def test_worst_verdict_ignores_skipped(self):
        rep = synthetic(
            results=[
                ProbeResult("a", "fail", "", skipped_reason="not applicable"),
                ProbeResult("b", "warn", ""),
            ]
        )
        assert rep.worst_verdict == "warn"

    def test_worst_verdict_escalates(self):
        rep = synthetic(
            results=[ProbeResult("a", "ok", ""), ProbeResult("b", "fail", "")]
        )
        assert rep.worst_verdict == "fail"

    def test_signal_to_noise_is_the_ratio(self):
        rep = synthetic(
            results=[
                ProbeResult("prior saturation", "ok", "", measurement=0.6, noise=False),
                ProbeResult("b", "ok", "", measurement=0.05, noise=True),
            ]
        )
        assert rep.signal_to_noise == pytest.approx(12.0)

    def test_zero_noise_is_infinite_signal_to_noise(self):
        rep = synthetic(
            results=[
                ProbeResult("prior saturation", "ok", "", measurement=0.6, noise=False),
                ProbeResult("b", "ok", "", measurement=0.0, noise=True),
            ]
        )
        assert rep.signal_to_noise == float("inf")

    def test_aggregates_are_none_without_the_probes(self):
        rep = synthetic()
        assert rep.noise_band is None
        assert rep.usable_range is None
        assert rep.signal_to_noise is None
        assert rep.threshold_is_meaningful() is None

    def test_margin_is_distance_to_the_boundary(self):
        assert synthetic(baseline_value=0.72, decision=Decision(0.5)).margin == pytest.approx(0.22)


class TestVerdict:
    def _rep(self, *, span, band, baseline=0.9, threshold=0.5):
        return synthetic(
            baseline_value=baseline,
            decision=Decision(threshold),
            results=[
                ProbeResult("prior saturation", "ok", "", measurement=span, noise=False),
                ProbeResult("noise", "ok", "", measurement=band, noise=True),
            ],
        )

    def test_usable_when_signal_beats_noise_and_the_margin_is_clear(self):
        rep = self._rep(span=0.7, band=0.02)
        assert rep.threshold_is_meaningful() is True
        assert "usable" in rep.verdict_line()
        assert "signal-to-noise" in rep.verdict_line()

    def test_not_usable_when_noise_matches_the_signal(self):
        rep = self._rep(span=0.03, band=0.05)
        assert rep.threshold_is_meaningful() is False
        assert "NOT USABLE" in rep.verdict_line()
        assert "measures its own" in rep.verdict_line()

    def test_unsafe_when_the_answer_sits_inside_the_noise_band(self):
        # Plenty of signal overall, but this answer is 0.01 from the boundary
        # while noise alone moves it 0.06.
        rep = self._rep(span=0.7, band=0.06, baseline=0.51, threshold=0.5)
        assert rep.threshold_is_meaningful() is False
        assert "NOT SAFE" in rep.verdict_line()
        assert "either way" in rep.verdict_line()

    def test_verdict_unavailable_without_measurements(self):
        assert "unavailable" in synthetic().verdict_line()

    def test_infinite_ratio_renders_as_a_word(self):
        rep = self._rep(span=0.7, band=0.0)
        assert "infinite" in rep.verdict_line()


class TestSuite:
    def test_runs_every_default_probe(self):
        rep = report((Signal(r"money back", 3.0),), bias=-1.5)
        assert len(rep.results) == len(DEFAULT_PROBES)

    def test_a_clean_question_comes_back_usable(self):
        rep = report((Signal(r"money back", 3.0),), bias=-1.5)
        assert rep.usable_range > 0.5
        assert rep.noise_band == pytest.approx(0.0, abs=1e-9)
        assert rep.threshold_is_meaningful() is True

    def test_a_saturated_question_fails(self):
        rep = report((Signal(r"money back", 0.3),), bias=4.0)
        saturation = next(r for r in rep.results if r.name == "prior saturation")
        assert saturation.verdict == "fail"
        assert rep.worst_verdict == "fail"

    def test_the_choice_probe_is_skipped_for_a_noul(self):
        rep = report((Signal(r"money back", 3.0),), bias=-1.5)
        option = next(r for r in rep.results if r.name == "option order flip")
        assert option.skipped

    def test_reports_what_it_spent(self):
        rep = report((Signal(r"money back", 3.0),), bias=-1.5)
        assert rep.ledger.requests > 0
        assert "requests" in rep.ledger.summary()

    def test_the_suite_is_cheap(self):
        # Cheaper than one exact Shapley run over six segments, which is the point:
        # you can afford to check every question you write.
        rep = report((Signal(r"money back", 3.0),), bias=-1.5)
        assert rep.ledger.requests <= 16

    def test_probes_can_be_overridden(self):
        rep = report((Signal(r"money back", 3.0),), bias=-1.5, probes=[prior_saturation])
        assert len(rep.results) == 1
        assert rep.results[0].name == "prior saturation"

    def test_summary_includes_the_probes_and_the_verdict(self):
        summary = report((Signal(r"money back", 3.0),), bias=-1.5).summary()
        assert "stability" in summary
        assert "probes" in summary
        assert "verdict" in summary
        assert "prior saturation" in summary

    def test_is_reproducible_under_a_seed(self):
        a = report((Signal(r"money back", 3.0),), bias=-1.5, seed=9)
        b = report((Signal(r"money back", 3.0),), bias=-1.5, seed=9)
        assert [r.measurement for r in a.results] == [r.measurement for r in b.results]

    def test_a_choice_question_runs_the_option_probe(self):
        question = Choice(
            instructions="Which team?",
            criteria={"billing": "money", "technical": "bugs", "sales": "pricing"},
        )
        client = Client(
            FakeTransport(
                signals=(Signal(r"money back", 2.0, label="billing"),),
                first_option_bonus=3.0,
            ),
            model="fake-1",
            limiter=RateLimiter.unlimited(),
        )

        async def go():
            return await stability(
                client,
                STATE,
                question,
                question_id="dept",
                segmenter="sentence",
                budget=Budget.unlimited(),
            )

        rep = run(go())
        option = next(r for r in rep.results if r.name == "option order flip")
        assert not option.skipped
        assert option.verdict == "fail"

    def test_a_failing_probe_is_named_even_when_wrapped(self):
        async def boom(ctx):
            raise RuntimeError("nope")

        from functools import partial

        rep = report(
            (Signal(r"money back", 3.0),), bias=-1.5, probes=[partial(boom)]
        )
        assert rep.results[0].name == "boom"
        assert "nope" in rep.results[0].skipped_reason
