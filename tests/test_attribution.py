"""The attribution engine.

The fake transport has *declared* causal structure, so these are not smoke
tests: the right answer is known in advance and the engine either finds it or it
does not. That is the whole reason the fake exists, and it is how the engine
stays falsifiable while hosted access is closed.
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
    JsonFieldSegmenter,
    Noul,
    RateLimiter,
    Score,
    Signal,
    Target,
    XRay,
    leave_one_out,
)

TICKET = (
    "I ordered the jacket on the 3rd and it still has not shipped. "
    "This is the second time I have written in. "
    "I have been a customer for years and normally everything is fine. "
    "At this point I would just like my money back."
)

SIGNALS = (
    Signal(pattern=r"money back", weight=2.6),
    Signal(pattern=r"second time", weight=0.4),
    Signal(pattern=r"normally everything is fine", weight=0.8, label="false"),
)

REFUND = Noul(instructions="The customer is asking for a refund.")


def explain(state=TICKET, question=REFUND, **kwargs):
    xray = XRay.fake(SIGNALS, bias=-1.2, model="fake-1")
    return xray.explain(state, question, segmenter="sentence", **kwargs)


class TestFindsPlantedEvidence:
    def test_the_strongest_segment_is_the_one_carrying_the_evidence(self):
        attribution = explain()
        assert "money back" in attribution.top(1)[0].segment.text

    def test_supporting_evidence_has_a_positive_delta(self):
        attribution = explain()
        top = attribution.top(1)[0]
        assert top.delta > 0
        assert top.supports

    def test_contrary_evidence_has_a_negative_delta(self):
        attribution = explain()
        opposing = attribution.opposing()
        assert len(opposing) == 1
        assert "normally everything is fine" in opposing[0].segment.text
        assert opposing[0].delta < 0

    def test_irrelevant_segments_measure_as_inert(self):
        attribution = explain()
        inert = [e for e in attribution.effects if e.magnitude < 1e-9]
        assert len(inert) == 1
        assert "still has not shipped" in inert[0].segment.text

    def test_ranking_orders_by_absolute_influence(self):
        magnitudes = [e.magnitude for e in explain().ranked()]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_ablated_value_and_delta_agree_with_the_baseline(self):
        attribution = explain()
        for effect in attribution.effects:
            assert effect.delta == pytest.approx(
                attribution.baseline_value - effect.ablated_value
            )


class TestRequestPlan:
    def test_issues_one_request_per_segment_plus_baseline_and_empty(self):
        transport = FakeTransport(signals=SIGNALS, bias=-1.2)
        xray = XRay(transport=transport, model="fake-1")
        attribution = xray.explain(TICKET, REFUND, segmenter="sentence")
        assert len(attribution.segments) == 4
        # 4 ablations + baseline + empty-state reference
        assert transport.calls == 6
        assert attribution.ledger.requests == 6

    def test_the_empty_reference_can_be_skipped(self):
        transport = FakeTransport(signals=SIGNALS, bias=-1.2)
        xray = XRay(transport=transport, model="fake-1")
        attribution = xray.explain(
            TICKET, REFUND, segmenter="sentence", include_empty=False
        )
        assert transport.calls == 5
        assert attribution.empty_value is None
        assert attribution.interaction_residual is None

    def test_repeated_states_are_served_from_cache(self):
        # Two identical sentences: ablating either produces the same request
        # body, so the second one must not reach the transport.
        state = "Please refund me. Please refund me. Unrelated closing line."
        transport = FakeTransport(signals=(Signal(r"refund", 1.0),))
        xray = XRay(transport=transport, model="fake-1")
        attribution = xray.explain(state, REFUND, segmenter="sentence")

        # 3 segments: baseline + 3 ablations + empty would be 5 requests, but
        # dropping either duplicate sentence yields the same state.
        assert transport.calls == 4
        assert attribution.ledger.avoided_requests == 1


class TestInteractionResidual:
    def test_independent_evidence_is_additive(self):
        attribution = explain()
        assert attribution.interaction_residual == pytest.approx(0.0, abs=0.05)
        assert attribution.is_additive()

    def test_redundant_evidence_shows_up_as_a_large_residual(self):
        # The same evidence twice. Removing either copy changes nothing because
        # the other still carries the decision, so leave-one-out reports both as
        # worthless while the total swing is large. That gap is the diagnostic.
        state = "I want my money back. Refund the money back to my card."
        attribution = explain(state=state)

        assert all(e.magnitude < 1e-9 for e in attribution.effects)
        assert attribution.total_swing > 0.4
        assert attribution.interaction_residual > 0.4
        assert not attribution.is_additive()

    def test_total_effect_sums_the_deltas(self):
        attribution = explain()
        assert attribution.total_effect == pytest.approx(
            sum(e.delta for e in attribution.effects)
        )


class TestTargets:
    def test_noul_tracks_probability_of_yes(self):
        attribution = explain()
        assert attribution.target.kind == "noul"
        assert attribution.target.describe() == "P(yes)"
        assert 0.0 <= attribution.baseline_value <= 1.0

    def test_choice_tracks_the_selected_option(self):
        question = Choice(
            instructions="Which team should handle this",
            criteria={"billing": "payments", "technical": "bugs", "sales": "pricing"},
        )
        xray = XRay.fake(
            (Signal(r"money back", 2.0, label="billing"),), model="fake-1"
        )
        attribution = xray.explain(TICKET, question, segmenter="sentence")

        assert attribution.target.kind == "choice_prob"
        assert attribution.target.label == "billing"
        assert attribution.baseline_answer.choice == "billing"
        assert "money back" in attribution.top(1)[0].segment.text

    def test_score_defaults_to_a_level_probability_not_the_interpolated_value(self):
        # Score values are documented as weakly calibrated numerically, so the
        # default target is calibrated probability mass on a level instead.
        question = Score(
            instructions="How frustrated the customer appears",
            criteria=["calm", "annoyed", "angry"],
        )
        xray = XRay.fake((Signal(r"second time", 2.0, label="2"),), model="fake-1")
        attribution = xray.explain(TICKET, question, segmenter="sentence")
        assert attribution.target.kind == "score_level_prob"

    def test_an_explicit_target_overrides_the_default(self):
        question = Choice(
            instructions="Which team should handle this",
            criteria={"billing": "payments", "technical": "bugs"},
        )
        xray = XRay.fake((Signal(r"money back", 2.0, label="billing"),), model="fake-1")
        attribution = xray.explain(
            TICKET,
            question,
            segmenter="sentence",
            target=Target("choice_prob", "technical"),
        )
        assert attribution.target.label == "technical"
        # The same evidence read from the losing option's axis flips sign.
        assert attribution.by_id(3).delta < 0


class TestStructuredState:
    STATE = {
        "ticket": {"body": "I would like my money back for the duplicate charge."},
        "notes": ["customer since 2019", "no prior refunds"],
    }

    def test_attributes_across_json_paths(self):
        xray = XRay.fake(SIGNALS, bias=-1.2, model="fake-1")
        attribution = xray.explain(
            self.STATE, REFUND, segmenter=JsonFieldSegmenter()
        )
        assert attribution.top(1)[0].segment.path == "ticket.body"

    def test_labels_are_usable_paths(self):
        xray = XRay.fake(SIGNALS, bias=-1.2, model="fake-1")
        attribution = xray.explain(self.STATE, REFUND, segmenter="field")
        assert {e.segment.label for e in attribution.effects} >= {
            "ticket.body",
            "notes.0",
        }


class TestAblationModes:
    def test_mask_mode_is_recorded_and_still_finds_the_evidence(self):
        attribution = explain(mode="mask")
        assert attribution.mode == "mask"
        assert "money back" in attribution.top(1)[0].segment.text

    def test_delete_and_mask_broadly_agree_on_the_top_segment(self):
        # Where they disagree, the effect is partly an ablation artifact. On
        # planted evidence they should not disagree.
        deleted = explain(mode="delete").top(1)[0].segment.id
        masked = explain(mode="mask").top(1)[0].segment.id
        assert deleted == masked


class TestBudgets:
    def test_preflight_refuses_a_plan_that_cannot_fit(self):
        with pytest.raises(BudgetExceeded, match="requests"):
            explain(budget=Budget(max_requests=3))

    def test_the_error_names_the_knob_that_fixes_it(self):
        with pytest.raises(BudgetExceeded, match="coarser segmenter"):
            explain(budget=Budget(max_requests=2))

    def test_a_cost_ceiling_is_enforced_before_spending(self):
        transport = FakeTransport(signals=SIGNALS)
        xray = XRay(transport=transport, model="fake-1")
        with pytest.raises(BudgetExceeded, match=r"\$"):
            xray.explain(
                TICKET, REFUND, segmenter="sentence", budget=Budget(max_usd=1e-12)
            )
        assert transport.calls == 0

    def test_spend_is_reported_from_the_services_own_token_count(self):
        attribution = explain()
        assert attribution.ledger.requests == 6
        assert attribution.ledger.input_tokens > 0
        assert attribution.ledger.usd > 0
        assert attribution.ledger.wall_seconds >= 0


class TestFailureHandling:
    def test_a_failed_ablation_degrades_the_map_instead_of_losing_the_run(self):
        # Fail after the baseline so the run is already underway.
        transport = _FlakyTransport(signals=SIGNALS, bias=-1.2, fail_on_call=3)
        xray = XRay(transport=transport, model="fake-1")
        attribution = xray.explain(TICKET, REFUND, segmenter="sentence", concurrency=1)

        assert len(attribution.failures) == 1
        assert len(attribution.effects) == 3
        assert attribution.baseline_value > 0

    def test_too_few_segments_is_a_clear_error(self):
        with pytest.raises(ValueError, match="at least two"):
            explain(state="One sentence only.")


class TestAsyncInterface:
    def test_aexplain_matches_explain(self):
        async def run():
            xray = XRay.fake(SIGNALS, bias=-1.2, model="fake-1")
            async with xray:
                return await xray.aexplain(TICKET, REFUND, segmenter="sentence")

        attribution = asyncio.run(run())
        assert "money back" in attribution.top(1)[0].segment.text

    def test_sync_explain_refuses_to_run_inside_a_loop(self):
        async def run():
            with pytest.raises(RuntimeError, match="active event loop"):
                XRay.fake(SIGNALS).explain(TICKET, REFUND, segmenter="sentence")

        asyncio.run(run())

    def test_leave_one_out_is_usable_directly(self):
        async def run():
            client = Client(
                FakeTransport(signals=SIGNALS, bias=-1.2),
                model="fake-1",
                limiter=RateLimiter.unlimited(),
            )
            return await leave_one_out(
                client, TICKET, REFUND, question_id="refund", segmenter="sentence"
            )

        attribution = asyncio.run(run())
        assert attribution.question_id == "refund"
        assert len(attribution.effects) == 4


class _FlakyTransport:
    """Wraps the fake and fails exactly one call, chosen by ordinal.

    Composition rather than inheritance: FakeTransport is a slots dataclass, so
    a subclass cannot add attributes.
    """

    def __init__(self, *, fail_on_call: int, **kwargs):
        self._inner = FakeTransport(**kwargs)
        self._fail_on_call = fail_on_call

    @property
    def calls(self) -> int:
        return self._inner.calls

    async def send(self, request):
        if self._inner.calls + 1 == self._fail_on_call:
            self._inner.calls += 1
            raise RuntimeError("simulated upstream failure")
        return await self._inner.send(request)

    async def aclose(self) -> None:
        await self._inner.aclose()
