"""The HTML reports.

Both reports are generated from state the caller does not necessarily trust and
then opened in a browser, so two properties are load-bearing and tested here
rather than assumed: everything is escaped, and no scripts are emitted.

Beyond that, the assertions are about whether the page leads with the right
conclusion. A reader who stops after the first box should not come away believing
the opposite of what the probes found.
"""

from __future__ import annotations

import asyncio

import pytest

from jev_xray import Budget, Client, FakeTransport, Noul, RateLimiter, Signal, to_html
from jev_xray.minimal import Decision
from jev_xray.report_html import stability_to_html
from jev_xray.shapley import shapley
from jev_xray.stability import stability

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


def client(signals=(), bias=0.0):
    return Client(
        FakeTransport(signals=tuple(signals), bias=bias),
        model="fake-1",
        limiter=RateLimiter.unlimited(),
    )


def stability_report(signals=(Signal(r"money back", 3.0),), bias=-1.5, **kwargs):
    async def go():
        return await stability(
            client(signals, bias),
            STATE,
            REFUND,
            question_id="refund",
            segmenter="sentence",
            budget=Budget.unlimited(),
            **kwargs,
        )

    return run(go())


def attribution_report(signals=(Signal(r"money back", 3.0),), bias=-1.5):
    async def go():
        return await shapley(
            client(signals, bias),
            STATE,
            REFUND,
            question_id="refund",
            segmenter="sentence",
            budget=Budget.unlimited(),
        )

    return run(go())


class TestSafety:
    HOSTILE = (
        '<script>alert(1)</script> and "quotes". '
        "I would like my money back. "
        "Another sentence here."
    )

    def test_an_attribution_report_emits_no_scripts(self):
        async def go():
            return await shapley(
                client((Signal(r"money back", 3.0),), -1.5),
                self.HOSTILE,
                REFUND,
                question_id="refund",
                segmenter="sentence",
                budget=Budget.unlimited(),
            )

        page = to_html(run(go()))
        assert "<script" not in page
        assert "&lt;script&gt;" in page

    def test_a_stability_report_emits_no_scripts(self):
        report = stability_report()
        page = stability_to_html(report)
        assert "<script" not in page.lower()

    def test_hostile_question_text_is_escaped(self):
        question = Noul(
            instructions='Is this <script>bad</script> or "fine"?',
            criteria={"true": "yes", "false": "no"},
        )

        async def go():
            return await stability(
                client((Signal(r"money back", 3.0),), -1.5),
                STATE,
                question,
                question_id="refund",
                segmenter="sentence",
                budget=Budget.unlimited(),
            )

        page = stability_to_html(run(go()))
        assert "<script>bad" not in page
        assert "&lt;script&gt;" in page

    def test_a_hostile_segment_is_escaped_in_the_heatmap(self):
        async def go():
            return await shapley(
                client((Signal(r"money back", 3.0),), -1.5),
                self.HOSTILE,
                REFUND,
                question_id="refund",
                segmenter="sentence",
                budget=Budget.unlimited(),
            )

        page = to_html(run(go()))
        assert "alert(1)" in page  # the text is shown
        assert "<script>alert" not in page  # but never as markup


class TestStabilityPage:
    def test_leads_with_the_verdict(self):
        page = stability_to_html(stability_report())
        assert page.index("verdict") < page.index("probes")

    def test_a_sound_question_is_toned_as_passing(self):
        page = stability_to_html(stability_report())
        assert 'class="verdict ok"' in page

    def test_a_question_the_input_cannot_decide_is_toned_as_failing(self):
        page = stability_to_html(stability_report(bias=4.0))
        assert 'class="verdict fail"' in page
        assert "cannot change this decision" in page

    def test_every_probe_appears_with_a_pill(self):
        report = stability_report()
        page = stability_to_html(report)
        for probe in report.results:
            assert probe.name in page
        assert 'class="pill skip"' in page  # the Choice probe, on a Noul

    def test_the_headline_numbers_are_present(self):
        report = stability_report()
        page = stability_to_html(report)
        assert f"{report.baseline_value:.4f}" in page
        assert "usable range" in page
        assert "noise band" in page

    def test_an_infinite_ratio_renders_as_a_word(self):
        page = stability_to_html(stability_report())
        assert "infinite" in page

    def test_the_model_is_named(self):
        page = stability_to_html(stability_report())
        assert "fake-1" in page

    def test_estimated_spend_is_labelled(self):
        report = stability_report()
        report.ledger.estimated_requests = 1
        assert "token counts estimated" in stability_to_html(report)

    def test_a_repo_link_adds_the_banner(self):
        page = stability_to_html(stability_report(), repo_url="https://example.test/x")
        assert "https://example.test/x" in page
        assert 'class="banner"' in page

    def test_the_title_can_be_overridden(self):
        page = stability_to_html(stability_report(), title="Quarterly review")
        assert "<title>Quarterly review</title>" in page

    def test_the_boundary_is_shown(self):
        page = stability_to_html(stability_report(decision=Decision(0.8)))
        assert "value &gt;= 0.8" in page or "value >= 0.8" in page

    def test_it_is_a_complete_document(self):
        page = stability_to_html(stability_report())
        assert page.startswith("<!doctype html>")
        assert page.rstrip().endswith("</html>")


class TestAttributionPage:
    def test_exact_shapley_is_described_as_exact(self):
        page = to_html(attribution_report())
        assert "Exact attribution" in page
        assert "not estimates" in page

    def test_a_sampled_run_is_described_as_sampled(self):
        async def go():
            return await shapley(
                client((Signal(r"money back", 3.0),), -1.5),
                STATE,
                REFUND,
                question_id="refund",
                segmenter="sentence",
                budget=Budget.unlimited(),
                exact=False,
                samples=6,
            )

        page = to_html(run(go()))
        assert "Sampled attribution" in page
        assert "error bars" in page

    def test_shapley_tooltips_do_not_claim_a_single_ablated_value(self):
        page = to_html(attribution_report())
        assert "averaged over every" in page

    def test_it_is_a_complete_document(self):
        page = to_html(attribution_report())
        assert page.startswith("<!doctype html>")
        assert page.rstrip().endswith("</html>")
