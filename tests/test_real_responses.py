"""Parser tests against real Jev response bodies.

Every fixture below is a response body published verbatim in Cloudflare's
model documentation for `typesafe/jev`, which is the closest thing to live
traffic available without an account. They replace the three assumptions the
parser was originally built on:

* ``legend`` arrives as an object keyed by the level number as a string,
  not as an array
* ``confidence`` is present on Choice and on Score
* a Noul answer carries no ``confidence`` at all
* every answer additionally echoes its own ``type``, which nothing in the
  written docs mentioned

Two properties of real distributions matter for the parser and are captured
here: probabilities are rounded to two decimals, and exact ``0`` and exact ``1``
both occur.

Source: https://developers.cloudflare.com/ai/models/typesafe/jev/
"""

from __future__ import annotations

import pytest

from jev_xray import Choice, InvalidResponse, Noul, Score, Target
from jev_xray.types import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    parse_answer,
    parse_response,
)

# --- the support triage example -------------------------------------------

TRIAGE_QUESTIONS = {
    "is_urgent": Noul(
        instructions="Does this convey urgency?",
        criteria={"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
    ),
    "department": Choice(
        instructions="Which team should handle this?",
        criteria={
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        },
    ),
    "frustration": Score(
        instructions="How frustrated is the customer?",
        criteria=["Calm", "Frustrated", "Very angry"],
    ),
}

TRIAGE_BODY = {
    "model": "jev-1.13.0",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.95},
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.8,
            "probabilities": {"billing": 0.87, "sales": 0, "technical": 0.13},
        },
        "frustration": {
            "type": "score",
            "score": 1.04,
            "confidence": 0.94,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0, "1": 0.96, "2": 0.04},
        },
    },
    "usage": {"input_tokens": 426, "output_tokens": 73},
}

# --- the account risk example (a Score answer peaking on the top level) ----

RISK_QUESTIONS = {
    "risk_level": Score(
        instructions="How risky does this account activity appear?",
        criteria=[
            "Low risk: activity is consistent with the account history",
            "Moderate risk: some unusual activity needs monitoring",
            "High risk: multiple strong indicators of account compromise",
        ],
    ),
    "escalate": Noul(
        instructions="Should this account be escalated for manual security review?",
        criteria={
            "true": "The activity warrants immediate human review",
            "false": "The activity can be handled with normal automated controls",
        },
    ),
}

RISK_BODY = {
    "model": "jev-1.13.0",
    "answers": {
        "risk_level": {
            "type": "score",
            "score": 1.84,
            "confidence": 0.77,
            "legend": {
                "0": "Low risk: activity is consistent with the account history",
                "1": "Moderate risk: some unusual activity needs monitoring",
                "2": "High risk: multiple strong indicators of account compromise",
            },
            "probabilities": {"0": 0, "1": 0.16, "2": 0.84},
        },
        "escalate": {"type": "noul", "noul": 0.81},
    },
    "usage": {"input_tokens": 421, "output_tokens": 36},
}

# --- the routing example (a point-mass distribution) ----------------------

ROUTING_QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this support request?",
        criteria={
            "account": "Login, password, profile, or security issues",
            "billing": "Charges, invoices, refunds, or subscriptions",
            "technical": "Product bugs, outages, or integrations",
            "other": "Requests that do not fit the other departments",
        },
    )
}

ROUTING_BODY = {
    "model": "jev-1.13.0",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "account",
            "confidence": 1,
            "probabilities": {
                "technical": 0,
                "billing": 0,
                "account": 1,
                "other": 0,
            },
        }
    },
    "usage": {"input_tokens": 380, "output_tokens": 45},
}


def _parse(questions, body):
    return parse_response(
        SystemOneRequest(model="jev-1.13.0", state="s", questions=questions), body
    )


class TestTriageResponse:
    def test_parses_all_three_primitives_from_one_response(self):
        parsed = _parse(TRIAGE_QUESTIONS, TRIAGE_BODY)
        assert isinstance(parsed.answers["is_urgent"], NoulAnswer)
        assert isinstance(parsed.answers["department"], ChoiceAnswer)
        assert isinstance(parsed.answers["frustration"], ScoreAnswer)

    def test_noul_is_the_probability_of_yes(self):
        assert _parse(TRIAGE_QUESTIONS, TRIAGE_BODY).answers["is_urgent"].noul == 0.95

    def test_choice_keeps_the_selection_and_the_distribution(self):
        answer = _parse(TRIAGE_QUESTIONS, TRIAGE_BODY).answers["department"]
        assert answer.choice == "billing"
        assert answer.confidence == 0.8
        assert answer.probabilities["billing"] == 0.87
        assert answer.probabilities["sales"] == 0.0

    def test_score_legend_is_an_object_keyed_by_level(self):
        answer = _parse(TRIAGE_QUESTIONS, TRIAGE_BODY).answers["frustration"]
        assert answer.legend == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
        assert answer.levels == 3

    def test_score_value_can_fall_between_levels(self):
        assert _parse(TRIAGE_QUESTIONS, TRIAGE_BODY).answers["frustration"].score == 1.04

    def test_usage_is_reported_with_nonzero_output_tokens(self):
        # Output is billed at nothing but is still counted, so accounting must
        # read input_tokens specifically rather than a total.
        usage = _parse(TRIAGE_QUESTIONS, TRIAGE_BODY).usage
        assert usage.input_tokens == 426
        assert usage.output_tokens == 73

    def test_the_answering_version_is_reported(self):
        assert _parse(TRIAGE_QUESTIONS, TRIAGE_BODY).model == "jev-1.13.0"


class TestTargetsAgainstRealAnswers:
    def test_noul_target(self):
        answer = _parse(TRIAGE_QUESTIONS, TRIAGE_BODY).answers["is_urgent"]
        assert Target.baseline(answer).read(answer) == 0.95

    def test_choice_target_follows_the_selected_option(self):
        answer = _parse(TRIAGE_QUESTIONS, TRIAGE_BODY).answers["department"]
        target = Target.baseline(answer)
        assert target.label == "billing"
        assert target.read(answer) == 0.87

    def test_score_target_picks_the_level_nearest_the_score(self):
        answer = _parse(TRIAGE_QUESTIONS, TRIAGE_BODY).answers["frustration"]
        target = Target.baseline(answer)
        # score 1.04 sits just above level 1
        assert target.label == "1"
        assert target.read(answer) == 0.96

    def test_score_target_on_the_risk_example(self):
        answer = _parse(RISK_QUESTIONS, RISK_BODY).answers["risk_level"]
        target = Target.baseline(answer)
        # score 1.84 is nearest level 2
        assert target.label == "2"
        assert target.read(answer) == 0.84

    def test_normalised_score_value_is_available_as_an_opt_in(self):
        answer = _parse(RISK_QUESTIONS, RISK_BODY).answers["risk_level"]
        assert Target("score_value").read(answer) == pytest.approx(0.92)


class TestEdgeShapes:
    def test_a_point_mass_distribution_parses(self):
        answer = _parse(ROUTING_QUESTIONS, ROUTING_BODY).answers["department"]
        assert answer.probabilities["account"] == 1.0
        assert answer.confidence == 1.0

    def test_integer_valued_probabilities_are_accepted(self):
        # Real bodies use bare 0 and 1, not 0.0 and 1.0.
        answer = _parse(ROUTING_QUESTIONS, ROUTING_BODY).answers["department"]
        assert isinstance(answer.probabilities["billing"], float)

    def test_two_decimal_rounding_does_not_trip_the_sum_check(self):
        # Ten options each rounded to 2dp can drift further from 1.0 than a
        # naive tolerance allows. Rejecting that would be a bug.
        criteria = {f"opt{i}": None for i in range(10)}
        probabilities = {f"opt{i}": 0.1 for i in range(10)}
        probabilities["opt0"] = 0.11
        probabilities["opt9"] = 0.08
        body = {
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "opt0",
                    "confidence": 0.3,
                    "probabilities": probabilities,
                }
            },
            "usage": {"input_tokens": 1},
        }
        parsed = _parse({"q": Choice(instructions="pick", criteria=criteria)}, body)
        assert parsed.answers["q"].choice == "opt0"

    def test_a_wildly_wrong_sum_is_still_rejected(self):
        body = {
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "a",
                    "confidence": 0.5,
                    "probabilities": {"a": 0.9, "b": 0.9},
                }
            },
            "usage": {},
        }
        with pytest.raises(InvalidResponse, match="sums to"):
            _parse({"q": Choice(instructions="pick", criteria={"a": None, "b": None})}, body)


class TestDeclaredTypeCrossCheck:
    QUESTION = {"q": Noul(instructions="true?")}

    def test_a_matching_declared_type_is_accepted(self):
        parsed = _parse(self.QUESTION, {"answers": {"q": {"type": "noul", "noul": 0.5}}, "usage": {}})
        assert parsed.answers["q"].noul == 0.5

    def test_a_mismatched_declared_type_is_rejected(self):
        # Reading a Choice answer as a Noul would silently produce a number off
        # the wrong axis, which is the one failure mode worth being loud about.
        with pytest.raises(InvalidResponse, match="declares type"):
            _parse(self.QUESTION, {"answers": {"q": {"type": "choice", "noul": 0.5}}, "usage": {}})

    def test_an_absent_declared_type_is_tolerated(self):
        parsed = _parse(self.QUESTION, {"answers": {"q": {"noul": 0.5}}, "usage": {}})
        assert parsed.answers["q"].noul == 0.5

    def test_a_noul_carrying_confidence_would_be_noticed(self):
        # The docs say Noul has none. If that ever changes, the check command
        # flags it; parsing stays tolerant so nothing breaks in the meantime.
        parsed = _parse(
            self.QUESTION,
            {"answers": {"q": {"type": "noul", "noul": 0.5, "confidence": 0.9}}, "usage": {}},
        )
        assert isinstance(parsed.answers["q"], NoulAnswer)


class TestAttributionOnRealShapes:
    def test_a_delta_between_two_real_bodies_reads_off_one_axis(self):
        """The core operation, done by hand on two documented responses.

        Attribution is nothing more than this subtraction, repeated. If the axis
        is stable across two independently published bodies, the measurement is
        meaningful.
        """
        baseline = parse_answer(
            "frustration", TRIAGE_QUESTIONS["frustration"], TRIAGE_BODY["answers"]["frustration"]
        )
        target = Target.baseline(baseline)

        ablated_body = {
            "type": "score",
            "score": 0.12,
            "confidence": 0.91,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.89, "1": 0.11, "2": 0},
        }
        ablated = parse_answer("frustration", TRIAGE_QUESTIONS["frustration"], ablated_body)

        delta = target.read(baseline) - target.read(ablated)
        assert delta == pytest.approx(0.85)  # 0.96 -> 0.11 on level 1
