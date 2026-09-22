"""The wire contract and the attribution target.

Response validation is deliberately strict. An answer whose labels do not match
the criteria that were sent is a bug worth surfacing loudly, because the whole
value of a bounded answer space is that you can trust the axis you are reading.
"""

from __future__ import annotations

import pytest

from jev_xray import (
    Choice,
    ChoiceAnswer,
    Client,
    FakeTransport,
    InvalidResponse,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    Target,
    question_from_wire,
)
from jev_xray.types import SystemOneRequest, is_alias, parse_response


class TestQuestionConstruction:
    def test_choice_needs_two_options(self):
        with pytest.raises(ValueError, match="at least two options"):
            Choice(instructions="pick", criteria={"only": "one"})

    def test_score_needs_two_levels(self):
        with pytest.raises(ValueError, match="at least two levels"):
            Score(instructions="rate", criteria=["single"])

    def test_wire_shapes_match_the_api(self):
        assert Noul(instructions="is it true").wire() == {
            "type": "noul",
            "instructions": "is it true",
        }
        assert Choice(instructions="which", criteria={"a": None, "b": "bee"}).wire() == {
            "type": "choice",
            "instructions": "which",
            "criteria": {"a": None, "b": "bee"},
        }
        assert Score(instructions="how much", criteria=["low", "high"]).wire() == {
            "type": "score",
            "instructions": "how much",
            "criteria": ["low", "high"],
        }

    def test_noul_criteria_are_omitted_when_absent(self):
        assert "criteria" not in Noul(instructions="x").wire()

    def test_instructions_may_be_structured(self):
        # An object puts the question in one field and the data it refers to in
        # others, which is a documented shape.
        question = Noul(instructions={"question": "is this urgent", "message": "help!"})
        assert question.wire()["instructions"]["question"] == "is this urgent"

    def test_a_request_needs_a_question(self):
        with pytest.raises(ValueError, match="at least one question"):
            SystemOneRequest(model="m", state="s", questions={}).wire()


class TestQuestionFromWire:
    def test_round_trips_each_type(self):
        for question in (
            Noul(instructions="true?"),
            Choice(instructions="which", criteria={"a": "A", "b": "B"}),
            Score(instructions="how much", criteria=["low", "high"]),
        ):
            assert question_from_wire(question.wire()).wire() == question.wire()

    def test_rejects_an_unknown_type(self):
        with pytest.raises(ValueError, match="unknown question type"):
            question_from_wire({"type": "ranking", "instructions": "x"})

    def test_requires_instructions(self):
        with pytest.raises(ValueError, match="instructions"):
            question_from_wire({"type": "noul"})

    def test_rejects_choice_criteria_of_the_wrong_shape(self):
        with pytest.raises(ValueError, match="option -> description"):
            question_from_wire({"type": "choice", "instructions": "x", "criteria": ["a", "b"]})

    def test_rejects_score_criteria_of_the_wrong_shape(self):
        with pytest.raises(ValueError, match="ordered array"):
            question_from_wire({"type": "score", "instructions": "x", "criteria": {"a": "b"}})


def _request(question):
    return SystemOneRequest(model="m", state="s", questions={"q": question})


class TestResponseValidation:
    NOUL = Noul(instructions="true?")
    CHOICE = Choice(instructions="which", criteria={"a": "A", "b": "B"})

    def test_parses_a_noul(self):
        parsed = parse_response(
            _request(self.NOUL),
            {"model": "jev-1.13.0", "answers": {"q": {"noul": 0.72}}, "usage": {"input_tokens": 10}},
        )
        assert isinstance(parsed.answers["q"], NoulAnswer)
        assert parsed.answers["q"].noul == 0.72
        assert parsed.usage.input_tokens == 10
        assert parsed.model == "jev-1.13.0"

    def test_parses_a_choice(self):
        parsed = parse_response(
            _request(self.CHOICE),
            {
                "answers": {
                    "q": {
                        "choice": "a",
                        "probabilities": {"a": 0.8, "b": 0.2},
                        "confidence": 0.6,
                    }
                },
                "usage": {"input_tokens": 4},
            },
        )
        answer = parsed.answers["q"]
        assert isinstance(answer, ChoiceAnswer)
        assert answer.choice == "a"

    def test_parses_a_score_legend_as_object_or_array(self):
        question = Score(instructions="how much", criteria=["low", "mid", "high"])
        body = {
            "answers": {
                "q": {
                    "score": 1.4,
                    "legend": ["low", "mid", "high"],
                    "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3},
                    "confidence": 0.4,
                }
            },
            "usage": {"input_tokens": 4},
        }
        parsed = parse_response(_request(question), body)
        answer = parsed.answers["q"]
        assert isinstance(answer, ScoreAnswer)
        assert answer.legend["1"] == "mid"

        body["answers"]["q"]["legend"] = {"0": "low", "1": "mid", "2": "high"}
        assert parse_response(_request(question), body).answers["q"].legend["2"] == "high"

    def test_rejects_missing_answers(self):
        with pytest.raises(InvalidResponse, match="missing"):
            parse_response(_request(self.NOUL), {"answers": {}, "usage": {}})

    def test_rejects_unexpected_answers(self):
        with pytest.raises(InvalidResponse, match="unexpected"):
            parse_response(
                _request(self.NOUL),
                {"answers": {"q": {"noul": 0.5}, "surprise": {"noul": 0.1}}, "usage": {}},
            )

    def test_rejects_a_distribution_that_does_not_sum_to_one(self):
        with pytest.raises(InvalidResponse, match="sums to"):
            parse_response(
                _request(self.CHOICE),
                {
                    "answers": {
                        "q": {
                            "choice": "a",
                            "probabilities": {"a": 0.8, "b": 0.8},
                            "confidence": 0.5,
                        }
                    },
                    "usage": {},
                },
            )

    def test_rejects_options_that_do_not_match_the_criteria_sent(self):
        with pytest.raises(InvalidResponse, match="expected"):
            parse_response(
                _request(self.CHOICE),
                {
                    "answers": {
                        "q": {
                            "choice": "a",
                            "probabilities": {"a": 0.5, "z": 0.5},
                            "confidence": 0.5,
                        }
                    },
                    "usage": {},
                },
            )

    def test_rejects_a_choice_outside_the_distribution(self):
        with pytest.raises(InvalidResponse, match="not an option"):
            parse_response(
                _request(self.CHOICE),
                {
                    "answers": {
                        "q": {
                            "choice": "z",
                            "probabilities": {"a": 0.5, "b": 0.5},
                            "confidence": 0.5,
                        }
                    },
                    "usage": {},
                },
            )

    def test_rejects_a_probability_out_of_range(self):
        with pytest.raises(InvalidResponse, match="probability"):
            parse_response(
                _request(self.NOUL), {"answers": {"q": {"noul": 1.7}}, "usage": {}}
            )

    def test_rejects_a_non_numeric_probability(self):
        with pytest.raises(InvalidResponse, match="must be a number"):
            parse_response(
                _request(self.NOUL), {"answers": {"q": {"noul": "high"}}, "usage": {}}
            )

    def test_rejects_a_body_that_is_not_an_object(self):
        with pytest.raises(InvalidResponse, match="must be an object"):
            parse_response(_request(self.NOUL), ["not", "an", "object"])


class TestTargets:
    def test_derives_the_axis_from_a_noul(self):
        assert Target.baseline(NoulAnswer(0.4)).kind == "noul"

    def test_derives_the_selected_option_from_a_choice(self):
        answer = ChoiceAnswer("b", {"a": 0.3, "b": 0.7}, 0.5)
        target = Target.baseline(answer)
        assert (target.kind, target.label) == ("choice_prob", "b")
        assert target.read(answer) == 0.7

    def test_derives_the_nearest_level_from_a_score(self):
        answer = ScoreAnswer(
            score=1.8,
            legend={"0": "low", "1": "mid", "2": "high"},
            probabilities={"0": 0.1, "1": 0.2, "2": 0.7},
            confidence=0.6,
        )
        target = Target.baseline(answer)
        assert (target.kind, target.label) == ("score_level_prob", "2")

    def test_score_value_is_normalised_to_the_unit_interval(self):
        answer = ScoreAnswer(
            score=1.0,
            legend={"0": "low", "1": "mid", "2": "high"},
            probabilities={"0": 0.2, "1": 0.6, "2": 0.2},
            confidence=0.5,
        )
        assert Target("score_value").read(answer) == pytest.approx(0.5)

    def test_reading_the_wrong_answer_type_is_an_error(self):
        with pytest.raises(TypeError):
            Target("noul").read(ChoiceAnswer("a", {"a": 1.0}, 1.0))

    def test_reading_an_absent_option_is_an_error(self):
        with pytest.raises(InvalidResponse, match="absent"):
            Target("choice_prob", "missing").read(ChoiceAnswer("a", {"a": 1.0}, 1.0))

    def test_descriptions_are_human_readable(self):
        assert Target("noul").describe() == "P(yes)"
        assert Target("choice_prob", "billing").describe() == "P(choice=billing)"
        assert Target("score_level_prob", "2").describe() == "P(level=2)"
        assert "normalised" in Target("score_value").describe()


class TestModelPinning:
    def test_aliases_are_recognised(self):
        assert is_alias("jev-latest")
        assert is_alias("jev-preview")
        assert not is_alias("jev-1.13.0")

    def test_the_client_warns_about_a_moving_alias(self):
        with pytest.warns(UserWarning, match="moving alias"):
            Client(FakeTransport(), model="jev-latest")

    def test_a_pinned_version_is_silent(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Client(FakeTransport(), model="jev-1.13.0")
