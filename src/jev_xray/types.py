"""The Jev wire contract, as typed Python.

Mirrors ``POST /v1/systemone``: a request carries one ``model``, one ``state``
and a map of named ``questions``; the response carries the answering ``model``,
one answer per question id, and token ``usage``.

Answer field names follow the published primitive reference: Choice returns
``choice`` / ``probabilities`` / ``confidence``, Score returns ``score`` /
``legend`` / ``probabilities`` / ``confidence``, and Noul returns ``noul``
alone with no separate confidence.

Response parsing here is deliberately permissive about container shapes (a
legend may arrive as an object keyed by level number or as an ordered array)
because the exact serialization was taken from the public docs and a community
HTTP adapter rather than from a live response. Verify against real traffic
before treating any of it as settled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence, Union

__all__ = [
    "State",
    "Instructions",
    "Question",
    "Choice",
    "Score",
    "Noul",
    "Answer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "NoulAnswer",
    "Usage",
    "SystemOneRequest",
    "SystemOneResponse",
    "Target",
    "InvalidResponse",
    "is_alias",
    "parse_answer",
    "parse_response",
    "question_from_wire",
]

# Jev accepts a string, a JSON object, or an array of text values as state.
# There is no image, audio or video input; non-text has to be pre-processed
# into text or structured fields by the caller.
State = Union[str, Mapping[str, Any], Sequence[Any]]

# Instructions accept the same three shapes. An object or array lets you put
# the question in one field and the data it refers to in others.
Instructions = Union[str, Mapping[str, Any], Sequence[Any]]

QuestionType = Literal["choice", "score", "noul"]

_ALIASES = frozenset({"jev-latest", "jev-preview"})


def is_alias(model: str) -> bool:
    """True for moving aliases whose target can change without notice.

    Anything tuned against an alias, thresholds especially, silently stops
    meaning what it meant when a new release ships behind the same name.
    """
    return model in _ALIASES


class InvalidResponse(ValueError):
    """The service returned something that is not a valid System One answer."""


# --------------------------------------------------------------------------
# Questions
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Choice:
    """Pick exactly one option from a named set with no order between them.

    ``criteria`` maps each option id to a description (or ``None``). Include an
    explicit escape option such as ``other`` when the list may not cover every
    input, otherwise the model has to force an answer into a bucket that does
    not fit.
    """

    instructions: Instructions
    criteria: Mapping[str, str | None]

    type: QuestionType = "choice"

    def __post_init__(self) -> None:
        if len(self.criteria) < 2:
            raise ValueError("a Choice needs at least two options")

    @property
    def options(self) -> tuple[str, ...]:
        return tuple(self.criteria)

    def wire(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": dict(self.criteria),
        }


@dataclass(frozen=True, slots=True)
class Score:
    """Place the state on an ordered rubric you define.

    ``criteria`` is the ordered list of level descriptions, lowest first. The
    returned ``score`` is a position along those levels and may fall between
    two of them.
    """

    instructions: Instructions
    criteria: Sequence[str]

    type: QuestionType = "score"

    def __post_init__(self) -> None:
        if len(self.criteria) < 2:
            raise ValueError("a Score needs at least two levels")

    @property
    def levels(self) -> int:
        return len(self.criteria)

    def wire(self) -> dict[str, Any]:
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": list(self.criteria),
        }


@dataclass(frozen=True, slots=True)
class Noul:
    """A yes/no judgment whose returned probability is itself the signal.

    ``criteria`` optionally clarifies what yes and no mean. Keep the polarity
    natural: a Noul whose ``true`` maps to "no" reads as a contradiction
    between instruction and criteria and measures worse.
    """

    instructions: Instructions
    criteria: Mapping[str, str] | None = None

    type: QuestionType = "noul"

    def wire(self) -> dict[str, Any]:
        body: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            body["criteria"] = dict(self.criteria)
        return body


Question = Union[Choice, Score, Noul]


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    choice: str
    probabilities: Mapping[str, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    score: float
    legend: Mapping[str, str]
    probabilities: Mapping[str, float]
    confidence: float

    @property
    def levels(self) -> int:
        return len(self.legend) or len(self.probabilities)


@dataclass(frozen=True, slots=True)
class NoulAnswer:
    """``noul`` is P(yes). Near 1 a strong yes, near 0 a strong no, near 0.5
    uncertain. Noul carries no separate confidence field."""

    noul: float


Answer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class SystemOneRequest:
    model: str
    state: State
    questions: Mapping[str, Question]

    def wire(self) -> dict[str, Any]:
        if not self.questions:
            raise ValueError("a request needs at least one question")
        return {
            "model": self.model,
            "state": self.state,
            "questions": {qid: q.wire() for qid, q in self.questions.items()},
        }


@dataclass(frozen=True, slots=True)
class SystemOneResponse:
    model: str
    answers: Mapping[str, Answer]
    usage: Usage


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_PROB_RANGE_EPSILON = 1e-6

# Observed responses report probabilities rounded to two decimals, including
# exact 0 and exact 1. Each rounded entry can be off by up to 0.005, so the
# acceptable error on the sum grows with the number of options. Being strict here
# would mean rejecting perfectly valid answers, which is a far worse failure than
# accepting a distribution that is a hair off.
_PROB_ROUNDING_HALF_STEP = 0.005
_PROB_SUM_FLOOR = 0.02


def _sum_tolerance(option_count: int) -> float:
    return max(_PROB_SUM_FLOOR, _PROB_ROUNDING_HALF_STEP * option_count)


def _as_probability(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InvalidResponse(f"{field} must be a number, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v) or not (
        0.0 - _PROB_RANGE_EPSILON <= v <= 1.0 + _PROB_RANGE_EPSILON
    ):
        raise InvalidResponse(f"{field} must be a probability in [0, 1], got {v}")
    return min(max(v, 0.0), 1.0)


def _as_distribution(raw: Any, field: str) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise InvalidResponse(f"{field} must be an object")
    dist = {str(k): _as_probability(v, f"{field}[{k}]") for k, v in raw.items()}
    if not dist:
        raise InvalidResponse(f"{field} must not be empty")
    total = sum(dist.values())
    tolerance = _sum_tolerance(len(dist))
    if abs(total - 1.0) > tolerance:
        raise InvalidResponse(
            f"{field} sums to {total:.6f}, expected 1.0 +/- {tolerance:.3f}"
        )
    return dist


def _check_declared_type(qid: str, question: Question, raw: Mapping[str, Any]) -> None:
    """Cross-check the answer's own ``type`` against the question we sent.

    Real answers carry a ``type`` field echoing the primitive. It is redundant
    with what we asked, which is exactly what makes it useful: if it ever
    disagrees, we are about to read a value off the wrong axis.
    """
    declared = raw.get("type")
    if declared is None:
        return
    if str(declared).lower() != question.type:
        raise InvalidResponse(
            f"answer for {qid!r} declares type {declared!r} but a "
            f"{question.type!r} question was sent"
        )


def _as_legend(raw: Any) -> dict[str, str]:
    """Accept a legend as an object keyed by level, or as an ordered array."""
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return {str(i): str(v) for i, v in enumerate(raw)}
    raise InvalidResponse("legend must be an object or an array")


def parse_answer(qid: str, question: Question, raw: Any) -> Answer:
    """Turn one raw answer body into its typed form.

    Rejects mismatches rather than coercing them: an answer whose labels do not
    match the criteria that were sent is a bug worth surfacing, not something
    to paper over.
    """
    if not isinstance(raw, Mapping):
        raise InvalidResponse(f"answer for {qid!r} must be an object")

    _check_declared_type(qid, question, raw)

    if isinstance(question, Noul):
        if "noul" not in raw:
            raise InvalidResponse(f"answer for {qid!r} is missing 'noul'")
        return NoulAnswer(noul=_as_probability(raw["noul"], f"{qid}.noul"))

    if isinstance(question, Choice):
        probabilities = _as_distribution(raw.get("probabilities"), f"{qid}.probabilities")
        expected = set(question.options)
        if set(probabilities) != expected:
            raise InvalidResponse(
                f"answer for {qid!r} covers options {sorted(probabilities)}, "
                f"expected {sorted(expected)}"
            )
        choice = raw.get("choice")
        if choice not in probabilities:
            raise InvalidResponse(f"answer for {qid!r} chose {choice!r}, which is not an option")
        return ChoiceAnswer(
            choice=str(choice),
            probabilities=probabilities,
            confidence=_as_probability(raw.get("confidence"), f"{qid}.confidence"),
        )

    if isinstance(question, Score):
        probabilities = _as_distribution(raw.get("probabilities"), f"{qid}.probabilities")
        if len(probabilities) != question.levels:
            raise InvalidResponse(
                f"answer for {qid!r} has {len(probabilities)} levels, "
                f"expected {question.levels}"
            )
        score = raw.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise InvalidResponse(f"{qid}.score must be a number")
        return ScoreAnswer(
            score=float(score),
            legend=_as_legend(raw.get("legend")),
            probabilities=probabilities,
            confidence=_as_probability(raw.get("confidence"), f"{qid}.confidence"),
        )

    raise InvalidResponse(f"unknown question type for {qid!r}")


def parse_response(request: SystemOneRequest, raw: Any) -> SystemOneResponse:
    if not isinstance(raw, Mapping):
        raise InvalidResponse("response body must be an object")

    answers_raw = raw.get("answers")
    if not isinstance(answers_raw, Mapping):
        raise InvalidResponse("response is missing an 'answers' object")

    missing = set(request.questions) - set(answers_raw)
    extra = set(answers_raw) - set(request.questions)
    if missing or extra:
        raise InvalidResponse(
            f"answer ids do not match the questions sent "
            f"(missing={sorted(missing)}, unexpected={sorted(extra)})"
        )

    answers = {
        qid: parse_answer(qid, question, answers_raw[qid])
        for qid, question in request.questions.items()
    }

    usage_raw = raw.get("usage") or {}
    if not isinstance(usage_raw, Mapping):
        raise InvalidResponse("'usage' must be an object")

    return SystemOneResponse(
        model=str(raw.get("model") or request.model),
        answers=answers,
        usage=Usage(
            input_tokens=int(usage_raw.get("input_tokens", 0)),
            output_tokens=int(usage_raw.get("output_tokens", 0)),
        ),
    )


# --------------------------------------------------------------------------
# Attribution target
# --------------------------------------------------------------------------

TargetKind = Literal["noul", "choice_prob", "score_level_prob", "score_value"]


@dataclass(frozen=True, slots=True)
class Target:
    """The single scalar an explanation tracks as the state is ablated.

    Attribution is only meaningful against a fixed axis: every ablated variant
    has to be read the same way as the baseline. ``Target`` names that axis.

    ``noul``
        P(yes) for a Noul question.
    ``choice_prob``
        Probability mass on one named option. Defaults to the option the
        baseline selected, so a positive delta means "this segment pushed the
        answer toward the decision you actually got".
    ``score_level_prob``
        Probability mass on one rubric level, defaulting to the baseline's
        nearest level. This is the default for Score because probabilities are
        calibrated while the interpolated score value is not.
    ``score_value``
        The raw fractional score, normalised to 0..1. Opt-in. The model's own
        guidance is that score levels are weak in numerical calibration and
        should not be interpolated to recover a magnitude; as a *relative*
        sensitivity signal it is still informative, but do not read the
        magnitude of a delta here as a quantity.
    """

    kind: TargetKind
    label: str | None = None

    @classmethod
    def baseline(cls, answer: Answer) -> "Target":
        """Derive the natural target from a baseline answer."""
        if isinstance(answer, NoulAnswer):
            return cls("noul")
        if isinstance(answer, ChoiceAnswer):
            return cls("choice_prob", answer.choice)
        if isinstance(answer, ScoreAnswer):
            nearest = min(
                answer.probabilities,
                key=lambda lvl: abs(_level_index(lvl) - answer.score),
            )
            return cls("score_level_prob", nearest)
        raise TypeError(f"cannot derive a target from {type(answer).__name__}")

    def read(self, answer: Answer) -> float:
        """Extract this target's scalar from an answer."""
        if self.kind == "noul":
            if not isinstance(answer, NoulAnswer):
                raise TypeError("target 'noul' needs a Noul answer")
            return answer.noul

        if self.kind == "choice_prob":
            if not isinstance(answer, ChoiceAnswer):
                raise TypeError("target 'choice_prob' needs a Choice answer")
            if self.label is None:
                raise ValueError("target 'choice_prob' needs an option label")
            if self.label not in answer.probabilities:
                raise InvalidResponse(f"option {self.label!r} absent from this answer")
            return answer.probabilities[self.label]

        if not isinstance(answer, ScoreAnswer):
            raise TypeError(f"target {self.kind!r} needs a Score answer")

        if self.kind == "score_level_prob":
            if self.label is None:
                raise ValueError("target 'score_level_prob' needs a level label")
            if self.label not in answer.probabilities:
                raise InvalidResponse(f"level {self.label!r} absent from this answer")
            return answer.probabilities[self.label]

        # score_value
        span = max(answer.levels - 1, 1)
        return min(max(answer.score / span, 0.0), 1.0)

    def describe(self) -> str:
        if self.kind == "noul":
            return "P(yes)"
        if self.kind == "choice_prob":
            return f"P(choice={self.label})"
        if self.kind == "score_level_prob":
            return f"P(level={self.label})"
        return "score (normalised)"


def _level_index(label: str) -> float:
    try:
        return float(label)
    except ValueError:
        return math.inf


def question_from_wire(body: Mapping[str, Any]) -> Question:
    """Rebuild a typed question from its JSON form.

    Lets a question set live in a file, which is how it ends up under review in
    a pull request rather than buried in a string literal.
    """
    if not isinstance(body, Mapping):
        raise ValueError("a question must be an object")

    kind = str(body.get("type", "")).lower()
    instructions = body.get("instructions")
    if instructions is None:
        raise ValueError("a question needs 'instructions'")
    criteria = body.get("criteria")

    if kind == "noul":
        if criteria is not None and not isinstance(criteria, Mapping):
            raise ValueError("Noul criteria must be an object of true/false descriptions")
        return Noul(
            instructions=instructions,
            criteria={str(k): str(v) for k, v in criteria.items()} if criteria else None,
        )

    if kind == "choice":
        if not isinstance(criteria, Mapping):
            raise ValueError("Choice criteria must be an object of option -> description")
        return Choice(
            instructions=instructions,
            criteria={str(k): (None if v is None else str(v)) for k, v in criteria.items()},
        )

    if kind == "score":
        if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
            raise ValueError("Score criteria must be an ordered array of level descriptions")
        return Score(instructions=instructions, criteria=[str(level) for level in criteria])

    raise ValueError(f"unknown question type {kind!r}; expected choice, score or noul")
