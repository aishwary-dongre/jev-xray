"""Put your question set under test.

A question is as much a part of your application as a function, and it can
regress the same way — when you reword it, when you change what goes into the
state, or when the provider moves a version pointer under you. Unlike a function
it has no type signature to protect it, so the only way to notice is to measure.

This plugin makes that a test:

    from jev_xray.pytest_plugin import assert_stable

    def test_refund_question_is_sound(jev):
        assert_stable(
            jev,
            state=TICKET,
            question=REFUND_QUESTION,
            threshold=0.8,
        )

The ``jev`` fixture is configured from the environment and **skips** when no key
is set, so a suite carrying these tests still passes on a machine without
credentials and in a fork's CI. Nothing here runs at import time and nothing
makes a request until a test asks for the fixture.

Failures carry the full probe report rather than a bare assertion, because
"paraphrase spread 0.23" tells you what to fix and "assert False" does not.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

import pytest

from .minimal import Decision
from .types import Question, State

__all__ = ["assert_stable", "assert_no_injection", "jev"]


def pytest_configure(config: Any) -> None:
    config.addinivalue_line(
        "markers",
        "jev_live: test that calls a real System One endpoint and is skipped "
        "without credentials",
    )


def _configured() -> bool:
    return any(
        os.environ.get(name)
        for name in ("TYPESAFE_API_KEY", "AI_GATEWAY_API_KEY", "LANGSMITH_API_KEY")
    ) or bool(os.environ.get("JEV_XRAY_BASE_URL"))


@pytest.fixture
def jev() -> Any:
    """An :class:`~jev_xray.XRay` built from the environment.

    Skips when nothing is configured. A question-quality test is worth having in
    the suite permanently, and it should not turn red on a laptop with no key or
    in a pull request from a fork that cannot see your secrets.

    Set ``JEV_XRAY_PROVIDER`` to pick a preset, or ``JEV_XRAY_BASE_URL`` to point
    at a local model.
    """
    if not _configured():
        pytest.skip(
            "no System One endpoint configured; set TYPESAFE_API_KEY, "
            "AI_GATEWAY_API_KEY, LANGSMITH_API_KEY or JEV_XRAY_BASE_URL"
        )

    from .explain import XRay

    return XRay(provider=os.environ.get("JEV_XRAY_PROVIDER") or None)


@pytest.fixture
def jev_fake() -> Any:
    """An offline :class:`~jev_xray.XRay` for testing the harness itself."""
    from .explain import XRay

    return XRay.fake()


def assert_stable(
    xray: Any,
    *,
    state: State,
    question: Question,
    question_id: str = "q",
    threshold: float = 0.5,
    segmenter: str = "auto",
    paraphrases: Sequence[str] | None = None,
    require_usable: bool = True,
    max_noise: float | None = None,
    min_signal_to_noise: float | None = None,
    **kwargs: Any,
) -> Any:
    """Fail the test unless this question is safe to threshold on.

    ``require_usable``
        fail when the noise band is as wide as the usable range, or when this
        answer sits inside a noise band of the boundary. This is the default
        check and the one worth having.
    ``max_noise``
        fail when any perturbation that should not matter moves the answer by
        more than this. Tighter and more specific than ``require_usable``.
    ``min_signal_to_noise``
        fail below a ratio of input-driven movement to noise-driven movement.

    Returns the report, so a test can make further assertions about individual
    probes.
    """
    report = xray.stability(
        state,
        question,
        question_id=question_id,
        segmenter=segmenter,
        decision=Decision(threshold=threshold),
        paraphrases=paraphrases,
        **kwargs,
    )

    problems: list[str] = []

    if require_usable and report.threshold_is_meaningful() is False:
        problems.append(report.verdict_line().strip())

    band = report.noise_band
    if max_noise is not None and band is not None and band > max_noise:
        worst = max(
            (r for r in report.ran if r.noise and r.measurement is not None),
            key=lambda r: abs(r.measurement),
            default=None,
        )
        where = f" (worst: {worst.name})" if worst else ""
        problems.append(
            f"noise band {band:.4f} exceeds max_noise {max_noise:.4f}{where}"
        )

    ratio = report.signal_to_noise
    if (
        min_signal_to_noise is not None
        and ratio is not None
        and ratio < min_signal_to_noise
    ):
        problems.append(
            f"signal-to-noise {ratio:.2f} is below the required "
            f"{min_signal_to_noise:.2f}"
        )

    if problems:
        raise AssertionError(
            f"question {question_id!r} is not safe to threshold at {threshold:g}\n\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\n"
            + report.summary()
        )

    return report


def assert_no_injection(
    xray: Any,
    *,
    state: State,
    question: Question,
    untrusted: Sequence[str],
    question_id: str = "q",
    segmenter: str = "auto",
    dangerous_direction: str = "support",
    method: str = "shapley",
    match_text: bool = False,
    **kwargs: Any,
) -> Any:
    """Fail the test if text you do not control drove the decision.

    Belongs in the suite of anything using a decision model as a guardrail. Give
    it a state carrying a realistic injection attempt and it fails when the
    attempt works, which is a regression test for an exploit rather than for a
    crash.

    ``untrusted`` are JSON field paths by default, fnmatch style. Set
    ``match_text`` to treat them as regexes against segment text instead, for a
    flat transcript.
    """
    from .injection import TrustBoundary, locate_injection

    attribution = xray.probe(
        state,
        question,
        method=method,
        question_id=question_id,
        segmenter=segmenter,
        **kwargs,
    )
    attribution = getattr(attribution, "attribution", attribution)

    boundary = (
        TrustBoundary.of(patterns=untrusted)
        if match_text
        else TrustBoundary.of(paths=untrusted)
    )
    result = locate_injection(
        attribution, boundary, dangerous_direction=dangerous_direction
    )

    if result.verdict == "fail":
        raise AssertionError(
            f"untrusted input drove the decision for question {question_id!r}\n\n"
            f"  {result.detail}\n\n" + result.summary()
        )

    return result
