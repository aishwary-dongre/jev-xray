"""Validate the wire contract against a live endpoint, for one cheap request.

Three things in ``types.py`` were inferred from the published docs and a
community HTTP adapter rather than from real traffic: the exact shape of a Score
``legend``, whether ``confidence`` appears where expected on Choice and Score,
and whether a Noul really carries no confidence field. Those are assumptions, and
assumptions in a parser turn into wrong explanations.

This asks one Choice, one Score and one Noul against a tiny shared state. Because
every question in a request is evaluated in parallel against the same state, that
is a *single* billed request of roughly a hundred tokens, and it exercises all
three answer shapes at once. Under a thousandth of a cent to find out whether the
parser is right.

It reports the raw body before attempting to parse, so a mismatch shows you what
the service actually sent instead of just failing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from .types import (
    Choice,
    InvalidResponse,
    Noul,
    Score,
    SystemOneRequest,
    Target,
    parse_answer,
)

__all__ = ["CheckResult", "build_probe", "run_check", "PROBE_STATE"]

PROBE_STATE = (
    "A customer writes: my card was charged twice for order 5512 and "
    "I would like one of the charges reversed."
)

PROBE_QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this message",
        criteria={
            "billing": "Payments, charges and refunds",
            "technical": "Bugs and integration faults",
            "other": "Anything the options above do not cover",
        },
    ),
    "urgency": Score(
        instructions="How urgent this message is",
        criteria=[
            "No time pressure at all",
            "Should be handled this week",
            "Needs attention today",
        ],
    ),
    "refund_requested": Noul(
        instructions="The customer is asking for money to be returned.",
    ),
}


@dataclass(slots=True)
class CheckResult:
    """What one probe request told us."""

    endpoint: str
    model_sent: str
    model_answered: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Mapping[str, Any] | None = None
    parsed: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    transport_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.transport_error is None and not self.problems

    def report(self) -> str:
        lines = [f"endpoint   {self.endpoint}", f"model sent {self.model_sent}"]

        if self.transport_error:
            lines += [
                "",
                f"FAILED     {self.transport_error}",
                "",
                _troubleshooting(self.transport_error),
            ]
            return "\n".join(lines)

        lines += [
            f"model back {self.model_answered or '(not reported)'}",
            f"latency    {self.latency_ms:.0f} ms",
            f"tokens     {self.input_tokens} in / {self.output_tokens} out",
            f"cost       ${self.input_tokens * 0.042 / 1_000_000:.8f}",
            "",
            "answers",
        ]
        for qid, summary in self.parsed.items():
            lines.append(f"  {qid:<18} {summary}")

        if self.problems:
            lines += ["", "contract mismatches"]
            lines += [f"  - {p}" for p in self.problems]
            lines += [
                "",
                "raw response body, so the parser can be corrected:",
                json.dumps(self.raw, indent=2)[:4000],
            ]
        else:
            lines += ["", "wire contract matches: all three answer shapes parsed clean."]

        return "\n".join(lines)


def build_probe(model: str) -> SystemOneRequest:
    return SystemOneRequest(model=model, state=PROBE_STATE, questions=PROBE_QUESTIONS)


async def run_check(transport: Any, model: str, endpoint: str) -> CheckResult:
    """Send the probe and describe what came back. Never raises."""
    request = build_probe(model)
    result = CheckResult(endpoint=endpoint, model_sent=model)

    started = time.monotonic()
    try:
        body = await transport.send(request)
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic, it reports
        result.transport_error = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        result.latency_ms = (time.monotonic() - started) * 1000.0

    result.raw = body

    if not isinstance(body, Mapping):
        result.problems.append("response body is not a JSON object")
        return result

    result.model_answered = body.get("model")
    usage = body.get("usage") or {}
    if isinstance(usage, Mapping):
        result.input_tokens = int(usage.get("input_tokens") or 0)
        result.output_tokens = int(usage.get("output_tokens") or 0)
    else:
        result.problems.append("'usage' is not an object")

    answers = body.get("answers")
    if not isinstance(answers, Mapping):
        result.problems.append("response has no 'answers' object")
        return result

    for qid, question in PROBE_QUESTIONS.items():
        if qid not in answers:
            result.problems.append(f"no answer for {qid!r}")
            continue
        try:
            answer = parse_answer(qid, question, answers[qid])
        except (InvalidResponse, TypeError, ValueError) as exc:
            result.problems.append(f"{qid}: {exc}")
            continue

        target = Target.baseline(answer)
        result.parsed[qid] = f"{target.describe()} = {target.read(answer):.4f}"

    # The specific inferences worth confirming explicitly.
    score_raw = answers.get("urgency")
    if isinstance(score_raw, Mapping):
        legend = score_raw.get("legend")
        if legend is None:
            result.problems.append("Score answer carried no 'legend'")
        elif not isinstance(legend, (Mapping, list)):
            result.problems.append(f"Score 'legend' was a {type(legend).__name__}")
        if "confidence" not in score_raw:
            result.problems.append("Score answer carried no 'confidence'")

    choice_raw = answers.get("department")
    if isinstance(choice_raw, Mapping) and "confidence" not in choice_raw:
        result.problems.append("Choice answer carried no 'confidence'")

    noul_raw = answers.get("refund_requested")
    if isinstance(noul_raw, Mapping) and "confidence" in noul_raw:
        result.problems.append(
            "Noul answer carried a 'confidence' field, which the docs say it "
            "should not; Target may need updating to use it"
        )

    return result


def _troubleshooting(error: str) -> str:
    lowered = error.lower()

    if "credit card" in lowered or "customer_verification" in lowered:
        return (
            "The key is valid; the account is not cleared to spend. Vercel AI\n"
            "Gateway requires a card on file before it serves any model request,\n"
            "including the free credits. Nothing about jev-xray changes this.\n"
            "\n"
            "Alternatives that do not need a card:\n"
            "  - run a local open reproduction (LitJev, openjev-sglang) and point\n"
            "    --endpoint at http://localhost:8000/v1/systemone\n"
            "  - wait for console.typesafe.ai signups to reopen"
        )

    if "401" in error:
        return (
            "The key was rejected. Check it matches the endpoint:\n"
            "  AI_GATEWAY_API_KEY for --provider vercel\n"
            "  TYPESAFE_API_KEY   for --provider typesafe"
        )

    if "403" in error:
        return (
            "The key was accepted but the request was refused. This is usually an\n"
            "account state problem rather than a bad key: unverified account, no\n"
            "payment method, exhausted credits, or the model not enabled for the\n"
            "team. The provider's message above is the authoritative reason."
        )
    if "404" in error:
        return (
            "That path did not exist. If you are on the Vercel gateway, the\n"
            "TypeSafe-compatible surface is https://ai-gateway.vercel.sh/typesafe\n"
            "and jev-xray appends /v1/systemone. Override the whole URL with\n"
            "--endpoint if the gateway composes it differently."
        )
    if "400" in error or "422" in error:
        return (
            "The body was rejected. Most likely the model id: the gateway may want\n"
            "'typesafe-ai/jev' where TypeSafe direct wants 'jev-1.13.0'.\n"
            "Retry with --model typesafe-ai/jev"
        )
    if "429" in error:
        return "Rate limited before a single request landed, which suggests a low account limit."
    return "Check the endpoint is reachable and the key is set in .env."
