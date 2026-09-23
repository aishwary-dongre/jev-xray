"""Command line entry point.

``jev-xray demo`` runs end to end with no key and no input file, against the
deterministic fake. It exists so the tool can be seen working while hosted
signups are closed, and so a contributor can verify their checkout in one
command.

``jev-xray explain`` takes a JSON file holding a ``state`` and a map of
``questions`` in the same shape the API accepts, so a question set can live in
version control and be reviewed like code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .attribution import Attribution
from .budget import Budget
from .explain import XRay
from .minimal import Decision
from .render import report, supports_color
from .transport.fake import Signal
from .types import Noul, question_from_wire

__all__ = ["main"]


_DEMO_STATE = """Hi - I ordered the blue jacket on the 3rd and it still hasn't shipped.
This is the second time I've had to write in about this order.
I've been a customer for four years and normally everything is fine.
At this point I'd just like my money back rather than wait any longer.
Your support page says orders ship within two business days."""

_DEMO_QUESTION = Noul(instructions="The customer is asking for a refund.")

# Planted evidence, so the demo has a verifiable right answer: one strong
# supporter, one mild supporter, one segment arguing the other way, and two that
# should come out inert.
_DEMO_SIGNALS = (
    Signal(pattern=r"money back", weight=2.6),
    Signal(pattern=r"second time", weight=0.4),
    Signal(pattern=r"normally everything is fine", weight=0.8, label="false"),
)
_DEMO_BIAS = -1.2


def _load_input(path: Path) -> tuple[Any, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc.strerror}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from None

    if not isinstance(payload, dict) or "state" not in payload or "questions" not in payload:
        raise SystemExit(
            f"{path} must be an object with 'state' and 'questions' keys, "
            "matching the shape the API accepts"
        )
    questions = payload["questions"]
    if not isinstance(questions, dict) or not questions:
        raise SystemExit("'questions' must be a non-empty object keyed by question id")
    return payload["state"], questions


def _split(result: Any) -> tuple[Any, Any]:
    """Separate the attribution map from the optional deep-probe extras."""
    deep = getattr(result, "attribution", None)
    if deep is not None:
        return deep, result
    return result, None


def _as_json(result: Any) -> str:
    attribution, deep = _split(result)
    payload: dict[str, Any] = {
            "model": attribution.model,
            "question_id": attribution.question_id,
            "method": "shapley" if hasattr(attribution, "exact") else "loo",
            "target": attribution.target.describe(),
            "baseline": attribution.baseline_value,
            "empty_state": attribution.empty_value,
            "ablation_mode": attribution.mode,
            "interaction_residual": attribution.interaction_residual,
            "segments": [
                {
                    "id": e.segment.id,
                    "label": e.segment.label,
                    "text": e.segment.text,
                    "effect": e.signed,
                    "ablated": e.ablated_value,
                    "std_error": e.std_error,
                }
                for e in attribution.ranked()
            ],
        "failures": len(attribution.failures),
        "cost": {
            "requests": attribution.ledger.requests,
            "avoided": attribution.ledger.avoided_requests,
            "input_tokens": attribution.ledger.input_tokens,
            "tokens_estimated": attribution.ledger.tokens_are_estimated,
            "usd": attribution.ledger.usd,
            "wall_seconds": attribution.ledger.wall_seconds,
        },
    }

    if hasattr(attribution, "exact"):
        payload["shapley"] = {
            "exact": attribution.exact,
            "coalitions_evaluated": attribution.coalitions_evaluated,
            "permutations": attribution.permutations,
            "efficiency_gap": attribution.efficiency_gap,
        }

    if deep is not None:
        payload["sufficient_evidence"] = {
            "found": deep.sufficient.found,
            "size": deep.sufficient.size,
            "of": deep.sufficient.total,
            "value": deep.sufficient.value,
            "quote": deep.sufficient.quote(),
        }
        payload["counterfactual"] = {
            "found": deep.flipping.found,
            "size": deep.flipping.size,
            "value": deep.flipping.value,
            "threshold": deep.flipping.decision.threshold,
            "remove": deep.flipping.quote(),
        }

    return json.dumps(payload, indent=2)


def _deep_blocks(deep: Any, color: bool) -> str:
    from .render import _dimmed  # local: rendering helpers are private by design

    parts = ["", _dimmed("smallest evidence that reproduces the answer", color),
             deep.sufficient.summary()]
    if deep.sufficient.found:
        parts.append(f'    "{deep.sufficient.quote()}"')

    parts += ["", _dimmed("smallest change that flips the decision", color),
              deep.flipping.summary()]
    if deep.flipping.found:
        parts.append(f'    remove: "{deep.flipping.quote()}"')

    decision = deep.flipping.decision
    prior = deep.attribution.unexplained_prior
    if decision.holds(prior) == decision.holds(deep.baseline_value):
        parts += [
            "",
            _dimmed("warning", color),
            f"  an empty state already answers {prior:.4f}, the same side of "
            f"{decision.describe()} as the full state.\n  this question cannot "
            f"discriminate: the answer is mostly a prior the input never touches.",
        ]
    return "\n".join(parts)


def _emit(result: Any, args: argparse.Namespace) -> None:
    attribution, deep = _split(result)
    if getattr(args, "html", None):
        from .report_html import to_html

        target = Path(args.html)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            to_html(
                attribution, repo_url=getattr(args, "html_link", None), deep=deep
            ),
            encoding="utf-8",
        )
        print(f"wrote {target}")
        if args.json:
            print(_as_json(result))
        return

    if args.json:
        print(_as_json(result))
        return

    color = False if args.no_color else supports_color()
    output = report(attribution, color=color)
    if deep is not None:
        output += "\n" + _deep_blocks(deep, color)
    print(output)


def _budget(args: argparse.Namespace) -> Budget:
    return Budget(max_requests=args.max_requests, max_usd=args.max_usd)


def _default_model(provider: str | None) -> str | None:
    """The model id to use when none was given explicitly.

    Naming a provider has to win over the generic ``JEV_XRAY_MODEL`` fallback.
    Otherwise a Jev version id left in .env gets sent to a gateway whose default
    model is something else entirely, and the failure looks like a bug in the
    tool rather than a mismatched model name.
    """
    if not provider:
        return None
    from .transport.http import resolve_provider

    try:
        _endpoint, model, _key_env = resolve_provider(provider)
    except ValueError:
        return None
    return model


def _run_demo(args: argparse.Namespace) -> int:
    xray = XRay.fake(_DEMO_SIGNALS, bias=_DEMO_BIAS, model="fake-jev-1")
    attribution = xray.explain(
        _DEMO_STATE,
        _DEMO_QUESTION,
        question_id="refund_requested",
        segmenter="sentence",
        mode=args.mode,
        budget=_budget(args),
    )
    if not args.json:
        print(
            "Deterministic fake, not the real model. The judgments are planted "
            "fixtures;\nthe segmentation, ablation, attribution and accounting "
            "are the real code path.\n"
        )
    _emit(attribution, args)
    return 0


def _run_explain(args: argparse.Namespace) -> int:
    state, raw_questions = _load_input(Path(args.input))

    qid = args.question
    if qid is None:
        if len(raw_questions) > 1:
            raise SystemExit(
                f"{args.input} defines {len(raw_questions)} questions; choose one with "
                f"-q, from: {', '.join(sorted(raw_questions))}"
            )
        qid = next(iter(raw_questions))
    elif qid not in raw_questions:
        raise SystemExit(
            f"no question {qid!r} in {args.input}; available: {', '.join(sorted(raw_questions))}"
        )

    try:
        question = question_from_wire(raw_questions[qid])
    except ValueError as exc:
        raise SystemExit(f"question {qid!r} is not valid: {exc}") from None

    if args.fake:
        xray = XRay.fake(model=args.model or "fake-jev-1")
    else:
        try:
            xray = XRay(
                model=args.model or _default_model(args.provider),
                endpoint=args.endpoint,
                provider=args.provider,
            )
        except ValueError as exc:
            raise SystemExit(
                f"{exc}\n\nNo hosted access yet? Run `jev-xray demo`, or point "
                "--endpoint at a local open reproduction serving /v1/systemone."
            ) from None

    extra: dict[str, Any] = {}
    if args.method == "deep":
        extra["decision"] = Decision(threshold=args.threshold)
    if args.method in ("shapley", "deep") and args.samples is not None:
        extra["samples"] = args.samples
        extra["exact"] = False

    result = xray.probe(
        state,
        question,
        method=args.method,
        question_id=qid,
        segmenter=args.segmenter,
        mode=args.mode,
        budget=_budget(args),
        concurrency=args.concurrency,
        **extra,
    )
    _emit(result, args)
    return 0


def _run_check(args: argparse.Namespace) -> int:
    import asyncio

    from .check import run_check
    from .transport.http import HttpTransport, resolve_provider

    endpoint, default_model, key_env = resolve_provider(args.provider)
    endpoint = args.endpoint or endpoint
    model = args.model or default_model

    try:
        transport = HttpTransport(
            endpoint=args.endpoint,
            provider=args.provider,
            # A diagnostic the user ran on purpose: surface the provider's own
            # error message rather than making them guess from a status code.
            include_error_detail=True,
        )
    except ValueError as exc:
        raise SystemExit(
            f"{exc}\n\nFor the Vercel gateway: create a key in the Vercel "
            f"dashboard under AI Gateway, then put it in .env as\n  {key_env}=..."
        ) from None

    async def run():
        try:
            return await run_check(transport, model, endpoint)
        finally:
            await transport.aclose()

    result = asyncio.run(run())
    print(result.report())
    return 0 if result.ok else 1


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=("delete", "mask"),
        default="delete",
        help="ablate by removing the segment, or by replacing it with a neutral placeholder",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=200,
        help="hard ceiling on requests for this run (default: 200)",
    )
    parser.add_argument(
        "--max-usd",
        type=float,
        default=0.05,
        help="hard ceiling on spend for this run (default: 0.05)",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    parser.add_argument(
        "--html",
        metavar="PATH",
        help="write a standalone HTML report instead of printing to the terminal",
    )
    parser.add_argument(
        "--html-link",
        metavar="URL",
        help="add a header bar to the HTML report linking back to the source",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-xray",
        description="Decision forensics for System One models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser(
        "demo", help="run a worked example offline, no API key required"
    )
    _add_common(demo)
    demo.set_defaults(func=_run_demo)

    explain = subparsers.add_parser(
        "explain", help="attribute one answer across the segments of its state"
    )
    explain.add_argument("input", help="JSON file with 'state' and 'questions'")
    explain.add_argument("-q", "--question", help="which question id to explain")
    explain.add_argument(
        "--segmenter",
        default="auto",
        help="auto, sentence, line, turn or field (default: auto)",
    )
    explain.add_argument("--model", help="model id; pin a version, not an alias")
    explain.add_argument("--endpoint", help="override the System One endpoint URL")
    explain.add_argument(
        "--provider",
        default="typesafe",
        help="typesafe, vercel, langsmith or local; default: typesafe",
    )
    explain.add_argument(
        "--concurrency", type=int, default=8, help="in-flight ablations (default: 8)"
    )
    explain.add_argument(
        "--method",
        choices=("loo", "shapley", "deep"),
        default="deep",
        help=(
            "loo: one request per segment, cheapest, blind to interaction. "
            "shapley: average marginal contribution, correct under interaction. "
            "deep: shapley plus minimal evidence and counterfactual (default)"
        ),
    )
    explain.add_argument(
        "--samples",
        type=int,
        help="force sampled Shapley with this many permutations instead of exact",
    )
    explain.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="the decision boundary a counterfactual has to cross (default: 0.5)",
    )
    explain.add_argument(
        "--fake",
        action="store_true",
        help="use the deterministic fake instead of a real model",
    )
    _add_common(explain)
    explain.set_defaults(func=_run_explain)

    check = subparsers.add_parser(
        "check",
        help="one cheap live request that validates the wire contract end to end",
    )
    check.add_argument(
        "--provider",
        default="vercel",
        help="typesafe (direct) or vercel (AI Gateway); default: vercel",
    )
    check.add_argument("--model", help="model id to send")
    check.add_argument("--endpoint", help="override the System One endpoint URL")
    check.set_defaults(func=_run_check)

    return parser


def _load_dotenv(path: Path | None = None) -> None:
    """Read .env into the environment for CLI use.

    Deliberately not a dependency and deliberately not done by the library: a
    program embedding jev-xray owns its own configuration. Existing environment
    variables always win, so an explicit export overrides the file.
    """
    path = path or Path.cwd() / ".env"
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip("'\"")
        if name and value and name not in os.environ:
            os.environ[name] = value


def main(argv: Sequence[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - a CLI should not traceback at a user
        print(f"jev-xray: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
