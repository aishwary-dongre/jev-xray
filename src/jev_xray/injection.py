"""Locating prompt injection by asking which part of the input did the work.

TypeSafe's own notes are explicit that state is treated as data, not as hostile:
text written to steer the model can steer it. Several projects in this ecosystem
use a decision model to approve or deny what an agent is allowed to do. If a
user-supplied field can move that decision, that is not a quirk, it is an
exploit.

Existing guardrails score *whether* a call is risky. None of them point at the
span that did the steering. Attribution already computes exactly that, so this
module is a pure function over a finished attribution — **no additional
requests**. It works with either estimator, and with Shapley it inherits correct
handling of an injection split across several sentences, which leave-one-out would
score as harmless.

The direction matters and is configurable. For a guardrail phrased as "is this
safe to execute", the dangerous movement is *toward* safe. Untrusted text arguing
that an action is fine is the attack; untrusted text arguing that it is dangerous
is, at worst, a user being cautious.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Iterable, Literal, Sequence

from .segment import Segment

__all__ = [
    "TrustBoundary",
    "InjectionFinding",
    "InjectionReport",
    "locate_injection",
    "DangerousDirection",
]

DangerousDirection = Literal["support", "oppose", "both"]

# Below this, an effect is indistinguishable from no effect and is not reported.
_MEASURABLE = 1e-4


@dataclass(frozen=True, slots=True)
class TrustBoundary:
    """Which parts of a state came from somewhere you do not control.

    Three ways to say it, because states come in different shapes:

    ``paths``
        fnmatch patterns against a segment's JSON path, so ``ticket.*`` or
        ``messages.*.body`` work. The natural choice for a structured state,
        where the trust boundary usually follows the schema.
    ``patterns``
        regexes against the segment text, for a flat text state where the only
        marker is the content itself, such as a transcript line beginning
        ``User:``.
    ``ids``
        explicit segment ids, for when you have already looked.

    A segment is untrusted if it matches any of them.
    """

    paths: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    ids: frozenset[int] = field(default_factory=frozenset)

    @classmethod
    def of(
        cls,
        *,
        paths: Iterable[str] = (),
        patterns: Iterable[str] = (),
        ids: Iterable[int] = (),
    ) -> "TrustBoundary":
        return cls(
            paths=tuple(paths), patterns=tuple(patterns), ids=frozenset(ids)
        )

    @property
    def is_empty(self) -> bool:
        return not (self.paths or self.patterns or self.ids)

    def is_untrusted(self, segment: Segment) -> bool:
        if segment.id in self.ids:
            return True
        if segment.path is not None and any(
            fnmatch(segment.path, pattern) for pattern in self.paths
        ):
            return True
        return any(
            re.search(pattern, segment.text, re.IGNORECASE) for pattern in self.patterns
        )

    def describe(self) -> str:
        parts = []
        if self.paths:
            parts.append("paths " + ", ".join(self.paths))
        if self.patterns:
            parts.append("matching " + ", ".join(self.patterns))
        if self.ids:
            parts.append("segments " + ", ".join(str(i) for i in sorted(self.ids)))
        return "; ".join(parts) or "nothing marked untrusted"


@dataclass(frozen=True, slots=True)
class InjectionFinding:
    """One untrusted segment that moved the decision."""

    segment: Segment
    effect: float
    share: float
    """This segment's share of the total absolute influence on the decision."""

    @property
    def label(self) -> str:
        return self.segment.label


@dataclass(slots=True)
class InjectionReport:
    verdict: Literal["ok", "warn", "fail"]
    detail: str
    findings: list[InjectionFinding]
    untrusted_influence: float
    total_influence: float
    dangerous_direction: DangerousDirection
    boundary: TrustBoundary
    top_overall_is_untrusted: bool = False

    @property
    def untrusted_share(self) -> float:
        if self.total_influence <= 0:
            return 0.0
        return self.untrusted_influence / self.total_influence

    @property
    def dangerous(self) -> list[InjectionFinding]:
        """Findings pushing in the direction that constitutes an attack."""
        if self.dangerous_direction == "both":
            return list(self.findings)
        if self.dangerous_direction == "oppose":
            return [f for f in self.findings if f.effect < 0]
        return [f for f in self.findings if f.effect > 0]

    def summary(self) -> str:
        lines = [
            f"  trust boundary: {self.boundary.describe()}",
            f"  untrusted text accounts for {self.untrusted_share:.0%} of the "
            f"influence on this decision",
            f"  {self.detail}",
        ]
        if self.dangerous:
            lines.append("  suspect spans, strongest first:")
            for finding in self.dangerous:
                lines.append(
                    f"    {finding.effect:+.4f}  [{finding.label}]  "
                    f"{finding.segment.preview(56)}"
                )
        return "\n".join(lines)


def locate_injection(
    attribution: object,
    boundary: TrustBoundary,
    *,
    dangerous_direction: DangerousDirection = "support",
    share_fail: float = 0.40,
    share_warn: float = 0.15,
) -> InjectionReport:
    """Report whether untrusted parts of the state drove the decision.

    Pure: consumes a finished :class:`~jev_xray.attribution.Attribution` or
    :class:`~jev_xray.shapley.ShapleyAttribution` and issues no requests.

    fail
        the single most influential segment overall is untrusted and pushes in the
        dangerous direction, or untrusted text accounts for more than
        ``share_fail`` of the total influence
    warn
        untrusted text accounts for more than ``share_warn``
    """
    effects = list(attribution.effects)  # type: ignore[attr-defined]
    if boundary.is_empty:
        return InjectionReport(
            verdict="ok",
            detail="no trust boundary supplied, so nothing was checked",
            findings=[],
            untrusted_influence=0.0,
            total_influence=sum(abs(e.signed) for e in effects),
            dangerous_direction=dangerous_direction,
            boundary=boundary,
        )

    total = sum(abs(e.signed) for e in effects)
    marked = [e for e in effects if boundary.is_untrusted(e.segment)]
    # Being untrusted is not a finding. Moving the decision is. A field the user
    # controls that measurably changed nothing is exactly what you want to see.
    untrusted = [e for e in marked if abs(e.signed) > _MEASURABLE]
    untrusted_influence = sum(abs(e.signed) for e in untrusted)

    findings = [
        InjectionFinding(
            segment=e.segment,
            effect=e.signed,
            share=(abs(e.signed) / total) if total > 0 else 0.0,
        )
        for e in sorted(untrusted, key=lambda e: abs(e.signed), reverse=True)
    ]

    ranked = sorted(effects, key=lambda e: abs(e.signed), reverse=True)
    top = ranked[0] if ranked else None
    top_is_untrusted = bool(top and boundary.is_untrusted(top.segment))

    report = InjectionReport(
        verdict="ok",
        detail="",
        findings=findings,
        untrusted_influence=untrusted_influence,
        total_influence=total,
        dangerous_direction=dangerous_direction,
        boundary=boundary,
        top_overall_is_untrusted=top_is_untrusted,
    )

    if not untrusted:
        report.detail = (
            "no untrusted segment measurably influenced the decision"
            if marked
            else "nothing in the state matched the trust boundary"
        )
        return report

    dangerous = report.dangerous
    share = report.untrusted_share

    top_dangerous = bool(
        top_is_untrusted
        and top is not None
        and _in_direction(top.signed, dangerous_direction)
    )

    if top_dangerous:
        report.verdict = "fail"
        report.detail = (
            "the single most influential span in this decision is untrusted and "
            "pushes it in the dangerous direction. treat this as an injection "
            "until proven otherwise"
        )
    elif share > share_fail and dangerous:
        report.verdict = "fail"
        report.detail = (
            f"untrusted text carries {share:.0%} of the influence, above the "
            f"{share_fail:.0%} bar, and some of it pushes in the dangerous direction"
        )
    elif share > share_warn and dangerous:
        report.verdict = "warn"
        report.detail = (
            f"untrusted text carries {share:.0%} of the influence. not conclusive, "
            f"but the decision is partly in the hands of whoever wrote it"
        )
    elif dangerous:
        report.verdict = "ok"
        report.detail = (
            f"untrusted text carries only {share:.0%} of the influence; the "
            f"decision rests on content you control"
        )
    else:
        report.verdict = "ok"
        report.detail = (
            "untrusted text influences the decision, but only against the "
            "dangerous direction, which is not an attack shape"
        )

    return report


def _in_direction(effect: float, direction: DangerousDirection) -> bool:
    if direction == "both":
        return effect != 0.0
    if direction == "oppose":
        return effect < 0.0
    return effect > 0.0
