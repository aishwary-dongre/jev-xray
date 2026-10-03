"""What changed when the model version moved under you.

TypeSafe ships `jev-latest` and `jev-preview` as moving aliases and says plainly
that a threshold tuned against one version should be pinned, because the answers
behind an alias can change without a change on your side. That is a warning with
no tool attached: nothing tells you *which* of your decisions changed, or by how
much.

This replays a corpus of states against two versions and reports the only two
things that matter.

**Flips** are decisions that changed side. These are not drift, they are
behaviour change: a refund that would now be declined, an action that would now
be allowed. One flip in a corpus is worth more attention than a large average
shift that crosses nothing.

**Shift** is how far the probability moved. It matters because a threshold sitting
inside the typical shift is no longer the threshold you tuned, even where nothing
flipped yet.

Needs no labels. It is a comparison, not an evaluation: it cannot say which
version is *right*, only what is different. That is usually the question you
actually have when a provider moves a pointer.

Cheap, because every question for one state goes in a single request. A hundred
states and eight questions is two hundred requests, not sixteen hundred.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

from .budget import Budget, BudgetExceeded, Ledger
from .client import Client
from .minimal import Decision
from .types import Question, State, Target

__all__ = ["QuestionDrift", "VersionDiff", "version_diff"]

Verdict = Literal["ok", "warn", "fail"]


@dataclass(slots=True)
class QuestionDrift:
    """How one question behaved across two versions."""

    question_id: str
    samples: int = 0
    flips: int = 0
    shifts: list[float] = field(default_factory=list)
    worst_flip: tuple[int, float, float] | None = None
    """(state index, baseline value, candidate value) for the widest flip."""
    worst_shift: tuple[int, float, float] | None = None
    errors: int = 0

    @property
    def flip_rate(self) -> float:
        return self.flips / self.samples if self.samples else 0.0

    @property
    def mean_abs_shift(self) -> float:
        return statistics.fmean(abs(s) for s in self.shifts) if self.shifts else 0.0

    @property
    def max_abs_shift(self) -> float:
        return max((abs(s) for s in self.shifts), default=0.0)

    @property
    def mean_shift(self) -> float:
        """Signed, so a systematic bias in one direction is visible."""
        return statistics.fmean(self.shifts) if self.shifts else 0.0

    @property
    def verdict(self) -> Verdict:
        if self.flips:
            return "fail"
        if self.max_abs_shift > 0.10:
            return "warn"
        return "ok"

    def line(self) -> str:
        symbol = {"ok": "ok", "warn": "!!", "fail": "XX"}[self.verdict]
        bits = [
            f"  {symbol}  {self.question_id}",
            f"      {self.flips}/{self.samples} decisions flipped",
            f"shift mean {self.mean_shift:+.4f}, worst {self.max_abs_shift:.4f}",
        ]
        line = bits[0] + "\n" + bits[1] + ", " + bits[2]
        if self.errors:
            line += f", {self.errors} errored"
        if self.worst_flip is not None:
            index, before, after = self.worst_flip
            line += (
                f"\n      widest flip at state {index}: "
                f"{before:.4f} -> {after:.4f}"
            )
        elif self.worst_shift is not None and self.max_abs_shift > 0.10:
            index, before, after = self.worst_shift
            line += (
                f"\n      largest move at state {index}: "
                f"{before:.4f} -> {after:.4f} (no flip)"
            )
        return line


@dataclass(slots=True)
class VersionDiff:
    baseline_model: str
    candidate_model: str
    decision: Decision
    drifts: list[QuestionDrift]
    states: int
    ledger: Ledger

    @property
    def total_flips(self) -> int:
        return sum(d.flips for d in self.drifts)

    @property
    def worst_verdict(self) -> Verdict:
        order = {"ok": 0, "warn": 1, "fail": 2}
        worst: Verdict = "ok"
        for drift in self.drifts:
            if order[drift.verdict] > order[worst]:
                worst = drift.verdict
        return worst

    @property
    def safe_to_migrate(self) -> bool:
        """No decision changed side and nothing moved more than 0.10."""
        return self.worst_verdict == "ok"

    def by_id(self, question_id: str) -> QuestionDrift:
        for drift in self.drifts:
            if drift.question_id == question_id:
                return drift
        raise KeyError(f"no drift recorded for {question_id!r}")

    def verdict_line(self) -> str:
        if self.total_flips:
            affected = [d.question_id for d in self.drifts if d.flips]
            return (
                f"  DO NOT MIGRATE BLIND. {self.total_flips} decision(s) changed "
                f"side across {self.states} state(s),\n  affecting: "
                f"{', '.join(affected)}. these are behaviour changes, not drift"
            )
        if self.worst_verdict == "warn":
            moved = max(self.drifts, key=lambda d: d.max_abs_shift)
            return (
                f"  no decision flipped, but {moved.question_id} moved by up to "
                f"{moved.max_abs_shift:.4f}.\n  any threshold tuned inside that "
                f"margin is no longer the threshold you tuned"
            )
        return (
            f"  no decision flipped and nothing moved more than 0.10 across "
            f"{self.states} state(s)"
        )

    def summary(self) -> str:
        lines = [
            f"{self.baseline_model}  ->  {self.candidate_model}",
            f"  states        {self.states}",
            f"  questions     {len(self.drifts)}",
            f"  boundary      {self.decision.describe()}",
            f"  spent         {self.ledger.summary()}",
            "",
            "per question",
        ]
        lines += [d.line() for d in self.drifts]
        lines += ["", "verdict", self.verdict_line()]
        return "\n".join(lines)


async def version_diff(
    baseline: Client,
    candidate: Client,
    states: Sequence[State],
    questions: Mapping[str, Question],
    *,
    decision: Decision | None = None,
    targets: Mapping[str, Target] | None = None,
    budget: Budget | None = None,
    ledger: Ledger | None = None,
) -> VersionDiff:
    """Replay states against two model versions and report what differs.

    Two requests per state regardless of how many questions you ask, since every
    question for one state is evaluated in a single call.

    The target for each question is derived from the **baseline** answer and then
    held fixed, so both versions are read off the same axis. Deriving it twice
    would compare a Choice's probability of one option against the probability of
    a different option and call the difference drift.
    """
    if not states:
        raise ValueError("version_diff needs at least one state")
    if not questions:
        raise ValueError("version_diff needs at least one question")

    decision = decision if decision is not None else Decision()
    budget = budget if budget is not None else Budget()
    ledger = ledger if ledger is not None else Ledger()

    drifts = {qid: QuestionDrift(question_id=qid) for qid in questions}
    resolved: dict[str, Target] = dict(targets) if targets else {}

    for index, state in enumerate(states):
        try:
            before = await baseline.ask(
                state, questions, ledger=ledger, budget=budget
            )
            after = await candidate.ask(
                state, questions, ledger=ledger, budget=budget
            )
        except BudgetExceeded:
            raise
        except Exception:  # noqa: BLE001 - one bad state must not lose the run
            for drift in drifts.values():
                drift.errors += 1
            continue

        for qid in questions:
            drift = drifts[qid]
            baseline_answer = before.answers[qid]

            # Fix the axis from the baseline, once, and reuse it.
            target = resolved.get(qid)
            if target is None:
                target = Target.baseline(baseline_answer)
                resolved[qid] = target

            try:
                left = target.read(baseline_answer)
                right = target.read(after.answers[qid])
            except Exception:  # noqa: BLE001 - e.g. an option that no longer exists
                drift.errors += 1
                continue

            shift = right - left
            drift.samples += 1
            drift.shifts.append(shift)

            if drift.worst_shift is None or abs(shift) > abs(
                drift.worst_shift[2] - drift.worst_shift[1]
            ):
                drift.worst_shift = (index, left, right)

            if decision.holds(left) != decision.holds(right):
                drift.flips += 1
                widest = (
                    drift.worst_flip is None
                    or abs(right - left)
                    > abs(drift.worst_flip[2] - drift.worst_flip[1])
                )
                if widest:
                    drift.worst_flip = (index, left, right)

    ledger.finish()
    return VersionDiff(
        baseline_model=baseline.model,
        candidate_model=candidate.model,
        decision=decision,
        drifts=[drifts[qid] for qid in questions],
        states=len(states),
        ledger=ledger,
    )
