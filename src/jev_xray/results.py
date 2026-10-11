"""The result types a renderer or consumer can be handed.

Two estimators produce two shapes. Leave-one-out yields a delta plus the single
ablated value behind it; Shapley yields an average contribution and, when
sampled, a standard error. One renderer handles both, and that worked by
duck-typing while every annotation claimed otherwise: each rendering function
said it took an ``Attribution`` and was routinely handed a
``ShapleyAttribution``.

That was not a harmless inaccuracy. A type checker reading those annotations
concluded the Shapley branch of ``render.table`` was dead, because an
``Attribution`` always carries an ``ablated_value`` and never ``None``. The branch
runs constantly. An annotation that makes live code look unreachable eventually
gets that code deleted by someone tidying up.

Both estimators already expose a common ``signed`` axis and agree on
``ablated_value`` and ``std_error`` being optional. Naming the union here is
enough to make that contract checkable without pushing a Protocol onto two
dataclasses that are otherwise plain.
"""

from __future__ import annotations

from .attribution import Attribution, SegmentEffect
from .shapley import ShapleyAttribution, ShapleyEffect

__all__ = ["AnyAttribution", "AnyEffect", "describe_failure"]

AnyAttribution = Attribution | ShapleyAttribution
"""Either estimator's completed evidence map."""

AnyEffect = SegmentEffect | ShapleyEffect
"""Either estimator's per-segment effect."""


def describe_failure(key: object) -> str:
    """Name whatever failed, across both estimators.

    Leave-one-out records a failure against the Segment whose ablation did not
    complete. Shapley records it against a coalition -- a set of kept segment
    ids -- because no single segment owns the request. Both land in
    ``failures``, and a renderer calling ``.label`` on the second one raises
    AttributeError at the moment you most want a report: after a run that partly
    failed.
    """
    label = getattr(key, "label", None)
    if isinstance(label, str):
        return label
    try:
        return f"coalition of {len(key)} segment(s)"  # type: ignore[arg-type]
    except TypeError:
        return str(key)
