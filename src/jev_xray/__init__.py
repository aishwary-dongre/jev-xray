"""jev-xray: decision forensics for System One models.

Jev returns a number and, by construction, cannot tell you why. This package
measures the why instead of asking for it: ablate part of the state, re-ask the
identical question, and read how far the answer moved.

    from jev_xray import XRay, Noul

    xray = XRay(model="jev-1.13.0")
    exp = xray.explain(ticket, Noul(instructions="The customer is asking for a refund."))

    exp.baseline_value      # 0.72
    exp.top(3)              # the three segments that moved it most
    exp.ledger.summary()    # what that cost

No hosted access? ``XRay.fake(...)`` and ``jev-xray demo`` run the whole path
offline against a deterministic fixture.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

from .attribution import Attribution, SegmentEffect, leave_one_out
from .budget import (
    PRICE_USD_PER_INPUT_TOKEN,
    Budget,
    BudgetExceeded,
    Ledger,
    RateLimiter,
    estimate_tokens,
)
from .cache import DiskCache, MemoryCache, NullCache
from .client import DEFAULT_MODEL, Client
from .explain import XRay
from .render import heatmap, report, table
from .report_html import to_html
from .segment import (
    JsonFieldSegmenter,
    LineSegmenter,
    Segment,
    SentenceSegmenter,
    TurnSegmenter,
    auto_segmenter,
    get_segmenter,
)
from .transport import FakeTransport, HttpTransport, Signal, Transport, TransportError
from .types import (
    Choice,
    ChoiceAnswer,
    InvalidResponse,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    Target,
    question_from_wire,
)

__all__ = [
    "__version__",
    # questions and answers
    "Choice",
    "Score",
    "Noul",
    "ChoiceAnswer",
    "ScoreAnswer",
    "NoulAnswer",
    "Target",
    "question_from_wire",
    "InvalidResponse",
    # probes
    "XRay",
    "leave_one_out",
    "Attribution",
    "SegmentEffect",
    # segmentation
    "Segment",
    "SentenceSegmenter",
    "LineSegmenter",
    "TurnSegmenter",
    "JsonFieldSegmenter",
    "auto_segmenter",
    "get_segmenter",
    # plumbing
    "Client",
    "DEFAULT_MODEL",
    "Transport",
    "TransportError",
    "HttpTransport",
    "FakeTransport",
    "Signal",
    "Budget",
    "BudgetExceeded",
    "Ledger",
    "RateLimiter",
    "estimate_tokens",
    "PRICE_USD_PER_INPUT_TOKEN",
    "MemoryCache",
    "DiskCache",
    "NullCache",
    # rendering
    "report",
    "heatmap",
    "table",
    "to_html",
]
