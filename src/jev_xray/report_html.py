"""A self-contained HTML report for one explanation.

The terminal heatmap is for the person running the probe. This is for everyone
else: the colleague reviewing a wrong decision, the reviewer on a pull request,
the ticket where someone asks why a refund was declined.

Design constraints, all deliberate:

* one file, no external assets, no JavaScript. It has to survive being emailed,
  committed, attached to an incident, or opened in two years.
* every piece of model input is HTML-escaped. A report is often generated from
  exactly the content you least trust, and a prompt-injection locator that is
  itself an injection vector would be embarrassing.
* the diagnostics are as prominent as the answer. A heatmap that quietly hides
  a large interaction residual is worse than no heatmap, because it looks
  authoritative while being unreliable.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

from .attribution import Attribution, SegmentEffect

__all__ = ["to_html"]

_EPSILON = 1e-4

_CSS = """
:root {
  --ink: #1a1d21;
  --muted: #6b7280;
  --line: #e5e7eb;
  --bg: #ffffff;
  --panel: #f9fafb;
  --pos: 34, 160, 90;
  --neg: 205, 60, 55;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px;
  background: var(--bg); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 900px; margin: 0 auto; }
h1 { font-size: 19px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 {
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin: 32px 0 12px; font-weight: 600;
}
.sub { color: var(--muted); font-size: 13px; margin-bottom: 28px; }
.sub code { background: var(--panel); padding: 1px 5px; border-radius: 3px; }

.headline { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 4px; }
.stat {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 8px; padding: 12px 16px; min-width: 132px;
}
.stat .k {
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); font-weight: 600;
}
.stat .v { font-size: 22px; font-weight: 650; font-variant-numeric: tabular-nums; }
.stat .v.sm { font-size: 15px; font-weight: 500; }

.state {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 20px 22px; white-space: pre-wrap; word-wrap: break-word;
  font: 15px/1.85 ui-monospace, SFMono-Regular, Menlo, monospace;
}
.seg { border-radius: 3px; padding: 1px 2px; }
.seg.inert { color: #9aa1aa; }

.legend { display: flex; gap: 18px; align-items: center; margin-top: 12px;
          font-size: 12px; color: var(--muted); flex-wrap: wrap; }
.swatch { display: inline-block; width: 26px; height: 11px;
          border-radius: 2px; vertical-align: -1px; margin-right: 6px; }

table.bars { width: 100%; border-collapse: collapse; }
table.bars td { padding: 5px 0; vertical-align: middle; border: 0; }
td.lbl { width: 172px; font-size: 13px; color: var(--muted);
         white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
         padding-right: 14px; }
td.num { width: 78px; text-align: right; font-size: 13px;
         font-variant-numeric: tabular-nums; padding-left: 12px; }
td.txt { font-size: 13px; color: var(--muted); padding-left: 14px;
         max-width: 300px; overflow: hidden; text-overflow: ellipsis;
         white-space: nowrap; }
.track { display: flex; height: 18px; }
.half { width: 50%; display: flex; }
.half.left { justify-content: flex-end; border-right: 1px solid var(--line); }
.half.right { justify-content: flex-start; }
.bar { height: 100%; border-radius: 2px; min-width: 1px; }
.bar.pos { background: rgba(var(--pos), 0.8); }
.bar.neg { background: rgba(var(--neg), 0.8); }

.note {
  border-left: 3px solid #d97706; background: #fffbeb;
  padding: 14px 18px; border-radius: 0 6px 6px 0; font-size: 14px;
}
.note.ok { border-left-color: #059669; background: #ecfdf5; }
.note b { font-weight: 650; }
.note p { margin: 0 0 8px; }
.note p:last-child { margin: 0; }

footer { margin-top: 36px; padding-top: 16px; border-top: 1px solid var(--line);
         font-size: 12px; color: var(--muted); }
footer code { background: var(--panel); padding: 1px 5px; border-radius: 3px; }
"""


def _shade(effect: SegmentEffect, strongest: float) -> str:
    """Background colour for one segment, scaled against the strongest effect."""
    if strongest <= _EPSILON or effect.magnitude < _EPSILON:
        return ""
    share = min(1.0, effect.magnitude / strongest)
    alpha = 0.12 + 0.5 * share
    channel = "--pos" if effect.signed > 0 else "--neg"
    return f"background: rgba(var({channel}), {alpha:.2f});"


def _heatmap(attribution: Attribution, strongest: float) -> str:
    """The state with each segment tinted by its influence. Text states only."""
    text = attribution.state
    if not isinstance(text, str):
        return ""

    effects = {e.segment.id: e for e in attribution.effects}
    out: list[str] = []
    cursor = 0

    for segment in sorted(attribution.segments, key=lambda s: s.start or 0):
        start, end = segment.start or 0, segment.end or 0
        if start > cursor:
            out.append(html.escape(text[cursor:start]))

        body = html.escape(text[start:end])
        effect = effects.get(segment.id)
        if effect is None:
            out.append(body)
        elif effect.magnitude < _EPSILON:
            out.append(f'<span class="seg inert" title="no measurable effect">{body}</span>')
        else:
            if effect.ablated_value is not None:
                tip = (
                    f"{segment.label}: removing this moves "
                    f"{attribution.target.describe()} by {-effect.signed:+.4f} "
                    f"(to {effect.ablated_value:.4f})"
                )
            else:
                tip = (
                    f"{segment.label}: contributes {effect.signed:+.4f} to "
                    f"{attribution.target.describe()}, averaged over every "
                    f"combination it appears in"
                )
            out.append(
                f'<span class="seg" style="{_shade(effect, strongest)}" '
                f'title="{html.escape(tip)}">{body}</span>'
            )
        cursor = end

    if cursor < len(text):
        out.append(html.escape(text[cursor:]))

    return f'<div class="state">{"".join(out)}</div>'


def _bars(attribution: Attribution, strongest: float, limit: int) -> str:
    rows: list[str] = []
    for effect in attribution.ranked()[:limit]:
        share = 0.0 if strongest <= _EPSILON else min(1.0, effect.magnitude / strongest)
        width = f"{share * 100:.1f}%"
        positive = effect.signed > 0

        left = f'<div class="bar neg" style="width:{width}"></div>' if not positive and share else ""
        right = f'<div class="bar pos" style="width:{width}"></div>' if positive and share else ""

        rows.append(
            "<tr>"
            f'<td class="lbl">{html.escape(effect.segment.label)}</td>'
            '<td><div class="track">'
            f'<div class="half left">{left}</div>'
            f'<div class="half right">{right}</div>'
            "</div></td>"
            f'<td class="num">{effect.signed:+.4f}</td>'
            f'<td class="txt">{html.escape(effect.segment.preview(64))}</td>'
            "</tr>"
        )
    return f'<table class="bars">{"".join(rows)}</table>'


def _diagnostics(attribution: Attribution) -> str:
    if getattr(attribution, "exact", False):
        return (
            f'<div class="note ok"><p><b>Exact attribution.</b> Every one of the '
            f"{attribution.coalitions_evaluated} possible combinations of this "
            f"evidence was evaluated, so these values are not estimates. They sum "
            f"to the total swing by construction &mdash; the "
            f"{attribution.efficiency_gap:+.4f} residual is floating-point error, "
            f"not uncertainty.</p><p>Interacting and redundant evidence is credited "
            f"correctly here, which leave-one-out cannot do.</p></div>"
        )

    if getattr(attribution, "permutations", 0):
        return (
            f'<div class="note"><p><b>Sampled attribution.</b> Estimated from '
            f"{attribution.permutations} random orderings over "
            f"{attribution.coalitions_evaluated} combinations, because exact "
            f"enumeration was too expensive. The per-segment error bars show how "
            f"far from settled each value is; a residual of "
            f"{attribution.efficiency_gap:+.4f} indicates overall convergence.</p>"
            f"</div>"
        )

    residual = attribution.interaction_residual
    if residual is None:
        return (
            '<div class="note"><p><b>No empty-state reference.</b> Without it there '
            "is no way to tell whether these effects account for the decision. "
            "Re-run with <code>include_empty=True</code>.</p></div>"
        )

    if abs(residual) <= 0.05:
        return (
            f'<div class="note ok"><p><b>These effects are additive.</b> '
            f"The individual deltas sum to {attribution.total_effect:+.4f}, against a "
            f"total swing of {attribution.total_swing:+.4f} between the full and empty "
            f"state. A residual of {residual:+.4f} means the segments act "
            f"independently, so leave-one-out is an adequate account of this "
            f"decision.</p></div>"
        )

    direction = (
        "The evidence is <b>redundant</b>: several segments carry the same signal, so "
        "removing any one of them changes little while the total effect is large. "
        "Leave-one-out under-reports every one of them."
        if residual > 0
        else "The segments <b>reinforce each other</b>, so each looks pivotal alone and "
        "the individual deltas over-count the total."
    )

    return (
        f'<div class="note"><p><b>These effects are not additive.</b> '
        f"They sum to {attribution.total_effect:+.4f} against a total swing of "
        f"{attribution.total_swing:+.4f}, leaving a residual of {residual:+.4f}.</p>"
        f"<p>{direction}</p>"
        f"<p>The ranking below is still informative, but do not read the magnitudes "
        f"as additive contributions.</p></div>"
    )


_BANNER_CSS = """
.banner {
  display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap;
  background: #1a1d21; color: #f9fafb; margin: -32px -32px 28px;
  padding: 14px 32px; font-size: 13px;
}
.banner .name { font-weight: 650; letter-spacing: -0.01em; }
.banner .desc { color: #9aa1aa; }
.banner .grow { flex: 1; }
.banner code { background: #2a2e34; padding: 2px 7px; border-radius: 3px;
               font-size: 12px; }
.banner a { color: #f9fafb; text-decoration: underline; text-underline-offset: 2px; }
"""


def _banner(repo_url: str) -> str:
    return (
        '<div class="banner">'
        '<span class="name">jev-xray</span>'
        '<span class="desc">which part of the input caused this decision</span>'
        '<span class="grow"></span>'
        "<span><code>jev-xray demo</code></span>"
        f'<span><a href="{html.escape(repo_url)}">source</a></span>'
        "</div>"
    )


def _minimal_blocks(deep: Any) -> str:
    """The because and the what-would-have-had-to-be-different."""
    blocks: list[str] = []

    s = deep.sufficient
    if s.found:
        blocks.append(
            "<h2>the smallest evidence that reproduces this answer</h2>"
            f'<div class="note ok"><p><b>{s.size} of {s.total} segments</b> are '
            f"enough. On their own they answer {s.value:.4f}, against "
            f"{s.baseline_value:.4f} for the whole state.</p>"
            f"<p><em>&ldquo;{html.escape(s.quote())}&rdquo;</em></p></div>"
        )
    else:
        blocks.append(
            "<h2>the smallest evidence that reproduces this answer</h2>"
            f'<div class="note"><p>No subset within {s.epsilon:g} of the answer was '
            f"found. This decision draws on most of the state rather than a "
            f"quotable part of it.</p></div>"
        )

    f = deep.flipping
    if f.found:
        blocks.append(
            "<h2>what would have had to be different</h2>"
            f'<div class="note"><p>Removing <b>{f.size} segment(s)</b> moves the '
            f"answer to {f.value:.4f}, crossing {f.decision.describe()}.</p>"
            f"<p><em>&ldquo;{html.escape(f.quote())}&rdquo;</em></p></div>"
        )
    else:
        blocks.append(
            "<h2>what would have had to be different</h2>"
            f'<div class="note ok"><p>Nothing found that flips '
            f"{f.decision.describe()}. The decision is robust to removing "
            f"evidence.</p></div>"
        )

    decision = f.decision
    prior = deep.attribution.unexplained_prior
    if decision.holds(prior) == decision.holds(deep.baseline_value):
        blocks.append(
            '<div class="note"><p><b>This question cannot discriminate.</b> An empty '
            f"state already answers {prior:.4f}, which falls on the same side of "
            f"{decision.describe()} as the full state. Most of the answer is a prior "
            f"the input never moves, so a threshold here will fire on almost "
            f"anything &mdash; including nothing at all.</p></div>"
        )

    return "".join(blocks)


def to_html(
    attribution: Attribution,
    *,
    limit: int = 12,
    title: str | None = None,
    repo_url: str | None = None,
    deep: Any | None = None,
) -> str:
    """Render one attribution as a standalone HTML document.

    ``repo_url`` adds a header bar naming the tool that produced the report. Worth
    setting on anything you hand to someone else, since a heatmap with no
    provenance invites the reader to trust it further than they should.
    """
    strongest = max((e.magnitude for e in attribution.effects), default=0.0)
    # Name the model that actually answered. Hardcoding a vendor here would put a
    # claim in the title that the run may not support.
    heading = title or (
        f"Why {attribution.model} answered {attribution.baseline_value:.2f}"
    )

    instructions = attribution.question.instructions
    question_text = instructions if isinstance(instructions, str) else repr(instructions)

    stats = [
        ("answer", f"{attribution.baseline_value:.4f}", attribution.target.describe()),
    ]
    if attribution.empty_value is not None:
        stats.append(
            ("empty state", f"{attribution.empty_value:.4f}", "with all evidence removed")
        )
        stats.append(("total swing", f"{attribution.total_swing:+.4f}", "full minus empty"))
    stats.append(("segments", str(len(attribution.segments)), f"{attribution.mode} ablation"))

    stat_html = "".join(
        f'<div class="stat"><div class="k">{html.escape(k)}</div>'
        f'<div class="v">{html.escape(v)}</div>'
        f'<div class="k" style="font-weight:400;text-transform:none;letter-spacing:0">'
        f"{html.escape(note)}</div></div>"
        for k, v, note in stats
    )

    heatmap = _heatmap(attribution, strongest)
    heatmap_block = ""
    if heatmap:
        heatmap_block = (
            "<h2>the state, tinted by influence</h2>"
            + heatmap
            + '<div class="legend">'
            '<span><span class="swatch" style="background:rgba(var(--pos),0.6)"></span>'
            "held the answer up</span>"
            '<span><span class="swatch" style="background:rgba(var(--neg),0.6)"></span>'
            "pushed against it</span>"
            '<span><span class="swatch" style="background:#e5e7eb"></span>no effect</span>'
            "<span>hover any span for its exact delta</span>"
            "</div>"
        )

    failures = ""
    if attribution.failures:
        items = "".join(
            f"<li>{html.escape(seg.label)}: {html.escape(why)}</li>"
            for seg, why in attribution.failures
        )
        failures = (
            f'<h2>failed ablations</h2><div class="note"><p>These segments have no '
            f"measured effect because their request did not complete, so the map is "
            f"incomplete.</p><ul>{items}</ul></div>"
        )

    ledger = attribution.ledger
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(heading)}</title>
<style>{_CSS}{_BANNER_CSS if repo_url else ""}</style></head>
<body><div class="wrap">

{_banner(repo_url) if repo_url else ""}
<h1>{html.escape(heading)}</h1>
<div class="sub">
  asked <code>{html.escape(question_text)}</code>
  &nbsp;&middot;&nbsp; question id <code>{html.escape(attribution.question_id)}</code>
  &nbsp;&middot;&nbsp; model <code>{html.escape(attribution.model)}</code>
</div>

<div class="headline">{stat_html}</div>

{heatmap_block}

<h2>ranked evidence</h2>
{_bars(attribution, strongest, limit)}

{_minimal_blocks(deep) if deep is not None else ""}

<h2>can you trust this map</h2>
{_diagnostics(attribution)}

{failures}

<footer>
  Measured by ablation: each segment was removed and the identical question asked
  again. Explains the model's sensitivity to its input, not its internal reasoning.
  <br>
  {ledger.requests} requests
  {f"&middot; {ledger.avoided_requests} avoided by cache" if ledger.avoided_requests else ""}
  &middot; {ledger.input_tokens:,} input tokens
  &middot; ${ledger.usd:.6f}
  {"&middot; token counts estimated, this host reported none, so spend is a projection at the configured price rather than a measurement" if ledger.tokens_are_estimated else ""}
  &middot; {ledger.wall_seconds:.2f}s
  &middot; generated {generated}
  &middot; <code>jev-xray</code>
</footer>

</div></body></html>"""
