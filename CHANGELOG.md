# Changelog

All notable changes to jev-xray are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [0.1.0] — 2026-10-10

First release. Everything below was built and validated in public over roughly
four weeks, starting from the day TypeSafe AI launched Jev (15 September 2026).

### Core attribution engine

- **Leave-one-out attribution** (`--method loo`): one request per segment,
  measures the answer shift when each span is removed, reports an interaction
  residual that flags when the simple method is unreliable.
- **Exact and sampled Shapley attribution** (`--method shapley`): averages each
  segment's marginal contribution over every context it could appear in. Exact
  over all 2^n subsets when affordable; permutation sampling with reported
  standard errors otherwise. The efficiency gap is zero by construction, which
  is the correctness property leave-one-out cannot provide.
- **Deep probe** (`--method deep`, default): Shapley plus the two smallest
  answers — minimal sufficient evidence (the shortest quote that reproduces the
  decision) and the minimal flipping set (the smallest removal that changes the
  decision). Both searches are free when Shapley has already evaluated every
  subset they need.
- **Coarse-to-fine drill-down** (`--method drill`): attributes coarsely and
  exactly, then re-attributes inside only the regions that carried weight. Makes
  exact attribution practical for states too long for 2^n in one pass. Every
  coalition in the fine pass still carries the rest of the document, so a
  sentence is measured in context rather than in an isolated paragraph.

### Stability probes

Six label-free probes that answer whether a question is safe to threshold on.
Label-free means no ground truth is needed — only perturbations that should not
change the answer.

- **Prior saturation**: asks the question against an empty state. If it lands on
  the same side of the boundary, the input cannot change the outcome.
- **Distractor drift**: appends decision-neutral filler at both ends. Measures
  whether irrelevant content moves the answer.
- **Order sensitivity**: reorders the same evidence and measures the spread.
- **Option order flip**: shuffles a Choice question's options. Flags when the
  branch your code takes depends on dictionary ordering.
- **Negation coherence**: swaps a Noul's true/false criteria. Checks whether the
  criteria are read at all, or only the instruction text.
- **Paraphrase spread**: rewords the question. Catches sensitivity to phrasing
  rather than content.

`jev-xray stability` exits non-zero when the question is not safe to threshold
on, so a question that regresses under a new model version can break CI.

### Injection locator

- Marks which parts of the state came from outside your control (JSON paths,
  regex patterns, or explicit segment IDs).
- Finds whether untrusted text drove the decision, at zero extra cost — it is a
  pure function over a finished attribution map.
- Direction-aware: untrusted text arguing against a dangerous action is a
  cautious user, not an attack, and is not flagged.
- `jev-xray explain --untrusted <path>` surfaces the trust boundary inline.

### Version diffing

- Replays a corpus against two model versions and reports flips (decisions that
  changed side) and shifts (probability movements) separately, because one flip
  is worth more attention than a large average shift that crosses nothing.
- Exits non-zero on any flip, so a version bump can gate a deploy.
- Costs two requests per state regardless of question count.

### pytest integration

- `assert_stable` and `assert_no_injection` register automatically via a
  `pytest11` entry point — no conftest wiring needed.
- The `jev` fixture skips when no endpoint is configured, so these tests can
  live in a suite permanently without turning red on a laptop without credentials
  or in a fork's CI.
- Failure messages carry the full probe report, not a bare assertion.

### Segmentation

- **Sentence**: guards against splitting on abbreviations (`Dr.`, `Inc.`, `e.g.`)
  and preserves closing quotes.
- **Line**, **Turn** (conversation speakers), **JSON field paths**.
- `WithinSpan`: restricts fine segmentation to one region of the text while
  ablation still rewrites the whole state — the correctness property behind
  coarse-to-fine.
- `reassemble`: rebuilds a text state with segments in a different order, used
  by the ordering probe.

### Infrastructure

- Provider presets: TypeSafe direct, Vercel AI Gateway, LangSmith LLM Gateway,
  local open reproductions.
- Token-bucket rate limiter, single-flight request coalescing, content-addressed
  cache with memory and disk backends.
- Budget ceilings (requests, tokens, USD) enforced before the first call.
- HTML reports for attribution and stability — self-contained, no scripts,
  everything escaped.
- GitHub Actions CI: test matrix (Python 3.11–3.14), lint (ruff + mypy), package
  build verification, smoke tests.
- Apache-2.0 license.

### Live validation (September–October 2026)

All four major features have been run against a hosted System One model
(SemIf via LangSmith):

- **Attribution** found a question whose empty-state prior is 0.963, making
  every threshold on it decoration regardless of how confident the answer looks.
- **Stability** confirmed that case: all six probes passed, but `input_can_decide`
  correctly blocked the "usable" verdict — the quiet failure mode that pure
  noise-ratio checks miss.
- **Injection locator** found a guardrail holding: an injected pre-approval claim
  pushed toward "safe" at +0.13 and was outvoted nine-to-one by the policy and
  the call itself. The empty-state prior on that guardrail is 0.989, meaning a
  retrieval bug that empties the state produces an approval.
- **Version diffing** is tested offline only — it needs two model versions, and
  only one free endpoint is currently available.

---

[Unreleased]: https://github.com/aishwary-dongre/jev-xray/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aishwary-dongre/jev-xray/releases/tag/v0.1.0
