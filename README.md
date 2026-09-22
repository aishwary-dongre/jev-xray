# jev-xray

**Decision forensics for System One models.**

[Jev](https://docs.typesafe.ai/) returns a typed decision with a calibrated
probability and, by construction, no explanation. There is no text channel for it
to justify itself through. When an answer is wrong — or right but surprising —
you get `0.62` and nothing to hold on to. TypeSafe names the consequence in its
own [jaggedness notes](https://docs.typesafe.ai/model-jaggedness/jev-1.13): as
state grows with unrelated detail, accuracy falls *and* it gets harder to tell
which part of the input produced a wrong answer.

jev-xray measures the *why* instead of asking for it. Ablate part of the state,
re-ask the identical question, read how far the answer moved.

```python
from jev_xray import XRay, Noul

xray = XRay(model="jev-1.13.0")
exp = xray.explain(ticket, Noul(instructions="The customer is asking for a refund."))

exp.baseline_value        # 0.7311
exp.top(3)                # the segments that moved it most, signed
exp.interaction_residual  # whether those effects are additive
exp.ledger.summary()      # 7 requests, 592 input tokens, $0.000025, 0.01s wall
```

## Why this is possible now

Occlusion attribution is textbook interpretability that has always been too
expensive to use. It needs tens to hundreds of forward passes per explanation. On
a frontier model at roughly $0.014 and 8.5s per call, one explanation costs
dollars and takes minutes, so nobody runs it on a routing decision.

At $0.042 per million input tokens with output free, the same explanation costs a
fraction of a cent and finishes in about the latency of a single call, because the
ablations are independent and go out concurrently.

Two properties make the deltas trustworthy in a way they never were for
generative models:

- **the answer axis is fixed** — every ablation is scored on the same bounded
  distribution you defined, so there is no parsing, no format drift and no
  refusal to handle
- **the probabilities are calibrated** — a 0.15 drop means something

## Try it without an API key

```bash
pip install -e ".[dev]"
jev-xray demo
```

`demo` runs the real code path — segmentation, ablation, concurrent evaluation,
attribution, budget accounting — against a deterministic fake with *planted*
evidence. The judgments are fixtures, not predictions. The point is that the
machinery is verifiable offline, which matters while TypeSafe signups are closed.

## Access

`jev-xray` speaks `POST /v1/systemone` and nothing else, so anything fronting
that contract works:

| Route | Set |
|---|---|
| TypeSafe direct | `TYPESAFE_API_KEY` |
| A gateway reselling Jev | `TYPESAFE_API_KEY` + `JEV_XRAY_BASE_URL` + `JEV_XRAY_MODEL` |
| A local open reproduction | `JEV_XRAY_BASE_URL=http://localhost:8000/v1/systemone` |

Copy `.env.example` to `.env`. `.env` is gitignored; the key is read from the
environment and never written to a config file.

Always pin a version. `jev-latest` and `jev-preview` move when a release ships,
which silently invalidates any threshold tuned against them — the client warns if
you pass one.

## What it does

**Segment** the state into the units an explanation can be written in: sentences,
lines, conversation turns, or JSON field paths using the same dot-and-index form
the docs use to point a question at part of a state.

**Ablate** each segment two ways, because they fail differently. `delete` removes
the span, which is faithful to "what if this had not been said" but shifts
everything after it. `mask` substitutes a short neutral placeholder, preserving
position at the cost of introducing text that was never there. Disagreement
between the two is a sign the measured effect is partly an ablation artifact.

**Attribute** by re-asking the identical question against each ablated state and
recording the signed change in one tracked scalar. Positive means the segment was
holding the answer up.

**Report** as a terminal heatmap over the original wording, plus a ranked table.

### Reading the interaction residual

Leave-one-out is the cheap estimator and it has a known blind spot: it cannot see
evidence that only counts jointly, and it reads *redundant* evidence as worthless
because removing either copy leaves the other carrying the decision.

`Attribution.interaction_residual` is the total swing between the full and empty
state minus the sum of the individual effects. Near zero means the segments act
independently and leave-one-out is an adequate account. Large means it is not:
the ranking is still informative but the magnitudes are not additive, and a
Shapley-style estimator over random subsets is the honest next step. The tool
tells you which situation you are in rather than letting you assume.

## Budgets are mandatory

Attribution is request-heavy on purpose, so a runaway probe is a real failure
mode. Every probe takes a `Budget` with ceilings on requests, tokens and dollars,
checked before the first call and again before each one. A pre-flight estimate
refuses an obviously impossible plan up front and names the knob that fixes it.
The rate limiter holds the fan-out inside the published 1,200 requests/minute and
250k tokens/second, with jittered backoff that honours `Retry-After`.

Every explanation reports what it actually spent, using the `input_tokens` the
service reported rather than an estimate.

## Limits

- Occlusion attribution explains the model's **sensitivity to its input**, not
  its internals. It is a behavioural explanation, not a mechanistic one.
- Deleting text perturbs the input distribution as well as removing content. The
  mask-versus-delete comparison quantifies that; it does not eliminate it.
- Calibration is TypeSafe's claim plus third-party studies on their own data, not
  a guarantee on yours.
- An explanation does not make an action safe. Deterministic permission checks
  belong upstream of every side effect, unchanged.
- Segment boundaries are the ceiling on resolution. Nothing finer than a segment
  can ever be attributed, and a segment spanning two independent pieces of
  evidence blurs them together.

## Status

Phase 1: core engine — transport, budget, cache, segmenters, leave-one-out
attribution, terminal report.

Next: Shapley sampling with coarse-to-fine drill-down, minimal sufficient and
flipping sets, then the label-free stability probes (paraphrase spread, option
order flip rate, negation coherence, distractor drift) and version diffing.

## License

Apache-2.0.
