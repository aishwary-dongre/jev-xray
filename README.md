# jev-xray

**Decision forensics for System One models.**

**[See a real report →](https://aishwary-dongre.github.io/jev-xray/)** — an actual
run, not a mockup.

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

## What it found on a real decision

A support ticket where the customer never says "refund" but does quote a refund
policy. Asked a hosted decision model whether she is asking for her money back:
**0.988**.

Delete the entire message and it still answers **0.963**.

The answer barely depends on the input, so no threshold on that question means
anything — and `0.988` alone would never have told you. The tool's own verdict:

```
smallest evidence that reproduces the answer
  0 of 6 segment(s) reproduce the answer: 0.9627 against 0.9876

smallest change that flips the decision
  no subset flips value >= 0.5; the decision is robust to removing evidence

warning
  an empty state already answers 0.9627, the same side of value >= 0.5 as the
  full state. this question cannot discriminate.
```

### Why the estimator matters

The same question, the same model, both estimators:

| sentence | leave-one-out | Shapley |
|---|---:|---:|
| Order A-104 arrived… the box was crushed | −0.0027 | **−0.0992** |
| The jacket itself looks fine, honestly. | −0.0039 | **−0.1861** |
| I've been shopping with you since 2019… | −0.0014 | **−0.0556** |
| I'm not sure whether to send it back… | −0.0100 | **+0.0291** |
| Your policy page says damaged items qualify… | +0.3949 | +0.4569 |
| Let me know what my options are. | +0.0000 | **−0.1245** |
| **sum of effects** | +0.3769 | +0.0207 |
| **total swing** | +0.0249 | +0.0249 |
| **unexplained** | −0.3520 | **−0.0000** |

Leave-one-out called five of six sentences inert. Shapley shows they were pushing
*against* the refund reading all along — one of them by 48× what leave-one-out
measured.

They were invisible for the same reason the question is broken: the prior is so
strong that removing any *single* sentence leaves the answer pinned near the
ceiling. Only contexts where several are already gone let it move. Note also that
leave-one-out's deltas sum to 15× the swing the state actually produces, while
Shapley's sum to it exactly.

64 requests, 8 seconds, a third of a thousandth of a dollar.

Measured on [SemIf](https://github.com/TheoLeeCJ/SemIf), an open reproduction, not
on Jev. What generalises is the class of bug, not the numbers.

## Try it in two minutes, free, no card

```bash
pip install -e ".[dev]"
jev-xray demo                        # offline, no account at all
```

Three estimators, selected with `--method`:

| | what it does | cost |
|---|---|---|
| `loo` | removes one segment at a time | n + 2 requests |
| `shapley` | average contribution over subsets; correct under interaction | 2ⁿ exact, or samples × (n+1) |
| `deep` | Shapley plus minimal evidence and counterfactual (**default**) | same as Shapley |

`deep` is the default because the extra two searches are usually free: exact
Shapley has already evaluated every subset they need.

For a real hosted model, LangChain runs [SemIf free through the LLM
Gateway](https://docs.langchain.com/langsmith/llm-gateway-decision-models). A
LangSmith API key is all it needs — no provider key, no purchase, no payment
method:

```bash
echo 'LANGSMITH_API_KEY=lsv2_pt_...' >> .env
jev-xray check   --provider langsmith                       # validate the contract
jev-xray explain examples/ticket.json -q refund_requested \
                 --provider langsmith --html report.html
```

## Access

`jev-xray` speaks `POST /v1/systemone` and nothing else, so anything fronting that
contract works. Presets:

| `--provider` | Endpoint | Default model | Key |
|---|---|---|---|
| `langsmith` | LangSmith LLM Gateway | `semif-qwen3.5-4b` | `LANGSMITH_API_KEY` |
| `typesafe` | TypeSafe direct | `jev-1.13.0` | `TYPESAFE_API_KEY` |
| `vercel` | Vercel AI Gateway | `jev-1.13.0` | `AI_GATEWAY_API_KEY` |
| `local` | `localhost:8000` | — | none |

Anything else: point `--endpoint` at it. Copy `.env.example` to `.env`; `.env` is
gitignored and keys are only ever read from the environment.

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

### Why leave-one-out is not enough

Removing one segment at a time is cheap and has two blind spots:

- **redundant evidence** — the same signal appears twice, so removing either copy
  leaves the other carrying the decision and both measure as worthless
- **complementary evidence** — two segments matter only together, so each looks
  individually decisive and their deltas sum to twice the real effect

Both are measurable. `interaction_residual` is the total swing minus the sum of
the individual effects. Near zero means the segments act independently. Far from
zero means they do not, and the magnitudes cannot be trusted.

Shapley fixes it by averaging each segment's contribution over every context it
could appear in. That is the only attribution satisfying **efficiency** — the
values sum exactly to the full-state answer minus the empty-state answer — so the
residual becomes zero by construction and turns into a check on the arithmetic
rather than a caveat on the result.

The cost is subsets: 2ⁿ of them, which is 64 requests for six segments. Exact when
that fits the budget, permutation sampling with reported standard errors when it
does not, and the report always says which it used.

### The two smallest answers

An attribution map is the right output for diagnosing a *question*. It is the
wrong one for explaining a *decision* — nobody wants six numbers, they want the
sentence that did it. So `deep` also searches for:

- **minimal sufficient evidence** — the smallest set of segments that reproduces
  the answer on its own. A quotable *because*.
- **minimal flipping set** — the smallest removal that changes the decision. The
  counterfactual, which is what an auditor or a customer actually asks for.

A sufficient set of **size zero** is the most important result either can return:
it means the empty state already produces the answer, so the question does not
need its input at all.

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

Working. 203 tests, and validated end to end against a live hosted model.

Six of those tests run against real response bodies published in [Cloudflare's
model docs](https://developers.cloudflare.com/ai/models/typesafe/jev/), so the
parser is checked against actual Jev output rather than a reading of the written
documentation. That is also how three guesses got corrected: a Score `legend`
arrives as an object keyed by level, `confidence` is present on Choice and Score
but absent on Noul, and every answer echoes its own `type` — which nothing in the
prose documentation mentions, and which the parser now cross-checks so it can
never read a value off the wrong axis.

**Not yet run against hosted Jev itself**, only against SemIf through LangSmith.
The wire contract is identical and `--provider typesafe` is wired, but nobody has
pointed it at the real thing yet.

Next: the label-free stability probes — paraphrase spread, option-order flip rate,
negation coherence, distractor drift — which measure whether a question is stable
enough to hang a threshold on at all. Then coarse-to-fine drill-down so Shapley
scales past a dozen segments, and version diffing for when `jev-latest` moves
under you.

## License

Apache-2.0.
