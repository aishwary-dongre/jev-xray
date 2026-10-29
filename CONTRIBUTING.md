# Contributing to jev-xray

Thanks for looking at contributing. This document covers how to get started,
how the project is structured, and what a good contribution looks like.

---

## Getting started

```bash
git clone https://github.com/aishwary-dongre/jev-xray.git
cd jev-xray
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Verify everything works:

```bash
pytest            # 449 tests, all offline, no key needed
jev-xray demo     # offline attribution demo
jev-xray demo --scenario injection  # injection locator demo
```

The full suite runs against a deterministic fake and makes no network calls, so
it passes on any machine.

---

## Running the linters

```bash
ruff check src tests   # style and correctness
mypy                   # type checking
```

Both run in CI on every push. A PR that passes locally passes in CI.

---

## Project layout

```
src/jev_xray/
  attribution.py    leave-one-out estimator
  shapley.py        exact and sampled Shapley estimator
  drill.py          coarse-to-fine for long states
  coalition.py      shared ablation engine (used by all estimators)
  minimal.py        minimal sufficient evidence and flipping sets
  stability.py      six label-free stability probes + StabilityReport
  injection.py      prompt injection locator (pure function over attribution)
  diff.py           version diffing across a corpus
  segment.py        sentence / line / turn / JSON-field segmenters
  types.py          Jev wire contract (Choice, Score, Noul, parsing)
  client.py         metered client (cache, rate limiter, single-flight coalescing)
  transport/        HTTP, fake, base
  budget.py         Budget / Ledger / RateLimiter / TokenBucket
  cache.py          Memory, Disk, Null caches
  explain.py        XRay facade
  render.py         terminal heatmap and table
  report_html.py    standalone HTML reports
  cli.py            jev-xray command
  pytest_plugin.py  assert_stable, assert_no_injection, jev fixture
  results.py        AnyAttribution / AnyEffect union types

tests/              449 tests, all offline
examples/           ready-to-run JSON fixtures
```

---

## What makes a good contribution

**The best contributions are ones the tests can prove.** Every feature in this
project has a fixture with planted evidence so the correct output is known before
running. If you add a new probe, estimator, or segmenter, build a fixture that
would catch a regression.

**Example fixtures are checked offline.** `tests/test_examples.py` loads every
`examples/*.json` the way the CLI does and confirms that the `-q` and
`--untrusted` names in its `_comment` command exist in the file. If you add a
fixture, put a runnable command in its `_comment` and the test covers it.

**Failing tests are as welcome as passing ones.** If you find a case the tool
handles wrong, a failing test that demonstrates it is a complete contribution
even without the fix.

**Honest diagnostics over confident wrong answers.** The whole point of this
project is telling you when a result is unreliable. A contribution that detects
an additional failure mode and says so is worth more than one that papers over it.

---

## Adding a new stability probe

1. Write an async function `async def my_probe(ctx: ProbeContext) -> ProbeResult`
   in `stability.py`.
2. If the perturbation should not change the answer (filler text, reordering,
   rewordings), set `noise=True` on the returned `ProbeResult`. This makes the
   measurement contribute to the noise band in the verdict.
3. Add it to `DEFAULT_PROBES` in `stability.py`.
4. Write two tests: one where the question passes, one rigged to fail in exactly
   the way the probe looks for. A probe that only returns `ok` is worthless.

---

## Adding a new segmenter

Extend `_SpanSegmenter` for text states or implement the `Segmenter` protocol
for structured ones. Register the name in `_REGISTRY` at the bottom of
`segment.py`. Add it to `auto_segmenter` if it has a natural trigger.

---

## Opening a pull request

- Run `pytest`, `ruff check src tests`, and `mypy` locally before pushing.
- Keep the commit message focused: what changed and why, not how.
- If the change touches a documented failure mode or a live validation finding,
  note whether that finding is still valid or has been superseded.

---

## Reporting a bug

Open an issue at https://github.com/aishwary-dongre/jev-xray/issues. Include:
- the `jev-xray` version (`jev-xray --version` or `python -c "import jev_xray; print(jev_xray.__version__)"`)
- the provider and model id (`--provider`, `--model`)
- the command you ran, with state and question redacted if sensitive
- the actual output and what you expected instead

---

## License

By contributing you agree that your contributions will be licensed under the
[Apache-2.0 license](LICENSE) that covers this project.
