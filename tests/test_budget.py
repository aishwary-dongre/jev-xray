"""Cost accounting, rate limiting and caching.

A probe that fans out is a probe that can run away, so these are the guard rails
rather than a nicety. The pre-flight check has to refuse an impossible plan
before the first request, and the ledger has to report real spend rather than
the estimate it used to decide.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from jev_xray import (
    PRICE_USD_PER_INPUT_TOKEN,
    Budget,
    BudgetExceeded,
    Client,
    DiskCache,
    FakeTransport,
    Ledger,
    MemoryCache,
    NullCache,
    Noul,
    RateLimiter,
    estimate_tokens,
)
from jev_xray.budget import TokenBucket
from jev_xray.cache import cache_key
from jev_xray.types import SystemOneRequest

QUESTION = Noul(instructions="The customer is asking for a refund.")


class TestPricing:
    def test_a_million_input_tokens_costs_about_four_cents(self):
        ledger = Ledger()
        ledger.record(input_tokens=1_000_000)
        assert ledger.usd == pytest.approx(0.042)

    def test_output_tokens_are_free(self):
        ledger = Ledger()
        ledger.record(input_tokens=1000, output_tokens=999_999)
        assert ledger.usd == pytest.approx(1000 * PRICE_USD_PER_INPUT_TOKEN)

    def test_the_published_rate_is_what_we_charge_against(self):
        assert PRICE_USD_PER_INPUT_TOKEN == 0.042 / 1_000_000


class TestEstimates:
    def test_estimates_scale_with_length(self):
        assert estimate_tokens("x" * 400) > estimate_tokens("x" * 40)

    def test_handles_structured_payloads(self):
        assert estimate_tokens({"a": "some text here"}) > 0

    def test_handles_nothing(self):
        assert estimate_tokens(None) == 0

    def test_never_returns_zero_for_real_content(self):
        assert estimate_tokens("hi") >= 1


class TestLedger:
    def test_counts_requests_separately_from_avoided_ones(self):
        ledger = Ledger()
        ledger.record(input_tokens=10)
        ledger.record_cache_hit()
        ledger.record_coalesced()
        assert ledger.requests == 1
        assert ledger.avoided_requests == 2

    def test_summary_mentions_spend_and_wall_time(self):
        ledger = Ledger()
        ledger.record(input_tokens=1234)
        summary = ledger.summary()
        assert "1 requests" in summary
        assert "1,234 input tokens" in summary
        assert "$" in summary
        assert "wall" in summary

    def test_wall_time_freezes_once_finished(self):
        ledger = Ledger()
        ledger.finish()
        first = ledger.wall_seconds
        time.sleep(0.01)
        assert ledger.wall_seconds == first


class TestBudgetEnforcement:
    def test_a_request_ceiling_stops_the_next_call(self):
        ledger = Ledger()
        ledger.record(input_tokens=1)
        with pytest.raises(BudgetExceeded, match="request budget"):
            ledger.check(Budget(max_requests=1))

    def test_a_token_ceiling_uses_the_projection(self):
        ledger = Ledger()
        with pytest.raises(BudgetExceeded, match="token budget"):
            ledger.check(Budget(max_input_tokens=100), about_to_spend_tokens=500)

    def test_a_cost_ceiling_uses_the_projection(self):
        ledger = Ledger()
        with pytest.raises(BudgetExceeded, match="cost budget"):
            ledger.check(Budget(max_usd=1e-9), about_to_spend_tokens=10_000)

    def test_unlimited_permits_anything(self):
        ledger = Ledger()
        ledger.record(input_tokens=10**9)
        ledger.check(Budget.unlimited(), about_to_spend_tokens=10**9)

    def test_defaults_are_conservative(self):
        budget = Budget()
        assert budget.max_requests == 200
        assert budget.max_usd == 0.05


class TestTokenBucket:
    def test_allows_a_burst_up_to_capacity(self):
        async def run():
            bucket = TokenBucket(rate=1.0, capacity=5.0)
            started = time.monotonic()
            for _ in range(5):
                await bucket.acquire(1.0)
            return time.monotonic() - started

        assert asyncio.run(run()) < 0.1

    def test_delays_once_the_bucket_is_empty(self):
        async def run():
            bucket = TokenBucket(rate=100.0, capacity=1.0)
            await bucket.acquire(1.0)
            started = time.monotonic()
            await bucket.acquire(1.0)
            return time.monotonic() - started

        # 1 token at 100/s is a 10ms wait; allow generous slack for scheduling.
        assert asyncio.run(run()) >= 0.004

    def test_an_oversized_request_does_not_deadlock(self):
        async def run():
            bucket = TokenBucket(rate=1000.0, capacity=10.0)
            await asyncio.wait_for(bucket.acquire(10_000), timeout=1.0)

        asyncio.run(run())

    def test_rejects_a_non_positive_rate(self):
        with pytest.raises(ValueError, match="positive"):
            TokenBucket(rate=0)

    def test_the_limiter_defaults_to_the_published_limits(self):
        async def run():
            await RateLimiter().acquire(100)

        asyncio.run(run())


class TestCaching:
    def _ask(self, client, state, ledger):
        return asyncio.run(
            client.ask(state, {"q": QUESTION}, ledger=ledger, budget=Budget.unlimited())
        )

    def test_an_identical_request_does_not_reach_the_transport(self):
        transport = FakeTransport()
        client = Client(transport, model="m", limiter=RateLimiter.unlimited())
        ledger = Ledger()

        self._ask(client, "same state", ledger)
        self._ask(client, "same state", ledger)

        assert transport.calls == 1
        assert ledger.cached_requests == 1
        assert ledger.requests == 1

    def test_a_different_state_is_a_miss(self):
        transport = FakeTransport()
        client = Client(transport, model="m", limiter=RateLimiter.unlimited())
        ledger = Ledger()

        self._ask(client, "state one", ledger)
        self._ask(client, "state two", ledger)

        assert transport.calls == 2

    def test_the_null_cache_never_hits(self):
        transport = FakeTransport()
        client = Client(
            transport, model="m", cache=NullCache(), limiter=RateLimiter.unlimited()
        )
        ledger = Ledger()

        self._ask(client, "same state", ledger)
        self._ask(client, "same state", ledger)

        assert transport.calls == 2

    def test_concurrent_identical_requests_are_coalesced(self):
        async def run():
            transport = FakeTransport(latency=0.01)
            client = Client(transport, model="m", limiter=RateLimiter.unlimited())
            ledger = Ledger()
            await asyncio.gather(
                *(
                    client.ask(
                        "same state",
                        {"q": QUESTION},
                        ledger=ledger,
                        budget=Budget.unlimited(),
                    )
                    for _ in range(5)
                )
            )
            return transport.calls, ledger

        calls, ledger = asyncio.run(run())
        assert calls == 1
        assert ledger.coalesced_requests == 4

    def test_a_failure_propagates_to_coalesced_followers(self):
        async def run():
            transport = FakeTransport(latency=0.01, fail_first=1)
            client = Client(transport, model="m", limiter=RateLimiter.unlimited())
            ledger = Ledger()
            return await asyncio.gather(
                *(
                    client.ask(
                        "same state",
                        {"q": QUESTION},
                        ledger=ledger,
                        budget=Budget.unlimited(),
                    )
                    for _ in range(3)
                ),
                return_exceptions=True,
            )

        results = asyncio.run(run())
        assert len(results) == 3
        assert all(isinstance(r, Exception) for r in results)

    def test_the_model_id_is_part_of_the_key(self):
        # A version diff must never read one version's answer for another's.
        left = SystemOneRequest(model="jev-1.13.0", state="s", questions={"q": QUESTION})
        right = SystemOneRequest(model="jev-1.14.0", state="s", questions={"q": QUESTION})
        assert cache_key(left) != cache_key(right)

    def test_the_key_is_stable_across_equal_requests(self):
        left = SystemOneRequest(model="m", state="s", questions={"q": QUESTION})
        right = SystemOneRequest(model="m", state="s", questions={"q": QUESTION})
        assert cache_key(left) == cache_key(right)

    def test_memory_cache_reports_its_size(self):
        cache = MemoryCache()
        cache.put("k", {"a": 1})
        assert len(cache) == 1


class TestDiskCache:
    def test_persists_across_instances(self, tmp_path):
        first = DiskCache(tmp_path / "c")
        first.put("abcd1234", {"answers": {}})
        second = DiskCache(tmp_path / "c")
        assert second.get("abcd1234") == {"answers": {}}

    def test_a_miss_returns_none(self, tmp_path):
        assert DiskCache(tmp_path / "c").get("nope0000") is None

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path):
        cache = DiskCache(tmp_path / "c", memoize=False)
        cache.put("deadbeef", {"answers": {}})
        path = (tmp_path / "c" / "de" / "deadbeef.json")
        path.write_text("{ not json", encoding="utf-8")
        assert cache.get("deadbeef") is None

    def test_entries_are_sharded_by_prefix(self, tmp_path):
        cache = DiskCache(tmp_path / "c")
        cache.put("ab000000", {"x": 1})
        assert (tmp_path / "c" / "ab" / "ab000000.json").exists()
