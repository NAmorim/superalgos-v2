"""SEP-3.19 / SEP-3.20: a follower that joins late records the provider's earlier signals as history, never trades them."""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sa_core.models import NostrEvent
from sa_core.power import ruleset_event
from sa_core.signals import build_commitment_event, build_private_signal_event, build_signal_event

from sa_engine.network.executor import SimulatedPaperExecutor
from sa_engine.network.follower import FollowerBot
from sa_engine.network.ledger import Ledger, LedgerRow
from tests.fake_relay import FakeRelay, matches
from tests.net_helpers import Identity, flat_feed
from tests.test_network_follower import NOW, NOW_MS, follower, signal

STEP_MS = 600_000  # ten minutes between signals: inside the stale limit (3 x 1 h + 30 s), so live ones execute


class _SpyExecutor(SimulatedPaperExecutor):
    """Records every intent that reaches the executor (the boundary to an exchange)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.refs: list[str] = []

    async def execute(self, ref: str, intent: Any, now_ms: int) -> Any:
        self.refs.append(ref)
        return await super().execute(ref, intent, now_ms)


class _Store:
    """The query half of a relay: answers REQ from a fixed set of stored events."""

    def __init__(self, events: Sequence[NostrEvent]) -> None:
        self.events = list(events)

    async def query(self, filters: Sequence[dict[str, Any]], timeout: float = 5.0) -> list[NostrEvent]:
        return [ev for ev in self.events if any(matches(ev, f) for f in filters)]


def chain(p: Identity, n: int, last_ms: int = NOW_MS) -> list[NostrEvent]:
    """seq 1..n, STEP_MS apart, the last created at ``last_ms``."""
    out: list[NostrEvent] = []
    for seq in range(1, n + 1):
        at = last_ms - (n - seq) * STEP_MS
        out.append(build_signal_event(p.device, signal(p, seq, out[-1].id if out else None, at_ms=at), p.delegation))
    return out


def spy_bot(tmp: Path, p: Identity, **kw: Any) -> tuple[FollowerBot, _SpyExecutor]:
    bot = follower(tmp, p, **kw)
    spy = _SpyExecutor("BINANCE:BTCUSDT", 10_000.0, flat_feed(50_000.0, NOW_MS - 1000))
    bot.executor = spy
    return bot, spy


def assert_contiguous(rows: Sequence[LedgerRow], n: int) -> None:
    by_seq = sorted(rows, key=lambda r: r.seq or 0)
    assert [r.seq for r in by_seq] == list(range(1, n + 1))
    assert all(r.status in ("accepted", "history") for r in by_seq), [(r.seq, r.status, r.reason) for r in by_seq]
    assert by_seq[0].prev is None
    assert all(b.prev == a.ref for a, b in itertools.pairwise(by_seq))  # no gap, no fork


# ---------------------------------------------------------------- unit


async def test_sep_3_19_backfill_records_the_earlier_chain_as_history_then_live_signals_follow_it(
    tmp_path: Path,
) -> None:
    p = Identity()
    events = chain(p, 9)
    join_s = events[5].created_at  # seq 1..5 were published before the follower subscribed
    bot, spy = spy_bot(tmp_path, p)
    added = await bot.backfill(_Store(events[:5]), join_s)  # type: ignore[arg-type]
    assert added == 5
    for ev in events[5:]:
        row = await bot.ingest(ev)
        assert row and row.status == "accepted"
    rows = bot.ledger.rows()
    assert_contiguous(rows, 9)
    history = [r for r in rows if r.seq is not None and r.seq <= 5]
    assert all(r.status == "history" and r.source == "backfill" and not r.placed for r in history)
    assert all("not executed" in r.reason for r in history)
    assert spy.refs == [e.id for e in events[5:]]
    assert bot.summary()["history"] == 5


async def test_sep_3_19_backfill_puts_history_in_chain_order_and_skips_what_is_newer_than_the_subscription(
    tmp_path: Path,
) -> None:
    p = Identity()
    events = chain(p, 6)
    bot, spy = spy_bot(tmp_path, p)
    store = _Store([events[i] for i in (3, 0, 2, 5, 1, 4)])  # the relay returns them newest-first or in any order
    assert await bot.backfill(store, events[4].created_at) == 4  # type: ignore[arg-type]
    assert_contiguous(bot.ledger.rows(), 4)
    assert len(bot.ledger.rows("history")) == 4
    assert bot.ledger.get(events[4].id) is None  # created at the boundary: the live subscription delivers it
    assert spy.refs == []


async def test_sep_3_19_a_gap_in_the_stored_history_is_recorded_as_a_gap(tmp_path: Path) -> None:
    p = Identity()
    events = chain(p, 5)
    bot, spy = spy_bot(tmp_path, p)
    await bot.backfill(_Store([events[0], events[1], events[3], events[4]]), int(NOW) + 1)  # type: ignore[arg-type]
    gap = bot.ledger.get(events[3].id)
    assert gap and gap.status == "rejected" and gap.reason.startswith("gap: seq 3..3")
    last = bot.ledger.get(events[4].id)
    assert last and last.status == "history"
    assert spy.refs == []


async def test_sep_3_19_history_is_verified_with_the_delegation_valid_when_it_was_created(tmp_path: Path) -> None:
    """A delegation that has expired since still covers what it signed while valid; a forged signal is rejected."""
    p = Identity(expires_at=int(NOW) - 1800)
    events = chain(p, 2, last_ms=NOW_MS - 3600_000)
    forged = events[1].model_copy(update={"content": events[1].content.replace("0.5", "0.9")})
    bot, spy = spy_bot(tmp_path, p)
    await bot.backfill(_Store([events[0], forged]), int(NOW))  # type: ignore[arg-type]
    ok = bot.ledger.get(events[0].id)
    assert ok and ok.status == "history"
    bad = bot.ledger.get(forged.id)
    assert bad and bad.status == "rejected" and bad.reason.startswith("verification failed")
    assert spy.refs == []


async def test_sep_3_19_private_history_is_opened_with_its_commitments(tmp_path: Path) -> None:
    p, me = Identity(), Identity()
    bot, spy = spy_bot(tmp_path, p, rules=p.private_rules(), me=me)
    stored: list[NostrEvent] = []
    prev: str | None = None
    for seq in (1, 2, 3):
        sig = signal(p, seq, prev, at_ms=NOW_MS - (4 - seq) * STEP_MS)
        commit = build_commitment_event(p.device, sig, p.delegation, sig.created_ms // 1000)
        stored += [build_private_signal_event(p.device, sig, p.delegation, me.pubkey, sig.created_ms // 1000), commit]
        prev = commit.id
    assert await bot.backfill(_Store(stored), int(NOW)) == 3  # type: ignore[arg-type]
    rows = sorted(bot.ledger.rows("history"), key=lambda r: r.seq or 0)
    assert [r.seq for r in rows] == [1, 2, 3] and rows[-1].ref == prev
    assert spy.refs == []


async def test_sep_3_20_a_private_history_signal_whose_commitment_arrives_live_is_still_history(
    tmp_path: Path,
) -> None:
    p, me = Identity(), Identity()
    bot, spy = spy_bot(tmp_path, p, rules=p.private_rules(), me=me)
    sig = signal(p, 1, None, at_ms=NOW_MS - STEP_MS)
    cipher = build_private_signal_event(p.device, sig, p.delegation, me.pubkey, sig.created_ms // 1000)
    await bot.backfill(_Store([cipher]), int(NOW))  # type: ignore[arg-type]
    pending = bot.ledger.get(cipher.id)
    assert pending and pending.status == "pending"
    row = await bot.ingest(build_commitment_event(p.device, sig, p.delegation, sig.created_ms // 1000))
    assert row and row.status == "history" and not row.placed
    assert spy.refs == []


async def test_sep_3_20_an_old_signal_pushed_on_the_live_route_is_recorded_as_history(tmp_path: Path) -> None:
    """A relay that replays old events on the live subscription cannot make the follower trade them."""
    p = Identity()
    events = chain(p, 3)
    bot, spy = spy_bot(tmp_path, p)
    await bot.backfill(_Store([]), events[2].created_at)  # type: ignore[arg-type]
    for ev in events:
        await bot.ingest(ev)
    assert [r.status for r in sorted(bot.ledger.rows(), key=lambda r: r.seq or 0)] == ["history", "history", "accepted"]
    assert spy.refs == [events[2].id]


async def test_sep_3_19_the_backfill_starts_at_the_ledgers_chain_head(tmp_path: Path) -> None:
    p = Identity()
    events = chain(p, 7)
    first, _ = spy_bot(tmp_path, p)
    for ev in events[:3]:
        await first.ingest(ev)  # a previous run traded seq 1..3
    assert first.ledger.oldest_head_ms(p.pubkey) == events[2].created_at * 1000
    first.ledger.close()
    store = _Store(events)
    seen: list[list[dict[str, Any]]] = []
    real = store.query

    async def spying(filters: Sequence[dict[str, Any]], timeout: float = 5.0) -> list[NostrEvent]:
        seen.append(list(filters))
        return await real(filters, timeout)

    store.query = spying  # type: ignore[method-assign]
    again, spy = spy_bot(tmp_path, p)
    assert await again.backfill(store, int(NOW) + 1) == 4  # type: ignore[arg-type]
    assert seen[0][0]["since"] == events[2].created_at
    statuses = {r.seq: r.status for r in again.ledger.rows()}
    assert statuses == {
        1: "accepted",
        2: "accepted",
        3: "accepted",
        4: "history",
        5: "history",
        6: "history",
        7: "history",
    }
    assert spy.refs == []
    assert Ledger(":memory:").oldest_head_ms(p.pubkey) is None


# ---------------------------------------------------------------- property


@settings(deadline=None, max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    n=st.integers(1, 9),
    join_offset_ms=st.integers(-STEP_MS, 9 * STEP_MS),
    replay_history_live=st.booleans(),
    relay_hides_history=st.booleans(),
)
def test_sep_3_20_property_no_signal_created_before_the_subscription_is_ever_executed(
    n: int,
    join_offset_ms: int,
    replay_history_live: bool,
    relay_hides_history: bool,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Random join times; the relay may also push old events live, or leave them out of the backfill."""
    p = Identity()
    events = chain(p, n)
    join_s = (events[0].created_at * 1000 + join_offset_ms) // 1000
    old = [ev for ev in events if ev.created_at < join_s]
    live = [ev for ev in events if ev.created_at >= join_s]

    async def go() -> tuple[FollowerBot, _SpyExecutor]:
        bot, spy = spy_bot(tmp_path_factory.mktemp("join"), p)
        await bot.backfill(_Store([] if relay_hides_history else events), join_s)  # type: ignore[arg-type]
        for ev in (old if replay_history_live or relay_hides_history else []) + live:
            await bot.ingest(ev)
        return bot, spy

    bot, spy = asyncio.run(go())
    assert not {e.id for e in old} & set(spy.refs)  # SEP-3.20
    assert spy.refs == [e.id for e in live]  # live signals executed as before, in chain order
    for ev in old:
        row = bot.ledger.get(ev.id)
        assert row and row.status == "history" and not row.placed
    assert_contiguous(bot.ledger.rows(), n)  # SEP-3.19: the ledger holds the whole chain


# ---------------------------------------------------------------- the relay session


@pytest.fixture
async def relay() -> AsyncIterator[FakeRelay]:
    r = FakeRelay()
    await r.start()
    yield r
    await r.stop()


async def test_sep_3_19_a_late_joiner_backfills_on_connect_and_executes_only_live_signals(
    relay: FakeRelay, tmp_path: Path
) -> None:
    p = Identity()
    await relay.inject(ruleset_event(p.device, p.rules(), p.delegation))
    now_ms = int(time.time() * 1000)
    events = chain(p, 5, last_ms=now_ms - 60_000)  # published before the follower existed
    for ev in events:
        await relay.inject(ev)
    bot = follower(tmp_path, p, rules=None, relay_url=relay.url, clock_s=time.time())
    spy = _SpyExecutor("BINANCE:BTCUSDT", 10_000.0, flat_feed(50_000.0, now_ms - 1000))
    bot.executor, bot.clock = spy, time.time
    task = asyncio.create_task(bot.run(asyncio.Event(), duration_s=3.0))
    await asyncio.wait_for(bot.ready.wait(), 30)
    assert len(bot.ledger.rows("history")) == 5  # recorded before the live subscription started
    for seq in range(6, 10):
        at = int(time.time() * 1000)
        ev = build_signal_event(p.device, signal(p, seq, events[-1].id, at_ms=at), p.delegation)
        events.append(ev)
        await relay.inject(ev)
        await asyncio.sleep(0.05)
    await task
    rows = bot.ledger.rows()
    assert_contiguous(rows, 9)
    assert sorted(r.seq or 0 for r in rows if r.status == "history" and r.source == "backfill") == [1, 2, 3, 4, 5]
    assert all(r.status == "accepted" for r in rows if (r.seq or 0) > 5)
    assert spy.refs == [e.id for e in events[5:]]


async def test_sep_3_19_a_signal_released_to_this_tier_after_the_join_is_history_not_lost(
    relay: FakeRelay, tmp_path: Path
) -> None:
    """A delayed tier gets a signal created before the join only after it: the live REQ reaches back the longest
    tier delay, so it is not lost (no gap), and it is recorded as history, not traded."""
    p = Identity()
    await relay.inject(ruleset_event(p.device, p.rules(), p.delegation))
    now_ms = int(time.time() * 1000)
    events = chain(p, 4, last_ms=now_ms - 60_000)
    for ev in events[:3]:
        await relay.inject(ev)
    bot = follower(tmp_path, p, rules=None, relay_url=relay.url, clock_s=time.time())
    spy = _SpyExecutor("BINANCE:BTCUSDT", 10_000.0, flat_feed(50_000.0, now_ms - 1000))
    bot.executor, bot.clock = spy, time.time
    task = asyncio.create_task(bot.run(asyncio.Event(), duration_s=2.5))
    await asyncio.wait_for(bot.ready.wait(), 30)
    await relay.inject(events[3])  # created a minute before the join, released to this tier now
    live = build_signal_event(p.device, signal(p, 5, events[3].id, at_ms=int(time.time() * 1000)), p.delegation)
    await asyncio.sleep(0.1)
    await relay.inject(live)
    await task
    assert_contiguous(bot.ledger.rows(), 5)
    late = bot.ledger.get(events[3].id)
    assert late and late.status == "history" and not late.placed
    assert spy.refs == [live.id]
