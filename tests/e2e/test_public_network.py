"""A public provider replaying a real week through the real relay to three followers at different power."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from e2e_helpers import (
    bundle_from_relay,
    describe_ms,
    floor_step,
    percentile,
    publish,
    query,
    replay_candles,
    verifiers_for,
)
from e2e_net import ROOT, TIMINGS, sa
from e2e_scenario import TIMEFRAME_S, Run
from sa_core.bundle import check_bundle, read_bundle, write_bundle
from sa_core.delegation import sign_delegation
from sa_core.keys import DeviceKey, SAKey
from sa_core.models import SIGNAL_KIND, Signal
from sa_core.signals import build_signal_event
from sa_core.trackrecord import compute

pytestmark = pytest.mark.e2e

# default tier table (sa_core.power.DEFAULT_TIERS): tier index -> delay_fraction of the timeframe
DELAY_FRACTION = {"t0": 0.0, "t2": 0.5, "zero": 2.0}


def signals_sent(run: Run) -> int:
    return int(run.provider_summary["signals"])


def test_the_replay_produced_enough_signals_for_the_claims(public_run: Run) -> None:
    n = signals_sent(public_run)
    assert n * 3 >= 100, f"{n} signals x 3 followers must give the 100 deliveries SEP-3.21 measures"
    assert (
        public_run.provider_summary["rejected"] == [] and public_run.provider_summary["sent"] >= n
    )
    assert public_run.provider_summary["anchored"] == n


def test_sep_3_43_every_follower_receives_every_public_signal(public_run: Run) -> None:
    n = signals_sent(public_run)
    for name in DELAY_FRACTION:
        rows = public_run.ledger(name).rows()
        assert sorted(r.seq for r in rows if r.status == "accepted" and r.seq is not None) == list(
            range(1, n + 1)
        ), f"{name} missed or rejected signals"
        assert all(r.status == "accepted" for r in rows), [
            r.reason for r in rows if r.status != "accepted"
        ]


def test_sep_3_21_delivery_is_in_tier_order_and_on_schedule(public_run: Run) -> None:
    by_name = {name: {r.seq: r for r in public_run.ledger(name).rows()} for name in DELAY_FRACTION}
    late: list[float] = []
    for seq in by_name["t0"]:
        t0, t2, zero = (by_name[n][seq] for n in ("t0", "t2", "zero"))
        assert t0.received_ms <= t2.received_ms <= zero.received_ms, (
            f"signal {seq}: tier order violated"
        )
        for name, row in (("t0", t0), ("t2", t2), ("zero", zero)):
            assert row.created_ms is not None
            scheduled = (
                row.created_ms + DELAY_FRACTION[name] * TIMEFRAME_S * 1000 / public_run.speed
            )
            lateness = row.received_ms - scheduled
            assert lateness >= -3, (
                f"{name} got signal {seq} {-lateness:.0f} ms before its release time"
            )
            late.append(lateness)
    p95 = percentile(late, 0.95)
    TIMINGS["public: delivery lateness p95 (ms)"] = p95
    TIMINGS["public: delivery lateness max (ms)"] = max(late)
    print(
        f"\ndelivery lateness vs scheduled release: {describe_ms(late)} over {len(late)} deliveries"
    )
    assert max(late) < 400, f"a delivery was {max(late):.0f} ms late: {describe_ms(late)}"


def test_sep_3_6_top_tier_publish_to_receive_latency_p95_is_under_5_seconds(
    public_run: Run,
) -> None:
    """SEP-3.6: from the signal's `created_ms` (set just before the provider publishes it, so this includes
    signing) to the receipt time in the top-tier follower's ledger, over every signal of the replay. Both are
    wall-clock ms on this machine. The transport is the real sarelay binary and real websockets on loopback."""
    latency = []
    for row in public_run.ledger("t0").rows():
        assert row.created_ms is not None and row.status == "accepted"
        latency.append(float(row.received_ms - row.created_ms))
    assert len(latency) >= 30, f"only {len(latency)} top-tier deliveries; the p95 would mean little"
    p95 = percentile(latency, 0.95)
    TIMINGS["public: top-tier publish-to-receive p95 (ms)"] = p95
    TIMINGS["public: top-tier publish-to-receive max (ms)"] = max(latency)
    print(
        f"\nSEP-3.6 top tier publish-to-receive: {describe_ms(latency)}; p95 = {p95:.1f} ms (limit 5000 ms)"
    )
    assert min(latency) >= -3, f"a signal was received {-min(latency):.0f} ms before it was created"
    assert p95 < 5000, (
        f"top-tier publish-to-receive p95 is {p95:.0f} ms, over 5 s: {describe_ms(latency)}"
    )


def test_followers_size_paper_orders_by_the_contract(public_run: Run) -> None:
    for name in DELAY_FRACTION:
        position = Decimal(0)
        fills = 0
        rows = sorted(
            (r for r in public_run.ledger(name).rows() if r.filled),
            key=lambda r: (r.filled_ms or 0, r.seq or 0),
        )
        for r in rows:
            assert r.signal_json and r.qty and r.equity_before and r.ref_price
            sig = Signal.model_validate_json(r.signal_json)
            qty = Decimal(r.qty)
            if sig.action == "open":
                want = floor_step(
                    Decimal(str(sig.size_fraction))
                    * Decimal(str(r.equity_before))
                    / Decimal(str(r.ref_price))
                )
                assert qty == want, f"{name} seq {r.seq}: open sized {qty}, contract says {want}"
                position += qty
            else:
                assert qty == position * Decimal(str(sig.size_fraction)) or qty == floor_step(
                    position * Decimal(str(sig.size_fraction))
                )
                position -= qty
            fills += 1
        assert fills > 0, f"{name} placed no orders"
        account = public_run.follower_summaries[name]["account"]
        assert Decimal(account["position_qty"]) == position
        assert account["fills"] == fills


def test_sep_3_19_a_late_joiner_backfills_the_chain_the_relay_holds(public_run: Run) -> None:
    """The `late` follower (T2) started after the relay held 10 signals: its ledger holds every seq, the earlier ones
    as history fetched on connect, the chain is the one the on-time follower saw, and the relay's bundle checked
    against the late follower's head is complete."""
    n = signals_sent(public_run)
    rows = sorted(public_run.ledger("late").rows(), key=lambda r: r.seq or 0)
    assert [r.seq for r in rows] == list(range(1, n + 1)), [
        (r.seq, r.status, r.reason) for r in rows
    ]
    assert all(r.status in ("history", "accepted") for r in rows), [
        r.reason for r in rows if r.status not in ("history", "accepted")
    ]
    history = [r for r in rows if r.status == "history"]
    live = [r for r in rows if r.status == "accepted"]
    assert len(history) >= 5 and len(live) >= 4, (
        f"{len(history)} history, {len(live)} live: move the join"
    )
    assert all(r.source == "backfill" for r in history)
    assert max(r.seq or 0 for r in history) < min(
        r.seq or 0 for r in live
    )  # history first, then live
    on_time = {r.seq: r.ref for r in public_run.ledger("t0").rows()}
    assert {r.seq: r.ref for r in rows} == on_time  # the same chain, no gap
    assert rows[0].prev is None and all(b.prev == a.ref for a, b in itertools.pairwise(rows))
    head = public_run.ledger("late").head(public_run.provider.pubkey, "ema-fast-1h")
    assert head == (n, rows[-1].ref)
    bundle = bundle_from_relay(public_run.relay.url, public_run.provider.pubkey, exported_at=0)
    check = check_bundle(bundle, expected_heads={"ema-fast-1h": head})
    assert check.complete and not check.gaps, check.problems + check.gaps
    print(f"\nSEP-3.19 late joiner: {len(history)} history (backfilled), {len(live)} live, of {n}")


def test_sep_3_20_the_late_joiner_never_traded_a_signal_created_before_it_joined(
    public_run: Run,
) -> None:
    rows = public_run.ledger("late").rows()
    history = [r for r in rows if r.status == "history"]
    live = [r for r in rows if r.status == "accepted"]
    assert history and all(not r.placed and not r.filled and r.qty is None for r in history)
    assert max(r.created_ms or 0 for r in history) < min(r.created_ms or 0 for r in live)
    assert any(r.placed for r in live), "live signals must still be executed"
    summary = public_run.follower_summaries["late"]
    assert summary["history"] == len(history)
    assert summary["placed"] == sum(
        r.placed for r in live
    )  # every order the executor saw came from a live signal
    assert summary["account"]["fills"] == sum(r.filled for r in live)


def test_sep_3_39_an_unregistered_pubkey_cannot_publish(public_run: Run) -> None:
    stranger_sa, stranger = SAKey.generate(), DeviceKey.generate()
    delegation = sign_delegation(stranger_sa, stranger.pubkey_hex, 4_000_000_000)
    import time

    sig = Signal(
        strategy_id="spam", seq=1, prev=None, instrument="BINANCE:BTCUSDT", side="buy", action="open", size_fraction=0.5,
        order_type="market", timeframe=TIMEFRAME_S, created_ms=int(time.time() * 1000),
        bar_open_ms=int(time.time() * 1000) - TIMEFRAME_S * 1000,
    )  # fmt: skip
    accepted, message = publish(public_run.relay.url, build_signal_event(stranger, sig, delegation))
    assert not accepted and message.startswith("blocked:") and "registered" in message, message
    assert query(public_run.relay.url, [{"authors": [stranger.pubkey_hex]}]) == []  # nothing stored


def test_sep_3_14_the_bundle_report_equals_the_one_recomputed_from_relay_events(
    public_run: Run, data_dir: Path, tsa_config: Path
) -> None:
    bundle = read_bundle(public_run.bundle)
    check = check_bundle(bundle)
    assert check.complete and len(check.items) == signals_sent(public_run)
    trust = Path(json.loads(tsa_config.read_text())["trust_dir"])
    candles = replay_candles(public_run.spec, data_dir)
    verifiers = verifiers_for(trust)
    from_file = compute(bundle, candles, verifiers=verifiers).to_json()
    from_relay = compute(
        bundle_from_relay(public_run.relay.url, bundle.provider, bundle.exported_at),
        candles,
        verifiers=verifiers,
    ).to_json()
    assert from_file == from_relay
    report = json.loads(from_file)
    assert (
        report["anchor_coverage"]["anchored"]
        == report["anchor_coverage"]["total"]
        == len(check.items)
    )
    assert report["signals"]["counted"] > 0 and report["completeness"]["complete"] is True
    print(
        f"\ntrack record: {report['trades']} trades, pnl {report['pnl']}, counted {report['signals']['counted']}/{report['signals']['total']}, excluded {report['signals']['excluded']}"
    )
    # the CLI computes the same bytes
    out = public_run.workdir / "record.json"
    sa(
        "record",
        public_run.bundle,
        "--replay-spec",
        public_run.spec_path,
        "--tsa-config",
        tsa_config,
        "--data-dir",
        data_dir,
        "-o",
        out,
    )
    assert out.read_text() == from_file


def edit_content(event: dict[str, Any]) -> None:
    event["content"] = event["content"].replace("0.5", "0.9", 1)


def test_sep_3_15_a_tampered_bundle_is_flagged(
    public_run: Run, tmp_path: Path, tsa_config: Path
) -> None:
    good = sa("verify", public_run.bundle, "--tsa-config", tsa_config)
    assert "bundle is complete" in good.stdout
    raw = json.loads(public_run.bundle.read_text())
    mutations: dict[str, Callable[[dict[str, Any]], object]] = {
        "dropped": lambda d: d["events"].pop(len(d["events"]) // 2),
        "reordered": lambda d: d["events"].insert(1, d["events"].pop(5)),
        "edited": lambda d: edit_content(d["events"][3]),
        "duplicated": lambda d: d["events"].append(d["events"][2]),
    }
    for label, mutate in mutations.items():
        doc = json.loads(json.dumps(raw))
        mutate(doc)
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(doc))
        res = sa("verify", path, "--tsa-config", tsa_config, check=False)
        assert res.returncode == 1 and "NOT complete" in res.stdout, (
            f"{label} bundle was not flagged:\n{res.stdout}"
        )
        assert check_bundle(read_bundle(path)).complete is False
    # cutting the tail is only visible against an external head (what followers saw)
    truncated = read_bundle(public_run.bundle).model_copy(
        update={"events": read_bundle(public_run.bundle).events[:-3]}
    )
    write_bundle(tmp_path / "tail.json", truncated)
    head = public_run.ledger("t0").head(truncated.provider, "ema-fast-1h")
    assert head is not None
    assert check_bundle(truncated).complete  # blind to the tail...
    assert not check_bundle(
        truncated, expected_heads={"ema-fast-1h": head}
    ).complete  # ...unless told the head


def test_sep_3_29_signal_files_get_the_same_checks(
    public_run: Run, tmp_path: Path, tsa_config: Path, data_dir: Path
) -> None:
    first = json.loads((public_run.state / "events.jsonl").read_text().splitlines()[0])["event"]
    good = tmp_path / "signal.json"
    good.write_text(json.dumps(first))
    zero = public_run.followers[2]
    args = ["follow", "--signal-file", good, "--provider", public_run.provider.pubkey, "--replay-spec", public_run.spec_path,
            "--tsa-config", tsa_config, "--data-dir", data_dir]  # fmt: skip
    ok = sa(*args, "--ledger", tmp_path / "ok.sqlite", home=zero.home)
    assert "accepted" in ok.stdout and "seq 1" in ok.stdout
    verified = sa("verify", good, "--tsa-config", tsa_config)
    assert "signal is valid" in verified.stdout
    bad = dict(first, content=first["content"].replace("0.5", "0.9", 1))
    (tmp_path / "tampered.json").write_text(json.dumps(bad))
    res = sa(
        *args[:2],
        tmp_path / "tampered.json",
        *args[3:],
        "--ledger",
        tmp_path / "bad.sqlite",
        home=zero.home,
        check=False,
    )
    assert res.returncode == 1 and "rejected" in res.stdout
    assert sa("verify", tmp_path / "tampered.json", check=False).returncode == 1


def test_the_public_relay_stores_signals_in_plaintext_as_the_control_for_the_private_run(
    public_run: Run,
) -> None:
    events = query(
        public_run.relay.url, [{"kinds": [SIGNAL_KIND], "authors": [public_run.provider.pubkey]}]
    )
    assert len(events) == signals_sent(public_run) and all(
        "size_fraction" in e.content for e in events
    )
    assert ROOT.exists()
