"""[t_c995ea84] F2 + F3/F5 — a gateway outage must not spend the card's failure budget, and an
admission whose receipt never settles is reconciled against the attempt's OWN result.

Three verdicts leave the submitter, distinguished only by exit code (``hermes_cli.kanban_db``):

* ``77`` admission refused by the owner  -> counted crash, exactly as before;
* ``76`` the gateway never admitted the run -> outcome ``gateway_unavailable``, NO failure counted;
* ``69`` admitted, then lost            -> counted crash, exactly as before.

Reconcile (F3): ``prompt.receipt`` answering ``unknown`` reads this run's ``worker_result`` —
gated by the claim lock AND the ``worker_bound`` pid, the same invariant that let it be written —
and returns its REAL exit code. Without one the old raise stands: never synthesize success.
Reconnect (F5): a socket that drops AFTER the owner admitted is re-dialed and the same
``admission_id`` re-polled; ``kanban.run`` is never put on the wire twice.

Every board below is a real ``kanban.db`` in a tmp HERMES_HOME: the reconcile and the reclaim
bookkeeping are exercised against real rows, not stubs.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from gateway import session_kanban
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_worker_client as kwc
from hermes_cli.gateway_client import GatewayClient, GatewayClientError


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def board(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a freshly initialised kanban DB; returns its path."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.delenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", raising=False)
    kb.init_db()
    return kb.kanban_db_path()


def _connect(db_path):
    """A board connection with row access by name — what every ``kanban_db``/dispatcher reader
    assumes (``kbc.connect``'s factory, taken straight)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _claimed(db_path, *, title="card"):
    """A task claimed for this run: ``running``, an open run, a host-prefixed claim lock.

    ``create_task``/``claim_task`` own their write transactions, so neither is wrapped here."""
    with _connect(db_path) as conn:
        task_id = kb.create_task(conn, title=title, assignee="a")
    with _connect(db_path) as conn:
        claimed = kb.claim_task(conn, task_id)
    assert claimed is not None and claimed.current_run_id
    return claimed


def _bind(db_path, task, *, pid, exit_code=None, last_output=""):
    """Write the events a real bound worker leaves: ``worker_bound`` always, ``worker_result``
    only when the caller wants the attempt to HAVE a result."""
    with _connect(db_path) as conn, kb.write_txn(conn):
        kb._append_event(conn, task.id, "worker_bound",
                         {"pid": pid, "claim_lock": task.claim_lock},
                         run_id=task.current_run_id)
        if exit_code is not None:
            kb._append_event(conn, task.id, "worker_result",
                             {"pid": pid, "claim_lock": task.claim_lock,
                              "exit_code": exit_code, "last_output": last_output},
                             run_id=task.current_run_id)


def _params(task, db_path):
    return {"board": "default", "task_id": task.id, "run_id": task.current_run_id,
            "claim_lock": task.claim_lock, "db": str(db_path)}


class _Owner:
    """Scripted owner: ``method -> [answer, ...]``; each call consumes the head and REPEATS the
    last remaining answer. An ``Exception`` entry is raised from ``send`` — a socket that dies
    mid-RPC, the transport failure F5 exists for."""

    def __init__(self, answers):
        self.inbox, self.sent = asyncio.Queue(), []
        self._answers = {m: list(a) for m, a in answers.items()}

    async def send(self, raw):
        req = json.loads(raw)
        self.sent.append(req)
        queue = self._answers.setdefault(req["method"], [])
        item = queue.pop(0)
        if not queue:
            queue.append(item)
        if isinstance(item, Exception):
            raise item
        await self.inbox.put(json.dumps({"id": req["id"], "result": item}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.inbox.get()


def _admitted(receipt):
    return {"session_id": "s1", "receipt": receipt}


async def _no_sleep(_seconds):
    return None


# --------------------------------------------------------------------------- F3: reconcile


@pytest.mark.asyncio
async def test_unknown_receipt_with_a_successful_result_completes(board):
    """M1 — receipt ``unknown`` + this run's ``worker_result(exit 0)`` -> the submitter completes
    with the REAL outcome instead of raising (the raise is what booked ``crashed`` before)."""
    task = _claimed(board)
    _bind(board, task, pid=4242, exit_code=0, last_output="all good")
    owner = _Owner({"kanban.run": [_admitted({"status": "unknown", "admission_id": "a1"})]})
    params = _params(task, board)

    async with GatewayClient(owner) as client:
        code = await kwc._submit_claim(client, params, str(board))

    assert code == 0
    # The failure path the dispatcher books ``crashed`` from is NOT taken.
    assert kbd._exit_code_kind(code) == ("clean_exit", 0)


@pytest.mark.asyncio
async def test_unknown_receipt_with_a_nonzero_result_returns_that_failure(board):
    """M2 — receipt ``unknown`` + ``worker_result(exit 1)`` -> the REAL failure is returned;
    no success is synthesized."""
    task = _claimed(board)
    _bind(board, task, pid=4242, exit_code=1, last_output="boom")
    owner = _Owner({"kanban.run": [_admitted({"status": "unknown", "admission_id": "a1"})]})
    params = _params(task, board)

    async with GatewayClient(owner) as client:
        code = await kwc._submit_claim(client, params, str(board))

    assert code == 1
    assert kbd._exit_code_kind(code)[0] == "nonzero_exit"  # -> the failure path, not success


@pytest.mark.asyncio
async def test_unknown_receipt_without_any_result_still_raises(board):
    """M3 — no ``worker_result`` -> the OLD behaviour is preserved: raise, so the run is booked
    as before (exit 69 = counted crash), never a synthesized outcome."""
    task = _claimed(board)
    _bind(board, task, pid=4242)  # bound, but no result was ever written
    owner = _Owner({"kanban.run": [_admitted({"status": "unknown", "admission_id": "a1"})]})
    params = _params(task, board)

    async with GatewayClient(owner) as client:
        with pytest.raises(GatewayClientError, match="execution unknown"):
            await kwc._submit_claim(client, params, str(board))

    # ... and that raise still maps to the counted (c) verdict, not the neutral gateway one:
    # the owner DID admit this run (``kanban.run`` answered), which is what tells (c) from (b).
    from hermes_cli.kanban_db import KANBAN_EXECUTION_UNKNOWN_EXIT_CODE
    with _connect(board) as conn, kb.write_txn(conn):
        kb._append_event(conn, task.id, "owner_admitted", {"run_id": task.current_run_id},
                         run_id=task.current_run_id)
    assert kwc._submit_exit_code(
        GatewayClientError("Kanban execution unknown; retained by owner"), board, params
    ) == KANBAN_EXECUTION_UNKNOWN_EXIT_CODE


@pytest.mark.asyncio
async def test_a_result_from_another_claim_is_not_reported(board):
    """F3's gate, the other direction: a ``worker_result`` carrying a DIFFERENT claim lock is
    another attempt's outcome, so it is refused and the old raise stands."""
    task = _claimed(board)
    with _connect(board) as conn, kb.write_txn(conn):
        kb._append_event(conn, task.id, "worker_bound",
                         {"pid": 4242, "claim_lock": task.claim_lock},
                         run_id=task.current_run_id)
        kb._append_event(conn, task.id, "worker_result",
                         {"pid": 4242, "claim_lock": "somebody-elses-claim",
                          "exit_code": 0, "last_output": ""},
                         run_id=task.current_run_id)
    owner = _Owner({"kanban.run": [_admitted({"status": "unknown", "admission_id": "a1"})]})

    async with GatewayClient(owner) as client:
        with pytest.raises(GatewayClientError, match="execution unknown"):
            await kwc._submit_claim(client, _params(task, board), str(board))


# --------------------------------------------------------------------------- F5: reconnect


@pytest.mark.asyncio
async def test_a_dropped_socket_is_repolled_by_admission_never_resubmitted(board, monkeypatch):
    """M4 — after the owner admitted, a socket that dies under the receipt poll is re-dialed and
    the SAME ``admission_id`` polled again. ``kanban.run`` is sent exactly once."""
    task = _claimed(board)
    _bind(board, task, pid=4242, exit_code=0, last_output="done")
    params = _params(task, board)
    dropped = GatewayClientError("Gateway disconnected; turn outcome is unknown. Do not resend.")
    first, second = (
        _Owner({"kanban.run": [_admitted({"status": "started", "admission_id": "a1"})],
                "prompt.receipt": [dropped]}),
        _Owner({"prompt.receipt": [{"status": "terminal", "admission_id": "a1"}]}),
    )
    dialed = []

    @asynccontextmanager
    async def connect(*_a, **_kw):
        dialed.append(len(dialed))
        # The real ``connect_gateway`` enters the client too: without that its reader never runs
        # and every RPC would wait for an answer nobody delivers.
        async with GatewayClient([first, second][len(dialed) - 1]) as client:
            yield client

    monkeypatch.setattr("hermes_cli.gateway_client.connect_gateway", connect)
    monkeypatch.setattr(kwc.asyncio, "sleep", _no_sleep)
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    assert await kwc.run(params, str(board)) == 0

    # M4: one submission across BOTH sockets — the turn is never sent twice.
    assert [r["method"] for r in first.sent] == ["kanban.run", "prompt.receipt"]
    assert [r["method"] for r in second.sent] == ["prompt.receipt"]
    # F5: the reconnect polls the admission the owner already accepted, by its id.
    assert second.sent[0]["params"]["admission_id"] == "a1"
    assert second.sent[0]["params"]["session_id"] == "s1"


@pytest.mark.asyncio
async def test_an_admitted_run_is_never_redialed_when_the_budget_is_spent(board, monkeypatch):
    """F5's bound: a gateway that never comes back keeps the OLD behaviour — raise — so the run
    is booked exactly as it was before the fix."""
    task = _claimed(board)
    _bind(board, task, pid=4242)
    params = _params(task, board)
    dropped = GatewayClientError("Gateway disconnected; turn outcome is unknown. Do not resend.")
    dials = {"n": 0}

    @asynccontextmanager
    async def connect(*_a, **_kw):
        dials["n"] += 1
        owner = _Owner({"kanban.run": [_admitted({"status": "started", "admission_id": "a1"})],
                        "prompt.receipt": [dropped]})
        async with GatewayClient(owner) as client:
            yield client

    monkeypatch.setattr("hermes_cli.gateway_client.connect_gateway", connect)
    monkeypatch.setattr(kwc.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(kwc, "RECEIPT_RECONNECT_S", 0.0)  # budget spent on the first failure
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    with pytest.raises(GatewayClientError, match="disconnected"):
        await kwc.run(params, str(board))
    # The window is armed by the first failure (one reconnect) and already spent by the next:
    # bounded, then the old behaviour — raise — instead of dialing forever.
    assert dials["n"] == 2


# --------------------------------------------------------------------------- F2: the exits


def test_the_three_verdicts_are_distinct_exit_codes(board):
    """F2's contract: (a) refused / (b) never admitted / (c) lost are three DIFFERENT codes,
    and only (b) is the neutral one."""
    from hermes_cli.kanban_db import (KANBAN_ADMISSION_REFUSED_EXIT_CODE,
                                      KANBAN_EXECUTION_UNKNOWN_EXIT_CODE,
                                      KANBAN_GATEWAY_UNAVAILABLE_EXIT_CODE)
    from hermes_state_runtime import RuntimeStoreError
    task = _claimed(board)
    params = _params(task, board)
    codes = {KANBAN_ADMISSION_REFUSED_EXIT_CODE, KANBAN_GATEWAY_UNAVAILABLE_EXIT_CODE,
             KANBAN_EXECUTION_UNKNOWN_EXIT_CODE}
    assert len(codes) == 3 and KANBAN_GATEWAY_UNAVAILABLE_EXIT_CODE == 76

    # (a) the owner's own decision code, whatever the board says.
    assert kwc._submit_exit_code(RuntimeStoreError("stale_kanban_claim"), board, params) \
        == KANBAN_ADMISSION_REFUSED_EXIT_CODE

    # (b) never admitted + a gateway failure.
    assert kwc._submit_exit_code(
        GatewayClientError("Gateway connection failed; no local fallback"), board, params
    ) == KANBAN_GATEWAY_UNAVAILABLE_EXIT_CODE

    # (c) admitted + a lost turn.
    with _connect(board) as conn, kb.write_txn(conn):
        kb._append_event(conn, task.id, "owner_admitted", {"run_id": task.current_run_id},
                         run_id=task.current_run_id)
    assert kwc._submit_exit_code(
        GatewayClientError("Gateway disconnected; turn outcome is unknown."), board, params
    ) == KANBAN_EXECUTION_UNKNOWN_EXIT_CODE

    # A missing env key is config, never the gateway's verdict: it stays the plain exit 1.
    assert kwc._submit_exit_code(KeyError("HERMES_KANBAN_DB"), board, params) == 1
    assert kwc._submit_exit_code(KeyError("HERMES_KANBAN_DB"), None, None) == 1


def _dead_worker_pid() -> int:
    proc = subprocess.Popen(["sleep", "0.05"])
    proc.wait()
    return proc.pid


def _park_as_dead_worker(conn, task, pid):
    """Leave the claim exactly as a crashed submitter leaves it: ``running``, a dead pid, no
    fingerprint (legacy row -> liveness is a plain ``/proc`` existence check)."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_started_at = NULL WHERE id = ?",
            (pid, task.id))


def _register_exit(pid, code):
    """Feed the reap registry the way a reaped child would: the raw wait status of ``code``."""
    kbd._recent_worker_exits[pid] = (kbd._wait_status_from_returncode(code), 10**9)


def test_gateway_unavailable_exit_releases_the_card_without_a_failure(board):
    """M5a — exit 76 -> outcome ``gateway_unavailable``, ``consecutive_failures`` untouched, the
    card released (NOT left ``running``), and the run visible for ``hermes kanban runs``."""
    task = _claimed(board)
    pid = _dead_worker_pid()
    _park_as_dead_worker(_connect(board), task, pid)
    _register_exit(pid, 76)

    conn = _connect(board)
    conn.row_factory = sqlite3.Row
    crashed = kbd.detect_crashed_workers(conn)

    assert crashed == []  # not a crash: never accounted, never in the breaker's path
    assert list(kbd.detect_crashed_workers._last_gateway_unavailable) == [task.id]
    run = dict(conn.execute("SELECT * FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                            (task.id,)).fetchone())
    assert run["outcome"] == "gateway_unavailable" and run["ended_at"]
    row = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task.id,)).fetchone())
    assert row["consecutive_failures"] == 0
    assert row["status"] == "ready"  # released, never left running
    assert row["last_failure_error"] in (None, "")  # nothing about the CARD failed
    kinds = [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task.id,))]
    assert "gateway_unavailable" in kinds and "crashed" not in kinds and "gave_up" not in kinds

    # M5a: the guard spaces the next spawn instead of burning one per tick, and never trips a
    # breaker (no ``gave_up``/``blocked`` anywhere on the card).
    assert kbd.check_respawn_guard(conn, task.id) == "infrastructure_cooldown"
    assert not [k for k in kinds if k in ("gave_up", "blocked")]


def test_owner_refusal_exit_still_counts_a_failure(board):
    """M5b — exit 77 (the owner refused) keeps the behaviour it had before this card: booked as a
    crash and counted against ``consecutive_failures``."""
    task = _claimed(board)
    pid = _dead_worker_pid()
    _park_as_dead_worker(_connect(board), task, pid)
    _register_exit(pid, 77)

    conn = _connect(board)
    conn.row_factory = sqlite3.Row
    crashed = kbd.detect_crashed_workers(conn)

    assert crashed == [task.id]
    run = dict(conn.execute("SELECT * FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                            (task.id,)).fetchone())
    assert run["outcome"] == "crashed"
    row = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (task.id,)).fetchone())
    assert row["consecutive_failures"] == 1


def test_a_gateway_outage_run_is_neutral_for_the_violation_budget(board):
    """F2 in the one place ``rate_limited`` is neutral too: an outage run neither consumes nor
    extends a protocol-violation streak."""
    task = _claimed(board)
    conn = _connect(board)
    conn.row_factory = sqlite3.Row
    with kb.write_txn(conn):
        for outcome, marker in (("crashed", {"protocol_violation": True}),
                                ("gateway_unavailable", {}),
                                ("crashed", {"protocol_violation": True})):
            conn.execute(
                "INSERT INTO task_runs (task_id, status, outcome, metadata, started_at, ended_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (task.id, outcome, outcome, json.dumps(marker), 100, 200))
    assert kbd._protocol_violation_streak(conn, task.id) == 2


# ------------------------------------------------------------------- R2 [t_dd382f5c]: the five
# round-2 fixes on this group — F5 livelock bounds, derived F6, the runtime_draining bucket,
# the gateway-unavailable ceiling, and the fail-closed gate.


@pytest.mark.asyncio
async def test_the_admitted_redial_loop_is_bounded_by_the_owner_wait_deadline(board, monkeypatch):
    """R2-1a — the livelock's hard bound: once the OWNER_WAIT_S deadline is spent, an admitted
    loop raises even when every poll 'succeeds' with ``unknown``. Before the fix this branch
    never consulted ``deadline`` and dialed forever (41+ dials past 50 s at a 0.5 s window)."""
    task = _claimed(board)
    _bind(board, task, pid=4242)  # bound, no result: every unknown poll ends in the F3 raise
    params = _params(task, board)
    dials = {"n": 0}

    @asynccontextmanager
    async def connect(*_a, **_kw):
        dials["n"] += 1
        owner = _Owner({"kanban.run": [_admitted({"status": "started", "admission_id": "a1"})],
                        "prompt.receipt": [{"status": "unknown", "admission_id": "a1"}]})
        async with GatewayClient(owner) as client:
            yield client

    monkeypatch.setattr("hermes_cli.gateway_client.connect_gateway", connect)
    monkeypatch.setattr(kwc.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(kwc, "OWNER_WAIT_S", 0.0)  # deadline = the moment run() starts
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    with pytest.raises(GatewayClientError, match="execution unknown"):
        await kwc.run(params, str(board))
    assert dials["n"] == 1  # admitted, polled, raised — and the loop ENDS at the deadline


@pytest.mark.asyncio
async def test_an_unknown_poll_does_not_rearm_the_reconnect_window():
    """R2-1b — the soft bound: a poll that answers ``unknown`` proves the SOCKET, not the
    outcome, so it must not close the ``RECEIPT_RECONNECT_S`` window. Any other status proves
    the turn alive and closes it exactly as F5 designed."""
    admission = kwc._Admission()
    admission.session_id = "s1"                 # __slots__ fields typed None by __init__; the
    admission.admission_id = "a1"               # runtime contract, not the annotation, is what
    admission.reconnect_deadline = 123.0        # this test arms and reads.

    class _Client:
        def __init__(self, receipt):
            self._receipt = receipt

        async def rpc(self, method, **kw):
            assert method == "prompt.receipt" and kw["admission_id"] == "a1"
            return dict(self._receipt)

    admission.reconnect_deadline = 123.0
    await kwc._poll_receipt(_Client({"status": "unknown"}), admission)
    assert admission.reconnect_deadline == 123.0  # unknown: window STAYS armed
    await kwc._poll_receipt(_Client({"status": "started"}), admission)
    assert admission.reconnect_deadline is None   # alive turn: healthy socket, window closed


def test_owner_wait_s_is_derived_from_the_unit_budget():
    """R2-2 — F6's false premise was a hand-asserted unit number. The constant may now only
    equal the SAME derivation that sizes ``TimeoutStopSec`` (gateway/restart.py through the
    gateway CLI's own config getters) plus the declared margin — so a config change moves the
    wait automatically and a stale comment cannot outrun the test. Where the host's installed
    unit is readable, the wait must also strictly outlive it."""
    from hermes_cli.gateway import _get_cron_drain_timeout, _get_restart_drain_timeout
    from gateway.restart import resolve_systemd_timeout_stop_sec

    stop = resolve_systemd_timeout_stop_sec(_get_restart_drain_timeout(), _get_cron_drain_timeout())
    assert kwc.OWNER_WAIT_S == pytest.approx(stop + kwc.OWNER_WAIT_MARGIN_S)
    assert kwc.OWNER_WAIT_S > stop  # the real margin is positive: never expire mid-drain
    unit = Path.home() / ".config" / "systemd" / "user" / "hermes-gateway.service"
    if unit.is_file():  # host-specific; skipped where no user unit is installed
        sec = [int(l.split("=", 1)[1]) for l in unit.read_text().splitlines()
               if l.strip().startswith("TimeoutStopSec=")]
        if sec:
            assert kwc.OWNER_WAIT_S > sec[0]


def test_runtime_draining_is_neutral_and_a_real_refusal_still_counts(board):
    """R2-3 — both directions. A draining host: no owner-refusal label, no failure spent
    (exit 76), still redialable while the drain lasts (F8). A real owner refusal: unchanged
    — exit 77 and counted."""
    from hermes_cli.kanban_db import (KANBAN_ADMISSION_REFUSED_EXIT_CODE,
                                      KANBAN_EXECUTION_UNKNOWN_EXIT_CODE,
                                      KANBAN_GATEWAY_UNAVAILABLE_EXIT_CODE)
    from hermes_state_runtime import RuntimeStoreError
    task = _claimed(board)
    params = _params(task, board)

    draining = RuntimeStoreError("runtime_draining")
    assert kwc._owner_decision_code(draining) is None          # not an owner's decision [F1]
    assert kwc._retryable_failure(draining) is True             # still redialable [F8]
    assert kwc._host_failure(draining) is True                  # the host's condition [F2]
    assert kwc._submit_exit_code(draining, board, params) == KANBAN_GATEWAY_UNAVAILABLE_EXIT_CODE

    # Admitted-then-drained keeps the counted verdict (69) — neutrality never covers a real turn.
    with _connect(board) as conn, kb.write_txn(conn):
        kb._append_event(conn, task.id, "owner_admitted", {"run_id": task.current_run_id},
                         run_id=task.current_run_id)
    assert kwc._submit_exit_code(draining, board, params) == KANBAN_EXECUTION_UNKNOWN_EXIT_CODE

    # A genuine owner refusal is untouched by the move: decision code, exit 77, counted by M5b.
    for code in ("stale_kanban_claim", "invalid_kanban_claim", "profile_mismatch"):
        assert kwc._owner_decision_code(RuntimeStoreError(code)) == code
        assert kwc._submit_exit_code(RuntimeStoreError(code), board, params) \
            == KANBAN_ADMISSION_REFUSED_EXIT_CODE


def _outage_runs(conn, task, n, *, base_ts):
    with kb.write_txn(conn):
        for i in range(n):
            conn.execute(
                "INSERT INTO task_runs (task_id, status, outcome, metadata, started_at, ended_at) "
                "VALUES (?, 'gateway_unavailable', 'gateway_unavailable', NULL, ?, ?)",
                (task.id, base_ts + i, base_ts + 500 + i))


def test_gateway_unavailable_streak_trips_a_ceiling_without_counting(board):
    """R2-4 — the ceiling: ``_GATEWAY_UNAVAILABLE_STREAK_LIMIT`` back-to-back outage runs past
    the cooldown switch the silent spaced retry into a sticky ``gave_up`` block (creator
    notify = the alert path), while ``consecutive_failures`` stays exactly zero."""
    task = _claimed(board)
    conn = _connect(board)
    conn.row_factory = sqlite3.Row
    past = int(time.time()) - 10_000  # every run well outside the cooldown
    _outage_runs(conn, task, kbd._GATEWAY_UNAVAILABLE_STREAK_LIMIT - 1, base_ts=past)
    assert kbd.check_respawn_guard(conn, task.id) is None  # below: F2 fall-through unchanged
    _outage_runs(conn, task, 1, base_ts=past + 100)
    assert kbd.check_respawn_guard(conn, task.id) == "gateway_unavailable_streak"

    with kb.write_txn(conn):  # production shape: the guard runs against a ready row
        conn.execute("UPDATE tasks SET status = 'ready', claim_lock = NULL, worker_pid = NULL "
                     "WHERE id = ?", (task.id,))
    res = kbd.DispatchResult()
    assert kbd._trip_gateway_unavailable_streak(conn, task.id, res) is True
    row = dict(conn.execute("SELECT status, consecutive_failures, last_failure_error "
                            "FROM tasks WHERE id = ?", (task.id,)).fetchone())
    assert row["status"] == "blocked"
    assert row["consecutive_failures"] == 0  # preserve_counter: the ceiling never books failures
    assert "gateway outage streak" in row["last_failure_error"]
    assert res.auto_blocked == [task.id]
    gave = conn.execute("SELECT payload FROM task_events WHERE task_id = ? AND kind = 'gave_up' "
                        "ORDER BY id DESC LIMIT 1", (task.id,)).fetchone()
    assert gave is not None
    payload = json.loads(gave["payload"])
    assert payload["sticky"] is True
    assert payload["gateway_unavailable_streak"] == kbd._GATEWAY_UNAVAILABLE_STREAK_LIMIT


def test_the_admission_gate_resolves_and_fails_closed(tmp_path, board):
    """R2-5 — the gate the docstring promises. The ONLY answer that permits a redial is a
    marker-free SELECT from a board that was actually read: a readable board with no marker
    still says False, everything unreadable — symlink loop, absent file, garbage path — says
    True (redial FORBIDDEN). The old raw-path shape answered 'no board file' -> False for the
    loop/absent cases: the exact fail-OPEN the audit found."""
    task = _claimed(board)
    params = _params(task, board)

    # The normal redial case is preserved: a real, readable board with no marker proves absence.
    assert kwc._owner_admitted(str(board), params) is False

    with _connect(board) as conn, kb.write_txn(conn):
        kb._append_event(conn, task.id, "owner_admitted", {"run_id": task.current_run_id},
                         run_id=task.current_run_id)

    # Same file, non-normalized raw string: the marker is still found through resolve().
    looped = Path(board).parent / "." / ".." / Path(board).name
    assert kwc._owner_admitted(str(looped), params) is True

    loop_a = tmp_path / "loop-a"
    loop_b = tmp_path / "loop-b"
    os.symlink(loop_b, loop_a)
    os.symlink(loop_a, loop_b)
    assert os.path.exists(loop_a) is False      # the old raw read: "missing" -> redial allowed
    assert kwc._owner_admitted(str(loop_a), params) is True  # unreadable -> fail CLOSED
    assert kwc._owner_admitted("", params) is True            # garbage path -> fail CLOSED
    assert kwc._owner_admitted(str(tmp_path / "nowhere.db"), params) is True  # absent -> CLOSED
