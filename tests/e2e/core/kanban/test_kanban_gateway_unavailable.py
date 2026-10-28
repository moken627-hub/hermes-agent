"""[t_c995ea84:M6] Integration — a gateway outage on a REAL card must end in a non-crash outcome
that ``hermes kanban runs`` shows, without spending the card's failure budget or leaving it
``running``. The card is archived afterwards.

Everything except the outage itself is the shipped code running for real: the card is created and
claimed through the CLI's own functions, bound to a real submitter process with the dispatcher's
own ``_set_worker_pid``, that submitter runs ``kanban_worker_client.main()`` to completion (exit
76 + its log trailer), the booking happens in a real ``hermes kanban dispatch`` tick, and the
verdict is read back with the real ``hermes kanban runs`` / ``archive`` commands.

The ONLY seam patched is the transport: ``connect_gateway`` raises the owner's own
``GatewayUnavailableError`` before anything is admitted — the "simulate the drop" the card allows.
Every line below that seam (exit-code verdict, trailer, classification, reclaim, bookkeeping) is
the code under review.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from tests.e2e.core.kanban._helpers import Board

pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="worker liveness reads /proc"),
    pytest.mark.live_system_guard_bypass,  # teardown may signal the spawned submitter
]

PY = sys.executable
GATEWAY_UNAVAILABLE_EXIT = 76  # KANBAN_GATEWAY_UNAVAILABLE_EXIT_CODE

# Bookkeeping on the scratch board through the real kanban_db functions, from a child process so
# no test process ever holds the board the dispatcher ticks against.
HELPER = """
import sqlite3, sys
from hermes_cli import kanban_db as kb

mode = sys.argv[1]
conn = sqlite3.connect(sys.argv[2])
conn.row_factory = sqlite3.Row
if mode == "claim":
    task = kb.claim_task(conn, sys.argv[3])
    assert task is not None, "claim refused"
    print("%s\\t%s" % (task.current_run_id, task.claim_lock))
elif mode == "bind":
    from hermes_cli.kanban_db_dispatch import _set_worker_pid
    # _set_worker_pid opens its own write transaction.
    _set_worker_pid(conn, sys.argv[3], int(sys.argv[4]),
                    run_id=int(sys.argv[5]), claim_lock=sys.argv[6])
    print("bound")
elif mode == "where":
    print(kb.kanban_db_path())
    print(kb.worker_log_path(sys.argv[3]))
    print(kb.get_current_board())
"""

# The real submitter, with one seam replaced: the dial refuses before any admission exists.
DRIVER = """
import sys, time
from contextlib import asynccontextmanager

from hermes_cli import gateway_client, kanban_worker_client

# A window for the parent to record this pid as the claim's worker before we exit.
time.sleep(2.0)


@asynccontextmanager
async def _outage(*_args, **_kwargs):
    raise gateway_client.GatewayUnavailableError(
        "Gateway inaccessible: simulated outage", "inaccessible")
    yield  # unreachable: the raise above is what makes this an async generator


gateway_client.connect_gateway = _outage
sys.exit(kanban_worker_client.main())
"""


def _run(board: Board, args: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run([PY, *args], cwd=str(board.root), env=board.env(),
                          capture_output=True, text=True, timeout=timeout, check=False)


def _helper(board: Board, *args: str) -> list[str]:
    proc = _run(board, ["-c", HELPER, *args])
    assert proc.returncode == 0, f"helper {args[0]} rc={proc.returncode}\n{proc.stderr}"
    return proc.stdout.strip().splitlines()


def test_a_gateway_outage_ends_in_a_visible_non_crash_outcome(tmp_path: Path) -> None:
    board = Board(tmp_path, "http://127.0.0.1:1/v1")
    # An assignee no profile answers for: the tick must book the reclaim, never spawn a worker
    # that would race the assertions below.
    tid = board.create("throwaway: gateway outage must not book a crash",
                       "--assignee", "no-such-profile-t995")
    db, log_path, current_board = _helper(board, "where", str(board.db_path), tid)
    log_path = Path(log_path)

    # 1. Claim the card the real way, then bind a real submitter pid to that claim.
    run_id, claim_lock = _helper(board, "claim", db, tid).pop().split("\t")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    driver = tmp_path / "gateway_outage_driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    env = board.env()
    env.update({
        "HERMES_KANBAN_DB": db,
        "HERMES_KANBAN_BOARD": current_board,
        "HERMES_KANBAN_TASK": tid,
        "HERMES_KANBAN_RUN_ID": str(run_id),
        "HERMES_KANBAN_CLAIM_LOCK": claim_lock,
    })
    with open(log_path, "ab") as log:
        proc = subprocess.Popen([PY, str(driver)], cwd=str(board.root), env=env,
                                stdout=subprocess.DEVNULL, stderr=log)
        _helper(board, "bind", db, tid, str(proc.pid), run_id, claim_lock)
        rc = proc.wait(timeout=120)

    # 2. The submitter's own verdict: exit 76, the F1 label kept, the trailer in the log.
    log_text = board.worker_log(tid)
    assert rc == GATEWAY_UNAVAILABLE_EXIT, f"submitter exited {rc}\n--- log ---\n{log_text[-4000:]}"
    assert f"[kanban-worker-exit] rc={GATEWAY_UNAVAILABLE_EXIT}" in log_text, log_text[-2000:]
    assert "Kanban submit failed (gateway):" in log_text  # the F1 label is untouched
    assert "Kanban admission refused by owner" not in log_text

    # 3. A REAL dispatch tick books it: neutral, never a crash, never counted, never running.
    tick = board.dispatch()
    assert tick.get("gateway_unavailable") == [tid], tick
    assert tick.get("crashed") == [], tick
    assert tick.get("spawned") == [], tick
    assert not board.events(tid, "respawn_guarded"), board.diag(tid)

    run = board.runs(tid)[-1]
    assert run["outcome"] == "gateway_unavailable", run
    task = board.task(tid)
    assert task["status"] == "ready", task["status"]          # released, NOT left running
    assert task["consecutive_failures"] == 0, task            # the budget was not spent
    assert not task["last_failure_error"], task["last_failure_error"]
    for kind in ("crashed", "gave_up", "blocked", "protocol_violation"):
        assert not board.events(tid, kind), f"{kind} event on an outage card:\n{board.diag(tid)}"

    # 4. Visible in `hermes kanban runs` (the binding condition), then archived.
    runs_out = board.cli("runs", tid).stdout
    assert re.search(r"\bgateway_unavailable\b", runs_out), runs_out
    board.cli("archive", tid)
    assert board.task(tid)["status"] == "archived"
