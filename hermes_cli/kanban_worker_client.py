"""Dispatcher subprocess: submit one claim to its profile owner; never run an agent."""
# Spawned as a bare ``sys.executable -m``: in a PM install that interpreter carries no dependencies
# until hermes_bootstrap selects the committed environment, so it must be the first import.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap":
        raise

import asyncio
import os
from pathlib import Path
import sqlite3
import sys
import time

# A worker the dispatcher spawns while `hermes update` drains the owner (or while the new owner is
# still starting) waits for it instead of exiting: the condition is the host's, not the card's, and
# a failed exit here spends the card's failure budget on an update window. Nothing is submitted
# until an owner is ready, so waiting can never run the claim twice.
# [t_83195837:F6] raised 180 -> 300 -> 360 by hand-asserted unit numbers; [t_dd382f5c:R2-2] the
# value is now DERIVED, never asserted. Measured 28/10 on THIS host, the gateway unit runs
# ``TimeoutStopSec=210`` (/root/.config/systemd/user/hermes-gateway.service) — exactly what
# ``gateway/restart.py::resolve_systemd_timeout_stop_sec()`` sizes the unit from here:
# ``max(restart_drain_timeout=180, cron_drain_timeout=30 + reserve=10) + headroom 30 = 210``.
# The stale "300" premise was a different host's number: at 360 the real margin over 210 was 150 s
# (a needlessly long blind wait); the earlier margin-zero worry only exists once the wait is
# pinned below the unit value. The wait must outlive the WHOLE stop: systemd may SIGKILL the
# cgroup at TimeoutStopSec, and the fresh owner still needs to boot and answer. So:
#     OWNER_WAIT_S = resolve_systemd_timeout_stop_sec(config) + OWNER_WAIT_MARGIN_S
# which on THIS host gives 210 + 60 = 270 s. A test binds the constant to the same config the
# unit generator reads, so a stale number cannot survive a config change unnoticed. The owner
# wait has exactly ONE derivation — no other site may hard-code the value.
OWNER_WAIT_MARGIN_S = 60.0
OWNER_WAIT_S_FALLBACK = 270.0  # documented source: THIS host's measured 210 s unit + 60 s margin


def resolve_owner_wait_s() -> float:
    """The submitter's owner-wait budget, derived from the SAME configured drain values that
    ``hermes_cli.gateway``'s unit generator writes into ``TimeoutStopSec`` (see the block above).

    Mirrors ``_get_restart_drain_timeout``/``_get_cron_drain_timeout`` (env override, then
    ``agent.*`` config, then parse-with-defaults) without importing the heavy gateway CLI —
    this runs in a bare dispatcher subprocess on every claim."""
    from hermes_cli.config import read_raw_config
    from gateway.restart import (DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT, parse_cron_drain_timeout,
                                 parse_restart_drain_timeout, resolve_systemd_timeout_stop_sec)
    cfg = read_raw_config()
    agent_cfg = cfg.get('agent', {}) if isinstance(cfg, dict) else {}
    drain_raw = os.getenv('HERMES_RESTART_DRAIN_TIMEOUT', '').strip()
    if not drain_raw:
        drain_raw = str(agent_cfg.get('restart_drain_timeout', DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT))
    drain = parse_restart_drain_timeout(drain_raw)
    cron_raw = os.getenv('HERMES_CRON_DRAIN_TIMEOUT')
    if cron_raw is None or not str(cron_raw).strip():
        cron = parse_cron_drain_timeout(agent_cfg.get('cron_drain_timeout')
                                        if 'cron_drain_timeout' in agent_cfg else None)
    else:
        cron = parse_cron_drain_timeout(cron_raw)
    return float(resolve_systemd_timeout_stop_sec(drain, cron)) + OWNER_WAIT_MARGIN_S


try:
    OWNER_WAIT_S = resolve_owner_wait_s()
except Exception:  # unreadable config: the documented fallback, never a silent 0-length wait
    OWNER_WAIT_S = OWNER_WAIT_S_FALLBACK
_TRANSIENT_OWNER_STATES = frozenset({'draining', 'starting'})
# Past the board lock wait: the owner's session setup and receipt before and after the write.
ADMISSION_ANSWER_MARGIN_S = 60.0

# [5ac:owner-transport-retry] an owner alive but thrashing under the host's RAM ceiling fails the
# websocket dial with one of these transport messages (upstream issue #139612 class of burn).
_TRANSPORT_FAILURE_PREFIXES = ('Gateway connection failed', 'Gateway connection timed out')


def _is_transport_failure(exc):  # [5ac:owner-transport-retry] host condition, not the card's
    return str(exc).startswith(_TRANSPORT_FAILURE_PREFIXES)


# [t_83195837:F1] the bounded wire codes the OWNER itself decides (gateway/session_kanban.
# admit_claim and the runtime store raise them). ONLY these may be reported as an owner refusal;
# every other failure here — missing env, dead socket, stopped gateway — is the host's condition
# and must stop wearing the "owner refused" label (that catch-all made every audit conclude the
# owner had refused when it had never been asked).
# [t_dd382f5c:R2-3] ``runtime_draining`` was in this set — but a host that is draining is NOT an
# owner making a decision about THIS card: calling it a refusal re-created the exact F1 mislabel
# and spent ``consecutive_failures`` (exit 77) on the host's own shutdown. It moved to the
# transient-refusal class below: redialable while the drain lasts, and on budget exhaustion it
# books the neutral ``gateway_unavailable`` (exit 76), never the owner-refused label.
_OWNER_DECISION_CODES = frozenset({'invalid_kanban_claim', 'stale_kanban_claim', 'profile_mismatch'})

# [t_83195837:F8] the one refusal the submitter may outlive: the owner's runtime drains for a
# restart and refuses BEFORE it admits anything, so redialing once the admission invariant below
# holds cannot run the claim twice. Deliberately OUTSIDE _OWNER_DECISION_CODES [t_dd382f5c:R2-3].
_TRANSIENT_REFUSAL_CODES = frozenset({'runtime_draining'})


def _owner_decision_code(exc):  # [t_83195837:F1] an owner's code, else None
    """``exc``'s owner decision code, or ``None`` when the failure is not an owner's decision.

    ``RuntimeStoreError.reason`` is the bounded wire code and ``GatewayRPCError`` carries that same
    code as its message; a message that merely mentions a code (``Gateway draining: ...``) is a host
    state, not a decision, and stays None."""
    code = getattr(exc, 'reason', None)
    if isinstance(code, str) and code in _OWNER_DECISION_CODES:
        return code
    text = str(exc)
    return text if text in _OWNER_DECISION_CODES else None


def _transient_refusal_code(exc):  # [t_dd382f5c:R2-3] the drain class is decided apart from F1
    """The transient-refusal wire code of ``exc``, or ``None``.

    Same two wire shapes as :func:`_owner_decision_code` (the runtime store's ``reason``
    attribute, the RPC error's bare code as its message) but keyed on
    ``_TRANSIENT_REFUSAL_CODES``: a draining host stays redialable (F8) while no longer
    counting as an owner decision (F1/F2)."""
    code = getattr(exc, 'reason', None)
    if isinstance(code, str) and code in _TRANSIENT_REFUSAL_CODES:
        return code
    text = str(exc)
    return text if text in _TRANSIENT_REFUSAL_CODES else None


def _retryable_failure(exc):  # [t_83195837:F4/F8] class gate only — the invariant is asserted apart
    """Whether ``exc`` is a class the submitter may redial AT ALL. The admission invariant is
    checked separately before every attempt, so a retryable class can never become a duplicate."""
    if isinstance(exc, OSError) or _is_transport_failure(exc):
        return True  # [5ac:owner-transport-retry] host transport: RAM-ceiling thrash, restart race
    return _transient_refusal_code(exc) is not None


def _owner_admitted(board_db, params):  # [t_83195837:F4] SELECT only — never writes the board
    """Has the owner already admitted THIS run?

    The duplicate-free invariant: ``admit_claim`` writes the ``owner_admitted`` marker in the SAME
    board transaction that accepts the claim, so no marker for this run means the claim was never
    accepted and a redial cannot run it twice.

    [t_dd382f5c:R2-5] the old shape read the RAW path and answered "no marker" (redial ALLOWED)
    whenever the file merely looked missing — a fail-OPEN the docstring had already promised
    against, because the owner is handed ``str(Path(board_db).resolve())`` (see
    :func:`_submit_claim`): any raw-vs-resolved divergence (loops, renames mid-flight,
    non-normalized paths) read an empty or absent file while the REAL board already held an
    ``owner_admitted`` marker. Now: the path is resolved the same way, and the ONLY answer that
    permits a redial is a marker-free SELECT from a board we actually read. Unreadable —
    unresolvable, absent, corrupt, directory, garbage — answers True (fail closed, no redial):
    absence of the FILE is no longer treated as proof of absence of the TURN."""
    task_id = str(params.get('task_id') or '')
    run_id = params.get('run_id')
    try:
        path = os.fspath(Path(board_db).resolve())
        if not os.path.exists(path):
            return True  # absent after RESOLVING = unreadable: absence of the file proves
                         # nothing about the turn — fail closed (and never let sqlite create it)
        conn = sqlite3.connect(path, timeout=2.0)
        try:
            if run_id is None:
                row = conn.execute(
                    "SELECT 1 FROM task_events WHERE task_id=? AND kind='owner_admitted' LIMIT 1",
                    (task_id,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM task_events WHERE task_id=? AND run_id=? AND kind='owner_admitted' LIMIT 1",
                    (task_id, int(run_id))).fetchone()
        finally:
            conn.close()
        return row is not None
    except Exception:  # unreadable/corrupt/absent board: never redial on an unproven invariant
        return True


async def _never_admitted(board_db, params):  # [t_83195837:F4] asserted before EVERY redial
    return not await asyncio.to_thread(_owner_admitted, board_db, params)


def admission_answer_budget():
    """Seconds to wait for ``kanban.run``'s answer: the longest the owner's admission write can
    wait on the board lock (``write_txn`` retries ``BEGIN IMMEDIATE`` ``_BUSY_MAX_RETRIES`` times,
    each under the busy timeout) plus ``ADMISSION_ANSWER_MARGIN_S``."""
    from hermes_cli import kanban_db_connect
    lock_wait = (kanban_db_connect._BUSY_MAX_RETRIES + 1) * kanban_db_connect._resolve_busy_timeout_ms() / 1000
    return lock_wait + ADMISSION_ANSWER_MARGIN_S


async def run(params, board_db):
    from hermes_state_runtime import RuntimeStoreError
    from hermes_cli.gateway_client import connect_gateway, GatewayClientError, GatewayUnavailableError
    if os.environ.get('HERMES_TUI_GATEWAY_URL'):
        raise GatewayClientError(LOCAL_OWNER_REQUIRED)
    deadline = time.monotonic() + OWNER_WAIT_S
    # [t_c995ea84:F5] the admission outlives every socket this loop opens: after the owner has
    # accepted the attempt the loop only ever re-polls it, never resubmits it.
    admission = _Admission()
    while True:
        try:
            async with connect_gateway() as client:
                # [5ac:owner-transport-retry] a socket this loop opens is the point of no return:
                # anything after it may already have submitted the claim — which is why every redial
                # below must first prove ON THE BOARD that the owner never admitted this run: the
                # gate is the owner_admitted marker, not "had we opened a socket yet" [t_83195837:F4].
                # kanban.run subscribes this socket to the card's session, but only receipts matter
                # here: discard the turn's events, or they fill the client's bounded queue and its
                # reader stops, with the next receipt behind it.
                drain = asyncio.create_task(_discard(client.events))
                try:
                    return await _submit_claim(client, params, board_db, admission)
                finally:
                    drain.cancel()
        except GatewayUnavailableError as exc:
            # [t_c995ea84:F5] no ready owner WHILE the turn already exists: re-dial and poll the
            # admission we own. The turn is the owner's; this loop never submits it again.
            if admission.admitted:
                # [t_dd382f5c:R2-1a] the ONE constant that bounds this loop is the owner-wait
                # deadline: an admitted run may keep re-dialing only inside it. Without this
                # check the branch looped unbounded (the F5 livelock: 41+ dials past 50 s).
                if time.monotonic() >= deadline or not _may_reconnect(admission):
                    raise
                await asyncio.sleep(1)
                continue
            # No ready owner to dial (drain/starting window). [t_83195837:F4] still assert the
            # admission invariant: on iteration >1 something may have reached the owner after the
            # previous proof, and an admitted run is never redialed.
            if exc.state not in _TRANSIENT_OWNER_STATES or time.monotonic() >= deadline \
                    or not await _never_admitted(board_db, params):
                raise
            await asyncio.sleep(1)
        except (GatewayClientError, OSError, RuntimeStoreError) as exc:
            # [t_c995ea84:F5] admitted: the socket died under a receipt we were polling (or the
            # admission answer itself was lost on the wire). Reconnect and poll by admission_id —
            # the one thing never allowed here is putting ``kanban.run`` on the wire twice.
            # When this budget is spent the old behaviour stands: raise, and the dispatcher books
            # the run exactly as it did before this fix.
            if admission.admitted:
                # [t_dd382f5c:R2-1a] bounded twice over: the owner-wait deadline caps the WHOLE
                # loop, the reconnect window below only ever trims it further (see R2-1b for the
                # arm/disarm rule the livelock broke).
                if time.monotonic() >= deadline or not _may_reconnect(admission):
                    raise
                await asyncio.sleep(1)
                continue
            # [5ac:owner-transport-retry] A refused/stalled dial before any submission is the host's
            # condition (RAM-ceiling thrash, restart race): redial within the same OWNER_WAIT_S budget
            # instead of spending the card's failure budget.
            # [t_83195837:F4] The redial gate is no longer `submitted` (socket opened == never
            # redial) but the board's own record: redial ONLY while no owner_admitted exists for this
            # run. That is what makes a redial provably duplicate-free — admit_claim answers a repeat
            # with stale_kanban_claim, and the runtime store answers a repeat request_id with the
            # ORIGINAL receipt instead of a second turn.
            # [t_83195837:F8] runtime_draining joins the transport class as transient: the owner
            # refuses it BEFORE it admits anything, so the same invariant gates it.
            # Failures that are neither transport nor a transient refusal ('Kanban execution unknown;
            # retained by owner', a non-transport refusal, a missing env key) keep raising, so the
            # owner still reports them and the work is never resent.
            if time.monotonic() >= deadline or not _retryable_failure(exc) \
                    or not await _never_admitted(board_db, params):
                raise
            await asyncio.sleep(1)


async def _discard(events):
    while True:
        await events.get()


# [t_c995ea84:F5] how long a socket that dropped AFTER the owner admitted may stay unreachable
# before the old behaviour (raise, so the card is booked as before) applies. It covers a restart
# drain plus the owner coming back up. It bounds CONNECTION FAILURE only: a poll that proves the
# turn alive (see :func:`_poll_receipt`) re-arms it, an ``unknown`` receipt does not
# [t_dd382f5c:R2-1b], and the whole admitted loop is additionally capped by the OWNER_WAIT_S
# deadline in run() [t_dd382f5c:R2-1a].
RECEIPT_RECONNECT_S = 300.0

# The submitter's own binding guard, hoisted so the exit-code verdict can name it exactly.
LOCAL_OWNER_REQUIRED = 'Kanban requires the assigned local profile owner'


class _Admission:
    """[t_c995ea84:F5] what the owner already accepted of THIS attempt, across reconnects.

    Once ``kanban.run`` has answered, the turn exists on the owner's side: every later socket
    re-polls that same admission by id. Resubmitting would run the claim twice — the one thing
    ``admit_claim`` answers with ``stale_kanban_claim`` and the ``owner_admitted`` marker exists
    to prevent. ``reconnect_deadline`` is ``None`` while a socket is healthy.
    """
    __slots__ = ('admission_id', 'reconnect_deadline', 'session_id')

    def __init__(self):
        self.session_id = None
        self.admission_id = None
        self.reconnect_deadline = None

    @property
    def admitted(self):
        return self.session_id is not None


def _may_reconnect(admission):
    """[t_c995ea84:F5] May a post-admission socket drop be re-dialed now?

    The first failure arms a ``RECEIPT_RECONNECT_S`` window; a poll that gets through (see
    :func:`_poll_receipt`) disarms it again. So the budget only ever fires on a gateway that
    cannot hold a single connection for that long — there the old behaviour stands: raise.
    """
    now = time.monotonic()
    if admission.reconnect_deadline is None:
        admission.reconnect_deadline = now + RECEIPT_RECONNECT_S
        return True
    return now < admission.reconnect_deadline


async def _poll_receipt(client, admission):
    """One ``prompt.receipt`` against the admission we already own. A poll that proves the turn
    is ALIVE (any status but ``unknown``) shows the socket is healthy again and closes the
    reconnect window; an ``unknown`` receipt does NOT [t_dd382f5c:R2-1b] — the socket answered
    but the settlement is still lost, so disarming there let the submitter re-arm the window on
    every raise→reconnect iteration and dial forever (the F5 livelock)."""
    receipt = await client.rpc('prompt.receipt', session_id=admission.session_id,
                               admission_id=admission.admission_id)
    if receipt.get('status') != 'unknown':
        admission.reconnect_deadline = None
    return receipt


async def _submit_claim(client, params, board_db, admission=None):
    from hermes_cli.gateway_client import GatewayClientError
    from gateway.session_kanban import worker_result, worker_result_for_claim
    admission = _Admission() if admission is None else admission
    if not admission.admitted:
        db = str(await asyncio.to_thread(Path(board_db).resolve))
        # The owner answers only after it checks the claim and writes its ``owner_admitted`` marker
        # in one board write transaction. The board (the cross-profile dispatch bus) can make that
        # wait out its busy timeout, so the 30 s default left the owner admitting an attempt whose
        # submitter had exited (the later-bound worker's submitter-liveness watcher ended it, and
        # the admission was left ``unknown``). The wait is still finite: once the marker commits,
        # ``enforce_max_runtime`` and TTL reclaim skip the run (``owner_reclaim_paused``), so an owner
        # that stalls after admitting is bounded only here. A hang-up fails the wait sooner.
        accepted = await client.rpc('kanban.run', _timeout=admission_answer_budget(), **dict(params, db=db))
        receipt = accepted['receipt']
        admission.session_id, admission.admission_id = accepted['session_id'], receipt['admission_id']
        print(f'Session: {admission.session_id}', file=sys.stderr)
    else:
        # [t_c995ea84:F5] a RECONNECTED socket: the turn already exists, so go straight back to
        # polling its receipt. ``kanban.run`` is never put on the wire a second time.
        receipt = await _poll_receipt(client, admission)
    while receipt['status'] not in {'terminal', 'unknown'}:
        await asyncio.sleep(.2)
        receipt = await _poll_receipt(client, admission)
    if receipt['status'] == 'unknown':
        # [t_c995ea84:F3] The owner lost the settlement, not necessarily the work: the attempt's
        # own ``worker_result`` (written only by the interpreter ``worker_bound`` recorded under
        # THIS claim) is the real outcome, so report it instead of crashing. Missing, mismatched
        # or malformed -> the old raise stands: never synthesize success, never guess.
        result = worker_result_for_claim(board_db, params)
        code = result.get('exit_code') if result is not None else None
        if result is None or not isinstance(code, int) or isinstance(code, bool):
            raise GatewayClientError('Kanban execution unknown; retained by owner')
        if result.get('last_output'):
            print(result['last_output'])
        return code
    result = worker_result(board_db, params)
    # The attempt's reply ends the log, as a one-shot worker's did: crash diagnostics quote it.
    if result.get('last_output'):
        print(result['last_output'])
    return result.get('exit_code', 1)


def main():
    # The dead-worker sweep books this process's exit from the trailer in its log (it never
    # reaps us), so every exit path writes it — the same contract as a quiet one-shot worker.
    from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER
    rc = 1
    try:
        rc = _submit()
    finally:
        print(f'\n{KANBAN_WORKER_EXIT_TRAILER}{int(rc)}', file=sys.stderr, flush=True)
    return rc


def _host_failure(exc):  # [t_c995ea84:F2] the HOST's gateway, never the owner and never config
    """Whether ``exc`` says the GATEWAY could not serve this run.

    Three readings, all host-side: the owner's own not-ready verdict (``GatewayUnavailableError``),
    the transport class F4/F8 already treats as redialable, and a socket that died or disagrees on
    protocol. Anything else (a missing env key, an owner decision) never reaches this predicate.
    """
    from hermes_cli.gateway_client import GatewayUnavailableError
    if isinstance(exc, GatewayUnavailableError):
        return True  # draining / starting / could not start — the gateway's own verdict
    if _retryable_failure(exc):
        return True  # transport, or a transient refusal already handled as (a) above
    return str(exc).startswith('Gateway ')


def _submit_exit_code(exc, board_db, params):  # [t_c995ea84:F2] (a)/(b)/(c), dispatcher-mapped
    """The exit code for one failed submission; the dispatcher maps each to its own bookkeeping.

    (a) 77 — the owner refused the admission: still the card's failure, counted exactly as before.
    (b) 76 — the gateway never admitted this run: neutral ``gateway_unavailable``, no failure spent.
    (c) 69 — admitted, then lost: counted, exactly as before.

    The BOARD decides "admitted": ``admit_claim`` writes ``owner_admitted`` in the same transaction
    that accepts the claim, and :func:`_owner_admitted` fails CLOSED (unreadable board reads as
    admitted), so a verdict this function cannot prove is booked the old way rather than guessed.
    """
    from hermes_cli.kanban_db import (KANBAN_ADMISSION_REFUSED_EXIT_CODE,
        KANBAN_GATEWAY_UNAVAILABLE_EXIT_CODE, KANBAN_EXECUTION_UNKNOWN_EXIT_CODE)
    if isinstance(exc, KeyError) or str(exc) == LOCAL_OWNER_REQUIRED:
        return 1  # missing dispatcher env / an explicit owner binding: a config error, not a verdict
    if _owner_decision_code(exc) is not None:
        return KANBAN_ADMISSION_REFUSED_EXIT_CODE  # (a)
    if board_db is None or params is None:
        return 1  # never even read the board: nothing to classify but the error itself
    if not _owner_admitted(board_db, params):
        return KANBAN_GATEWAY_UNAVAILABLE_EXIT_CODE if _host_failure(exc) else 1  # (b)
    return KANBAN_EXECUTION_UNKNOWN_EXIT_CODE  # (c)


def _submit():
    params = board_db = None
    try:
        params = {'board': os.environ['HERMES_KANBAN_BOARD'], 'task_id': os.environ['HERMES_KANBAN_TASK'],
            'run_id': int(os.environ['HERMES_KANBAN_RUN_ID']), 'claim_lock': os.environ['HERMES_KANBAN_CLAIM_LOCK']}
        board_db = os.environ['HERMES_KANBAN_DB']
        # Owner startup must never inherit this worker's task/tool/goal identity.
        for key in list(os.environ):
            # The busy timeout is a host tuning knob, not task identity: the answer budget reads it.
            if key.startswith('HERMES_KANBAN_') and key not in ('HERMES_KANBAN_HOME', 'HERMES_KANBAN_BUSY_TIMEOUT_MS'):
                os.environ.pop(key)
        for key in ('HERMES_SESSION_SOURCE', 'HERMES_TENANT', 'TERMINAL_CWD'):
            os.environ.pop(key, None)
        return asyncio.run(run(params, board_db))
    except (KeyError, ValueError, OSError) as exc:
        # [t_83195837:F1] label split: an owner's own decision code may be called a refusal;
        # everything else — a missing env key, a dead socket, a stopped gateway — was never the
        # owner's answer. The EXIT CODE now carries the class too [t_c995ea84:F2]: only a gateway
        # that never admitted the run exits 76, so a gateway outage stops being booked as a crash.
        _code = _owner_decision_code(exc)
        if _code is not None:
            print(f'Kanban admission refused by owner: {_code}', file=sys.stderr)
        else:
            print(f'Kanban submit failed (gateway): {exc}', file=sys.stderr)
        return _submit_exit_code(exc, board_db, params)


if __name__ == '__main__':
    raise SystemExit(main())
