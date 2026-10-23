"""Kanban worker sessions resolve SMART approvals non-interactively (t_b1d88c90).

Owner directive 16/10/2026: the smart-guardian aux LLM is the SOLE approver for kanban
one-shot workers. A gateway-dispatched worker session has ``HERMES_KANBAN_TASK`` set, and
the guardian step must NOT fall through to an operator prompt (Telegram card) or a gateway
timeout wait on DENY or ESCALATE — both resolve fail-closed instantly.

Invariants under test:
  1. kanban session + guardian DENY  -> instant BLOCKED (smart_denied), no gateway prompt.
  2. kanban session + guardian ESCALATE -> instant fail-closed BLOCKED, no gateway prompt.
  3. kanban session + guardian APPROVE -> approved (unchanged behavior).
  4. NON-kanban gateway session + DENY -> legacy owner-override prompt still happens.
  5. NON-kanban gateway session + ESCALATE -> legacy prompt still happens.
  6. kanban session + manual mode (smart=False) -> untouched: still prompts (patch only
     makes the SMART lane stricter).
"""

import os
import unittest
from unittest.mock import patch

from tools import approval

_KANBAN_ENV = {"HERMES_KANBAN_TASK": "t_unittest"}


def _make_wait_mock():
    """Gateway wait mock: a human answers 'once' as soon as asked."""
    def _wait(session_key, notify_cb, data, surface="gateway"):
        return {"resolved": True, "choice": "once", "reason": None, "cancelled": None}
    return _wait


def _run_decision(verdict: str, *, kanban: bool, smart: bool = True, session_key: str = "s-unit"):
    env = dict(_KANBAN_ENV) if kanban else {}
    with patch.dict(os.environ, env):
        if not kanban:
            os.environ.pop("HERMES_KANBAN_TASK", None)  # restored by patch.dict on exit
        with patch("tools.approval._smart_verdict", return_value=verdict) as _sv, \
             patch("tools.approval._gateway_notify_cb", return_value=(lambda *a, **k: None)), \
             patch("tools.approval._await_gateway_decision", side_effect=_make_wait_mock()) as wait, \
             patch("tools.approval._present_with_selected_transport",
                   return_value={"selected": False}) as transport:
            res = approval._human_decision(
                approval._COMMAND_GATE, command="rm -rf /tmp/unit", description="recursive delete",
                pattern_key="danger:rm-rf", pattern_keys=["danger:rm-rf"], session_key=session_key,
                approval_callback=None, is_cli=False, is_gateway=True, is_ask=False, smart=smart)
    return res, wait, transport


class TestKanbanGuardianOnlyFlag(unittest.TestCase):
    def test_flag_reads_live_env(self):
        with patch.dict(os.environ, _KANBAN_ENV):
            assert approval._kanban_guardian_only() is True
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_KANBAN_TASK", None)
            assert approval._kanban_guardian_only() is False

    def test_empty_value_is_not_kanban(self):
        with patch.dict(os.environ, {"HERMES_KANBAN_TASK": ""}):
            assert approval._kanban_guardian_only() is False


class TestKanbanSmartLaneNonInteractive(unittest.TestCase):
    def test_deny_resolves_instantly_without_operator(self):
        res, wait, transport = _run_decision("deny", kanban=True, session_key="s-k-deny")
        assert res["approved"] is False
        assert res.get("smart_denied") is True
        wait.assert_not_called()
        transport.assert_not_called()

    def test_escalate_fails_closed_without_operator(self):
        res, wait, transport = _run_decision("escalate", kanban=True, session_key="s-k-esc")
        assert res["approved"] is False
        assert res.get("outcome") == "blocked"
        assert res.get("smart_escalated") is True
        assert "non-interactively" in res["message"]
        wait.assert_not_called()
        transport.assert_not_called()

    def test_approve_unchanged(self):
        res, wait, transport = _run_decision("approve", kanban=True, session_key="s-k-appro")
        assert res["approved"] is True
        wait.assert_not_called()
        transport.assert_not_called()


class TestNonKanbanBehaviorPreserved(unittest.TestCase):
    def test_deny_still_offers_owner_override(self):
        res, wait, _transport = _run_decision("deny", kanban=False, session_key="s-n-deny")
        # the mock answers "once" -> granted; the invariant is that the operator WAS asked
        wait.assert_called_once()
        assert res["approved"] is True

    def test_escalate_still_prompts(self):
        _res, wait, _transport = _run_decision("escalate", kanban=False, session_key="s-n-esc")
        wait.assert_called_once()

    def test_manual_mode_in_kanban_still_prompts(self):
        # smart=False (approvals.mode: manual) — the patch must not silence manual approvals.
        _res, wait, _transport = _run_decision("escalate", kanban=True, smart=False,
                                             session_key="s-k-manual")
        wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
