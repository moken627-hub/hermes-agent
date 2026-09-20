"""Display label for the provider that actually serves a session's model.

5ac patch (2026-09-20). A *named* custom provider (``providers.<name>`` in config.yaml, e.g.
``bai``) resolves to the generic runtime kind ``custom``:
``resolve_runtime_provider()`` returns ``provider='custom'``,
``source='custom_provider:<name>'``, ``requested_provider='<name>'``. Every UI surface that
printed ``agent.provider`` therefore read ``Provider: custom`` and hid which endpoint actually
served the model.

:func:`display_provider_label` maps that back to the configured display name (``B.AI``) so the
session trailer, gateway session info and ACP/TUI panels all agree. Routing is untouched — this
is display only; callers keep using the runtime ``custom`` provider for credentials/clients.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _norm_url(url: Any) -> str:
    return str(url or "").strip().rstrip("/").lower()


def _configured_providers() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config_readonly

        providers = load_config_readonly().get("providers")
        return providers if isinstance(providers, dict) else {}
    except Exception:
        logger.debug("provider display label: config unavailable", exc_info=True)
        return {}


def display_provider_label(
    provider: Optional[str],
    requested: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """Label for *provider*: a named custom provider's display name, else the id unchanged.

    ``provider`` is the runtime kind (``custom`` for user-defined providers), ``requested`` the
    configured provider id (``bai``), ``base_url`` the endpoint the session actually talks to.
    Falls back to the requested id, then to the runtime kind, so an unlabeled endpoint still
    shows something truthful instead of an invented name.
    """
    kind = str(provider or "").strip()
    if kind != "custom":
        return kind

    requested_id = str(requested or "").strip()
    if requested_id.lower() in {"custom", "auto"}:
        requested_id = ""

    providers = _configured_providers()
    entry: Any = None
    if requested_id:
        candidate = providers.get(requested_id)
        if isinstance(candidate, dict):
            entry = candidate
    if entry is None and base_url:
        wanted = _norm_url(base_url)
        for value in providers.values():
            if isinstance(value, dict) and wanted and _norm_url(value.get("base_url")) == wanted:
                entry = value
                break

    if isinstance(entry, dict):
        name = str(entry.get("name") or "").strip()
        if name:
            return name
    return requested_id or kind
