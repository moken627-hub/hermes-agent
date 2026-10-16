"""Plugin slash commands of the classic CLI on the shared gateway (``slash.exec``).

On main a plugin's ``/command`` ran in this process. Rewired, the plugin's hooks, tools and MCP servers
live in the gateway, so its command runs there too (``gateway/session_commands.py``, the messaging
gateway's own plugin dispatch): ``ctx.call_mcp`` reaches the servers the gateway connected, and this
process never imports the plugin to run it. The names come from the gateway's ``commands.catalog``,
read once per session and gateway life (``/new``, resume and a redial read again) and again once when
a typed name misses, so a plugin the gateway enabled since still dispatches; a cached name a plugin
reload dropped since is re-read once when the gateway refuses it, then dispatched again (:data:`GONE`).
A catalog whose ``plugin_discovery_error`` is set lists no plugins because the gateway could not load
them, not because there are none: it is treated as a failed read and never cached. Off the gateway
(tests, the TUI slash worker's shadow CLI) main's dispatch applies.
"""
from __future__ import annotations

import threading

from hermes_cli.gateway_client import GatewayClientError

_CATEGORY = "Plugin commands"  # tui_gateway/command_discovery.py
GONE = object()  # run(): the gateway no longer has the command; the caller dispatches the name again
_GONE_LINE = "  /{}: the gateway no longer has this plugin command; nothing was run."
_STOPPED_LINE = "  /{}: stopped waiting. The command keeps running in the gateway and its reply will not be shown."


class CatalogUnavailable(Exception):
    """The gateway's command list could not be read (already printed): run nothing for this input."""


def _boundary(cli, connection):
    return connection, getattr(connection, "epoch", None), getattr(cli, "session_id", None)


def _read(cli, connection):
    """Fetch the catalog's plugin commands and cache them for the session and gateway life that served
    them (raises GatewayClientError, also when the gateway reports its plugin discovery failed). The
    result is published only if neither moved during the call: a reply from epoch E that arrives after
    a concurrent redial is never relabelled E+1 (``call_tagged``)."""
    session = getattr(cli, "session_id", None)
    catalog, epoch = connection.call_tagged("commands.catalog")
    catalog = catalog or {}
    if catalog.get("plugin_discovery_error"):  # absent on older gateways: today's behaviour
        raise GatewayClientError(f"plugin command discovery failed: {catalog['plugin_discovery_error']}")
    listed = catalog.get("plugin_commands")
    if isinstance(listed, dict):  # every plugin command, also one a display row (Ink's /logs) hides
        found = {str(name).lstrip("/"): str(desc or "") for name, desc in listed.items()}
    else:  # older gateways: the display category
        found = {}
        for category in catalog.get("categories") or []:
            if isinstance(category, dict) and category.get("name") == _CATEGORY:
                for row in category.get("pairs") or []:
                    if isinstance(row, (list, tuple)) and len(row) == 2 and isinstance(row[0], str):
                        found[row[0].lstrip("/")] = str(row[1] or "")
    served = (connection, epoch, session)
    if _boundary(cli, connection) == served:
        cli._gateway_plugin_commands = (served, found)
    return found


def _cached(cli, connection):
    if getattr(connection, "closed", False):  # hung up: the next call redials, maybe to another life
        return None
    boundary, found = getattr(cli, "_gateway_plugin_commands", None) or (None, None)
    return found if boundary == _boundary(cli, connection) else None


def names(cli):
    """``{name: description}`` of the gateway's plugin commands for completion, or None off the
    gateway. A failed read degrades quietly to none; dispatch uses :func:`lookup`."""
    from hermes_cli.gateway_side import connection_for
    connection = connection_for(cli)
    if connection is None:
        return None
    found = _cached(cli, connection)
    if found is None:
        try:
            found = _read(cli, connection)
        except GatewayClientError:
            return {}  # not cached: the next read asks again
    return found


def lookup(cli, name):
    """Whether ``/name`` is one of the gateway's plugin commands; None off the gateway. A name the
    cached list lacks re-reads it once. A failed read is not evidence the name is no plugin: it is
    printed and :class:`CatalogUnavailable` raised, so no skill, bundle or expansion stands in."""
    from hermes_cli.gateway_side import connection_for
    connection = connection_for(cli)
    if connection is None:
        return None
    found = _cached(cli, connection)
    if found is not None and name in found:
        return True
    try:
        return name in _read(cli, connection)
    except GatewayClientError as exc:
        from cli import _cprint
        _cprint(f"  /{name}: the gateway's command list is unavailable ({exc}); nothing was run.")
        raise CatalogUnavailable(name) from exc


def run(cli, name, arg):
    """True when the gateway answered ``/name arg`` (the local handler must not run); :data:`GONE`
    when it no longer has that plugin command and a fresh catalog read agrees (a plugin reload, e.g.
    ``hermes plugins disable``, dropped it in this session and gateway life, which no cache boundary
    sees): nothing ran, and the caller dispatches the name again (bundle, skill, prefix)."""
    from hermes_cli.gateway_side import connection_for
    connection = connection_for(cli)
    if connection is None:
        return False
    from cli import _cprint
    from hermes_cli.gateway_client import GatewayRPCError
    from hermes_cli.gateway_connection import CallStopped
    # No answer deadline: main's CLI waited for the handler, which may wait minutes (ctx.call_mcp). On
    # a socket of its own, so the session's prompts and interrupts never queue behind it.
    wake = cli._plugin_command_wait = threading.Event()  # set by Ctrl+C (cli_tui_mixin)
    try:
        reply = connection.call_until("slash.exec", wake, session_id=cli.session_id,
                                      command=name + (" " + arg if arg else ""))
    except (CallStopped, KeyboardInterrupt) as exc:
        _cprint(_STOPPED_LINE.format(name))
        if isinstance(exc, KeyboardInterrupt):
            raise
        return True
    except GatewayRPCError as exc:
        if str(exc) != "unsupported_command":
            _cprint(f"  /{name}: {exc}")
            return True
        reply = {"type": "unsupported"}
    except GatewayClientError as exc:
        _cprint(f"  /{name}: {exc}")
        return True
    finally:
        cli._plugin_command_wait = None
    reply = reply or {}
    if reply.get("type", "exec") != "exec":  # it lost the plugin since lookup: a refusal or skill reply
        if _dropped(cli, connection, name):
            return GONE
        _cprint(_GONE_LINE.format(name))
        return True
    output = str(reply.get("output") or "")
    if output:
        _cprint(output)
    return True


def _dropped(cli, connection, name):
    """Whether the gateway's catalog, read again with the cached copy dropped, no longer lists
    ``name`` (a failed read: not known, so False)."""
    cli._gateway_plugin_commands = None
    try:
        return name not in _read(cli, connection)
    except GatewayClientError:
        return False


def report_gone(name, *known):
    """Print that ``/name`` is no longer a plugin command, unless one of the built-in commands or a
    ``known`` skill or bundle name starts with it (then prefix expansion decides); True when printed."""
    from hermes_cli.commands import COMMANDS
    if any(key.startswith("/" + name) for group in (COMMANDS, *known) for key in group):
        return False
    from cli import _cprint
    _cprint(_GONE_LINE.format(name))
    return True
