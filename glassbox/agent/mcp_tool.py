"""IronClaw / MCP tool wrapper around the AutoFit pipeline.

Two callable surfaces are exposed:

- :func:`auto_fit_tool` — direct Python entry. Accepts either a CSV path or
  base64-encoded CSV bytes and returns a JSON-safe dict.
- ``python -m glassbox.agent.mcp_tool`` — CLI shim. Reads a JSON request from
  stdin (or ``--input``) and writes a JSON response to stdout. This is useful
  for direct IronClaw-style tool calls.

Request schema::

    {
        "csv_path": "data/sample.csv",        # or
        "csv_b64": "<base64-encoded CSV>",
        "target_column": "purchased",
        "task": "auto"|"classification"|"regression",
        "search": "random"|"grid",
        "time_budget": 120
    }

Response schema is the AutoFit report (see ``glassbox.agent.report``) wrapped
as ``{"ok": true, "report": ...}`` on success or ``{"ok": false, "error": ...}``
on failure.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import sys
from typing import Any

from glassbox.agent.autofit import auto_fit, auto_fit_from_bytes
from glassbox.agent.report import make_json_safe

_VALID_TASKS = {"auto", "classification", "regression"}
_VALID_SEARCHES = {"random", "grid"}


def _coerce_int(value: Any, name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def auto_fit_tool(
    csv_path: str | None = None,
    csv_b64: str | None = None,
    target_column: str | None = None,
    task: str = "auto",
    search: str = "random",
    time_budget: int = 120,
) -> dict[str, Any]:
    """Run AutoFit and return a JSON-safe report.

    Exactly one of ``csv_path`` or ``csv_b64`` must be supplied. ``csv_b64`` is
    the inline path: the agent passes the CSV body as base64 when it cannot or
    should not reference a host filesystem path.
    """
    if not target_column:
        raise ValueError("target_column is required.")
    if task not in _VALID_TASKS:
        raise ValueError(f"task must be one of {sorted(_VALID_TASKS)}.")
    if search not in _VALID_SEARCHES:
        raise ValueError(f"search must be one of {sorted(_VALID_SEARCHES)}.")

    has_path = bool(csv_path)
    has_bytes = bool(csv_b64)
    if has_path == has_bytes:
        raise ValueError("Provide exactly one of csv_path or csv_b64.")

    if has_bytes:
        try:
            csv_bytes = base64.b64decode(csv_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("csv_b64 is not valid base64.") from exc
        report = auto_fit_from_bytes(
            csv_bytes,
            target_column=target_column,
            task=task,
            search=search,
            time_budget=time_budget,
        )
    else:
        report = auto_fit(
            csv_path,
            target_column=target_column,
            task=task,
            search=search,
            time_budget=time_budget,
        )

    return make_json_safe(report)


def _handle_request(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        report = auto_fit_tool(
            csv_path=payload.get("csv_path"),
            csv_b64=payload.get("csv_b64"),
            target_column=payload.get("target_column"),
            task=payload.get("task", "auto"),
            search=payload.get("search", "random"),
            time_budget=_coerce_int(payload.get("time_budget"), "time_budget", 120),
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}

    return {"ok": True, "report": report}


def _read_request(args: argparse.Namespace) -> dict[str, Any]:
    if args.input:
        raw = args.input
    else:
        raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("Empty request: pass JSON via stdin or --input.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Request is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object.")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m glassbox.agent.mcp_tool",
        description="IronClaw MCP entry for the GlassBox AutoFit pipeline.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="JSON request string. Defaults to reading stdin.",
    )
    args = parser.parse_args(argv)

    try:
        payload = _read_request(args)
    except ValueError as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": str(exc)}) + "\n")
        return 2

    response = _handle_request(payload)
    sys.stdout.write(json.dumps(response) + "\n")
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
