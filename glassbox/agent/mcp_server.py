"""IronClaw-compatible MCP server exposing GlassBox AutoFit.

Two transports are supported, selected via ``--transport``:

* ``stdio`` (default) — child-process MCP transport. Used by hosts like Claude
  Desktop, mcp-inspector, and the existing FastMCP tests in this repo.
* ``streamable-http`` — HTTP transport on ``--host``/``--port`` (default
  ``127.0.0.1:8765``). This is what IronClaw's URL-based registration expects:
  ``ironclaw mcp add glassbox http://localhost:8765/mcp``.

Run standalone::

    python -m glassbox.agent.mcp_server                          # stdio
    python -m glassbox.agent.mcp_server --transport streamable-http \\
        --host 127.0.0.1 --port 8765                              # HTTP for IronClaw

The single tool, ``auto_fit``, is a thin FastMCP wrapper around
:func:`glassbox.agent.mcp_tool.auto_fit_tool` so all input validation, base64
decoding, and JSON-safe coercion stay in one place.
"""

from __future__ import annotations

import argparse
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - guidance for unconfigured envs
    raise ImportError(
        "The 'mcp' package is required to run the IronClaw MCP server. "
        "Install with: pip install -e .[mcp]"
    ) from exc

from glassbox.agent.mcp_tool import auto_fit_tool

app = FastMCP("glassbox-automl")


@app.tool()
def auto_fit(
    target_column: str,
    csv_path: str | None = None,
    csv_b64: str | None = None,
    task: str = "auto",
    search: str = "random",
    time_budget: int = 120,
) -> dict[str, Any]:
    """Run end-to-end AutoML on a CSV and return a JSON report.

    Provide exactly one of ``csv_path`` (filesystem) or ``csv_b64`` (base64-
    encoded CSV bytes; use this when the agent has no host filesystem).

    The returned report includes the EDA summary, model leaderboard, the best
    model with its hyperparameters and cross-validation score, feature
    importances, and an ``explanation`` array of short natural-language bullets
    suitable for the agent to read back to the user.
    """
    return auto_fit_tool(
        csv_path=csv_path,
        csv_b64=csv_b64,
        target_column=target_column,
        task=task,
        search=search,
        time_budget=time_budget,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m glassbox.agent.mcp_server",
        description="GlassBox AutoFit MCP server (stdio or HTTP).",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport. Use 'streamable-http' for IronClaw URL registration.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP transports.")
    parser.add_argument("--port", type=int, default=8765, help="Port for HTTP transports.")
    args = parser.parse_args(argv)

    if args.transport != "stdio":
        app.settings.host = args.host
        app.settings.port = args.port

    app.run(transport=args.transport)


if __name__ == "__main__":
    main()
