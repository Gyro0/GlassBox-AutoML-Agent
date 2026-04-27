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

The single tool, ``auto_fit``, is a path-mode FastMCP wrapper around
:func:`glassbox.agent.mcp_tool.auto_fit_tool`. The lower-level helper still
supports inline CSV bytes for direct Python use, but the MCP surface is kept
path-only so agents do not try to invent base64 payloads when a local CSV path
is available.
"""

from __future__ import annotations

import argparse
from typing import Annotated, Any, Literal

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - guidance for unconfigured envs
    raise ImportError(
        "The 'mcp' package is required to run the IronClaw MCP server. "
        "Install with: pip install -e .[mcp]"
    ) from exc

from pydantic import Field

from glassbox.agent.mcp_tool import auto_fit_tool

app = FastMCP("glassbox-automl")


@app.tool()
def auto_fit(
    target_column: Annotated[
        str,
        Field(
            description=(
                "Exact CSV header name of the column to predict. Case-sensitive; "
                "must match a header from the CSV at csv_path. "
                "Example: 'median_house_value'."
            ),
        ),
    ],
    csv_path: Annotated[
        str,
        Field(
            description=(
                "Filesystem path to the local CSV, visible to this server. "
                "For uploads from the IronClaw web demo, use 'data/_uploaded.csv'."
            ),
        ),
    ],
    task: Annotated[
        Literal["auto", "classification", "regression"],
        Field(
            description=(
                "Problem type. Use 'auto' to infer from the target column dtype, "
                "or set explicitly to 'classification' or 'regression'."
            ),
        ),
    ] = "auto",
    search: Annotated[
        Literal["random", "grid"],
        Field(
            description=(
                "Hyperparameter search strategy: 'random' samples within "
                "time_budget; 'grid' enumerates a small fixed grid."
            ),
        ),
    ] = "random",
    time_budget: Annotated[
        int,
        Field(
            ge=1,
            le=300,
            description=(
                "Soft wall-clock cap (seconds) for random search. "
                "Typical values: 15 for quick demos, 60-120 for real runs."
            ),
        ),
    ] = 120,
) -> dict[str, Any]:
    """Run end-to-end AutoML on a CSV and return a JSON report.

    Always supply ``csv_path`` and ``target_column`` — both are required and
    must be drawn from the user's context, never invented or left empty.

    The returned report includes the EDA summary, model leaderboard, the best
    model with its hyperparameters and cross-validation score, feature
    importances, and an ``explanation`` array of short natural-language bullets
    suitable for the agent to read back to the user.
    """
    return auto_fit_tool(
        csv_path=csv_path,
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
