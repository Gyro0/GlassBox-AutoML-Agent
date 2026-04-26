"""Tests for the FastMCP wrapper that registers with IronClaw."""

from __future__ import annotations

import asyncio
import csv

import pytest

pytest.importorskip("mcp")

from glassbox.agent import mcp_server  # noqa: E402  (after importorskip)


def _write_csv(tmp_path, headers: list[str], rows: list[list[object]]) -> str:
    path = tmp_path / "data.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return str(path)


def test_app_is_named_glassbox_automl() -> None:
    assert mcp_server.app.name == "glassbox-automl"


def test_auto_fit_tool_is_registered_on_app() -> None:
    tools = asyncio.run(mcp_server.app.list_tools())
    names = {tool.name for tool in tools}
    assert "auto_fit" in names


def test_auto_fit_runs_through_the_wrapper(tmp_path) -> None:
    csv_path = _write_csv(
        tmp_path,
        ["x1", "x2", "target"],
        [
            [1, 10, 6], [2, 20, 8], [3, 30, 10], [4, 40, 12],
            [5, 50, 14], [6, 60, 16], [7, 70, 18], [8, 80, 20],
            [9, 90, 22], [10, 100, 24],
        ],
    )

    report = mcp_server.auto_fit(
        target_column="target",
        csv_path=csv_path,
        task="regression",
        search="grid",
        time_budget=10,
    )

    assert report["task"] == "regression"
    assert report["cv_score"] > 0.9
    assert isinstance(report["explanation"], list)
    assert any("Selected" in line for line in report["explanation"])
