"""Tests for the IronClaw/MCP tool wrapper."""

from __future__ import annotations

import base64
import io
import json
from contextlib import redirect_stdout

import pytest

from glassbox.agent.mcp_tool import _handle_request, auto_fit_tool, main


_CSV_BYTES = (
    b"x1,x2,target\n"
    b"1,10,6\n"
    b"2,20,8\n"
    b"3,30,10\n"
    b"4,40,12\n"
    b"5,50,14\n"
    b"6,60,16\n"
    b"7,70,18\n"
    b"8,80,20\n"
    b"9,90,22\n"
    b"10,100,24\n"
)


def _csv_path(tmp_path) -> str:
    path = tmp_path / "regression.csv"
    path.write_bytes(_CSV_BYTES)
    return str(path)


def test_auto_fit_tool_path_returns_json_safe_report(tmp_path) -> None:
    report = auto_fit_tool(
        csv_path=_csv_path(tmp_path),
        target_column="target",
        task="regression",
        search="grid",
        time_budget=10,
    )

    json.dumps(report)
    assert report["task"] == "regression"
    assert report["cv_score"] > 0.9
    assert report["feature_names"] == ["x1", "x2"]
    assert isinstance(report["explanation"], list)
    assert any("Selected" in line for line in report["explanation"])


def test_auto_fit_tool_bytes_matches_path_pipeline(tmp_path) -> None:
    encoded = base64.b64encode(_CSV_BYTES).decode("ascii")
    report_bytes = auto_fit_tool(
        csv_b64=encoded,
        target_column="target",
        task="regression",
        search="grid",
        time_budget=10,
    )
    report_path = auto_fit_tool(
        csv_path=_csv_path(tmp_path),
        target_column="target",
        task="regression",
        search="grid",
        time_budget=10,
    )

    assert report_bytes["best_model"] == report_path["best_model"]
    assert report_bytes["feature_names"] == report_path["feature_names"]


def test_auto_fit_tool_requires_exactly_one_input() -> None:
    with pytest.raises(ValueError):
        auto_fit_tool(target_column="target")
    with pytest.raises(ValueError):
        auto_fit_tool(
            csv_path="x.csv",
            csv_b64=base64.b64encode(_CSV_BYTES).decode("ascii"),
            target_column="target",
        )


def test_auto_fit_tool_rejects_invalid_task() -> None:
    with pytest.raises(ValueError):
        auto_fit_tool(
            csv_b64=base64.b64encode(_CSV_BYTES).decode("ascii"),
            target_column="target",
            task="bogus",
        )


def test_handle_request_wraps_errors_as_json() -> None:
    response = _handle_request({"target_column": "target"})
    assert response["ok"] is False
    assert "error" in response
    assert response["error_type"] == "ValueError"


def test_cli_shim_writes_json_response(tmp_path) -> None:
    request = json.dumps(
        {
            "csv_path": _csv_path(tmp_path),
            "target_column": "target",
            "task": "regression",
            "search": "grid",
            "time_budget": 10,
        }
    )

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = main(["--input", request])

    assert exit_code == 0
    response = json.loads(buffer.getvalue())
    assert response["ok"] is True
    assert response["report"]["task"] == "regression"


def test_cli_shim_reports_invalid_json() -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = main(["--input", "not-json"])

    assert exit_code == 2
    response = json.loads(buffer.getvalue())
    assert response["ok"] is False
