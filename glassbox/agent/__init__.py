"""Agent-level interfaces for GlassBox AutoML."""

from glassbox.agent.autofit import AutoFit, auto_fit, auto_fit_from_bytes
from glassbox.agent.report import generate_report

__all__ = [
    "AutoFit",
    "auto_fit",
    "auto_fit_from_bytes",
    "generate_report",
]
