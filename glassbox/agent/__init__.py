"""Agent-level interfaces for GlassBox AutoML."""

from glassbox.agent.autofit import AutoFit, auto_fit
from glassbox.agent.report import generate_report

__all__ = [
    "AutoFit",
    "auto_fit",
    "generate_report",
]
