"""
exporters/html_exporter.py – Single-file HTML dashboard using Jinja2.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")


def render_dashboard(state: Dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("dashboard.html")
    return template.render(**state)
