"""Smoke test: the Windows launcher really boots Streamlit.

`streamlit` is often missing from PATH even when it is installed, which is why
run.bat exists. This test runs the launcher exactly as a user would, waits for
Streamlit's health endpoint, and shuts the server tree down afterwards.
"""
from __future__ import annotations

import os
import platform
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "run.bat"
STARTUP_TIMEOUT = 90  # seconds; first run imports pandas, langgraph, etc.


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _health(port: int) -> str:
    url = f"http://127.0.0.1:{port}/_stcore/health"
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.read().decode().strip()


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            capture_output=True,
        )
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()


@pytest.mark.skipif(platform.system() != "Windows", reason="run.bat is the Windows launcher")
def test_run_bat_serves_streamlit_health() -> None:
    assert LAUNCHER.exists(), "run.bat is missing from the repo root"

    port = _free_port()
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            [
                "cmd", "/c", str(LAUNCHER),
                "--server.headless", "true",
                "--server.address", "127.0.0.1",
                "--server.port", str(port),
                "--browser.gatherUsageStats", "false",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.time() + STARTUP_TIMEOUT
            last_error: Exception | None = None
            while time.time() < deadline:
                if process.poll() is not None:
                    break
                try:
                    if _health(port) == "ok":
                        return
                except Exception as exc:  # server not listening yet
                    last_error = exc
                time.sleep(1)
        finally:
            _stop(process)
            log.seek(0)
            output = log.read()

    pytest.fail(f"health endpoint never returned 'ok' ({last_error}).\n{output}")
