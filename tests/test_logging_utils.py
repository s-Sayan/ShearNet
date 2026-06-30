"""Tests for logging configuration helpers (pure, no heavy deps)."""

import io
import logging

from shearnet.logging_utils import ansi, configure_logging, get_logger, supports_color


def test_ansi_respects_no_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert ansi("\033[1m") == ""


def test_ansi_respects_force_color(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    assert ansi("\033[1m") == "\033[1m"


def test_supports_color_false_for_non_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert supports_color(io.StringIO()) is False


def test_logger_level_filtering():
    buf = io.StringIO()
    configure_logging(level=logging.WARNING, stream=buf, force=True)
    log = get_logger("shearnet.test")
    log.info("info-should-be-hidden")
    log.warning("warn-should-show")
    out = buf.getvalue()
    assert "hidden" not in out
    assert "warn-should-show" in out
    # restore default config for other tests
    configure_logging(level=logging.INFO, force=True)
