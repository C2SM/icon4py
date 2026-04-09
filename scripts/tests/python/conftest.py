"""Shared fixtures for dev-scripts tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def scripts_dir() -> Path:
    """Return the ``scripts/`` directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture()
def repo_root(scripts_dir: Path) -> Path:
    """Return the repository root."""
    return scripts_dir.parent
