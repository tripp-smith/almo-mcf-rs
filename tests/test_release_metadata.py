from __future__ import annotations

import re
from pathlib import Path


def _read_project_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    match = re.search(r"^version\s*=\s*\"([^\"]+)\"", text, re.MULTILINE)
    assert match is not None, "version not found in pyproject.toml"
    return match.group(1)


def test_changelog_mentions_current_version() -> None:
    version = _read_project_version()
    changelog = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
    text = changelog.read_text(encoding="utf-8")
    pattern = rf"^## \[{re.escape(version)}\]"
    assert re.search(pattern, text, re.MULTILINE), "CHANGELOG entry missing for current version"
