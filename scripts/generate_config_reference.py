#!/usr/bin/env python
"""Regenerate docs/configuration.md from the settings declared in app/config.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import FILE_SETTINGS, SECRET_SETTINGS, Setting  # noqa: E402

CONFIG_REFERENCE_PATH = "docs/configuration.md"

_REFERENCE_HEADER = """<!-- Generated from app/config.py by scripts/generate_config_reference.py. -->
<!-- Run that script after changing SETTINGS; CI checks this file is current. -->

# Configuration Reference

Seshat reads every non-secret setting from one TOML file

`SESHAT_CONFIG_FILE` selects the file and defaults to `/etc/seshat/seshat.toml`.
Start from [`seshat.example.toml`](../seshat.example.toml).

## File settings

| Setting | Default | Purpose |
| --- | --- | --- |
"""

_SECRETS_HEADER = """
## Secrets

Provided as environment variables; the Compose stack reads them from `.env`.
See [`.env.example`](../.env.example).

| Variable | Required | Purpose |
| --- | --- | --- |
"""


def render_reference() -> str:
    """Render ``docs/configuration.md`` from the declared settings."""
    lines = [_REFERENCE_HEADER]
    for setting in FILE_SETTINGS:
        lines.append(
            f"| `{setting.label}` | {_default_cell(setting)} | {setting.purpose} |\n"
        )
    lines.append(_SECRETS_HEADER)
    for setting in SECRET_SETTINGS:
        required = "yes" if setting.required else "no"
        lines.append(f"| `{setting.label}` | {required} | {setting.purpose} |\n")
    return "".join(lines)


def _default_cell(setting: Setting) -> str:
    if setting.required:
        return "required"
    if setting.kind == "integer":
        return f"`{setting.default}`"
    return f'`"{setting.default}"`' if setting.default else "empty"


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / CONFIG_REFERENCE_PATH
    target.write_text(render_reference(), encoding="utf-8")
    print(f"wrote {target}")
