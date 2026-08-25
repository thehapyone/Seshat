"""The configuration file, and the checked-in files that describe it.

Configuration used to be declared once in the loader, again in compose.yaml,
again in .env.example and again in the docs, with nothing tying those together;
settings drifted out of the deployment unnoticed. These tests are that tie.
"""

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from app.config import (
    CONFIG_PATH_VARIABLE,
    FILE_SETTINGS,
    SECRET_SETTINGS,
    ConfigurationError,
    Settings,
)
from scripts.generate_config_reference import CONFIG_REFERENCE_PATH, render_reference

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "seshat.example.toml"
EXAMPLE_ENV = REPO_ROOT / ".env.example"
COMPOSE = REPO_ROOT / "compose.yaml"

# Compose derives SESHAT_DATABASE_URL from this password so the URL and the
# database container can never disagree; it is the only secret input that is not
# itself a setting.
COMPOSE_ONLY_SECRET = "SESHAT_POSTGRES_PASSWORD"

_COMMENTED_SETTING = re.compile(r"^(\s*)#\s(\w+ = )", re.MULTILINE)


def uncommented(text: str) -> str:
    """The example file with every commented-out setting enabled."""
    return _COMMENTED_SETTING.sub(r"\1\2", text)


def value_at(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = document
    for key in path:
        node = node[key]
    return node


def has_path(document: dict[str, Any], path: tuple[str, ...]) -> bool:
    node: Any = document
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return True


def test_load_reads_the_file_named_by_the_config_variable(
    tmp_path: Path, base_secrets: dict[str, str]
) -> None:
    path = tmp_path / "seshat.toml"
    path.write_text(
        """
        [embedding]
        base_url = "https://api.openai.com/v1"
        model = "text-embedding-3-small"
        dimension = 1536

        [chunking]
        size = 256
        overlap = 32
        """,
        encoding="utf-8",
    )

    settings = Settings.load({**base_secrets, CONFIG_PATH_VARIABLE: str(path)})

    assert settings.chunk_size == 256
    assert settings.chunk_overlap == 32
    assert settings.config_origin == str(path)


def test_a_missing_file_names_the_path_it_looked_for(
    tmp_path: Path, base_secrets: dict[str, str]
) -> None:
    missing = tmp_path / "absent.toml"
    with pytest.raises(ConfigurationError, match=str(missing)):
        Settings.load({**base_secrets, CONFIG_PATH_VARIABLE: str(missing)})


def test_malformed_toml_is_reported_as_a_configuration_error(
    tmp_path: Path, base_secrets: dict[str, str]
) -> None:
    path = tmp_path / "seshat.toml"
    path.write_text("[embedding\nmodel =", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="not valid TOML"):
        Settings.load({**base_secrets, CONFIG_PATH_VARIABLE: str(path)})


def test_the_example_config_starts_the_service_as_shipped(
    base_secrets: dict[str, str],
) -> None:
    document = tomllib.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    settings = Settings.from_document(document, env=base_secrets, origin=str(EXAMPLE_CONFIG))

    assert settings.conversion_configured is True
    assert settings.docling_base_url == "http://docling:5001"


def test_the_example_config_covers_every_file_setting(base_secrets: dict[str, str]) -> None:
    """Every setting is reachable from the example, commented or not."""
    document = tomllib.loads(uncommented(EXAMPLE_CONFIG.read_text(encoding="utf-8")))

    missing = [".".join(s.path) for s in FILE_SETTINGS if not has_path(document, s.path)]
    assert not missing, f"seshat.example.toml does not mention: {', '.join(missing)}"

    # The commented-out values are the documented defaults, so enabling them all
    # must still produce a valid configuration.
    Settings.from_document(
        document, env={**base_secrets, "AZURE_OCR_API_KEY": "a-key"}, origin=str(EXAMPLE_CONFIG)
    )


def test_the_example_env_lists_exactly_the_secrets() -> None:
    names = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", EXAMPLE_ENV.read_text(), re.MULTILINE))
    expected = {s.variable for s in SECRET_SETTINGS} - {"SESHAT_DATABASE_URL"}

    assert names == expected | {COMPOSE_ONLY_SECRET}


def test_compose_passes_only_secrets_to_seshat() -> None:
    """A non-secret in the environment would be silently ignored at runtime.

    Reads the literal keys under the seshat service rather than a resolved
    ``docker compose config``, so YAML indirection such as an anchor merged into
    the service is not covered. That is a deliberate trade: resolving would make
    the unit suite depend on Docker, and the loader ignores stray variables by
    construction, so what this guards against is a person re-adding a setting
    here and expecting it to take effect.
    """
    text = COMPOSE.read_text()
    seshat_service = text[text.index("  seshat:") : text.index("\n  database:")]
    passed = set(re.findall(r"^      ([A-Z][A-Z0-9_]*):", seshat_service, re.MULTILINE))

    assert passed == {s.variable for s in SECRET_SETTINGS}


def test_the_configuration_reference_is_generated_from_the_settings() -> None:
    reference = REPO_ROOT / CONFIG_REFERENCE_PATH

    assert reference.read_text(encoding="utf-8") == render_reference(), (
        "docs/configuration.md is stale; run scripts/generate_config_reference.py"
    )


def test_the_example_config_shows_the_real_defaults(base_secrets: dict[str, str]) -> None:
    """A commented-out value claims to be the default, so it has to be one."""
    shipped = tomllib.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    enabled = tomllib.loads(uncommented(EXAMPLE_CONFIG.read_text(encoding="utf-8")))

    stale = {}
    for setting in FILE_SETTINGS:
        # Settings the example sets outright are deployment values, not defaults,
        # and a setting whose default is empty has no value worth showing.
        if has_path(shipped, setting.path) or not setting.default:
            continue
        shown = value_at(enabled, setting.path)
        if shown != setting.default:
            stale[".".join(setting.path)] = (shown, setting.default)

    assert not stale, f"seshat.example.toml shows values that are not the default: {stale}"


def test_a_directory_at_the_config_path_explains_the_bind_mount(
    tmp_path: Path, base_secrets: dict[str, str]
) -> None:
    """Docker creates a directory when the mount source is missing; say so."""
    target = tmp_path / "seshat.toml"
    target.mkdir()

    with pytest.raises(ConfigurationError, match="is a directory") as error:
        Settings.load({**base_secrets, CONFIG_PATH_VARIABLE: str(target)})

    message = str(error.value)
    assert "remove the seshat.toml directory" in message
    assert "copy seshat.example.toml to seshat.toml" in message
