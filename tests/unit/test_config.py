"""Configuration validation."""

from copy import deepcopy
from typing import Any

import pytest

from app.config import ConfigurationError, Settings


def build(
    config: dict[str, Any], secrets: dict[str, str], **overrides: Any
) -> Settings:
    """Build settings from *config* deep-merged with *overrides*."""
    document = deepcopy(config)
    for section, values in overrides.items():
        target = document
        for key in section.split("__"):
            target = target.setdefault(key, {})
        target.update(values)
    return Settings.from_document(document, env=secrets, origin="seshat.toml")


def test_defaults_are_applied(base_secrets: dict[str, str]) -> None:
    settings = Settings.from_document(
        {
            "embedding": {
                "base_url": "https://api.openai.com/v1",
                "model": "text-embedding-3-small",
                "dimension": 1536,
            }
        },
        env=base_secrets,
    )

    assert settings.db_schema == "seshat"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120
    assert settings.max_canonical_block_bytes == 64_000
    assert settings.retrieval_mode == "hybrid"
    assert settings.embedding_batch_size == 64
    assert settings.default_top_k == 8
    assert settings.max_top_k == 50
    assert settings.default_scan_limit == 25
    assert settings.max_scan_limit == 100
    assert settings.max_scan_payload_bytes == 256_000
    assert settings.source_storage_dir == "/var/lib/seshat/sources"
    assert settings.log_level == "INFO"
    assert settings.vector_table == "seshat_vectors"


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("embedding", "base_url"),
        ("embedding", "model"),
        ("embedding", "dimension"),
    ],
)
def test_required_file_settings_are_reported_by_name(
    base_config: dict[str, Any], base_secrets: dict[str, str], section: str, key: str
) -> None:
    base_config[section].pop(key)
    with pytest.raises(ConfigurationError, match=f"{section}.{key}"):
        Settings.from_document(base_config, env=base_secrets, origin="seshat.toml")


@pytest.mark.parametrize(
    "name",
    [
        "SESHAT_DATABASE_URL",
        "SESHAT_API_TOKEN",
        "SESHAT_CURSOR_SIGNING_KEY",
        "SESHAT_EMBEDDING_API_KEY",
    ],
)
def test_required_secrets_are_reported_by_name(
    base_config: dict[str, Any], base_secrets: dict[str, str], name: str
) -> None:
    base_secrets.pop(name)
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_document(base_config, env=base_secrets)


def test_an_unknown_key_fails_startup_instead_of_being_ignored(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    """A mistyped key would otherwise fall back to its default and change nothing."""
    base_config["chunking"]["siz"] = 900
    base_config["retreival"] = {"mode": "vector"}

    with pytest.raises(ConfigurationError) as error:
        Settings.from_document(base_config, env=base_secrets, origin="seshat.toml")

    message = str(error.value)
    assert "chunking.siz" in message
    assert "retreival" in message
    assert "seshat.toml" in message


def test_an_environment_variable_never_overrides_a_file_setting(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    """Non-secret settings have exactly one source, so stale env stays inert."""
    settings = Settings.from_document(
        base_config,
        env={**base_secrets, "SESHAT_CHUNK_SIZE": "4096", "SESHAT_RETRIEVAL_MODE": "hybrid"},
    )

    assert settings.chunk_size == 128
    assert settings.retrieval_mode == "vector"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"database": {"schema": "public; drop table"}}, "database.schema"),
        ({"embedding": {"base_url": "not-a-url"}}, "embedding.base_url"),
        ({"embedding": {"dimension": 0}}, "embedding.dimension"),
        ({"embedding": {"dimension": "1536"}}, "embedding.dimension"),
        ({"embedding": {"dimension": True}}, "embedding.dimension"),
        ({"embedding": {"model": 12}}, "embedding.model"),
        ({"chunking": {"size": 256, "overlap": 512}}, "chunking.overlap"),
        ({"retrieval": {"default_top_k": 30, "max_top_k": 10}}, "retrieval.default_top_k"),
        ({"retrieval": {"mode": "lexical"}}, "retrieval.mode"),
        ({"storage": {"source_dir": "relative/path"}}, "storage.source_dir"),
        ({"logging": {"level": "CHATTY"}}, "logging.level"),
    ],
)
def test_invalid_values_are_rejected(
    base_config: dict[str, Any],
    base_secrets: dict[str, str],
    overrides: dict[str, Any],
    expected: str,
) -> None:
    for section, values in overrides.items():
        base_config.setdefault(section, {}).update(values)
    with pytest.raises(ConfigurationError, match=expected):
        Settings.from_document(base_config, env=base_secrets, origin="seshat.toml")


@pytest.mark.parametrize(
    "overrides",
    [
        {"SESHAT_DATABASE_URL": "mysql://db/seshat"},
        {"SESHAT_API_TOKEN": "short"},
        {"SESHAT_CURSOR_SIGNING_KEY": "short"},
    ],
)
def test_invalid_secrets_are_rejected(
    base_config: dict[str, Any], base_secrets: dict[str, str], overrides: dict[str, str]
) -> None:
    name = next(iter(overrides))
    with pytest.raises(ConfigurationError, match=name):
        Settings.from_document(base_config, env={**base_secrets, **overrides})


def test_secrets_are_never_echoed(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    with pytest.raises(ConfigurationError) as error:
        Settings.from_document(base_config, env={**base_secrets, "SESHAT_API_TOKEN": "tiny"})
    assert "tiny" not in str(error.value)

    settings = Settings.from_document(base_config, env=base_secrets)
    redacted = str(settings.redacted())
    assert settings.api_token not in redacted
    assert settings.cursor_signing_key not in redacted
    assert settings.embedding_api_key not in redacted


def test_database_urls_are_translated_for_sqlalchemy(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    settings = Settings.from_document(
        base_config,
        env={**base_secrets, "SESHAT_DATABASE_URL": "postgres://user:pass@db:5432/seshat"},
    )

    assert settings.async_database_url == "postgresql+asyncpg://user:pass@db:5432/seshat"
    assert settings.sync_database_url == "postgresql://user:pass@db:5432/seshat"


def test_trailing_slash_is_stripped_from_embedding_url(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    settings = build(
        base_config, base_secrets, embedding={"base_url": "https://resource.example.com/openai/v1/"}
    )

    assert settings.embedding_base_url == "https://resource.example.com/openai/v1"


def test_upload_and_conversion_defaults(settings: Settings) -> None:
    assert settings.max_upload_bytes == 50 * 1024 * 1024
    assert settings.max_filename_characters == 255
    assert settings.document_converter == "docling"
    assert settings.docling_base_url == ""
    assert settings.docling_timeout_seconds == 120
    # An hour, so a slow scanned PDF is not cut off by Seshat while Docling is
    # still working on it.
    assert settings.docling_conversion_deadline_seconds == 3_600
    assert settings.docling_poll_interval_seconds == 5
    assert settings.azure_ocr_endpoint == ""
    assert settings.azure_ocr_model_id == "prebuilt-layout"
    assert settings.azure_ocr_timeout_seconds == 300
    assert settings.conversion_configured is False


def test_a_configured_docling_url_enables_conversion(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    settings = build(
        base_config,
        base_secrets,
        converter__docling={"base_url": "http://docling:5001/", "timeout_seconds": 45},
    )

    assert settings.docling_base_url == "http://docling:5001"
    assert settings.docling_timeout_seconds == 45
    assert settings.conversion_configured is True


def test_async_conversion_bounds_are_configurable(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    settings = build(
        base_config,
        base_secrets,
        converter__docling={"conversion_deadline_seconds": 7_200, "poll_interval_seconds": 15},
    )

    assert settings.docling_conversion_deadline_seconds == 7_200
    assert settings.docling_poll_interval_seconds == 15


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("conversion_deadline_seconds", 30),
        ("conversion_deadline_seconds", 90_000),
        ("conversion_deadline_seconds", "an hour"),
        ("poll_interval_seconds", 0),
        ("poll_interval_seconds", 600),
    ],
)
def test_out_of_range_async_conversion_bounds_are_rejected(
    base_config: dict[str, Any], base_secrets: dict[str, str], key: str, value: Any
) -> None:
    with pytest.raises(ConfigurationError, match=f"converter.docling.{key}"):
        build(base_config, base_secrets, converter__docling={key: value})


def test_the_deadline_must_cover_a_single_request(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    """A per-request timeout longer than the whole deadline can never be reached."""
    with pytest.raises(
        ConfigurationError, match="converter.docling.conversion_deadline_seconds"
    ):
        build(
            base_config,
            base_secrets,
            converter__docling={"timeout_seconds": 600, "conversion_deadline_seconds": 120},
        )


@pytest.mark.parametrize("value", ["docling:5001", "ftp://docling", "/relative"])
def test_a_docling_url_must_be_absolute_http(
    base_config: dict[str, Any], base_secrets: dict[str, str], value: str
) -> None:
    with pytest.raises(ConfigurationError, match="converter.docling.base_url"):
        build(base_config, base_secrets, converter__docling={"base_url": value})


def test_an_unknown_document_converter_is_rejected(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    with pytest.raises(ConfigurationError, match="converter.backend"):
        build(base_config, base_secrets, converter={"backend": "textract"})


def test_azure_document_converter_requires_endpoint_and_key(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    endpoint = "https://resource.cognitiveservices.azure.com"

    with pytest.raises(ConfigurationError, match="converter.azure.endpoint"):
        build(base_config, base_secrets, converter={"backend": "azure"})

    with pytest.raises(ConfigurationError, match="converter.azure.endpoint"):
        Settings.from_document(
            {**deepcopy(base_config), "converter": {"backend": "azure"}},
            env={**base_secrets, "AZURE_OCR_API_KEY": "a-key"},
            origin="seshat.toml",
        )

    with pytest.raises(ConfigurationError, match="AZURE_OCR_API_KEY"):
        build(
            base_config,
            base_secrets,
            converter={"backend": "azure", "azure": {"endpoint": endpoint}},
        )


def test_a_configured_azure_converter_enables_conversion(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    document = deepcopy(base_config)
    document["converter"] = {
        "backend": "azure",
        "azure": {
            "endpoint": "https://resource.cognitiveservices.azure.com/",
            "model_id": "prebuilt-read",
            "timeout_seconds": 45,
        },
    }
    settings = Settings.from_document(
        document, env={**base_secrets, "AZURE_OCR_API_KEY": "a-key"}, origin="seshat.toml"
    )

    assert settings.document_converter == "azure"
    assert settings.azure_ocr_endpoint == "https://resource.cognitiveservices.azure.com"
    assert settings.azure_ocr_api_key == "a-key"
    assert settings.azure_ocr_model_id == "prebuilt-read"
    assert settings.azure_ocr_timeout_seconds == 45
    assert settings.conversion_configured is True
    # Docling stays optional and unrelated when Azure is the selected backend.
    assert settings.docling_base_url == ""


@pytest.mark.parametrize(
    "value", ["resource.cognitiveservices.azure.com", "ftp://resource", "/relative"]
)
def test_an_azure_endpoint_must_be_absolute_http(
    base_config: dict[str, Any], base_secrets: dict[str, str], value: str
) -> None:
    with pytest.raises(ConfigurationError, match="converter.azure.endpoint"):
        build(base_config, base_secrets, converter__azure={"endpoint": value})


@pytest.mark.parametrize("value", [1, "not-a-number", 50 * 1024 * 1024 + 1])
def test_upload_limits_are_bounded(
    base_config: dict[str, Any], base_secrets: dict[str, str], value: Any
) -> None:
    with pytest.raises(ConfigurationError, match="limits.max_upload_bytes"):
        build(base_config, base_secrets, limits={"max_upload_bytes": value})


@pytest.mark.parametrize("value", [1, "not-a-number", 1_000_001])
def test_canonical_block_limit_is_bounded(
    base_config: dict[str, Any], base_secrets: dict[str, str], value: Any
) -> None:
    with pytest.raises(ConfigurationError, match="limits.max_canonical_block_bytes"):
        build(base_config, base_secrets, limits={"max_canonical_block_bytes": value})


@pytest.mark.parametrize(
    ("scan", "expected"),
    [
        (
            {"default_limit": 20, "max_limit": 10},
            "scan.default_limit",
        ),
        (
            {"max_payload_bytes": 32_000},
            "scan.max_payload_bytes",
        ),
    ],
)
def test_scan_limits_preserve_page_invariants(
    base_config: dict[str, Any],
    base_secrets: dict[str, str],
    scan: dict[str, int],
    expected: str,
) -> None:
    with pytest.raises(ConfigurationError, match=expected):
        build(base_config, base_secrets, scan=scan)


def test_redacted_settings_report_the_file_without_secrets(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    document = deepcopy(base_config)
    document["converter"] = {
        "backend": "docling",
        "docling": {"base_url": "http://docling:5001"},
        "azure": {"endpoint": "https://resource.cognitiveservices.azure.com"},
    }
    settings = Settings.from_document(
        document,
        env={**base_secrets, "AZURE_OCR_API_KEY": "a-secret-azure-key"},
        origin="/etc/seshat/seshat.toml",
    )

    redacted = settings.redacted()

    assert redacted["config_origin"] == "/etc/seshat/seshat.toml"
    assert redacted["converter.docling.base_url"] == "http://docling:5001"
    assert redacted["converter.backend"] == "docling"
    assert redacted["converter.azure.endpoint"] == "https://resource.cognitiveservices.azure.com"
    assert redacted["limits.max_upload_bytes"] == 50 * 1024 * 1024
    # Secrets are absent by construction: redacted() is built from the file
    # settings, so a new secret cannot leak into it by being forgotten here.
    assert not any(key.isupper() for key in redacted)
    serialized = str(redacted)
    assert settings.api_token not in serialized
    assert settings.embedding_api_key not in serialized
    assert settings.azure_ocr_api_key not in serialized


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"chunking": {"size": {"tokens": 800}}}, "chunking.size in .* must be an integer"),
        ({"retrieval": {"mode": {"kind": "hybrid"}}}, "retrieval.mode in .* must be a string"),
    ],
)
def test_a_setting_written_as_a_section_is_a_type_error(
    base_config: dict[str, Any],
    base_secrets: dict[str, str],
    overrides: dict[str, Any],
    expected: str,
) -> None:
    """Not a silent fallback to the default, which the old env loader could not see."""
    for section, values in overrides.items():
        base_config[section].update(values)
    with pytest.raises(ConfigurationError, match=expected):
        Settings.from_document(base_config, env=base_secrets, origin="seshat.toml")


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"embedding": {"base_url": "https://user:secret@host.example/v1"}}, "credentials"),
        ({"embedding": {"base_url": "https://host.example/v1?api-key=secret"}}, "query string"),
        ({"embedding": {"base_url": "https://host.example/v1#secret"}}, "query string"),
        ({"converter": {"docling": {"base_url": "http://user:secret@docling:5001"}}}, "credentials"),
        (
            {"converter": {"azure": {"endpoint": "https://user:secret@resource.example"}}},
            "credentials",
        ),
    ],
)
def test_an_endpoint_url_may_not_carry_a_credential(
    base_config: dict[str, Any],
    base_secrets: dict[str, str],
    overrides: dict[str, Any],
    expected: str,
) -> None:
    """Endpoint URLs reach the startup log, so a credential in one would be printed."""
    for section, values in overrides.items():
        base_config.setdefault(section, {}).update(values)

    with pytest.raises(ConfigurationError, match=expected) as error:
        Settings.from_document(base_config, env=base_secrets, origin="seshat.toml")

    assert "secret" not in str(error.value)


def test_a_credential_in_an_endpoint_url_cannot_reach_the_startup_log(
    base_config: dict[str, Any], base_secrets: dict[str, str]
) -> None:
    """The guard above is what keeps redacted() safe for URL-shaped settings."""
    base_config["embedding"]["base_url"] = "https://user:secret@host.example/v1"

    with pytest.raises(ConfigurationError):
        Settings.from_document(base_config, env=base_secrets)

    # The database URL legitimately carries a password, and stays out of the view.
    settings = Settings.from_document(base_config | {"embedding": {
        "base_url": "https://host.example/v1",
        "model": "text-embedding-3-large",
        "dimension": 12,
    }}, env=base_secrets)
    assert "pass" not in str(settings.redacted())
