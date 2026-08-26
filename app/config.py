"""Configuration for Seshat."""

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

RetrievalMode = Literal["hybrid", "vector"]
DocumentConverterBackend = Literal["docling", "azure"]

VECTOR_TABLE_NAME = "seshat_vectors"
MINIMUM_API_TOKEN_LENGTH = 16
MINIMUM_CURSOR_SIGNING_KEY_LENGTH = 32
MAXIMUM_SEARCH_COLLECTIONS = 10
DEFAULT_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAXIMUM_LISTED_SOURCES = 200
DEFAULT_SOURCE_STORAGE_DIR = "/var/lib/seshat/sources"

CONFIG_PATH_VARIABLE = "SESHAT_CONFIG_FILE"
DEFAULT_CONFIG_PATH = "/etc/seshat/seshat.toml"

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_SCHEMA_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_POSTGRES_SCHEMES = ("postgres://", "postgresql://")


class ConfigurationError(ValueError):
    """Raised when required configuration is missing or invalid."""


class _Missing:
    """Sentinel distinguishing an absent key from a present empty value."""


_MISSING = _Missing()


@dataclass(frozen=True, slots=True)
class Setting:
    """One declared setting: where it comes from, and what values it accepts.

    Exactly one of ``path`` (a key in the TOML file) and ``variable`` (an
    environment variable holding a secret) is set.
    """

    attribute: str
    kind: Literal["string", "integer"]
    purpose: str
    path: tuple[str, ...] = ()
    variable: str = ""
    default: str | int | None = None
    minimum: int | None = None
    maximum: int | None = None
    choices: tuple[str, ...] = ()
    normalize: Literal["", "lower", "upper", "url"] = ""

    @property
    def secret(self) -> bool:
        return bool(self.variable)

    @property
    def required(self) -> bool:
        return self.default is None

    @property
    def label(self) -> str:
        """How the setting is named to whoever has to change it."""
        return self.variable or ".".join(self.path)

    def describe(self, origin: str) -> str:
        if self.secret:
            return f"Environment variable {self.variable}"
        return f"{self.label} in {origin}"


def _secret(
    attribute: str, variable: str, purpose: str, *, default: str | None = None
) -> Setting:
    return Setting(
        attribute=attribute,
        kind="string",
        purpose=purpose,
        variable=variable,
        default=default,
    )


SETTINGS: tuple[Setting, ...] = (
    # Secrets. Kept out of the configuration file so that file stays safe to
    # commit, template, and mount read-only into the container.
    _secret(
        "database_url",
        "SESHAT_DATABASE_URL",
        "PostgreSQL connection URL, including the password.",
    ),
    _secret(
        "api_token",
        "SESHAT_API_TOKEN",
        f"Bearer token every API request must present; at least "
        f"{MINIMUM_API_TOKEN_LENGTH} characters.",
    ),
    _secret(
        "cursor_signing_key",
        "SESHAT_CURSOR_SIGNING_KEY",
        f"Server-only key for scan cursor integrity; at least "
        f"{MINIMUM_CURSOR_SIGNING_KEY_LENGTH} characters.",
    ),
    _secret(
        "embedding_api_key",
        "SESHAT_EMBEDDING_API_KEY",
        "Credential for the embedding endpoint.",
    ),
    _secret(
        "azure_ocr_api_key",
        "AZURE_OCR_API_KEY",
        "Credential for the Azure Document Intelligence resource. Required when "
        "`converter.backend` is `azure`.",
        default="",
    ),
    Setting(
        attribute="db_schema",
        path=("database", "schema"),
        kind="string",
        default="seshat",
        purpose="PostgreSQL schema holding Seshat's tables.",
    ),
    Setting(
        attribute="embedding_base_url",
        path=("embedding", "base_url"),
        kind="string",
        normalize="url",
        purpose="OpenAI-compatible embedding endpoint.",
    ),
    Setting(
        attribute="embedding_model",
        path=("embedding", "model"),
        kind="string",
        purpose="Embedding model, or provider deployment name.",
    ),
    Setting(
        attribute="embedding_dimension",
        path=("embedding", "dimension"),
        kind="integer",
        minimum=8,
        maximum=4096,
        purpose="Embedding dimension requested from the provider.",
    ),
    Setting(
        attribute="embedding_batch_size",
        path=("embedding", "batch_size"),
        kind="integer",
        default=64,
        minimum=1,
        maximum=2048,
        purpose="Texts sent per embedding request.",
    ),
    Setting(
        attribute="chunk_size",
        path=("chunking", "size"),
        kind="integer",
        default=800,
        minimum=64,
        maximum=8192,
        purpose="Target search and scan passage size in tokens.",
    ),
    Setting(
        attribute="chunk_overlap",
        path=("chunking", "overlap"),
        kind="integer",
        default=120,
        minimum=0,
        maximum=4096,
        purpose="Tokens shared between splits of oversized content. Must be smaller than `chunking.size`.",
    ),
    Setting(
        attribute="retrieval_mode",
        path=("retrieval", "mode"),
        kind="string",
        default="hybrid",
        choices=("hybrid", "vector"),
        normalize="lower",
        purpose="`hybrid` combines vector and keyword search; `vector` uses vector search alone.",
    ),
    Setting(
        attribute="default_top_k",
        path=("retrieval", "default_top_k"),
        kind="integer",
        default=8,
        minimum=1,
        maximum=200,
        purpose="Results returned when a search omits `top_k`.",
    ),
    Setting(
        attribute="max_top_k",
        path=("retrieval", "max_top_k"),
        kind="integer",
        default=50,
        minimum=1,
        maximum=200,
        purpose="Largest `top_k` a search may request. Must be at least `retrieval.default_top_k`.",
    ),
    Setting(
        attribute="default_scan_limit",
        path=("scan", "default_limit"),
        kind="integer",
        default=25,
        minimum=1,
        maximum=200,
        purpose="Canonical blocks returned when a scan omits `limit`.",
    ),
    Setting(
        attribute="max_scan_limit",
        path=("scan", "max_limit"),
        kind="integer",
        default=100,
        minimum=1,
        maximum=200,
        purpose="Largest canonical-block count accepted for one scan page.",
    ),
    Setting(
        attribute="max_scan_payload_bytes",
        path=("scan", "max_payload_bytes"),
        kind="integer",
        default=256_000,
        minimum=1_024,
        maximum=8_000_000,
        purpose="Largest aggregate UTF-8 text payload returned by one scan page.",
    ),
    Setting(
        attribute="max_document_bytes",
        path=("limits", "max_document_bytes"),
        kind="integer",
        default=8_000_000,
        minimum=1_024,
        maximum=64_000_000,
        purpose="Largest extracted text accepted from one document.",
    ),
    Setting(
        attribute="max_canonical_block_bytes",
        path=("limits", "max_canonical_block_bytes"),
        kind="integer",
        default=64_000,
        minimum=1_024,
        maximum=1_000_000,
        purpose="Largest rendered canonical block accepted during ingestion.",
    ),
    Setting(
        attribute="max_upload_bytes",
        path=("limits", "max_upload_bytes"),
        kind="integer",
        default=DEFAULT_MAX_UPLOAD_BYTES,
        minimum=1_024,
        maximum=DEFAULT_MAX_UPLOAD_BYTES,
        purpose="Largest upload accepted; the maximum is 50 MiB.",
    ),
    Setting(
        attribute="max_filename_characters",
        path=("limits", "max_filename_characters"),
        kind="integer",
        default=255,
        minimum=16,
        maximum=1_024,
        purpose="Longest accepted upload filename.",
    ),
    # Original upload bytes are kept here so a source can be reopened and
    # cross-checked after a restart. It must be an absolute path on a durable
    # mount; a relative path would follow the process's working directory.
    Setting(
        attribute="source_storage_dir",
        path=("storage", "source_dir"),
        kind="string",
        default=DEFAULT_SOURCE_STORAGE_DIR,
        purpose="Absolute path where original uploads are kept.",
    ),
    Setting(
        attribute="document_converter",
        path=("converter", "backend"),
        kind="string",
        default="docling",
        choices=("docling", "azure"),
        normalize="lower",
        purpose="Which service converts PDF and Office uploads.",
    ),
    # Docling is optional: without it, text and Markdown uploads still work and
    # formats that need conversion fail per job with an explicit message.
    Setting(
        attribute="docling_base_url",
        path=("converter", "docling", "base_url"),
        kind="string",
        default="",
        normalize="url",
        purpose="docling-serve instance. Empty rejects uploads that need conversion.",
    ),
    # Docling conversion is submitted as an asynchronous task and polled, so the
    # bound that matters is the overall deadline rather than any single request's
    # timeout. An hour leaves room for a large scanned PDF on CPU.
    Setting(
        attribute="docling_timeout_seconds",
        path=("converter", "docling", "timeout_seconds"),
        kind="integer",
        default=120,
        minimum=5,
        maximum=900,
        purpose="Timeout for one HTTP request to Docling (submit, poll, or result fetch).",
    ),
    Setting(
        attribute="docling_conversion_deadline_seconds",
        path=("converter", "docling", "conversion_deadline_seconds"),
        kind="integer",
        default=3_600,
        minimum=60,
        maximum=86_400,
        purpose=(
            "Total time one document's conversion may take, measured from submission and "
            "across restarts. Must be at least `converter.docling.timeout_seconds`."
        ),
    ),
    Setting(
        attribute="docling_poll_interval_seconds",
        path=("converter", "docling", "poll_interval_seconds"),
        kind="integer",
        default=5,
        minimum=1,
        maximum=300,
        purpose=(
            "Delay between Docling task status polls. Must not exceed "
            "`converter.docling.conversion_deadline_seconds`."
        ),
    ),
    Setting(
        attribute="azure_ocr_endpoint",
        path=("converter", "azure", "endpoint"),
        kind="string",
        default="",
        normalize="url",
        purpose=(
            "Azure Document Intelligence resource URL. Required when "
            "`converter.backend` is `azure`."
        ),
    ),
    Setting(
        attribute="azure_ocr_model_id",
        path=("converter", "azure", "model_id"),
        kind="string",
        default="prebuilt-layout",
        purpose="Analysis model used for conversion.",
    ),
    Setting(
        attribute="azure_ocr_timeout_seconds",
        path=("converter", "azure", "timeout_seconds"),
        kind="integer",
        default=300,
        minimum=5,
        maximum=900,
        purpose="Total time allowed for one document's analysis, including polling.",
    ),
    Setting(
        attribute="log_level",
        path=("logging", "level"),
        kind="string",
        default="INFO",
        choices=LOG_LEVELS,
        normalize="upper",
        purpose="Root log level.",
    ),
)

FILE_SETTINGS: tuple[Setting, ...] = tuple(s for s in SETTINGS if not s.secret)
SECRET_SETTINGS: tuple[Setting, ...] = tuple(s for s in SETTINGS if s.secret)


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable service settings."""

    database_url: str
    db_schema: str
    api_token: str
    cursor_signing_key: str
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_dimension: int
    embedding_batch_size: int
    chunk_size: int
    chunk_overlap: int
    default_top_k: int
    max_top_k: int
    retrieval_mode: RetrievalMode
    default_scan_limit: int
    max_scan_limit: int
    max_scan_payload_bytes: int
    max_document_bytes: int
    max_canonical_block_bytes: int
    max_upload_bytes: int
    max_filename_characters: int
    source_storage_dir: str
    document_converter: DocumentConverterBackend
    docling_base_url: str
    docling_timeout_seconds: int
    docling_conversion_deadline_seconds: int
    docling_poll_interval_seconds: int
    azure_ocr_endpoint: str
    azure_ocr_api_key: str
    azure_ocr_model_id: str
    azure_ocr_timeout_seconds: int
    log_level: str
    config_origin: str = ""

    @property
    def conversion_configured(self) -> bool:
        """Whether uploads that need a converter (PDF, Office) can be processed."""
        if self.document_converter == "azure":
            return bool(self.azure_ocr_endpoint and self.azure_ocr_api_key)
        return bool(self.docling_base_url)

    @property
    def vector_table(self) -> str:
        return VECTOR_TABLE_NAME

    @property
    def async_database_url(self) -> str:
        """SQLAlchemy asyncpg URL used by the vector store."""
        return _replace_scheme(self.database_url, "postgresql+asyncpg://")

    @property
    def sync_database_url(self) -> str:
        """SQLAlchemy psycopg2 URL used by the vector store for DDL."""
        return _replace_scheme(self.database_url, "postgresql://")

    def redacted(self) -> dict[str, object]:
        """Return a log-safe view of the settings."""
        view: dict[str, object] = {
            "config_origin": self.config_origin or "(defaults only)"
        }
        for setting in FILE_SETTINGS:
            value = getattr(self, setting.attribute)
            view[setting.label] = "(unset)" if value == "" else value
        return view

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any] | None = None,
        *,
        env: Mapping[str, str] | None = None,
        origin: str = "the configuration file",
    ) -> "Settings":
        """Build settings from a parsed TOML *document* plus secrets from *env*."""
        table = document or {}
        environment = os.environ if env is None else env

        # Checked before anything else: a mistyped key would otherwise fall back
        # to its default and change nothing visible.
        unknown = _unknown_keys(table)
        if unknown:
            raise ConfigurationError(
                f"{origin} contains settings Seshat does not recognize: "
                f"{', '.join(unknown)}."
            )

        values: dict[str, Any] = {"config_origin": origin}
        for setting in SETTINGS:
            if setting.secret:
                values[setting.attribute] = _read_secret(setting, environment)
            else:
                values[setting.attribute] = _read_option(setting, table, origin)

        _validate(values, origin)
        return cls(**values)

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Read the configuration file named by ``SESHAT_CONFIG_FILE``."""
        environment = os.environ if env is None else env
        path = Path(
            (environment.get(CONFIG_PATH_VARIABLE) or "").strip() or DEFAULT_CONFIG_PATH
        )
        origin = str(path)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            raise ConfigurationError(
                f"Configuration file {origin} does not exist. Copy seshat.example.toml to it, "
                f"or point {CONFIG_PATH_VARIABLE} at the file to use."
            ) from None
        except IsADirectoryError:
            # Docker creates a directory when a bind mount's source file is
            # missing, so this is what skipping the copy step looks like.
            raise ConfigurationError(
                f"Configuration file {origin} is a directory, not a file. If it is a bind "
                "mount, remove the seshat.toml directory, copy seshat.example.toml to "
                "seshat.toml, and recreate the container."
            ) from None
        except OSError as exc:
            raise ConfigurationError(
                f"Configuration file {origin} could not be read: {exc.strerror}."
            ) from None
        try:
            document = tomllib.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:
            raise ConfigurationError(
                f"Configuration file {origin} is not valid UTF-8."
            ) from None
        except tomllib.TOMLDecodeError as exc:
            raise ConfigurationError(
                f"Configuration file {origin} is not valid TOML: {exc}."
            ) from None
        return cls.from_document(document, env=environment, origin=origin)


def _validate(values: dict[str, Any], origin: str) -> None:
    """Apply the checks that span settings or exceed a type and a range."""
    if not values["database_url"].startswith(_POSTGRES_SCHEMES):
        raise ConfigurationError(
            "SESHAT_DATABASE_URL must be a postgres:// or postgresql:// URL."
        )

    if len(values["api_token"]) < MINIMUM_API_TOKEN_LENGTH:
        raise ConfigurationError(
            f"SESHAT_API_TOKEN must be at least {MINIMUM_API_TOKEN_LENGTH} characters."
        )

    if len(values["cursor_signing_key"]) < MINIMUM_CURSOR_SIGNING_KEY_LENGTH:
        raise ConfigurationError(
            f"SESHAT_CURSOR_SIGNING_KEY must be at least "
            f"{MINIMUM_CURSOR_SIGNING_KEY_LENGTH} characters."
        )

    if not _SCHEMA_PATTERN.match(values["db_schema"]):
        raise ConfigurationError(
            f"database.schema in {origin} must be a lowercase SQL identifier "
            "(letters, digits and underscores, not starting with a digit)."
        )

    _require_http_url(
        values["embedding_base_url"],
        "embedding.base_url",
        origin,
        "an OpenAI-compatible base endpoint (for example https://<resource>/openai/v1)",
    )

    if values["chunk_overlap"] >= values["chunk_size"]:
        raise ConfigurationError(
            f"chunking.overlap in {origin} must be smaller than chunking.size."
        )

    if values["default_top_k"] > values["max_top_k"]:
        raise ConfigurationError(
            f"retrieval.default_top_k in {origin} must not exceed retrieval.max_top_k."
        )

    if values["default_scan_limit"] > values["max_scan_limit"]:
        raise ConfigurationError(
            f"scan.default_limit in {origin} must not exceed scan.max_limit."
        )

    if values["max_scan_payload_bytes"] < values["max_canonical_block_bytes"]:
        raise ConfigurationError(
            f"scan.max_payload_bytes in {origin} must not be smaller than "
            "limits.max_canonical_block_bytes."
        )

    if not values["source_storage_dir"].startswith("/"):
        raise ConfigurationError(
            f"storage.source_dir in {origin} must be an absolute path."
        )

    if values["docling_base_url"]:
        _require_http_url(
            values["docling_base_url"],
            "converter.docling.base_url",
            origin,
            "a docling-serve instance (for example http://docling:5001)",
        )

    if (
        values["docling_conversion_deadline_seconds"]
        < values["docling_timeout_seconds"]
    ):
        raise ConfigurationError(
            f"converter.docling.conversion_deadline_seconds in {origin} must not be smaller "
            "than converter.docling.timeout_seconds."
        )

    if (
        values["docling_poll_interval_seconds"]
        > values["docling_conversion_deadline_seconds"]
    ):
        raise ConfigurationError(
            f"converter.docling.poll_interval_seconds in {origin} must not exceed "
            "converter.docling.conversion_deadline_seconds."
        )

    if values["azure_ocr_endpoint"]:
        _require_http_url(
            values["azure_ocr_endpoint"],
            "converter.azure.endpoint",
            origin,
            "an Azure Document Intelligence resource (for example "
            "https://<resource>.cognitiveservices.azure.com)",
        )

    if values["document_converter"] == "azure" and not (
        values["azure_ocr_endpoint"] and values["azure_ocr_api_key"]
    ):
        raise ConfigurationError(
            f"converter.azure.endpoint in {origin} and environment variable AZURE_OCR_API_KEY "
            "are both required when converter.backend is 'azure'."
        )


def _require_http_url(value: str, label: str, origin: str, expected: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ConfigurationError(
            f"{label} in {origin} must be an absolute http(s) URL pointing at {expected}."
        )

    if parsed.username or parsed.password:
        raise ConfigurationError(
            f"{label} in {origin} must not embed credentials. Put the credential in its "
            "environment variable instead; endpoint URLs are written to the startup log."
        )
    if parsed.query or parsed.fragment:
        raise ConfigurationError(
            f"{label} in {origin} must be a base URL, without a query string or fragment."
        )


def _known_paths() -> set[tuple[str, ...]]:
    return {setting.path for setting in FILE_SETTINGS}


def _unknown_keys(document: Mapping[str, Any]) -> list[str]:
    """Dotted names present in *document* that no setting declares."""
    known = _known_paths()
    tables = {path[:index] for path in known for index in range(1, len(path))}
    found: list[str] = []

    def walk(table: Mapping[str, Any], prefix: tuple[str, ...]) -> None:
        for key, value in table.items():
            path = (*prefix, key)
            if path in known:
                continue
            if isinstance(value, Mapping) and path in tables:
                walk(value, path)
            else:
                found.append(".".join(path))

    walk(document, ())
    return sorted(found)


def _read_secret(setting: Setting, env: Mapping[str, str]) -> str:
    value = (env.get(setting.variable) or "").strip()
    if value:
        return value
    if setting.required:
        raise ConfigurationError(
            f"Environment variable {setting.variable} is required but was empty or unset."
        )
    return str(setting.default)


def _read_option(
    setting: Setting, document: Mapping[str, Any], origin: str
) -> str | int:
    raw = _lookup(document, setting.path)
    if raw is _MISSING:
        return _default_of(setting, origin)
    if setting.kind == "integer":
        # bool is an int subclass in Python, and `true` is not a count.
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ConfigurationError(f"{setting.describe(origin)} must be an integer.")
        return _bounded(setting, raw, origin)
    if not isinstance(raw, str):
        raise ConfigurationError(f"{setting.describe(origin)} must be a string.")
    value = raw.strip()
    if not value:
        return _default_of(setting, origin)
    return _normalized(setting, value, origin)


def _lookup(document: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """The value at *path*, or ``_MISSING``.

    A table found at *path* is returned rather than treated as absent, so writing
    a setting as a section is a type error instead of a silent fallback.
    """
    node: Any = document
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return _MISSING
        node = node[key]
    return node


def _default_of(setting: Setting, origin: str) -> str | int:
    if setting.required:
        raise ConfigurationError(
            f"{setting.describe(origin)} is required but was not set."
        )
    assert setting.default is not None
    return setting.default


def _bounded(setting: Setting, value: int, origin: str) -> int:
    if setting.minimum is not None and value < setting.minimum:
        raise ConfigurationError(
            f"{setting.describe(origin)} must be between {setting.minimum} and {setting.maximum}."
        )
    if setting.maximum is not None and value > setting.maximum:
        raise ConfigurationError(
            f"{setting.describe(origin)} must be between {setting.minimum} and {setting.maximum}."
        )
    return value


def _normalized(setting: Setting, value: str, origin: str) -> str:
    if setting.normalize == "lower":
        value = value.lower()
    elif setting.normalize == "upper":
        value = value.upper()
    elif setting.normalize == "url":
        value = value.rstrip("/")
    if setting.choices and value not in setting.choices:
        allowed = ", ".join(f"'{choice}'" for choice in setting.choices)
        raise ConfigurationError(
            f"{setting.describe(origin)} must be one of {allowed}."
        )
    return value


def _replace_scheme(dsn: str, scheme: str) -> str:
    for candidate in _POSTGRES_SCHEMES:
        if dsn.startswith(candidate):
            return f"{scheme}{dsn[len(candidate) :]}"
    return dsn
