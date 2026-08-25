# Contributing to Seshat

## Development setup

Seshat requires Python 3.12 or newer and `uv`.

```bash
uv sync --extra dev
uv run pytest -q
git config core.hooksPath .githooks   # once per clone; see below
```

Git does not share hooks through a clone, so the last line is what enables
[`.githooks/pre-commit`](.githooks/pre-commit) for you. It replaces `.git/hooks`,
which holds only Git's samples.

The PostgreSQL integration tests run when `SESHAT_TEST_DATABASE_URL` points
to a PostgreSQL database with the `vector` extension. Unit tests use in-memory
fakes and do not require external services.

## Configuration

Non-secret settings are declared once, in the `SETTINGS` tuple in
[`app/config.py`](app/config.py); secrets are declared there too and read from
the environment. Adding or changing a setting means:

1. Edit `SETTINGS`.
2. Add it to [`seshat.example.toml`](seshat.example.toml), commented out at its
   default unless a deployment must set it.
3. Run `python scripts/generate_config_reference.py` to refresh
   [`docs/configuration.md`](docs/configuration.md).

Before opening a pull request, run the test suite and build the image locally:

```bash
docker build -t seshat:local .
```

Keep changes focused, update the README when the API or configuration changes,
and do not commit credentials, source documents, generated files, or local
environment files.