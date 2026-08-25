"""Bearer-token authentication for the versioned API surface."""

import hmac

from fastapi import HTTPException, Request, status

_UNAUTHORIZED_HEADERS = {"WWW-Authenticate": "Bearer"}


def extract_bearer_token(authorization_header: str | None) -> str | None:
    """Return the bearer token from an Authorization header, or None."""
    if not authorization_header:
        return None
    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def token_matches(expected: str, supplied: str | None) -> bool:
    """Constant-time comparison that tolerates a missing token."""
    if not supplied:
        return False
    return hmac.compare_digest(expected, supplied)


async def require_api_token(request: Request) -> None:
    """FastAPI dependency guarding every ``/v1`` route."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Not ready.")
    supplied = extract_bearer_token(request.headers.get("authorization"))
    if not token_matches(settings.api_token, supplied):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid service API token is required.",
            headers=_UNAUTHORIZED_HEADERS,
        )
