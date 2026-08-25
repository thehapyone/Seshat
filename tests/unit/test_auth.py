"""Authentication for the versioned API surface."""

import pytest
from httpx import AsyncClient

from app.auth import extract_bearer_token, token_matches
from tests.conftest import API_TOKEN, auth_headers

DOCUMENT = {
    "collection_id": "example-collection",
    "external_id": "manual",
    "text": "Check the inlet pressure sensor.",
}
SEARCH = {"query": "pressure", "collection_ids": ["example-collection"]}


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, None),
        ("", None),
        ("Basic abc", None),
        ("Bearer", None),
        ("Bearer   ", None),
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),
    ],
)
def test_extract_bearer_token(header: str | None, expected: str | None) -> None:
    assert extract_bearer_token(header) == expected


def test_token_matches_rejects_empty_and_wrong_tokens() -> None:
    assert token_matches("expected-token", "expected-token")
    assert not token_matches("expected-token", "expected-toke")
    assert not token_matches("expected-token", None)
    assert not token_matches("expected-token", "")


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/v1/documents/text", DOCUMENT),
        ("get", "/v1/jobs/00000000-0000-0000-0000-000000000000", None),
        ("post", "/v1/search", SEARCH),
    ],
)
async def test_v1_requires_a_valid_token(
    client: AsyncClient, method: str, path: str, body: dict | None
) -> None:
    for headers in ({}, auth_headers("wrong-token-value-000000")):
        response = await client.request(method, path, json=body, headers=headers)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"


async def test_health_probes_stay_unauthenticated(client: AsyncClient) -> None:
    assert (await client.get("/health/live")).status_code == 200
    assert (await client.get("/health/ready")).status_code == 200


async def test_valid_token_is_accepted(client: AsyncClient) -> None:
    response = await client.post("/v1/search", json=SEARCH, headers=auth_headers(API_TOKEN))
    assert response.status_code == 200
