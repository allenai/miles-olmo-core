"""Generation metadata must pass through without blocking control on JSON conversion."""

import gzip
import json

import httpx
import pytest

from miles.router.router import MilesRouter


@pytest.mark.parametrize(
    "body,status,content_type",
    [
        (b'{ "value": 1.234567890123456789, "duplicate": 1, "duplicate": 2 }', 200, "application/json"),
        (b"upstream error", 503, "text/plain"),
        (b"\x00\xff", 200, "application/octet-stream"),
    ],
)
def test_proxy_preserves_bytes_status_and_headers(body, status, content_type, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("proxy must not deserialize or serialize the response")

    monkeypatch.setattr(json, "loads", forbidden)
    monkeypatch.setattr(json, "dumps", forbidden)
    router = object.__new__(MilesRouter)
    response = router.build_proxy_response(
        {
            "response_body": body,
            "status_code": status,
            "headers": {
                "content-type": content_type,
                "content-length": "999",
                "transfer-encoding": "chunked",
                "content-encoding": "gzip",
                "x-request-id": "test",
            },
        }
    )
    assert response.body == body
    assert response.status_code == status
    assert response.headers["content-type"] == content_type
    assert response.headers["content-length"] == str(len(body))
    assert response.headers["x-request-id"] == "test"
    assert "transfer-encoding" not in response.headers
    assert "content-encoding" not in response.headers


async def test_proxy_returns_decoded_httpx_body_without_stale_encoding():
    # HTTPX transparently decompresses before the router constructs its response.
    body = b'{"text":"hello"}'
    upstream = httpx.Response(
        200, content=gzip.compress(body), headers={"Content-Encoding": "gzip", "Content-Type": "application/json"}
    )
    router = object.__new__(MilesRouter)
    response = router.build_proxy_response(
        {"response_body": await upstream.aread(), "status_code": 200, "headers": dict(upstream.headers)}
    )
    assert response.body == body
    assert "content-encoding" not in response.headers
