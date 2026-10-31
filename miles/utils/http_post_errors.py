"""Transport HTTP failures as data across Ray, without replaying the request."""

from dataclasses import dataclass

import httpx


class SerializableHTTPStatusError(httpx.HTTPStatusError):
    """HTTPX's keyword-only constructor otherwise breaks Ray exception unpickling."""

    def __reduce__(self):
        return (_restore_status_error, (str(self), self.request, self.response))


def _restore_status_error(message, request, response):
    return SerializableHTTPStatusError(message, request=request, response=response)


@dataclass(frozen=True)
class HTTPPostFailure:
    error_type: type[httpx.HTTPError]
    message: str
    method: str
    url: str
    status_code: int | None = None
    body: bytes = b""

    @classmethod
    def capture(cls, error: httpx.HTTPError, *, method: str, url: str):
        # No request payload or headers: generation prompts and credentials must
        # not be copied into error envelopes. Bound unexpected server error pages.
        status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
        body = error.response.content[:65536] if isinstance(error, httpx.HTTPStatusError) else b""
        return cls(type(error), str(error), method.upper(), url, status, body)

    def exception(self) -> httpx.HTTPError:
        request = httpx.Request(self.method, self.url)
        if self.status_code is not None:
            response = httpx.Response(self.status_code, content=self.body, request=request)
            return SerializableHTTPStatusError(self.message, request=request, response=response)
        if issubclass(self.error_type, httpx.RequestError):
            return self.error_type(self.message, request=request)
        return self.error_type(self.message)
