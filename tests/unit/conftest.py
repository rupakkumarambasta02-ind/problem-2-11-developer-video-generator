import httpx
import pytest

try:  # anthropic>=1 and openai>=2 talk through httpx2, which respx cannot patch; bridge it to httpx
    import httpx2
except ImportError:  # older SDKs use httpx directly
    httpx2 = None


@pytest.fixture(autouse=True)
def _bridge_httpx2(monkeypatch):
    if httpx2 is None:
        return

    def handle(self, request):
        try:
            with httpx.Client() as c:
                r = c.send(httpx.Request(request.method, str(request.url), headers=list(request.headers.items()),
                                         content=request.read()))
        except httpx.TimeoutException as e:
            raise httpx2.ReadTimeout(str(e), request=request) from e
        return httpx2.Response(r.status_code, headers=[(k, v) for k, v in r.headers.items()
                                                      if k.lower() not in ("content-encoding", "content-length")],
                               content=r.content, request=request)

    monkeypatch.setattr(httpx2.HTTPTransport, "handle_request", handle)
