# Task: Add Rate Limiting and Auth to API

## Task

Add per-IP rate limiting and optional API key authentication to `sentinel/api.py`. Tighten CORS from `allow_origins=["*"]` to configurable origins.

## Context

- `sentinel/api.py` defines a FastAPI app via `create_app()` with 5 endpoints: POST `/api/analyze`, POST `/api/report`, GET `/api/patterns`, GET `/api/stats`, GET `/api/health`.
- CORS is currently set to `allow_origins=["*"]` -- wide open.
- There is no rate limiting or authentication.
- The project uses optional dependencies -- FastAPI is not required at runtime. Imports are inside `create_app()`.
- `sentinel/config.py` may or may not exist yet. If it does, use `get_config().rate_limit_rpm` and `get_config().cors_origins`. If it doesn't, define local defaults: `rate_limit_rpm=60`, `cors_origins=["http://localhost:3000"]`.

## What To Do

### 1. Add in-memory per-IP rate limiting

Implement a simple token bucket rate limiter directly in `api.py` (do NOT add `slowapi` as a dependency -- keep it stdlib-compatible aside from FastAPI itself).

```python
class RateLimiter:
    """In-memory per-IP token bucket rate limiter."""
    def __init__(self, rpm: int = 60):
        self.rpm = rpm
        self._buckets: dict[str, list] = {}  # ip -> [tokens, last_refill_time]
    
    def allow(self, ip: str) -> bool:
        """Return True if request is allowed, False if rate limited."""
        ...
```

- Each IP gets `rpm` tokens per minute.
- Tokens refill at a rate of `rpm / 60` per second.
- When tokens are exhausted, return 429 Too Many Requests.

Add the rate limiter as FastAPI middleware or dependency:
- Create a dependency function `check_rate_limit` that extracts client IP from `request.client.host`.
- Apply it to all 5 endpoints.
- Return HTTP 429 with body `{"detail": "Rate limit exceeded. Try again later."}` when limited.

### 2. Add optional API key authentication

- Look for `X-API-Key` header on incoming requests.
- If no API key is configured (env var `SENTINEL_API_KEY` is not set), skip auth entirely -- all requests pass.
- If `SENTINEL_API_KEY` is set, require all requests to include a matching `X-API-Key` header.
- Return HTTP 401 with body `{"detail": "Invalid or missing API key."}` on auth failure.

Implement as a FastAPI dependency function `check_api_key`.

### 3. Tighten CORS

- Change `allow_origins=["*"]` to a configurable list.
- Try to import `get_config` from `sentinel.config`. If import fails (config module doesn't exist yet), fall back to `["http://localhost:3000", "http://localhost:8080"]`.
- Keep `allow_methods=["*"]` and `allow_headers=["*"]` -- those are fine.

### 4. Add tests in `tests/test_api_ratelimit.py`

Write tests using FastAPI's `TestClient` (from `starlette.testclient`):

- **Rate limiting tests:**
  - Sending `rpm + 1` requests in rapid succession to `/api/health` returns 429 on the last request.
  - After waiting (or mocking time), tokens refill and requests succeed again.
  - Different IPs have independent rate limits.

- **Auth tests:**
  - When `SENTINEL_API_KEY` env var is not set, requests without `X-API-Key` header succeed.
  - When `SENTINEL_API_KEY=test-key-123` is set, requests without header return 401.
  - When `SENTINEL_API_KEY=test-key-123` is set, requests with correct `X-API-Key: test-key-123` succeed.
  - When `SENTINEL_API_KEY=test-key-123` is set, requests with wrong key return 401.

- **CORS tests:**
  - Verify `Access-Control-Allow-Origin` header in responses matches configured origins.

Use `monkeypatch` to set/unset `SENTINEL_API_KEY` environment variable. Use `monkeypatch` or direct instantiation to control rate limit RPM for testing (set to a low value like 5).

## Acceptance Criteria

- [ ] Per-IP rate limiting returns 429 when limit exceeded.
- [ ] Optional API key auth via `X-API-Key` header, controlled by `SENTINEL_API_KEY` env var.
- [ ] CORS origins are configurable, no longer hardcoded to `["*"]`.
- [ ] All 5 existing endpoints still work when rate limit is not exceeded and auth passes.
- [ ] All new and existing tests pass.

## Constraints

- Do NOT add `slowapi` or any new pip dependency. Implement rate limiting with stdlib data structures.
- FastAPI and uvicorn are optional deps -- guard imports appropriately.
- Do not change endpoint signatures or response formats.
- The `RateLimiter` class must be defined inside `api.py` (or at module level within it) so it's self-contained.

## Test Command

```bash
python -m pytest tests/test_api_ratelimit.py tests/test_core.py -v
```
