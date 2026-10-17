#!/usr/bin/env python3
"""Read-only checks of the explicitly selected CI token. Never publish or log it."""

import json
import math
import os
from email.utils import parsedate_to_datetime
import socket
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener


DATASETS = frozenset({
    "ust-archive/schedule",
    "ust-archive/catalog",
    "ust-archive/credit-transfer",
    "ust-archive/ust-space",
})
ENDPOINT = "https://huggingface.co"
ATTEMPTS = 3
TIMEOUT_SECONDS = 10
MAX_RETRY_DELAY = 10
MAX_RESPONSE_BYTES = 1024 * 1024
TRANSIENT_HTTP = frozenset({408, 429, 500, 502, 503, 504})


class NoRedirect(HTTPRedirectHandler):
    """Do not forward authorization to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PreflightError(Exception):
    """Contains only locally constructed, sanitized diagnostics."""


def http_diagnostic(stage, status):
    if status == 401 and stage == "identity":
        category = "authentication"
        detail = "HF_TOKEN was rejected by the identity endpoint"
    elif status in (401, 403):
        category = "authorization"
        detail = "access denied; check resource scopes and organization policy"
    elif status == 404:
        category = "missing-resource-or-inaccessible"
        detail = "resource is missing or hidden from this identity; verify ID and access"
    elif status == 429:
        category = "rate-limit"
        detail = "Hub request rate limit reached"
    elif status in TRANSIENT_HTTP:
        category = "service-unavailable"
        detail = "temporary Hub service failure"
    else:
        category = "http-error"
        detail = "unexpected Hub response"
    return f"{stage}: {category} (HTTP {status}): {detail}"


def transient_network(reason):
    # Certificate/configuration failures and permanent DNS failures are not retries.
    if isinstance(reason, ssl.SSLError):
        return False
    return isinstance(reason, (TimeoutError, ConnectionError)) or (
        isinstance(reason, socket.gaierror) and reason.errno == socket.EAI_AGAIN
    )


def retry_delay(header, attempt):
    delay = 2 ** (attempt - 1)
    if header is None:
        return delay
    if header.isascii() and header.isdecimal():
        if len(header) > 2:
            return None
        requested = int(header)
    else:
        try:
            requested = max(0, math.ceil(parsedate_to_datetime(header).timestamp() - time.time()))
        except (ValueError, TypeError, OverflowError):
            return delay
    return max(delay, requested) if requested <= MAX_RETRY_DELAY else None


def check(path, stage, token, *, open_url, sleep, stderr, expect_json=False):
    for attempt in range(1, ATTEMPTS + 1):
        retry_after = None
        request = Request(
            ENDPOINT + path,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            method="GET",
        )
        try:
            with open_url(request, timeout=TIMEOUT_SECONDS) as response:
                status = response.status
                if not 200 <= status < 300:
                    # urllib raises HTTPError normally; also handle custom transports.
                    raise HTTPError(request.full_url, status, "", response.headers, None)
                if not expect_json:
                    return None
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise PreflightError(f"{stage}: invalid-response: metadata exceeds size limit")
                try:
                    result = json.loads(body)
                except (ValueError, UnicodeError):
                    raise PreflightError(f"{stage}: invalid-response: expected JSON metadata") from None
                if not isinstance(result, dict):
                    raise PreflightError(f"{stage}: invalid-response: expected metadata object")
                return result
        except HTTPError as error:
            diagnostic = http_diagnostic(stage, error.code)
            retryable = error.code in TRANSIENT_HTTP
            if error.headers:
                retry_after = error.headers.get("Retry-After")
            error.close()
        except URLError as error:
            diagnostic = f"{stage}: network: Hub connection failed (details withheld)"
            retryable = transient_network(error.reason)
        except (TimeoutError, ConnectionError) as error:
            diagnostic = f"{stage}: network: Hub connection failed (details withheld)"
            retryable = transient_network(error)
        except ssl.SSLError:
            raise PreflightError(f"{stage}: network: TLS verification failed (details withheld)") from None

        if not retryable or attempt == ATTEMPTS:
            raise PreflightError(f"{diagnostic}; stopped after {attempt} attempt(s)")
        # Do not retry sooner than the server requests. Long waits exceed this
        # fail-fast check's budget and require a later owner-authorized run.
        delay = retry_delay(retry_after, attempt)
        if delay is None:
            raise PreflightError(f"{diagnostic}; Retry-After exceeds preflight retry budget")
        print(f"HF preflight: {diagnostic}; retry {attempt + 1}/{ATTEMPTS} in {delay}s", file=stderr)
        sleep(delay)
    raise AssertionError("unreachable")


def main(argv=None, *, environ=None, open_url=None, sleep=time.sleep, stdout=None, stderr=None):
    argv = sys.argv[1:] if argv is None else argv
    environ = os.environ if environ is None else environ
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    if len(argv) != 1 or argv[0] not in DATASETS:
        print("HF preflight: configuration: expected one supported active dataset ID", file=stderr)
        return 1
    dataset = argv[0]
    token = environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF preflight: missing-credential: HF_TOKEN is absent or empty", file=stderr)
        return 1
    if not token.isascii() or any(character.isspace() or ord(character) < 32 for character in token):
        print("HF preflight: authentication: HF_TOKEN has an invalid format", file=stderr)
        return 1
    if open_url is None:
        open_url = build_opener(NoRedirect()).open
    options = dict(open_url=open_url, sleep=sleep, stderr=stderr)
    try:
        identity = check("/api/whoami-v2", "identity", token, expect_json=True, **options)
        if not isinstance(identity.get("name"), str) or not identity["name"]:
            raise PreflightError("identity: invalid-response: authenticated identity is missing")
        metadata = check(f"/api/datasets/{dataset}", "dataset read", token, expect_json=True, **options)
        if metadata.get("id") != dataset:
            raise PreflightError("dataset read: invalid-response: dataset ID does not match")
        # The official Hub auth_check(write=True) uses this GET endpoint. It
        # checks content write authorization without creating or uploading files.
        check(f"/api/datasets/{dataset}/auth-check/write", "write authorization", token, **options)
    except PreflightError as error:
        print(f"HF preflight failed: {error}", file=stderr)
        return 1
    except Exception:
        # Never print remote bodies, URLs, request headers, identity payloads,
        # exception strings or tracebacks, which may echo a credential.
        print("HF preflight failed: internal-error: unexpected failure (details withheld)", file=stderr)
        return 1
    print(
        f"HF preflight: identity, dataset read access, and content write authorization passed "
        f"for {dataset}; publication is not verified.", file=stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
