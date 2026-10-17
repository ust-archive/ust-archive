# Dataset access preflight

The four active Schedule, Catalog, Credit Transfer, and ust.space workflows run
`scripts/hf_preflight.py` before downloading data or checking out the updater.
Faculty and inactive workflows are outside this change.

The script uses only the job's explicitly selected `HF_TOKEN` environment variable.
It does not read a local token store or fall back to anonymous access. It sends
three read-only requests to `https://huggingface.co`:

1. `GET /api/whoami-v2` validates identity authentication.
2. `GET /api/datasets/{dataset}` checks dataset metadata/read access.
3. `GET /api/datasets/{dataset}/auth-check/write` checks content write authorization.

The last endpoint is the GET used by the official
[`HfApi.auth_check(write=True)`](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.auth_check)
([implementation](https://github.com/huggingface/huggingface_hub/blob/v2.2.0/src/huggingface_hub/hf_api.py)).
It does not upload content or create a repository. Successful public reads alone
do not establish write access. Passing all three checks establishes readiness at
that moment; it does not verify a subsequent publish or valid generated data.

Failures return a nonzero exit code, so the normal Actions success dependency
blocks subsequent download, generation, and upload steps. Existing schedules,
dataset IDs, visibility, updater commands, and upload semantics are preserved.
The workflow repository and updater use separate checkout paths (`workflow` and
`scraper`) to make the preflight script available without overwriting a checkout.

Diagnostics distinguish missing credentials, identity authentication failure,
authorization denial, missing-or-inaccessible resources, rate limits, transient
service failures, and network failures. In particular, a 404 does not establish
that a private dataset was deleted. Organization policies can also cause 403s;
see [HF token policies](https://huggingface.co/docs/hub/security-tokens) and
[organization access](https://huggingface.co/docs/hub/organizations-security).

Only 408, 429, 500, 502, 503, 504, temporary DNS errors, connection failures, and
timeouts can retry, with at most three attempts per endpoint, a ten-second socket
timeout, and one/two-second backoff. `Retry-After` is honored up to ten seconds;
longer requested waits fail without retrying early. Persistent HTTP permission
errors, TLS errors, redirects, and malformed responses fail immediately. The
script rejects redirects so credentials cannot follow a redirected URL.

Logs contain fixed diagnostics and HTTP status codes. They never include the
token, identity payload, response body, arbitrary exception text, or traceback.

## Local verification

Run the mocked suite without credentials or network access:

```sh
python -B -m unittest discover -s tests -v
```

The suite covers missing/invalid credentials, inaccessible private datasets,
read-success/write-denial, transient retry/exhaustion, permanent failure,
redaction, redirect rejection, and workflow gating. These checks do not publish.

## Owner verification

The owner must securely confirm that the intended repository/job selects a
nonempty `HF_TOKEN` and that its token/resource/organization settings permit the
required reads and writes. Do not copy the separate rankings repository's token,
print credentials, or use an upload to test access.

After credential repair and explicit authorization for a live workflow run,
verify the intended dataset revision and record/term coverage, or a deliberately
verified valid no-op. A green preflight alone does not complete this repair.
