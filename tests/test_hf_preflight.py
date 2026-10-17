"""No network, credential store, or publishing calls are used by these tests."""

import io
import json
from pathlib import Path
import ssl
import sys
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hf_preflight


TOKEN = "hf_mock_credential_not_a_real_token"
DATASET = "ust-archive/ust-space"


class Response(io.BytesIO):
    def __init__(self, payload=None, *, status=200, headers=None):
        super().__init__(json.dumps(payload).encode())
        self.status = status
        self.headers = headers or {}


def failure(status, headers=None):
    # All untrusted fields deliberately contain the dummy secret.
    return HTTPError(
        f"https://huggingface.co/{TOKEN}", status, TOKEN,
        headers or {"X-Error-Message": TOKEN}, io.BytesIO(TOKEN.encode()),
    )


class PreflightTests(unittest.TestCase):
    def run_check(self, results=(), *, token=TOKEN, dataset=DATASET):
        request = Mock(side_effect=results)
        sleep = Mock()
        output = io.StringIO()
        code = hf_preflight.main(
            [dataset], environ={"HF_TOKEN": token}, open_url=request,
            sleep=sleep, stdout=output, stderr=output,
        )
        self.assertNotIn(TOKEN, output.getvalue())
        return code, output.getvalue(), request, sleep

    def success(self):
        return [Response({"name": TOKEN}), Response({"id": DATASET}), Response()]

    def test_absent_or_blank_credential_fails_before_network(self):
        for token in ("", " \n\t "):
            with self.subTest(token=repr(token)):
                code, output, request, sleep = self.run_check(token=token)
                self.assertEqual(code, 1)
                self.assertIn("missing-credential", output)
                request.assert_not_called()
                sleep.assert_not_called()

    def test_invalid_credential_stops_before_public_read(self):
        code, output, request, sleep = self.run_check([failure(401)])
        self.assertEqual(code, 1)
        self.assertIn("authentication", output)
        self.assertIn("HTTP 401", output)
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_read_and_write_are_separate_read_only_requests(self):
        code, output, request, _ = self.run_check(self.success())
        self.assertEqual(code, 0)
        self.assertIn("content write authorization", output)
        self.assertIn("publication is not verified", output)
        self.assertEqual(
            [call.args[0].full_url for call in request.call_args_list],
            ["https://huggingface.co/api/whoami-v2",
             f"https://huggingface.co/api/datasets/{DATASET}",
             f"https://huggingface.co/api/datasets/{DATASET}/auth-check/write"],
        )
        for call in request.call_args_list:
            self.assertEqual(call.args[0].get_method(), "GET")
            self.assertIsNone(call.args[0].data)
            self.assertEqual(call.args[0].get_header("Authorization"), f"Bearer {TOKEN}")
            self.assertEqual(call.kwargs["timeout"], 10)

    def test_inaccessible_private_dataset_fails_without_retry_or_write_check(self):
        for status, category in ((401, "authorization"), (403, "authorization"),
                                 (404, "missing-resource-or-inaccessible")):
            with self.subTest(status=status):
                code, output, request, sleep = self.run_check(
                    [Response({"name": "mock-user"}), failure(status)],
                )
                self.assertEqual(code, 1)
                self.assertIn(category, output)
                self.assertIn(f"HTTP {status}", output)
                self.assertNotIn("deleted", output)
                self.assertEqual(request.call_count, 2)
                sleep.assert_not_called()

    def test_public_read_does_not_bypass_write_authorization(self):
        for status in (401, 403, 404):
            with self.subTest(status=status):
                code, output, request, sleep = self.run_check(
                    [Response({"name": "mock-user"}), Response({"id": DATASET}), failure(status)],
                )
                self.assertEqual(code, 1)
                self.assertIn("write authorization", output)
                self.assertEqual(request.call_count, 3)
                sleep.assert_not_called()

    def test_transient_http_retries_are_bounded(self):
        for status, category in ((408, "service-unavailable"), (429, "rate-limit"),
                                 (500, "service-unavailable"), (502, "service-unavailable"),
                                 (503, "service-unavailable"), (504, "service-unavailable")):
            with self.subTest(status=status):
                code, output, request, sleep = self.run_check([failure(status) for _ in range(3)])
                self.assertEqual(code, 1)
                self.assertIn(category, output)
                self.assertEqual(request.call_count, 3)
                self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 2])

    def test_temporary_network_failure_is_distinct_and_bounded(self):
        code, output, request, sleep = self.run_check(
            [URLError(ConnectionRefusedError(TOKEN)) for _ in range(3)],
        )
        self.assertEqual(code, 1)
        self.assertIn("network", output)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_retry_can_recover_without_repeating_completed_checks(self):
        code, _, request, sleep = self.run_check(
            [Response({"name": "mock-user"}), failure(503), Response({"id": DATASET}), Response()],
        )
        self.assertEqual(code, 0)
        self.assertEqual(request.call_count, 4)
        sleep.assert_called_once_with(1)

    def test_retry_after_honored_within_budget(self):
        code, _, _, sleep = self.run_check(
            [failure(429, {"Retry-After": "4"}), *self.success()],
        )
        self.assertEqual(code, 0)
        sleep.assert_called_once_with(4)

    def test_long_retry_after_fails_without_retrying_early(self):
        code, output, request, sleep = self.run_check([failure(429, {"Retry-After": "120"})])
        self.assertEqual(code, 1)
        self.assertIn("rate-limit", output)
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()

    def test_http_date_retry_after_is_honored_or_exceeds_budget(self):
        with patch.object(hf_preflight.time, "time", return_value=0):
            self.assertEqual(hf_preflight.retry_delay("Thu, 01 Jan 1970 00:00:04 GMT", 1), 4)
            self.assertIsNone(hf_preflight.retry_delay("Thu, 01 Jan 1970 00:01:00 GMT", 1))
        self.assertEqual(hf_preflight.retry_delay(TOKEN, 1), 1)

    def test_invalid_header_credential_fails_without_request(self):
        for token in (TOKEN + "\nsecond-line", TOKEN + "\x00", TOKEN + "\N{SNOWMAN}"):
            code, output, request, _ = self.run_check(token=token)
            self.assertEqual(code, 1)
            self.assertIn("authentication", output)
            request.assert_not_called()

    def test_permanent_http_and_tls_failures_are_not_retried(self):
        for error in (failure(400), failure(301), failure(501),
                      URLError(ssl.SSLCertVerificationError(TOKEN))):
            with self.subTest(error=type(error).__name__):
                code, _, request, sleep = self.run_check([error])
                self.assertEqual(code, 1)
                self.assertEqual(request.call_count, 1)
                sleep.assert_not_called()

    def test_malformed_or_wrong_metadata_fails_closed(self):
        for response in (Response({"name": "mock-user"}), Response([TOKEN])):
            code, output, _, _ = self.run_check([Response({"name": "mock-user"}), response])
            self.assertEqual(code, 1)
            self.assertIn("invalid-response", output)

    def test_unexpected_exception_is_sanitized(self):
        code, output, _, _ = self.run_check([RuntimeError(TOKEN)])
        self.assertEqual(code, 1)
        self.assertIn("internal-error", output)
        self.assertNotIn("Traceback", output)

    def test_unknown_dataset_and_faculty_cannot_be_requested(self):
        for dataset in ("ust-archive/faculty", TOKEN):
            code, output, request, sleep = self.run_check(dataset=dataset)
            self.assertEqual(code, 1)
            self.assertIn("configuration", output)
            request.assert_not_called()
            sleep.assert_not_called()

    def test_redirects_are_not_followed_with_authorization(self):
        handler = hf_preflight.NoRedirect()
        self.assertIsNone(handler.redirect_request(None, None, 302, "mock", {}, "https://example.com"))


if __name__ == "__main__":
    unittest.main()
