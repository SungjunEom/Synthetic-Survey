import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from datasets import Dataset
from openai import OpenAI

from survey.cli import main
from survey.provider import OpenAIProvider, ProviderError
from survey.storage import load_plan, read_json
from tests.helpers import FakeProvider, reply, row


class ProviderTests(unittest.TestCase):
    def adapter(self, handler):
        provider = OpenAIProvider("test-not-a-real-key", 2.0)
        provider.client.close()
        provider.client = OpenAI(api_key="test-not-a-real-key", max_retries=0,
                                 http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.addCleanup(provider.client.close)
        return provider

    def test_real_sdk_serialization_and_metadata(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, headers={"x-request-id": "req-fixture"}, json={
                "id": "chat-fixture", "object": "chat.completion", "created": 1, "model": "fixture-snapshot",
                "system_fingerprint": "fp-fixture", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"answers":{"Q1":3}}'}}]})

        provider = self.adapter(handler)
        result = provider.complete([{"role": "user", "content": "test"}], {"type": "object"},
                                   {"model": "fake", "temperature": None, "send_seed": False, "max_tokens": 100}, 42)
        self.assertNotIn("temperature", requests[0])
        self.assertNotIn("seed", requests[0])
        self.assertEqual(requests[0]["max_completion_tokens"], 100)
        self.assertTrue(requests[0]["response_format"]["json_schema"]["strict"])
        self.assertEqual(result.request_id, "req-fixture")
        self.assertEqual(result.system_fingerprint, "fp-fixture")
        self.assertEqual(result.usage["total_tokens"], 15)

    def test_http_error_classification_without_sdk_retries(self):
        for code, retryable in [(429, True), (500, True), (401, False), (400, False)]:
            calls = []

            def handler(request):
                calls.append(request)
                return httpx.Response(code, json={"error": {"message": "server error", "type": "error"}})

            provider = self.adapter(handler)
            with self.subTest(code=code), self.assertRaises(ProviderError) as context:
                provider.complete([], {}, {"model": "fake", "temperature": 0.8, "send_seed": True, "max_tokens": 100}, 1)
            self.assertEqual(context.exception.retryable, retryable)
            self.assertEqual(context.exception.fatal, not retryable)
            self.assertEqual(len(calls), 1)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "run"
        self.output = io.StringIO()
        self.stdout = contextlib.redirect_stdout(self.output)
        self.stderr = contextlib.redirect_stderr(self.output)
        self.stdout.__enter__()
        self.stderr.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.addCleanup(self.stderr.__exit__, None, None, None)

    def test_local_dataset_dry_run_needs_no_key(self):
        dataset = Dataset.from_list([row(age=99, education_level="무학")])
        with patch("survey.cli.load_source", return_value=(dataset, {"kind": "fixture"})), patch("survey.cli.api_key") as key:
            code = main(["--n", "1", "--personas-file", "fixture.jsonl", "--sex", "여자", "--education", "무학",
                         "--age", "99", "--run-dir", str(self.path), "--dry-run", "--seed", "42"])
        self.assertEqual(code, 0, self.output.getvalue())
        key.assert_not_called()
        manifest, cohort = load_plan(self.path)
        self.assertEqual(cohort[0]["persona"]["age"], 99)
        self.assertEqual(manifest["settings"]["lang"], "ko")
        self.assertIn("해외여행", manifest["questions"][2]["text"])

    def test_custom_dry_run_resume_and_summary(self):
        q = Path(self.temp.name) / "questions.json"
        q.write_text('[{"id":"Q1","text":"Test","type":"scale"}]')
        code = main(["--custom", "--n", "2", "--dry-run", "--run-dir", str(self.path), "--questions-file", str(q)])
        self.assertEqual(code, 0, self.output.getvalue())
        provider = FakeProvider([reply(), reply()])
        with patch("survey.cli.api_key", return_value="fake"), patch("survey.cli.OpenAIProvider", return_value=provider):
            self.assertEqual(main(["--resume", str(self.path)]), 0, self.output.getvalue())
        self.assertEqual(main(["--summarize", str(self.path)]), 0)
        self.assertEqual(read_json(self.path / "summary.json")["successful"], 2)
        self.assertEqual(main(["--resume", str(self.path), "--seed", "1"]), 2)

    def test_bad_inputs_are_rejected_before_loading_data(self):
        cases = [["--n", "0"], ["--n", "1", "--sex", "non-binary"], ["--n", "1", "--randomize"],
                 ["--n", "1", "--age", "40-25"], ["--n", "1", "--timeout", "nan"],
                 ["--n", "1", "--custom", "--province", "서울"], ["--retry-failed"],
                 ["--n", "1", "--max-attempts", "0"]]
        with patch("survey.cli.load_source") as load:
            for args in cases:
                with self.subTest(args=args):
                    self.assertEqual(main(args + ["--dry-run"]), 2)
            load.assert_not_called()
