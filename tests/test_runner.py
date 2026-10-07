import csv
import json
import tempfile
import unittest
from pathlib import Path

from survey.provider import ProviderError
from survey.questions import Question
from survey.runner import execute_run
from survey.storage import export_run, load_plan, load_records, read_json, run_lock, write_json
from tests.helpers import FakeProvider, plan, reply


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "run"

    def test_success_exports_count_people_and_resume_skips(self):
        manifest, cohort = plan(self.path)
        provider = FakeProvider([reply(3), reply(5)])
        summary = execute_run(self.path, provider, sleep=lambda _: None)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["demographics_successful"]["sex"], {"여자": 2})
        self.assertEqual(summary["questions"]["Q1"]["mean"], 4)
        self.assertEqual(summary["unique_personas_successful"], 2)
        with (self.path / "respondents.csv").open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 2)
        self.assertEqual(len((self.path / "results.jsonl").read_text().splitlines()), 2)
        never_called = FakeProvider([])
        execute_run(self.path, never_called)
        self.assertEqual(never_called.calls, [])
        self.assertEqual(load_plan(self.path), (manifest, cohort))

    def test_invalid_json_and_answers_retry_then_succeed(self):
        manifest, cohort = plan(self.path, n=1)
        provider = FakeProvider([reply(content='{"answers":{"Q1":2,"Q1":4}}'), reply(999), reply(4)])
        delays = []
        summary = execute_run(self.path, provider, sleep=delays.append)
        self.assertEqual(summary["successful"], 1)
        self.assertEqual(delays, [1.5, 3.0])
        record = next(iter(load_records(self.path, manifest, cohort).values()))
        self.assertEqual(len(record["attempts"]), 3)
        self.assertEqual(record["answers"], {"Q1": 4})
        self.assertEqual(len({c["seed"] for c in provider.calls}), 3)

    def test_failed_records_are_explicit_and_retry_is_opt_in(self):
        manifest, cohort = plan(self.path, n=1, max_attempts=2)
        summary = execute_run(self.path, FakeProvider([reply(content="{}"), reply(content="{}")]), sleep=lambda _: None)
        self.assertEqual((summary["successful"], summary["failed"]), (0, 1))
        self.assertEqual((self.path / "results.jsonl").read_text(), "")
        execute_run(self.path, FakeProvider([]))
        summary = execute_run(self.path, FakeProvider([reply()]), retry_failed=True)
        self.assertEqual(summary["successful"], 1)
        record = next(iter(load_records(self.path, manifest, cohort).values()))
        self.assertEqual(len(record["attempts"]), 3)

    def test_refusal_is_terminal_but_other_respondents_continue(self):
        plan(self.path)
        summary = execute_run(self.path, FakeProvider([reply(refusal="refused"), reply()]))
        self.assertEqual((summary["successful"], summary["failed"]), (1, 1))

    def test_transient_error_retries_fatal_error_stops(self):
        plan(self.path)
        provider = FakeProvider([ProviderError("HTTP_429", True), reply(), ProviderError("HTTP_401", False, True)])
        summary = execute_run(self.path, provider, sleep=lambda _: None)
        self.assertEqual((summary["successful"], summary["failed"]), (1, 1))

    def test_fatal_error_preserves_unattempted_cohort(self):
        plan(self.path)
        summary = execute_run(self.path, FakeProvider([ProviderError("HTTP_400", False, True)]))
        self.assertEqual((summary["failed"], summary["pending"]), (1, 1))

    def test_interruption_keeps_completed_records(self):
        plan(self.path)
        with self.assertRaises(KeyboardInterrupt):
            execute_run(self.path, FakeProvider([reply(), KeyboardInterrupt()]))
        summary = read_json(self.path / "summary.json")
        self.assertEqual((summary["successful"], summary["pending"]), (1, 1))
        provider = FakeProvider([reply(4)])
        self.assertEqual(execute_run(self.path, provider)["successful"], 2)
        self.assertEqual(len(provider.calls), 1)

    def test_resume_preserves_attempt_budget_and_other_seeds(self):
        manifest, cohort = plan(self.path, max_attempts=2)
        def interrupt(_):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            execute_run(self.path, FakeProvider([reply(999)]), sleep=interrupt)
        provider = FakeProvider([reply(999), reply()])
        summary = execute_run(self.path, provider, sleep=lambda _: None)
        self.assertEqual((summary["failed"], summary["successful"]), (1, 1))
        other = Path(self.temp.name) / "other"
        plan(other, max_attempts=2)
        clean = FakeProvider([reply(), reply()])
        execute_run(other, clean)
        self.assertEqual(clean.calls[1]["seed"], provider.calls[1]["seed"])

    def test_truncation_is_not_success(self):
        plan(self.path, n=1, max_attempts=1)
        summary = execute_run(self.path, FakeProvider([reply(finish_reason="length")]))
        self.assertEqual(summary["failed"], 1)

    def test_plan_tampering_and_output_corruption_are_rejected(self):
        manifest, cohort = plan(self.path, n=1)
        execute_run(self.path, FakeProvider([reply()]))
        record_path = next((self.path / "records").glob("*.json"))
        record = read_json(record_path)
        record["answers"]["Q1"] = 999
        write_json(record_path, record)
        with self.assertRaises(ValueError):
            export_run(self.path, manifest, cohort)
        cohort[0]["persona"]["age"] = 20
        write_json(self.path / "cohort.json", cohort)
        with self.assertRaisesRegex(ValueError, "integrity"):
            load_plan(self.path)

    def test_existing_directory_and_concurrent_runner_are_rejected(self):
        plan(self.path)
        with self.assertRaises(FileExistsError):
            plan(self.path)
        with run_lock(self.path):
            with self.assertRaisesRegex(ValueError, "already in use"):
                execute_run(self.path, FakeProvider([]))

    def test_multiple_questions_count_people_and_preserve_null_and_lists(self):
        questions = [Question(id="Q1", text="Scale", type="scale"),
                     Question(id="Q2", text="Multi", type="multi", options=["a", "b"]),
                     Question(id="Q3", text="Missing", type="number", allow_na=True),
                     Question(id="Q4", text="Large number", type="number")]
        plan(self.path, questions=questions)
        raw = {"answers": {"Q1": 3, "Q2": ["a", "b"], "Q3": None, "Q4": 1e308}}
        summary = execute_run(self.path, FakeProvider([reply(content=json.dumps(raw)), reply(content=json.dumps(raw))]))
        self.assertEqual(summary["demographics_successful"]["sex"], {"여자": 2})
        self.assertEqual(summary["questions"]["Q2"]["counts"], {"a": 2, "b": 2})
        self.assertEqual(summary["questions"]["Q3"]["answered"], 0)
        self.assertEqual(summary["questions"]["Q3"]["not_applicable"], 2)
        self.assertEqual(summary["questions"]["Q4"]["mean"], 1e308)
        with (self.path / "answers.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 8)
        self.assertEqual(json.loads(rows[1]["answer"]), ["a", "b"])
        self.assertEqual(rows[2]["is_na"], "True")
