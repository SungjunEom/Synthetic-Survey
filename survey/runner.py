"""Durable run planning and sequential execution with bounded, audited retries."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import time
import uuid
from pathlib import Path

from survey import __version__
from survey.personas import Persona, file_hash
from survey.prompts import PROMPT_VERSION, build_messages
from survey.provider import ProviderError
from survey.questions import Question, response_schema, strict_json, validate_answers, validate_questions
from survey.storage import (digest, export_run, load_plan, load_records, now,
                            run_lock, save_plan, write_json)


def derived_seed(seed: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:4], "big") % (2**31)


def implementation_hash() -> str:
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))})


def prepare_run(directory: Path, personas: list[Persona], questions: list[Question],
                settings: dict, source: dict, sampling: dict, experiment: dict):
    run_id = str(uuid.uuid4())
    cohort = []
    for i, persona in enumerate(personas):
        messages = build_messages(persona, questions, settings["lang"], settings["survey_date"])
        cohort.append({"respondent_id": str(uuid.uuid5(uuid.UUID(run_id), str(i))),
                       "sample_index": i, "request_seed": derived_seed(settings["seed"], f"respondent:{i}"),
                       "persona": persona.model_dump(), "messages": messages, "prompt_sha256": digest(messages)})
    versions = {}
    for package in ("openai", "pydantic", "PyYAML", "datasets", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {"format_version": 2, "run_id": run_id, "created_at": now(), "version": __version__,
                "implementation_sha256": implementation_hash(), "python": platform.python_version(),
                "dependencies": versions, "prompt_version": PROMPT_VERSION, "source": source,
                "sampling": sampling, "experiment": experiment, "settings": settings,
                "questions": [q.model_dump() for q in questions], "response_schema": response_schema(questions)}
    manifest["questionnaire_sha256"] = digest(manifest["questions"])
    save_plan(directory, manifest, cohort)
    export_run(directory, manifest, cohort)
    return manifest, cohort


def execute_run(directory: Path, provider, retry_failed: bool = False, sleep=time.sleep) -> dict:
    with run_lock(directory):
        manifest, cohort = load_plan(directory)
        if manifest["implementation_sha256"] != implementation_hash():
            raise ValueError("Implementation changed since this run was planned; use the original code to resume")
        records = load_records(directory, manifest, cohort)
        questions = validate_questions(manifest["questions"])
        settings = manifest["settings"]
        stop = False
        try:
            for task in cohort:
                rid = task["respondent_id"]
                record = records.get(rid)
                if record and (record["status"] == "success" or (record["status"] == "failed" and not retry_failed)):
                    continue
                if record is None:
                    record = {"respondent_id": rid, "manifest_sha256": manifest["manifest_sha256"],
                              "status": "pending", "attempts": [], "cycle_start": 0}
                elif record["status"] == "failed":
                    record["cycle_start"] = len(record["attempts"])
                    record["status"] = "pending"
                while len(record["attempts"]) - record["cycle_start"] < settings["max_attempts"]:
                    attempt_seed = derived_seed(task["request_seed"], str(len(record["attempts"])))
                    attempt = {"started_at": now(), "seed": attempt_seed}
                    retryable = True
                    try:
                        reply = provider.complete(task["messages"], manifest["response_schema"], settings, attempt_seed)
                        attempt["response"] = reply.to_dict()
                        if reply.refusal:
                            retryable = False
                            raise ValueError("Model refused this response")
                        if reply.finish_reason != "stop":
                            retryable = reply.finish_reason == "length"
                            raise ValueError(f"Incomplete response: {reply.finish_reason}")
                        if not reply.content:
                            raise ValueError("Empty response")
                        answers = validate_answers(strict_json(reply.content), questions)
                    except ProviderError as exc:
                        attempt["error"] = exc.kind
                        attempt["request_id"] = exc.request_id
                        retryable, stop = exc.retryable, exc.fatal
                    except ValueError as exc:
                        attempt["error"] = str(exc)
                    else:
                        record["answers"] = answers
                        record["status"] = "success"
                    attempt["finished_at"] = now()
                    record["attempts"].append(attempt)
                    exhausted = len(record["attempts"]) - record["cycle_start"] >= settings["max_attempts"]
                    if record["status"] != "success" and (exhausted or not retryable):
                        record["status"] = "failed"
                    write_json(directory / "records" / f"{rid}.json", record)
                    if record["status"] != "pending":
                        break
                    count = len(record["attempts"]) - record["cycle_start"]
                    sleep(min(30.0, 1.5 * 2 ** (count - 1)))
                # Covers a crash after the last attempt was persisted but before advancing.
                if record["status"] == "pending":
                    record["status"] = "failed"
                    write_json(directory / "records" / f"{rid}.json", record)
                print(f"[{task['sample_index'] + 1}/{len(cohort)}] {rid}: {record['status']}", flush=True)
                if stop:
                    break
        finally:
            summary = export_run(directory, manifest, cohort)
        return summary
