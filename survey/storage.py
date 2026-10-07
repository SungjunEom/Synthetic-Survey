"""Atomic experiment artifacts, locking, and exports derived from validated records."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from survey.questions import strict_json, validate_answers, validate_questions


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(encode(value).encode()).hexdigest()


def atomic_text(path: Path, content: str):
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, data):
    atomic_text(path, encode(data) + "\n")


def read_json(path: Path):
    return strict_json(path.read_text(encoding="utf-8"))


@contextmanager
def run_lock(directory: Path):
    # Advisory OS locks are released on process exit; no stale PID-file recovery needed.
    import fcntl
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("This run is already in use by another process") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def save_plan(directory: Path, manifest: dict, cohort: list[dict]):
    # An explicit directory must be new: never overwrite another experiment.
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "records").mkdir()
    manifest["cohort_sha256"] = digest(cohort)
    manifest["manifest_sha256"] = digest(manifest)
    write_json(directory / "cohort.json", cohort)
    write_json(directory / "manifest.json", manifest)


def load_plan(directory: Path):
    manifest = read_json(directory / "manifest.json")
    expected = manifest.pop("manifest_sha256")
    if digest(manifest) != expected:
        raise ValueError("Manifest integrity check failed; do not edit an existing run")
    manifest["manifest_sha256"] = expected
    if manifest["format_version"] != 2:
        raise ValueError("Unsupported run format")
    cohort = read_json(directory / "cohort.json")
    if digest(cohort) != manifest["cohort_sha256"]:
        raise ValueError("Cohort integrity check failed")
    return manifest, cohort


def load_records(directory: Path, manifest: dict, cohort: list[dict]) -> dict:
    questions = validate_questions(manifest["questions"])
    tasks = {t["respondent_id"]: t for t in cohort}
    records = {}
    for path in sorted((directory / "records").glob("*.json")):
        record = read_json(path)
        rid = record["respondent_id"]
        if rid not in tasks or path.stem != rid or record["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError(f"Record does not belong to this run: {path.name}")
        if record["status"] not in {"success", "failed", "pending"}:
            raise ValueError(f"Unknown record status: {path.name}")
        if record["status"] == "success":
            validate_answers({"answers": record["answers"]}, questions)
        records[rid] = record
    return records


def demographic_counts(tasks: list[dict]) -> dict:
    fields = ["sex", "education", "politics", "nationality", "province", "age_band"]
    counts = {field: Counter() for field in fields}
    for task in tasks:
        p = task["persona"]
        values = {**p, "province": p["attributes"].get("province"),
                  "age_band": f"{p['age'] // 10 * 10}-{p['age'] // 10 * 10 + 9}"}
        for field in fields:
            counts[field][str(values.get(field) if values.get(field) is not None else "unspecified")] += 1
    return {field: dict(count) for field, count in counts.items()}


def export_run(directory: Path, manifest: dict, cohort: list[dict]) -> dict:
    records = load_records(directory, manifest, cohort)
    successful = [t for t in cohort if records.get(t["respondent_id"], {}).get("status") == "success"]
    failed = [t for t in cohort if records.get(t["respondent_id"], {}).get("status") == "failed"]
    results = []
    answer_rows, respondent_rows = [], []
    for task in cohort:
        rid, p = task["respondent_id"], task["persona"]
        record = records.get(rid, {})
        base = {"respondent_id": rid, "persona_uuid": p["uuid"], "age": p["age"], "sex": p["sex"],
                "nationality": p["nationality"], "education": p["education"], "politics": p["politics"],
                "province": p["attributes"].get("province"), "district": p["attributes"].get("district")}
        respondent_rows.append({**base, "status": record.get("status", "pending"),
                                "attempts": len(record.get("attempts", []))})
        if record.get("status") != "success":
            continue
        answers = [{"id": q["id"], "question": q["text"], "answer": record["answers"][q["id"]]}
                   for q in manifest["questions"]]
        results.append({"respondent_id": rid, "persona": p, "answers": answers,
                        "meta": {"manifest_sha256": manifest["manifest_sha256"],
                                 "request_seed": task["request_seed"], "response": record["attempts"][-1]["response"]}})
        for answer in answers:
            value = answer["answer"]
            answer_rows.append({**base, "q_id": answer["id"], "question": answer["question"],
                                "answer": encode(value) if isinstance(value, list) else value,
                                "is_na": value is None})

    def csv_output(name, rows, fields):
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        atomic_text(directory / name, buffer.getvalue())

    base_fields = ["respondent_id", "persona_uuid", "age", "sex", "nationality", "education", "politics", "province", "district"]
    csv_output("respondents.csv", respondent_rows, base_fields + ["status", "attempts"])
    csv_output("answers.csv", answer_rows, base_fields + ["q_id", "question", "answer", "is_na"])
    atomic_text(directory / "results.jsonl", "".join(encode(r) + "\n" for r in results))
    summaries = {}
    for q in manifest["questions"]:
        values = [records[t["respondent_id"]]["answers"][q["id"]] for t in successful]
        answered = [v for v in values if v is not None]
        info = {"respondents": len(values), "answered": len(answered), "not_applicable": len(values) - len(answered)}
        if q["type"] in {"single", "multi"}:
            options = [x for v in answered for x in v] if q["type"] == "multi" else answered
            counts = Counter(options)
            info["counts"] = {option: counts[option] for option in q["options"]}
        elif q["type"] in {"scale", "number"}:
            # Scale before summing to avoid overflow for large finite input values.
            info["mean"] = sum(value / len(answered) for value in answered) if answered else None
            if q["type"] == "scale":
                info["counts"] = dict(Counter(map(str, answered)))
        summaries[q["id"]] = info
    summary = {"planned": len(cohort), "successful": len(successful), "failed": len(failed),
               "pending": len(cohort) - len(successful) - len(failed),
               "unique_personas_planned": len({t["persona"]["uuid"] for t in cohort}),
               "unique_personas_successful": len({t["persona"]["uuid"] for t in successful}),
               "demographics_planned": demographic_counts(cohort),
               "demographics_successful": demographic_counts(successful),
               "demographics_failed": demographic_counts(failed), "questions": summaries,
               "interpretation": "Unweighted synthetic model responses, not human population estimates."}
    write_json(directory / "summary.json", summary)
    return summary
