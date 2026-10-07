"""Command-line interface. Planning needs no OpenAI credentials."""
from __future__ import annotations

import argparse
import os
import secrets
import sys
import uuid
from datetime import date
from pathlib import Path

from survey.personas import (age_bounds, custom_personas, from_nemotron, load_source,
                             normalize_filters, sample_dataset)
from survey.provider import OpenAIProvider
from survey.questions import load_questions
from survey.runner import execute_run, prepare_run
from survey.storage import export_run, load_plan, run_lock

ROOT = Path(__file__).resolve().parent.parent


def temperature(value):
    if value.lower() == "default":
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Temperature must be 0-2 or 'default'") from exc
    if not 0 <= number <= 2:
        raise argparse.ArgumentTypeError("Temperature must be 0-2 or 'default'")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run validated synthetic surveys using South Korean Nemotron personas.")
    p.add_argument("--n", type=int, help="Number of respondents (required for new runs)")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--nemotron-korea", action="store_true", help="Use Nemotron (the default)")
    source.add_argument("--custom", action="store_true", help="Use illustrative custom personas instead of Nemotron")
    p.add_argument("--personas-file", help="Local Nemotron-shaped JSON/JSONL file, instead of downloading")
    p.add_argument("--dataset-revision", default="main", help="Hugging Face commit/ref; resolved and saved as a commit")
    p.add_argument("--questions-file", help="Questionnaire YAML/JSON; default follows --lang")
    p.add_argument("--lang", choices=["ko", "en"], default="ko")
    p.add_argument("--survey-date", default=date.today().isoformat(), help="ISO date anchoring relative survey questions")
    p.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini-2024-07-18"))
    p.add_argument("--temperature", type=temperature, default=0.8, help="0-2, or 'default' to omit the API parameter")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--timeout", type=float, default=90.0)
    p.add_argument("--seed", type=int, help="Sampling seed; generated and saved if omitted")
    p.add_argument("--no-model-seed", action="store_true", help="Omit the optional seed parameter for incompatible models")
    p.add_argument("--api-key-file", default="api_key.txt")
    p.add_argument("--out-dir", default="out", help="Parent for new run directories")
    p.add_argument("--run-dir", help="Exact NEW directory to create")
    p.add_argument("--dry-run", action="store_true", help="Save cohort, exact prompts and metadata without model calls")
    p.add_argument("--resume", metavar="RUN_DIR", help="Execute pending respondents using an existing saved plan")
    p.add_argument("--retry-failed", action="store_true", help="With --resume, grant failed respondents another attempt budget")
    p.add_argument("--summarize", metavar="RUN_DIR", help="Rebuild exports without model calls")
    p.add_argument("--with-replacement", action="store_true", help="Explicitly permit repeated personas")
    for name in ("age", "sex", "education", "marital-status", "housing-type", "occupation", "province", "district"):
        p.add_argument(f"--{name}")
    p.add_argument("--politics", help="Explicit experimental political override, never a demographic filter")
    p.add_argument("--mbti", help="Custom mode only")
    p.add_argument("--nationality", help="Custom mode only")
    p.add_argument("--randomize", action="store_true", help="Custom mode only; does not invent political orientation")
    return p


def api_key(path: str) -> str:
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if not value and Path(path).is_file():
        value = Path(path).read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError("Set OPENAI_API_KEY or provide --api-key-file; --dry-run needs no key")
    return value


def show_summary(directory: Path, summary: dict):
    print(f"Run: {directory.resolve()}")
    print(f"Respondents: {summary['successful']} successful, {summary['failed']} failed, {summary['pending']} pending")
    print(f"Unique personas: {summary['unique_personas_planned']} planned, {summary['unique_personas_successful']} successful")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.resume or args.summarize:
            if args.resume and args.summarize:
                raise ValueError("Choose either --resume or --summarize")
            allowed = {"--resume", "--api-key-file", "--retry-failed"} if args.resume else {"--summarize"}
            supplied = {token.split("=", 1)[0] for token in argv if token.startswith("--")}
            if supplied - allowed:
                raise ValueError("Saved runs are immutable; resume/summarize cannot override experiment settings")
            directory = Path(args.resume or args.summarize)
            manifest, cohort = load_plan(directory)
            if args.summarize:
                with run_lock(directory):
                    summary = export_run(directory, manifest, cohort)
            else:
                provider = OpenAIProvider(api_key(args.api_key_file), manifest["settings"]["timeout"])
                summary = execute_run(directory, provider, retry_failed=args.retry_failed)
            show_summary(directory, summary)
            return 1 if args.resume and (summary["failed"] or summary["pending"]) else 0

        if args.retry_failed:
            raise ValueError("--retry-failed requires --resume")
        if args.n is None or args.n < 1:
            raise ValueError("--n must be a positive integer")
        if args.max_attempts < 1 or args.max_tokens < 1 or not 0 < args.timeout < float("inf"):
            raise ValueError("Attempts, token limit and timeout must be positive and finite")
        survey_date = date.fromisoformat(args.survey_date).isoformat()
        seed = args.seed if args.seed is not None else secrets.randbits(31)
        questions_path = args.questions_file or ROOT / ("questions.ko.yaml" if args.lang == "ko" else "questions.yaml")
        questions = load_questions(questions_path)
        filters = {}
        if args.custom:
            if args.personas_file or args.with_replacement or any(getattr(args, x) for x in ("marital_status", "housing_type", "occupation", "province", "district")):
                raise ValueError("Dataset filters, --personas-file and --with-replacement require Nemotron mode")
            fields = {k: getattr(args, k) for k in ("age", "sex", "education", "nationality", "mbti", "politics")}
            age_bounds(args.age, minimum=18)
            personas = custom_personas(args.n, seed, fields, args.randomize)
            source = {"kind": "custom", "fields": fields, "randomized": args.randomize}
            sampling = {"sampled_rows": args.n, "unique_personas": args.n, "replacement": False}
        else:
            if args.mbti or args.nationality or args.randomize:
                raise ValueError("--mbti, --nationality and --randomize require --custom; use --politics for an explicit override")
            filters = normalize_filters({"age": args.age, "sex": args.sex, "education_level": args.education,
                                         "marital_status": args.marital_status, "housing_type": args.housing_type,
                                         "occupation": args.occupation, "province": args.province, "district": args.district})
            # Check credentials before a potentially large dataset download, but not during planning.
            if not args.dry_run:
                api_key(args.api_key_file)
            dataset, source = load_source(args.personas_file, args.dataset_revision)
            rows, sampling = sample_dataset(dataset, args.n, seed, filters, args.with_replacement)
            personas = [from_nemotron(row, args.politics) for row in rows]
        settings = {"model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens,
                    "max_attempts": args.max_attempts, "timeout": args.timeout, "seed": seed,
                    "send_seed": not args.no_model_seed, "lang": args.lang, "survey_date": survey_date}
        directory = Path(args.run_dir) if args.run_dir else Path(args.out_dir) / f"run-{uuid.uuid4().hex}"
        if not args.dry_run:
            key = api_key(args.api_key_file)
        manifest, cohort = prepare_run(directory, personas, questions, settings, source, sampling,
                                       {"filters": filters, "politics_override": args.politics})
        print(f"Saved run plan: {directory.resolve()}", flush=True)
        if args.dry_run:
            print(f"Planned {len(cohort)} respondents; seed={seed}. No model requests made.")
            print(f"Execute with: python synthetic_survey.py --resume {directory}")
            return 0
        summary = execute_run(directory, OpenAIProvider(key, args.timeout))
        show_summary(directory, summary)
        return 1 if summary["failed"] or summary["pending"] else 0
    except KeyboardInterrupt:
        print("Interrupted. Completed records are saved; use --resume with the run directory.", file=sys.stderr)
        return 130
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
