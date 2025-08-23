#!/usr/bin/env python3
from __future__ import annotations

import argparse, os, json, random, time, sys, csv, uuid, datetime, pathlib, re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from tqdm import tqdm

import yaml  # pyyaml
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

# --- OpenAI official SDK (https://platform.openai.com/docs/libraries) ---
try:
    from openai import OpenAI  # SDK v1+
except Exception as e:
    print("Couldn't import OpenAI SDK. Did you install requirements.txt?")
    raise

# --------------------------- Config & Constants ---------------------------

MBTI_TYPES = [
    "INTJ","INTP","ENTJ","ENTP",
    "INFJ","INFP","ENFJ","ENFP",
    "ISTJ","ISFJ","ESTJ","ESFJ",
    "ISTP","ISFP","ESTP","ESFP"
]

SEX_CHOICES = ["male", "female", "non-binary"]
EDU_CHOICES = ["High school", "Associate's", "Bachelor's", "Master's", "Doctorate"]
NATIONALITIES = ["United States", "South Korea", "United Kingdom", "Canada", "Australia", "India", "Germany", "France", "Japan", "Brazil"]

US_POLITICS = ["Democrat", "Republican"]
KR_POLITICS = ["진보", "보수"]

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# --------------------------- Data Models ---------------------------------

class Persona(BaseModel):
    mbti: str = Field(..., description="One of the 16 MBTI types.")
    age: int = Field(..., ge=18, le=90)
    sex: str = Field(..., description="male, female, or non-binary")
    nationality: str
    education: str
    politics: Optional[str] = Field(None, description="US: Democrat/Republican; KR: 진보/보수; otherwise optional.")

    @field_validator("mbti")
    @classmethod
    def check_mbti(cls, v):
        if v.upper() not in MBTI_TYPES:
            raise ValueError(f"mbti must be one of {MBTI_TYPES}")
        return v.upper()

    @field_validator("sex")
    @classmethod
    def check_sex(cls, v):
        if v not in SEX_CHOICES:
            raise ValueError(f"sex must be one of {SEX_CHOICES}")
        return v

    @field_validator("education")
    @classmethod
    def check_edu(cls, v):
        if v not in EDU_CHOICES:
            raise ValueError(f"education must be one of {EDU_CHOICES}")
        return v

class QA(BaseModel):
    id: str
    question: str
    answer: Any

class SurveyResult(BaseModel):
    respondent_id: str
    persona: Persona
    answers: List[QA]
    meta: Dict[str, Any] = Field(default_factory=dict)

# --------------------------- Helpers -------------------------------------

def load_api_key(path: str) -> Optional[str]:
    # Prefer environment, else fallback to file (single line).
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()
    p = pathlib.Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None

def load_questions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    qs = data.get("questions", data)
    # Normalize
    out = []
    for i, q in enumerate(qs, start=1):
        qid = q.get("id") or f"Q{i}"
        out.append({
            "id": qid,
            "text": q.get("text") or q.get("question"),
            "type": q.get("type", "free"),
            "options": q.get("options")
        })
    return out

def random_persona(seed: Optional[int]=None, nationality: Optional[str]=None, politics: Optional[str]=None,
                   mbti: Optional[str]=None, age: Optional[int]=None, sex: Optional[str]=None, education: Optional[str]=None) -> Persona:
    rnd = random.Random(seed if seed is not None else random.randrange(1<<30))
    nat = nationality or rnd.choice(NATIONALITIES)
    # politics rules by nationality
    pol = politics
    if not pol:
        if nat == "United States":
            pol = rnd.choice(US_POLITICS)
        elif nat == "South Korea":
            pol = rnd.choice(KR_POLITICS)
        else:
            pol = None
    return Persona(
        mbti=(mbti or rnd.choice(MBTI_TYPES)).upper(),
        age=age if age is not None else rnd.randint(18, 80),
        sex=sex or rnd.choice(SEX_CHOICES),
        nationality=nat,
        education=education or rnd.choice(EDU_CHOICES),
        politics=pol
    )

def build_system_prompt() -> str:
    return (
        "You are a *single* survey respondent. Answer **only as the persona provided**."
        "Be brief, realistic, and consistent with the persona traits (age, education, politics, nationality)."
        "If a question is irrelevant to the persona or country context, answer 'N/A' briefly."
        "Return *only* valid JSON; do not include extra commentary."
    )

def build_user_prompt(persona: Persona, questions: List[Dict[str, Any]], answer_language: Optional[str]=None) -> str:
    # We let the model respond in the language of the question text by default.
    persona_lines = [
        f"MBTI: {persona.mbti}",
        f"Age: {persona.age}",
        f"Sex: {persona.sex}",
        f"Nationality: {persona.nationality}",
        f"Education: {persona.education}",
        f"Political opinion: {persona.politics or 'N/A'}",
    ]
    guidelines = (
        "Answer the following questionnaire as this persona."
        "Where options are provided, pick exactly one for 'single', allow multiple (<=3) for 'multi'."
        "For 'scale', answer an integer from 1 to 5. For 'number', return a number."
        "Keep free-text answers to one short sentence."
    )
    # Create a minimal self-contained instruction for JSON shape
    schema_hint = (
        "Output JSON with this shape:"
        "{"
        '  "respondent": { "mbti": "...", "age": 0, "sex": "...", "nationality": "...", "education": "...", "politics": "..." },'
        '  "answers": [ {"id": "Q1", "question": "...", "answer": <string|number|array> }, ... ]'
        "}"
        "Ensure the 'respondent' object **matches exactly** the persona above."
    )
    lines = [
        "Persona:",
        *persona_lines,
        "",
        guidelines,
        "",
        schema_hint,
        "",
        "Questions:"
    ]
    for q in questions:
        opt = ""
        if q.get("options"):
            opt = f" Options: {q['options']}"
        lines.append(f"- {q['id']} ({q['type']}): {q['text']}{opt}")
    return "".join(lines)

def call_model(client: OpenAI, model: str, persona: Persona, questions: List[Dict[str, Any]], temperature: float=0.8, seed: Optional[int]=None, max_tokens: int=800) -> Dict[str, Any]:
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(persona, questions)

    # We request strict JSON output using Chat Completions + response_format
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        seed=seed
    )
    content = resp.choices[0].message.content
    return json.loads(content)

def summarize_to_dataframe(results: List[SurveyResult]) -> pd.DataFrame:
    # Flatten into rows per answer
    rows = []
    for r in results:
        for qa in r.answers:
            rows.append({
                "respondent_id": r.respondent_id,
                "mbti": r.persona.mbti,
                "age": r.persona.age,
                "sex": r.persona.sex,
                "nationality": r.persona.nationality,
                "education": r.persona.education,
                "politics": r.persona.politics,
                "q_id": qa.id,
                "question": qa.question,
                "answer": qa.answer,
            })
    return pd.DataFrame(rows)

def coerce_answers(raw_json: Dict[str, Any], questions: List[Dict[str, Any]]) -> List[QA]:
    # Minimal coercion so rows are consistent
    q_by_id = {q["id"]: q for q in questions}
    answers = []
    for item in raw_json.get("answers", []):
        qid = str(item.get("id"))
        question_text = item.get("question") or q_by_id.get(qid, {}).get("text", "")
        ans = item.get("answer")
        answers.append(QA(id=qid, question=question_text, answer=ans))
    return answers

def pick_age(spec: Optional[str], rnd: random.Random, default_min: int = 18, default_max: int = 90) -> int:
    """Parse --age which can be a single int ('30') or a range '25-40'.
    Returns a sampled age within [default_min, default_max].
    """
    if spec is None:
        # Original default behavior was ~18-80; we'll keep a broad default within validation bounds.
        return rnd.randint(default_min, 80)
    s = str(spec).strip()
    # Single integer
    if s.isdigit():
        val = int(s)
        if val < default_min or val > default_max:
            raise ValueError(f"--age must be between {default_min}-{default_max}")
        return val
    # Range MIN-MAX
    m = re.match(r'^(\d+)\s*-\s*(\d+)$', s)
    if m:
        lo, hi = sorted([int(m.group(1)), int(m.group(2))])
        lo = max(lo, default_min)
        hi = min(hi, default_max)
        if lo > hi:
            raise ValueError(f"Invalid --age range after clamping: {lo}-{hi}")
        return rnd.randint(lo, hi)
    raise ValueError('Invalid --age format. Use a single integer like "30" or a range like "25-40".')

def main():
    parser = argparse.ArgumentParser(description="Run a synthetic survey via OpenAI's Chat Completions API.")
    parser.add_argument("--questions-file", default="questions.yaml", help="YAML or JSON file with a 'questions' list.")
    parser.add_argument("--api-key-file", default="api_key.txt", help="Plaintext API key file (1 line).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use (text-capable).")
    parser.add_argument("--n", type=int, required=True, help="Number of synthetic respondents to generate.")
    parser.add_argument("--randomize", action="store_true", help="Randomize unspecified persona fields for each respondent.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    # Manual persona overrides
    parser.add_argument("--mbti")
    parser.add_argument("--age", type=str, help="Age or range MIN-MAX (e.g., 25-40). You can still pass a single integer.")
    parser.add_argument("--sex", choices=SEX_CHOICES)
    parser.add_argument("--nationality")
    parser.add_argument("--education", choices=EDU_CHOICES)
    parser.add_argument("--politics")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=800)
    args = parser.parse_args()

    # Load key
    key = load_api_key(args.api_key_file)
    if not key:
        print("OpenAI API key not found. Put it in api_key.txt or set OPENAI_API_KEY env var.")
        sys.exit(2)

    client = OpenAI(api_key=key)

    # Load questionnaire
    questions = load_questions(args.questions_file)
    if not questions:
        print("No questions found.")
        sys.exit(2)

    rnd = random.Random(args.seed)
    out_dir = pathlib.Path("out")
    out_dir.mkdir(exist_ok=True)

    results: List[SurveyResult] = []
    ts = int(time.time())
    jsonl_path = out_dir / f"results_{ts}.jsonl"
    csv_path = out_dir / f"results_{ts}.csv"

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for i in tqdm(range(args.n), desc="Surveying"):
            # Determine age for this respondent (range or single int)
            age_for_resp = pick_age(args.age, rnd)

            # Persona handling
            if args.randomize:
                pers = random_persona(
                    seed=rnd.randrange(1<<30),
                    nationality=args.nationality,
                    politics=args.politics,
                    mbti=args.mbti,
                    age=age_for_resp,
                    sex=args.sex,
                    education=args.education
                )
            else:
                # If user didn't randomize, still allow unspecified fields to be auto-filled once
                pers = random_persona(
                    seed=(args.seed if args.seed is not None else rnd.randrange(1<<30)),
                    nationality=args.nationality,
                    politics=args.politics,
                    mbti=args.mbti,
                    age=age_for_resp,
                    sex=args.sex,
                    education=args.education
                )

            # Call model with retry/backoff
            for attempt in range(4):
                try:
                    raw = call_model(client, args.model, pers, questions, temperature=args.temperature, seed=rnd.randrange(1<<30), max_tokens=args.max_tokens)
                    break
                except Exception as e:
                    wait = 1.5 * (attempt + 1)
                    if attempt == 3:
                        raise
                    time.sleep(wait)

            # Validate/coerce
            try:
                respondent = {
                    "mbti": pers.mbti,
                    "age": pers.age,
                    "sex": pers.sex,
                    "nationality": pers.nationality,
                    "education": pers.education,
                    "politics": pers.politics
                }
                answers = coerce_answers(raw, questions)
                survey_res = SurveyResult(
                    respondent_id=str(uuid.uuid4())[:8],
                    persona=Persona(**respondent),
                    answers=answers,
                    meta={"model": args.model}
                )
            except ValidationError as ve:
                print("Validation error on model output:", ve)
                continue

            # Persist JSONL row
            jf.write(json.dumps(survey_res.model_dump(), ensure_ascii=False) + "")
            results.append(survey_res)

    # Build CSV
    df = summarize_to_dataframe(results)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Print tiny summaries
    def count_col(col):
        return df[[col]].value_counts().reset_index(name="count")

    print("--- Summary ---")
    if not df.empty:
        for col in ["nationality", "politics", "mbti", "education", "sex"]:
            vc = count_col(col).head(10)
            print(f"{col} (top 10):")
            print(vc.to_string(index=False))
        print(f"Saved {len(results)} respondents to:")
        print(f"  JSONL: {jsonl_path}")
        print(f"  CSV:   {csv_path}")
    else:
        print("No rows to summarize.")

if __name__ == "__main__":
    main()
