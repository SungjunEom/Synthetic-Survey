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
    mbti: Optional[str] = Field(None, description="One of the 16 MBTI types.")
    age: int = Field(..., ge=18, le=90)
    sex: str = Field(..., description="male, female, or non-binary; or 한국어 성별 (남자/여자).")
    nationality: str
    education: str
    politics: Optional[str] = Field(None, description="US: Democrat/Republican; KR: 진보/보수; otherwise optional.")

    # Nemotron-Korea specific fields
    uuid: Optional[str] = None
    persona: Optional[str] = None
    professional_persona: Optional[str] = None
    sports_persona: Optional[str] = None
    arts_persona: Optional[str] = None
    travel_persona: Optional[str] = None
    culinary_persona: Optional[str] = None
    family_persona: Optional[str] = None
    cultural_background: Optional[str] = None
    skills_and_expertise: Optional[str] = None
    skills_and_expertise_list: Optional[str] = None
    hobbies_and_interests: Optional[str] = None
    hobbies_and_interests_list: Optional[str] = None
    career_goals_and_ambitions: Optional[str] = None
    marital_status: Optional[str] = None
    military_status: Optional[str] = None
    family_type: Optional[str] = None
    housing_type: Optional[str] = None
    education_level: Optional[str] = None
    bachelors_field: Optional[str] = None
    occupation: Optional[str] = None
    district: Optional[str] = None
    province: Optional[str] = None

    @field_validator("mbti")
    @classmethod
    def check_mbti(cls, v):
        if v is None:
            return v
        if v.upper() not in MBTI_TYPES:
            raise ValueError(f"mbti must be one of {MBTI_TYPES}")
        return v.upper()

    @field_validator("sex")
    @classmethod
    def check_sex(cls, v):
        valid_sexes = SEX_CHOICES + ["남자", "여자"]
        if v not in valid_sexes:
            raise ValueError(f"sex must be one of {valid_sexes}")
        return v

    @field_validator("education")
    @classmethod
    def check_edu(cls, v):
        valid_edus = EDU_CHOICES + ['초등학교', '4년제 대학교', '고등학교', '2~3년제 전문대학', '중학교', '대학원', '무학']
        if v not in valid_edus:
            raise ValueError(f"education must be one of {valid_edus}")
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

# --------------------------- Prompts (EN/KO) ---------------------------

def build_system_prompt(lang: str = "en") -> str:
    if lang.lower() == "ko":
        return (
            "당신은 *단 한 명의* 설문 응답자입니다. 반드시 **제공된 페르소나로만** 답하세요."
            "페르소나 특성(나이, 학력, 정치 성향, 국적)에 맞게 간결하고 현실적으로 답변하세요."
            "질문이 페르소나나 해당 국가 맥락과 무관하면 'N/A'로 짧게 답하세요."
            "반드시 *유효한 JSON만* 반환하고, 추가 설명/코멘트는 포함하지 마세요."
        )
    # default: English
    return (
        "You are a *single* survey respondent. Answer **only as the persona provided**."
        "Be brief, realistic, and consistent with the persona traits (age, education, politics, nationality)."
        "If a question is irrelevant to the persona or country context, answer 'N/A' briefly."
        "Return *only* valid JSON; do not include extra commentary."
    )

def build_user_prompt(persona: Persona, questions: List[Dict[str, Any]],
                      answer_language: Optional[str]=None, lang: str = "en") -> str:
    is_nemotron = getattr(persona, "persona", None) is not None

    if lang.lower() == "ko":
        if is_nemotron:
            persona_lines = [
                f"설명: {persona.persona}",
                f"나이: {persona.age}세",
                f"성별: {persona.sex}",
                f"국적: {persona.nationality}",
                f"학력: {persona.education_level or persona.education}",
                f"정치 성향: {persona.politics or 'N/A'}",
            ]
            if persona.occupation:
                persona_lines.append(f"직업: {persona.occupation}")
            if persona.province or persona.district:
                loc = f"{persona.province or ''} {persona.district or ''}".strip()
                persona_lines.append(f"거주지: {loc}")
            if persona.marital_status:
                persona_lines.append(f"결혼 상태: {persona.marital_status}")
            if persona.family_type:
                persona_lines.append(f"가족 형태: {persona.family_type}")
            if persona.housing_type:
                persona_lines.append(f"주거 형태: {persona.housing_type}")
            if persona.cultural_background:
                persona_lines.append(f"문화적 배경: {persona.cultural_background}")
            if persona.professional_persona:
                persona_lines.append(f"직업적 상세: {persona.professional_persona}")
            if persona.skills_and_expertise:
                persona_lines.append(f"보유 기술 및 전문성: {persona.skills_and_expertise}")
            if persona.hobbies_and_interests:
                persona_lines.append(f"취미 및 관심사: {persona.hobbies_and_interests}")
            if persona.career_goals_and_ambitions:
                persona_lines.append(f"진로 목표 및 포부: {persona.career_goals_and_ambitions}")
            
            schema_hint = (
                "다음 JSON 형식으로 출력하세요:"
                "{"
                '  "respondent": { "uuid": "...", "age": 0, "sex": "...", "nationality": "...", "education": "...", "politics": "..." },'
                '  "answers": [ {"id": "Q1", "question": "...", "answer": <string|number|array> }, ... ]'
                "}"
                "'respondent' 객체는 **위 페르소나와 정확히 일치**해야 합니다."
            )
        else:
            persona_lines = [
                f"MBTI: {persona.mbti}",
                f"나이: {persona.age}",
                f"성별: {persona.sex}",
                f"국적: {persona.nationality}",
                f"학력: {persona.education}",
                f"정치 성향: {persona.politics or 'N/A'}",
            ]
            schema_hint = (
                "다음 JSON 형식으로 출력하세요:"
                "{"
                '  "respondent": { "mbti": "...", "age": 0, "sex": "...", "nationality": "...", "education": "...", "politics": "..." },'
                '  "answers": [ {"id": "Q1", "question": "...", "answer": <string|number|array> }, ... ]'
                "}"
                "'respondent' 객체는 **위 페르소나와 정확히 일치**해야 합니다."
            )

        guidelines = (
            "아래 설문지를 위 페르소나로서 답하세요."
            "선택지가 있는 문항은 'single'이면 **하나만**, 'multi'이면 **여러 개(최대 3개)** 선택하세요."
            "'scale'은 1~5의 **정수**로, 'number'는 **숫자**로 답하세요."
            "서술형 문항은 **한 문장으로 짧게** 답하세요."
        )
        lines = [
            "페르소나:",
            *persona_lines,
            "",
            guidelines,
            "",
            schema_hint,
            "",
            "질문:"
        ]
        out = []
        for q in questions:
            opt = f" 선택지: {q['options']}" if q.get("options") else ""
            out.append(f"- {q['id']} ({q['type']}): {q['text']}{opt}")
        return "\n".join(lines + out)

    # default: English
    if is_nemotron:
        persona_lines = [
            f"Description: {persona.persona}",
            f"Age: {persona.age}",
            f"Sex: {persona.sex}",
            f"Nationality: {persona.nationality}",
            f"Education: {persona.education_level or persona.education}",
            f"Political opinion: {persona.politics or 'N/A'}",
        ]
        if persona.occupation:
            persona_lines.append(f"Occupation: {persona.occupation}")
        if persona.province or persona.district:
            loc = f"{persona.province or ''} {persona.district or ''}".strip()
            persona_lines.append(f"Location: {loc}")
        if persona.marital_status:
            persona_lines.append(f"Marital status: {persona.marital_status}")
        if persona.family_type:
            persona_lines.append(f"Family type: {persona.family_type}")
        if persona.housing_type:
            persona_lines.append(f"Housing type: {persona.housing_type}")
        if persona.cultural_background:
            persona_lines.append(f"Cultural background: {persona.cultural_background}")
        if persona.professional_persona:
            persona_lines.append(f"Professional profile: {persona.professional_persona}")
        if persona.skills_and_expertise:
            persona_lines.append(f"Skills and expertise: {persona.skills_and_expertise}")
        if persona.hobbies_and_interests:
            persona_lines.append(f"Hobbies and interests: {persona.hobbies_and_interests}")
        if persona.career_goals_and_ambitions:
            persona_lines.append(f"Career goals: {persona.career_goals_and_ambitions}")

        schema_hint = (
            "Output JSON with this shape:"
            "{"
            '  "respondent": { "uuid": "...", "age": 0, "sex": "...", "nationality": "...", "education": "...", "politics": "..." },'
            '  "answers": [ {"id": "Q1", "question": "...", "answer": <string|number|array> }, ... ]'
            "}"
            "Ensure the 'respondent' object **matches exactly** the persona above."
        )
    else:
        persona_lines = [
            f"MBTI: {persona.mbti}",
            f"Age: {persona.age}",
            f"Sex: {persona.sex}",
            f"Nationality: {persona.nationality}",
            f"Education: {persona.education}",
            f"Political opinion: {persona.politics or 'N/A'}",
        ]
        schema_hint = (
            "Output JSON with this shape:"
            "{"
            '  "respondent": { "mbti": "...", "age": 0, "sex": "...", "nationality": "...", "education": "...", "politics": "..." },'
            '  "answers": [ {"id": "Q1", "question": "...", "answer": <string|number|array> }, ... ]'
            "}"
            "Ensure the 'respondent' object **matches exactly** the persona above."
        )

    guidelines = (
        "Answer the following questionnaire as this persona."
        "Where options are provided, pick exactly one for 'single', allow multiple (<=3) for 'multi'."
        "For 'scale', answer an integer from 1 to 5. For 'number', return a number."
        "Keep free-text answers to one short sentence."
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
    out = []
    for q in questions:
        opt = ""
        if q.get("options"):
            opt = f" Options: {q['options']}"
        out.append(f"- {q['id']} ({q['type']}): {q['text']}{opt}")
    return "\n".join(lines + out)


def call_model(client: OpenAI, model: str, persona: Persona, questions: List[Dict[str, Any]],
               temperature: float=0.8, seed: Optional[int]=None, max_tokens: int=800,
               lang: str = "en") -> Dict[str, Any]:
    system_prompt = build_system_prompt(lang=lang)
    user_prompt = build_user_prompt(persona, questions, lang=lang)

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
            row_dict = {
                "respondent_id": r.respondent_id,
                "mbti": r.persona.mbti,
                "age": r.persona.age,
                "sex": r.persona.sex,
                "nationality": r.persona.nationality,
                "education": r.persona.education,
                "politics": r.persona.politics,
            }
            # Dynamically add other fields if they are set (e.g. from Nemotron)
            p_dict = r.persona.model_dump()
            for k, v in p_dict.items():
                if k not in row_dict and v is not None:
                    row_dict[k] = v

            row_dict.update({
                "q_id": qa.id,
                "question": qa.question,
                "answer": qa.answer,
            })
            rows.append(row_dict)
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

def load_nemotron_personas(
    n: int,
    seed: Optional[int] = None,
    sex: Optional[str] = None,
    age_spec: Optional[str] = None,
    education: Optional[str] = None,
    politics: Optional[str] = None,
    randomize_politics: bool = False,
    marital_status: Optional[str] = None,
    housing_type: Optional[str] = None,
    occupation: Optional[str] = None,
    province: Optional[str] = None,
    district: Optional[str] = None
) -> List[Persona]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("The 'datasets' package is required for Nemotron personas. Please install it using: pip install datasets")
        sys.exit(1)

    print("Loading nvidia/Nemotron-Personas-Korea dataset from Hugging Face...")
    try:
        dataset = load_dataset("nvidia/Nemotron-Personas-Korea", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure you have internet access and the 'datasets' package is installed correctly.")
        sys.exit(1)

    df = dataset.to_pandas()

    # Apply sex filter
    if sex:
        sex_map = {"male": "남자", "female": "여자", "non-binary": "남자"}
        target_sex = sex_map.get(sex.lower(), sex)
        df = df[df["sex"] == target_sex]

    # Apply age filter
    if age_spec:
        s = str(age_spec).strip()
        if s.isdigit():
            target_age = int(s)
            df = df[df["age"] == target_age]
        else:
            m = re.match(r'^(\d+)\s*-\s*(\d+)$', s)
            if m:
                lo, hi = sorted([int(m.group(1)), int(m.group(2))])
                df = df[(df["age"] >= lo) & (df["age"] <= hi)]
            else:
                raise ValueError(f"Invalid age format: {age_spec}")

    # Apply education filter
    if education:
        edu_map = {
            "high school": "고등학교",
            "associate's": "2~3년제 전문대학",
            "bachelor's": "4년제 대학교",
            "master's": "대학원",
            "doctorate": "대학원"
        }
        target_edu = edu_map.get(education.lower(), education)
        df = df[df["education_level"] == target_edu]

    # Apply marital status filter
    if marital_status:
        marital_map = {
            "married": "배우자있음",
            "single": "미혼",
            "unmarried": "미혼",
            "widowed": "사별",
            "divorced": "이혼"
        }
        target_marital = marital_map.get(marital_status.lower(), marital_status)
        df = df[df["marital_status"] == target_marital]

    # Apply housing type filter
    if housing_type:
        housing_map = {
            "apartment": "아파트",
            "villa": "다세대주택",
            "multi-family": "연립주택",
            "house": "단독주택",
            "single-family": "단독주택"
        }
        target_housing = housing_map.get(housing_type.lower(), housing_type)
        df = df[df["housing_type"] == target_housing]

    # Apply occupation filter (case-insensitive substring match)
    if occupation:
        df = df[df["occupation"].str.contains(occupation, case=False, na=False)]

    # Apply province filter (case-insensitive substring match)
    if province:
        df = df[df["province"].str.contains(province, case=False, na=False)]

    # Apply district filter (case-insensitive substring match)
    if district:
        df = df[df["district"].str.contains(district, case=False, na=False)]

    if len(df) == 0:
        print("No personas match the specified filters in the Nemotron dataset.")
        sys.exit(1)

    # Sample rows
    sampled_df = df.sample(n=n, random_state=seed, replace=(len(df) < n))

    # Convert to Persona objects
    personas = []
    import random as py_random
    rnd = py_random.Random(seed)
    for _, row in sampled_df.iterrows():
        pol = politics
        if not pol and randomize_politics:
            pol = rnd.choice(KR_POLITICS)

        pers = Persona(
            mbti=None,
            age=int(row["age"]),
            sex=row["sex"],
            nationality="South Korea",
            education=row["education_level"],
            politics=pol,
            uuid=row["uuid"],
            persona=row["persona"],
            professional_persona=row.get("professional_persona"),
            sports_persona=row.get("sports_persona"),
            arts_persona=row.get("arts_persona"),
            travel_persona=row.get("travel_persona"),
            culinary_persona=row.get("culinary_persona"),
            family_persona=row.get("family_persona"),
            cultural_background=row.get("cultural_background"),
            skills_and_expertise=row.get("skills_and_expertise"),
            skills_and_expertise_list=row.get("skills_and_expertise_list"),
            hobbies_and_interests=row.get("hobbies_and_interests"),
            hobbies_and_interests_list=row.get("hobbies_and_interests_list"),
            career_goals_and_ambitions=row.get("career_goals_and_ambitions"),
            marital_status=row.get("marital_status"),
            military_status=row.get("military_status"),
            family_type=row.get("family_type"),
            housing_type=row.get("housing_type"),
            education_level=row.get("education_level"),
            bachelors_field=row.get("bachelors_field"),
            occupation=row.get("occupation"),
            district=row.get("district"),
            province=row.get("province")
        )
        personas.append(pers)

    return personas

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
    parser.add_argument("--lang", choices=["en", "ko"], default="en", help="Prompt language (en or ko).")
    # Nemotron-Korea argument
    parser.add_argument("--nemotron-korea", action="store_true", help="Use nvidia/Nemotron-Personas-Korea dataset from Hugging Face.")
    # Nemotron-specific filters
    parser.add_argument("--marital-status", help="Filter Nemotron personas by marital status (e.g. married, single, widowed, divorced, or 한국어).")
    parser.add_argument("--housing-type", help="Filter Nemotron personas by housing type (e.g. apartment, house, villa, or 한국어).")
    parser.add_argument("--occupation", help="Filter Nemotron personas by occupation (substring match).")
    parser.add_argument("--province", help="Filter Nemotron personas by province (substring match).")
    parser.add_argument("--district", help="Filter Nemotron personas by district (substring match).")
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

    # Prepare cohort of personas
    if args.nemotron_korea:
        personas = load_nemotron_personas(
            n=args.n,
            seed=args.seed,
            sex=args.sex,
            age_spec=args.age,
            education=args.education,
            politics=args.politics,
            randomize_politics=args.randomize,
            marital_status=args.marital_status,
            housing_type=args.housing_type,
            occupation=args.occupation,
            province=args.province,
            district=args.district
        )
    else:
        personas = []
        for i in range(args.n):
            age_for_resp = pick_age(args.age, rnd)
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
                pers = random_persona(
                    seed=(args.seed if args.seed is not None else rnd.randrange(1<<30)),
                    nationality=args.nationality,
                    politics=args.politics,
                    mbti=args.mbti,
                    age=age_for_resp,
                    sex=args.sex,
                    education=args.education
                )
            personas.append(pers)

    results: List[SurveyResult] = []
    ts = int(time.time())
    jsonl_path = out_dir / f"results_{ts}.jsonl"
    csv_path = out_dir / f"results_{ts}.csv"

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for pers in tqdm(personas, desc="Surveying"):
            # Call model with retry/backoff
            for attempt in range(4):
                try:
                    raw = call_model(
                        client, args.model, pers, questions,
                        temperature=args.temperature,
                        seed=rnd.randrange(1<<30),
                        max_tokens=args.max_tokens,
                        lang=args.lang,
                    )
                    break
                except Exception as e:
                    wait = 1.5 * (attempt + 1)
                    if attempt == 3:
                        raise
                    time.sleep(wait)

            # Validate/coerce
            try:
                answers = coerce_answers(raw, questions)
                survey_res = SurveyResult(
                    respondent_id=str(uuid.uuid4())[:8],
                    persona=pers,
                    answers=answers,
                    meta={"model": args.model}
                )
            except ValidationError as ve:
                print("Validation error on model output:", ve)
                continue

            # Persist JSONL row
            jf.write(json.dumps(survey_res.model_dump(), ensure_ascii=False) + "\n")
            results.append(survey_res)

    # Build CSV
    df = summarize_to_dataframe(results)
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Print tiny summaries
    def count_col(col):
        return df[[col]].value_counts().reset_index(name="count")

    print("--- Summary ---")
    if not df.empty:
        cols_to_summarize = ["nationality", "politics", "mbti", "education", "sex"]
        for extra in ["occupation", "province", "education_level"]:
            if extra in df.columns and not df[extra].isna().all():
                cols_to_summarize.append(extra)
        
        for col in cols_to_summarize:
            if col in df.columns:
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
