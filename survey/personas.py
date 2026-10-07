"""Sample Arrow-backed data without materializing narrative columns in pandas."""
from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DATASET = "nvidia/Nemotron-Personas-Korea"
SEX_MAP = {"male": "남자", "female": "여자"}
EDUCATION_MAP = {"high school": "고등학교", "associate's": "2~3년제 전문대학",
                 "bachelor's": "4년제 대학교", "master's": "대학원", "doctorate": "대학원"}
EDUCATION_KR = {"무학", "초등학교", "중학교", "고등학교", "2~3년제 전문대학", "4년제 대학교", "대학원"}
MARITAL_MAP = {"married": "배우자있음", "single": "미혼", "unmarried": "미혼", "widowed": "사별", "divorced": "이혼"}
HOUSING_MAP = {"apartment": "아파트", "villa": "다세대주택", "multi-family": "연립주택",
               "house": "단독주택", "single-family": "단독주택"}
MBTI = [a+b+c+d for a in "IE" for b in "NS" for c in "FT" for d in "JP"]


class Persona(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    uuid: str = Field(min_length=1)
    age: int = Field(ge=18, le=120)
    sex: str = Field(min_length=1)
    nationality: str = Field(min_length=1)
    education: str = Field(min_length=1)
    politics: str | None = None
    mbti: str | None = None
    # Preserve every source field, including all narrative facets.
    attributes: dict[str, Any] = Field(default_factory=dict)


def age_bounds(spec: str | None, minimum: int = 19) -> tuple[int, int] | None:
    if spec is None:
        return None
    match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", spec)
    if not match:
        raise ValueError("Age must be an integer or an ascending range such as 25-40")
    lo, hi = int(match[1]), int(match[2] or match[1])
    if not minimum <= lo <= hi <= 120:
        raise ValueError(f"Age range must be ascending and within {minimum}-120")
    return lo, hi


def normalize_filters(values: dict) -> dict:
    filters = {k: v.strip() for k, v in values.items() if v is not None}
    if any(not v for v in filters.values()):
        raise ValueError("Filters must not be blank")
    if "age" in filters:
        filters["age"] = age_bounds(filters["age"])
    for field, mapping in (("sex", SEX_MAP), ("education_level", EDUCATION_MAP),
                           ("marital_status", MARITAL_MAP), ("housing_type", HOUSING_MAP)):
        if field in filters:
            filters[field] = mapping.get(filters[field].lower(), filters[field])
    if filters.get("sex") not in {None, "남자", "여자"}:
        raise ValueError("Nemotron has only 남자/male and 여자/female sex categories; non-binary is unavailable")
    if filters.get("education_level") not in EDUCATION_KR | {None}:
        raise ValueError("Unknown Nemotron education category")
    return filters


def matches(values: dict, filters: dict) -> bool:
    for field, target in filters.items():
        value = values.get(field)
        if field == "age":
            if not isinstance(value, int) or not target[0] <= value <= target[1]:
                return False
        elif field in {"occupation", "province", "district"}:
            if not isinstance(value, str) or target.casefold() not in value.casefold():
                return False
        elif value != target:
            return False
    return True


def sample_dataset(dataset, n: int, seed: int, filters: dict, replacement: bool = False):
    if n < 1:
        raise ValueError("Respondent count must be positive")
    required = {"uuid", "age", "sex", "education_level", "persona"} | set(filters)
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
    total = len(dataset)
    if filters:
        columns = list(filters)

        def keep_batch(*batches):
            return [matches(dict(zip(columns, row)), filters) for row in zip(*batches)]

        dataset = dataset.filter(keep_batch, input_columns=columns, batched=True,
                                 desc="Filtering demographics")
    eligible = len(dataset)
    if not eligible:
        raise ValueError("No personas match the requested filters")
    if n > eligible and not replacement:
        raise ValueError(f"Requested {n} unique personas, but only {eligible} are eligible; use --with-replacement explicitly")
    rng = random.Random(seed)
    indices = [rng.randrange(eligible) for _ in range(n)] if replacement else rng.sample(range(eligible), n)
    selected = list(dataset.select(indices))
    ids = [row["uuid"] for row in selected]
    if not replacement and len(set(ids)) != n:
        raise ValueError("Source contains duplicate selected UUIDs; repair the source instead of treating them as distinct people")
    return selected, {"population_rows": total, "eligible_rows": eligible,
                      "sampled_rows": n, "unique_personas": len(set(ids)), "replacement": replacement}


def from_nemotron(row: dict, politics: str | None = None) -> Persona:
    if type(row.get("age")) is not int or not 19 <= row["age"] <= 120:
        raise ValueError(f"Invalid adult age in persona {row.get('uuid')}")
    if row.get("sex") not in {"남자", "여자"} or row.get("education_level") not in EDUCATION_KR:
        raise ValueError(f"Invalid demographics in persona {row.get('uuid')}")
    if not isinstance(row.get("persona"), str) or not row["persona"].strip():
        raise ValueError(f"Missing narrative in persona {row.get('uuid')}")
    if row.get("country", "대한민국") != "대한민국":
        raise ValueError("Nemotron Korea records must have country 대한민국")
    return Persona(uuid=row["uuid"], age=row["age"], sex=row["sex"], nationality="South Korea",
                   education=row["education_level"], politics=politics, attributes=row)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source(personas_file: str | None, revision: str):
    from datasets import load_dataset
    if personas_file:
        path = Path(personas_file).resolve()
        digest = file_hash(path)
        dataset = load_dataset("json", data_files=str(path), split="train")
        return dataset, {"kind": "local", "path": str(path), "sha256": digest}
    from huggingface_hub import HfApi
    # Resolve mutable refs before loading so provenance identifies exactly what was sampled.
    resolved = HfApi().dataset_info(DATASET, revision=revision).sha
    if not resolved:
        raise ValueError("Could not resolve the dataset revision")
    dataset = load_dataset(DATASET, revision=resolved, split="train")
    return dataset, {"kind": "huggingface", "dataset": DATASET, "split": "train",
                     "requested_revision": revision, "revision": resolved,
                     "fingerprint": dataset._fingerprint}


def custom_personas(n: int, seed: int, fields: dict, randomize: bool) -> list[Persona]:
    """Convenience experiment generator; not a population model."""
    rng = random.Random(seed)
    bounds = age_bounds(fields.get("age"), minimum=18)
    mbti = fields.get("mbti")
    if mbti and mbti.upper() not in MBTI:
        raise ValueError("Invalid MBTI")
    if fields.get("sex") and fields["sex"] not in {"male", "female", "non-binary", "남자", "여자"}:
        raise ValueError("Invalid custom persona sex")

    def draw():
        return dict(age=rng.randint(*(bounds or ((18, 80) if randomize else (35, 35)))),
                    sex=fields.get("sex") or (rng.choice(["male", "female"]) if randomize else "female"),
                    nationality=fields.get("nationality") or "South Korea",
                    education=fields.get("education") or (rng.choice(sorted(EDUCATION_KR)) if randomize else "4년제 대학교"),
                    mbti=mbti.upper() if mbti else (rng.choice(MBTI) if randomize else None),
                    politics=fields.get("politics"))
    fixed = draw()
    return [Persona(uuid=f"custom-{i+1:06d}", **(draw() if randomize else fixed)) for i in range(n)]
