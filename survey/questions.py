"""Questionnaire definitions and independent validation of model output."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


def finite_number(value) -> bool:
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject ambiguous definitions rather than silently keeping the last value."""


def unique_mapping(loader, node):
    pairs = loader.construct_pairs(node, deep=True)
    result = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError("YAML mapping keys must be unique strings")
        result[key] = value
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    text: str
    type: Literal["free", "single", "multi", "scale", "number"] = "free"
    options: list[str] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    min_choices: int | None = None
    max_choices: int | None = None
    allow_na: bool = False

    @model_validator(mode="after")
    def check_definition(self):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", self.id):
            raise ValueError("Question IDs must start with a letter and use letters, digits, _ or -")
        if not self.text.strip():
            raise ValueError(f"{self.id}: text must not be blank")
        if self.type in {"single", "multi"}:
            if not self.options or any(not x.strip() for x in self.options):
                raise ValueError(f"{self.id}: choices require nonempty string options")
            if len(set(self.options)) != len(self.options):
                raise ValueError(f"{self.id}: duplicate options")
        elif self.options is not None:
            raise ValueError(f"{self.id}: options only apply to single/multi questions")
        if self.type == "multi":
            self.min_choices = 1 if self.min_choices is None else self.min_choices
            self.max_choices = min(3, len(self.options)) if self.max_choices is None else self.max_choices
            if not 0 <= self.min_choices <= self.max_choices <= len(self.options):
                raise ValueError(f"{self.id}: invalid choice limits")
        elif self.min_choices is not None or self.max_choices is not None:
            raise ValueError(f"{self.id}: choice limits only apply to multi questions")
        if self.type == "scale":
            self.minimum = 1 if self.minimum is None else self.minimum
            self.maximum = 5 if self.maximum is None else self.maximum
            if type(self.minimum) is not int or type(self.maximum) is not int:
                raise ValueError(f"{self.id}: scale bounds must be integers")
        if self.type in {"scale", "number"}:
            for bound in (self.minimum, self.maximum):
                if bound is not None and not finite_number(bound):
                    raise ValueError(f"{self.id}: bounds must be finite numbers")
            if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
                raise ValueError(f"{self.id}: minimum exceeds maximum")
        elif self.minimum is not None or self.maximum is not None:
            raise ValueError(f"{self.id}: numeric bounds only apply to scale/number")
        return self


def validate_questions(data) -> list[Question]:
    if isinstance(data, dict):
        if set(data) != {"questions"}:
            raise ValueError("Questionnaire must contain only a 'questions' list")
        data = data["questions"]
    if not isinstance(data, list) or not data:
        raise ValueError("Questionnaire must be a nonempty list")
    questions = [Question.model_validate(q) for q in data]
    if len({q.id for q in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
    return questions


def load_questions(path: str | Path) -> list[Question]:
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    try:
        data = yaml.load(content, Loader=UniqueKeyLoader) if path.suffix.lower() in {".yaml", ".yml"} else strict_json(content)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid questionnaire YAML: {exc}") from exc
    return validate_questions(data)


def object_schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def response_schema(questions: list[Question]) -> dict:
    properties = {}
    for q in questions:
        if q.type == "single":
            schema = {"type": "string", "enum": q.options}
        elif q.type == "multi":
            schema = {"type": "array", "items": {"type": "string", "enum": q.options},
                      "minItems": q.min_choices, "maxItems": q.max_choices}
        elif q.type in {"scale", "number"}:
            schema = {"type": "integer" if q.type == "scale" else "number"}
            for key in ("minimum", "maximum"):
                if getattr(q, key) is not None:
                    schema[key] = getattr(q, key)
        else:
            schema = {"type": "string"}
        properties[q.id] = {"anyOf": [schema, {"type": "null"}]} if q.allow_na else schema
    return object_schema({"answers": object_schema(properties)})


def strict_json(content: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid_constant)


def validate_answers(raw, questions: list[Question]) -> dict:
    if not isinstance(raw, dict) or set(raw) != {"answers"}:
        raise ValueError("Response must contain exactly an 'answers' object")
    answers = raw["answers"]
    if not isinstance(answers, dict) or set(answers) != {q.id for q in questions}:
        raise ValueError("Answers must contain every question ID exactly once, with no unknown IDs")
    for q in questions:
        a = answers[q.id]
        if a is None and q.allow_na:
            continue
        valid = False
        if q.type == "free":
            valid = isinstance(a, str) and bool(a.strip())
        elif q.type == "single":
            valid = isinstance(a, str) and a in q.options
        elif q.type == "multi":
            valid = (isinstance(a, list) and all(isinstance(x, str) and x in q.options for x in a)
                     and len(set(a)) == len(a) and q.min_choices <= len(a) <= q.max_choices)
        elif q.type in {"scale", "number"}:
            valid = type(a) is int if q.type == "scale" else type(a) in {int, float}
            valid = (valid and finite_number(a)
                     and (q.minimum is None or a >= q.minimum)
                     and (q.maximum is None or a <= q.maximum))
        if not valid:
            raise ValueError(f"{q.id}: invalid {q.type} answer")
    return {q.id: answers[q.id] for q in questions}
