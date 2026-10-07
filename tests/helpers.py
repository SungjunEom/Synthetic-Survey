import json

from survey.personas import from_nemotron
from survey.provider import Reply
from survey.questions import Question
from survey.runner import prepare_run


def row(i=1, **changes):
    return {"uuid": f"fixture-{i}", "age": 30, "sex": "여자", "education_level": "고등학교",
            "persona": "테스트용 가상의 응답자", "province": "서울", "district": "서울-강남구",
            "occupation": "개발자 [웹]", "marital_status": "미혼", "housing_type": "아파트",
            "country": "대한민국", "travel_persona": "여행 맥락", "family_persona": "가족 맥락",
            "sports_persona": "운동 맥락", "arts_persona": "예술 맥락", "culinary_persona": "음식 맥락",
            "professional_persona": "직장 맥락", "hobbies_and_interests_list": ["독서", "걷기"], **changes}


def reply(value=3, **changes):
    return Reply(**({"content": json.dumps({"answers": {"Q1": value}}), "finish_reason": "stop",
                    "model": "fake-model", "usage": {"total_tokens": 12}} | changes))


class FakeProvider:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def complete(self, messages, schema, settings, seed):
        self.calls.append({"messages": messages, "schema": schema, "settings": settings, "seed": seed})
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def plan(path, n=2, max_attempts=3, questions=None):
    return prepare_run(path, [from_nemotron(row(i, age=99 if i == 0 else 30)) for i in range(n)],
                       questions or [Question(id="Q1", text="삶의 만족도", type="scale")],
                       {"model": "fake-model", "temperature": 0.8, "seed": 42, "lang": "ko",
                        "survey_date": "2026-10-07", "max_tokens": 4096, "max_attempts": max_attempts,
                        "timeout": 10.0, "send_seed": True},
                       {"kind": "fixture"}, {"sampled_rows": n}, {})
