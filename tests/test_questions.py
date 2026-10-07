import tempfile
import unittest
from pathlib import Path

from survey.questions import (Question, load_questions, response_schema, strict_json,
                              validate_answers, validate_questions)


class QuestionTests(unittest.TestCase):
    def test_every_question_type_and_na(self):
        qs = validate_questions([
            {"id": "S", "text": "Scale", "type": "scale"},
            {"id": "C", "text": "Choice", "type": "single", "options": ["a", "b"]},
            {"id": "M", "text": "Multi", "type": "multi", "options": ["a", "b"]},
            {"id": "N", "text": "Number", "type": "number", "minimum": 0},
            {"id": "F", "text": "Free", "allow_na": True},
        ])
        good = {"S": 3, "C": "a", "M": ["a", "b"], "N": 1.5, "F": None}
        self.assertEqual(validate_answers({"answers": good}, qs), good)
        for key, invalid in [("S", 999), ("S", True), ("S", 3.0), ("S", "3"),
                             ("C", "invalid"), ("C", None), ("M", ["a", "a"]),
                             ("M", []), ("M", [42]), ("N", -1), ("N", float("inf")),
                             ("N", True), ("N", 10**400), ("F", " "), ("F", 2)]:
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                validate_answers({"answers": good | {key: invalid}}, qs)

    def test_reject_missing_unknown_and_wrong_shapes(self):
        qs = [Question(id="Q1", text="Test", type="scale")]
        for raw in ({}, [], {"answers": []}, {"answers": {}}, {"answers": {"Q1": 3, "Q2": 4}},
                    {"answers": {"Q1": 3}, "respondent": {}}, {"answers": [42]}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_answers(raw, qs)

    def test_strict_json_rejects_duplicate_keys_and_constants(self):
        for raw in ('{"answers":{"Q1":1,"Q1":2}}', '{"x":NaN}', '{"x":Infinity}', '{'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                strict_json(raw)

    def test_invalid_definitions(self):
        for change in ({"text": " "}, {"id": "bad id"}, {"type": "unknown"},
                       {"type": "single"}, {"type": "single", "options": ["a", "a"]},
                       {"type": "single", "options": [1]}, {"options": ["a"]},
                       {"type": "scale", "minimum": 6}, {"type": "scale", "minimum": 1.5},
                       {"type": "number", "maximum": float("nan")}, {"min_choices": 1},
                       {"type": "multi", "options": ["a"], "max_choices": 2}, {"typo": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Question.model_validate({"id": "Q1", "text": "Test"} | change)
        with self.assertRaises(ValueError):
            validate_questions([{"id": "Q1", "text": "First"}, {"id": "Q1", "text": "Second"}])

    def test_load_list_or_wrapper_yaml_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            for name, content in [("q.json", '[{"id":"Q1","text":"Test"}]'),
                                  ("q.YAML", 'questions:\n  - id: Q1\n    text: Test\n')]:
                path = Path(d) / name
                path.write_text(content)
                self.assertEqual(load_questions(path)[0].id, "Q1")

    def test_schema_requires_all_questions_and_constrains_choices(self):
        qs = [Question(id="Q1", text="Choice", type="single", options=["예", "아니오"]),
              Question(id="Q2", text="Scale", type="scale", minimum=0, maximum=10, allow_na=True)]
        schema = response_schema(qs)
        answers = schema["properties"]["answers"]
        self.assertFalse(answers["additionalProperties"])
        self.assertEqual(answers["required"], ["Q1", "Q2"])
        self.assertEqual(answers["properties"]["Q1"]["enum"], ["예", "아니오"])
        self.assertEqual(answers["properties"]["Q2"]["anyOf"][0]["maximum"], 10)

    def test_duplicate_yaml_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "q.yaml"
            path.write_text('questions:\n  - id: Q1\n    text: First\n    text: Second\n')
            with self.assertRaisesRegex(ValueError, "unique"):
                load_questions(path)
