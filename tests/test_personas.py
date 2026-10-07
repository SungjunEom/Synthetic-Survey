import unittest
from unittest.mock import patch

from datasets import Dataset

from survey.personas import (DATASET, age_bounds, custom_personas, from_nemotron,
                             load_source, normalize_filters, sample_dataset)
from survey.prompts import build_messages
from survey.questions import Question
from tests.helpers import row


class PersonaTests(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset.from_list([row(1), row(2, age=99, sex="남자", education_level="무학"),
                                         row(3, age=42, province="경기", occupation="교사")])

    def test_age99_and_all_facets_reach_both_prompts(self):
        source = row(age=99)
        persona = from_nemotron(source)
        self.assertEqual(persona.attributes, source)
        for lang in ("ko", "en"):
            messages = build_messages(persona, [Question(id="Q1", text="Test")], lang, "2026-10-07")
            for field in ("travel_persona", "family_persona", "sports_persona", "arts_persona", "culinary_persona"):
                self.assertIn(source[field], messages[1]["content"])
        for age in (18, "99", True, 121):
            with self.subTest(age=age), self.assertRaises(ValueError):
                from_nemotron(row(age=age))

    def test_korean_english_filters_and_literal_substrings(self):
        for filters in ({"sex": "female", "age": "25-40"}, {"sex": "여자", "occupation": "[웹]"},
                        {"education_level": "High school", "province": "서울", "marital_status": "single"}):
            selected, stats = sample_dataset(self.dataset, 1, 1, normalize_filters(filters))
            self.assertEqual(selected[0]["uuid"], "fixture-1")
            self.assertEqual(stats["eligible_rows"], 1)
        selected, _ = sample_dataset(self.dataset, 1, 1, normalize_filters({"education_level": "무학", "age": "99"}))
        self.assertEqual(selected[0]["age"], 99)

    def test_bad_filters_fail_and_no_silent_replacement(self):
        for sex in ("non-binary", "unknown", " "):
            with self.assertRaises(ValueError):
                normalize_filters({"sex": sex})
        for age in ("18", "40-25", "25-130", "abc"):
            with self.assertRaises(ValueError):
                age_bounds(age)
        with self.assertRaisesRegex(ValueError, "only 3"):
            sample_dataset(self.dataset, 4, 1, {})
        with self.assertRaisesRegex(ValueError, "No personas"):
            sample_dataset(self.dataset, 1, 1, {"province": "없는곳"})

    def test_seeded_sampling_and_explicit_replacement(self):
        first, _ = sample_dataset(self.dataset, 3, 42, {})
        second, _ = sample_dataset(self.dataset, 3, 42, {})
        self.assertEqual(first, second)
        self.assertEqual(len({r["uuid"] for r in first}), 3)
        repeated, stats = sample_dataset(self.dataset, 3, 42, {"age": (99, 99)}, True)
        self.assertEqual(len(repeated), 3)
        self.assertEqual(stats["unique_personas"], 1)
        self.assertTrue(stats["replacement"])

    def test_source_duplicates_are_not_distinct_people(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            sample_dataset(Dataset.from_list([row(), row()]), 2, 1, {})

    def test_resolve_remote_revision_before_load(self):
        with patch("huggingface_hub.HfApi") as api, patch("datasets.load_dataset", return_value=self.dataset) as load:
            api.return_value.dataset_info.return_value.sha = "abc123"
            _, source = load_source(None, "main")
        load.assert_called_once_with(DATASET, revision="abc123", split="train")
        self.assertEqual(source["revision"], "abc123")

    def test_custom_profiles_dont_randomize_politics(self):
        personas = custom_personas(5, 42, {}, True)
        self.assertTrue(all(p.politics is None for p in personas))
        fixed = custom_personas(3, 42, {"age": "25-40", "politics": "진보"}, False)
        self.assertEqual(len({p.age for p in fixed}), 1)
        self.assertTrue(all(p.politics == "진보" for p in fixed))
