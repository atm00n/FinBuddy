import unittest
from unittest.mock import patch

import model


class DynamicCategorizationPromptTests(unittest.TestCase):
    def test_parse_category_response_accepts_custom_categories(self):
        self.assertEqual(
            model._parse_category_response(
                '["anna", "Unknown"]',
                expected_count=2,
                valid_categories=["Food", "Misc", "anna"],
            ),
            ["anna", "Misc"],
        )

    def test_system_prompt_includes_custom_categories_and_learning_examples(self):
        prompt = model._build_system_prompt(
            categories=["Food", "Misc", "anna"],
            examples=[
                {
                    "original_description": "nayanbhattad007@okaxis",
                    "description": "Anna transfer",
                    "category": "anna",
                }
            ],
        )

        self.assertIn("Food, Misc, anna", prompt)
        self.assertIn("nayanbhattad007@okaxis", prompt)
        self.assertIn('category "anna"', prompt)

    def test_local_fallback_stays_unchanged_when_ai_is_unavailable(self):
        with patch.object(model, "_get_api_key", return_value=None):
            with patch.object(model, "_call_gemma", side_effect=RuntimeError("offline")):
                self.assertEqual(
                    model._categorize_batch_internal(["uber ride"]),
                    ["Transport"],
                )


if __name__ == "__main__":
    unittest.main()
