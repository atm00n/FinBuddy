import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

import assistant


class FakeGeminiModels:
    def __init__(self, text="Use the Food budget first."):
        self.text = text
        self.request = None

    def generate_content(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(text=self.text)


class FakeGeminiClient:
    def __init__(self, text="Use the Food budget first."):
        self.models = FakeGeminiModels(text)


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeUrlResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class AssistantProviderTests(unittest.TestCase):
    def test_uses_gemini_first(self):
        client = FakeGeminiClient()
        fake_types = SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig)

        with patch.object(assistant, "types", fake_types):
            result = assistant._call_gemini(client, "context")

        self.assertEqual(result, "Use the Food budget first.")
        self.assertEqual(client.models.request["model"], "gemini-2.5-flash")
        self.assertEqual(client.models.request["contents"], "context")
        self.assertIn(
            "careful personal finance assistant",
            client.models.request["config"].kwargs["system_instruction"],
        )

    def test_calls_gemma_through_ollama_generate_endpoint(self):
        requests = []

        def fake_urlopen(http_request, timeout):
            requests.append((http_request, timeout))
            return FakeUrlResponse({"response": "Cut the top category first."})

        with patch("assistant.request.urlopen", side_effect=fake_urlopen):
            result = assistant._call_gemma("context")

        self.assertEqual(result, "Cut the top category first.")
        self.assertEqual(requests[0][1], 20)
        self.assertEqual(requests[0][0].full_url, "http://localhost:11434/api/generate")
        payload = json.loads(requests[0][0].data.decode("utf-8"))
        self.assertEqual(payload["model"], "gemma4")
        self.assertFalse(payload["stream"])

    def test_gemini_failure_falls_back_to_gemma(self):
        transactions = [
            {
                "date": "2026-05-01",
                "description": "Swiggy dinner",
                "amount": -500.0,
                "category": "Food",
            }
        ]

        with patch("assistant.get_gemini_client", return_value=FakeGeminiClient("")):
            with patch("assistant._call_gemma", return_value="Gemma advice"):
                result = assistant.ask_assistant(
                    "Where am I overspending?",
                    api_key="gemini-test",
                    transactions=transactions,
                    budgets={"Food": 300.0},
                    period_label="May 2026",
                )

        self.assertEqual(result, "Gemma advice")

    def test_gemini_metadata_reports_ai_source(self):
        transactions = [
            {
                "date": "2026-05-01",
                "description": "Salary",
                "amount": 50000.0,
                "category": "Income",
            }
        ]

        with patch("assistant.get_gemini_client", return_value=object()):
            with patch("assistant._call_gemini", return_value="Gemini advice"):
                result = assistant.ask_assistant_with_metadata(
                    "How am I doing?",
                    api_key="gemini-test",
                    transactions=transactions,
                    budgets={},
                    period_label="May 2026",
                )

        self.assertEqual(result["content"], "Gemini advice")
        self.assertTrue(result["uses_ai"])
        self.assertEqual(result["source"], "Gemini")

    def test_gemma_metadata_reports_local_ai_source(self):
        transactions = [
            {
                "date": "2026-05-01",
                "description": "Swiggy dinner",
                "amount": -500.0,
                "category": "Food",
            }
        ]

        with patch("assistant.get_gemini_client", return_value=object()):
            with patch("assistant._call_gemini", side_effect=RuntimeError("gemini down")):
                with patch("assistant._call_gemma", return_value="Gemma advice"):
                    result = assistant.ask_assistant_with_metadata(
                        "Where am I overspending?",
                        api_key="gemini-test",
                        transactions=transactions,
                        budgets={"Food": 300.0},
                        period_label="May 2026",
                    )

        self.assertEqual(result["content"], "Gemma advice")
        self.assertTrue(result["uses_ai"])
        self.assertEqual(result["source"], "Local Gemma")

    def test_all_ai_failures_are_visible_before_rule_based_fallback(self):
        transactions = [
            {
                "date": "2026-05-01",
                "description": "Swiggy dinner",
                "amount": -500.0,
                "category": "Food",
            }
        ]

        with patch("assistant.get_gemini_client", return_value=FakeGeminiClient("")):
            with patch("assistant._call_gemma", side_effect=RuntimeError("ollama offline")):
                result = assistant.ask_assistant(
                    "Where am I overspending?",
                    api_key="gemini-test",
                    transactions=transactions,
                    budgets={"Food": 300.0},
                    period_label="May 2026",
                )

        self.assertIn("AI providers failed", result)
        self.assertIn("Gemini: Gemini returned an empty response", result)
        self.assertIn("Gemma: ollama offline", result)
        self.assertIn("Food is over budget", result)

    def test_rule_fallback_metadata_reports_not_using_ai(self):
        transactions = [
            {
                "date": "2026-05-01",
                "description": "Swiggy dinner",
                "amount": -500.0,
                "category": "Food",
            }
        ]

        with patch("assistant.get_gemini_client", return_value=FakeGeminiClient("")):
            with patch("assistant._call_gemma", side_effect=RuntimeError("ollama offline")):
                result = assistant.ask_assistant_with_metadata(
                    "Where am I overspending?",
                    api_key="gemini-test",
                    transactions=transactions,
                    budgets={"Food": 300.0},
                    period_label="May 2026",
                )

        self.assertFalse(result["uses_ai"])
        self.assertEqual(result["source"], "Local FinBuddy rules")
        self.assertIn("AI providers failed", result["content"])
        self.assertIn("Food is over budget", result["content"])

    def test_empty_transactions_metadata_reports_not_using_ai(self):
        result = assistant.ask_assistant_with_metadata(
            "Where am I overspending?",
            api_key="gemini-test",
            transactions=[],
            budgets={"Food": 300.0},
            period_label="May 2026",
        )

        self.assertFalse(result["uses_ai"])
        self.assertEqual(result["source"], "Local FinBuddy rules")
        self.assertIn("I do not have any transactions", result["content"])


if __name__ == "__main__":
    unittest.main()
