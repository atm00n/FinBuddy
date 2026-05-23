"""
assistant.py - AI Finance Assistant powered by Gemini with Gemma fallback.

Passes filtered transaction context to the model so answers are grounded in the
user's actual spending data for the selected period.
"""

import json
import os
from urllib import request
from typing import Optional, TypedDict

try:
    from google import genai
    from google.genai import types

    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    genai = None
    types = None

from tracker import budget_status, compute_summary, recent_transactions_text
from utils import get_budgets, get_transactions

_SYSTEM_PROMPT = """You are a careful personal finance assistant for FinBuddy.
Use only the financial numbers and transactions provided in the prompt.
If the data is sparse or insufficient, say that clearly instead of guessing.
Respond with short, practical action steps grounded in the provided budget, spending, and income data.
Never invent transactions, categories, trends, or percentages.
Use Rs for currency and keep the response under 220 words."""

_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEFAULT_GEMMA_MODEL = "gemma4"
_DEFAULT_OLLAMA_HOST = "http://localhost:11434"
_GEMINI_SOURCE = "Gemini"
_GEMMA_SOURCE = "Local Gemma"
_RULES_SOURCE = "Local FinBuddy rules"


class AssistantResponse(TypedDict):
    content: str
    uses_ai: bool
    source: str


def get_gemini_client(api_key: Optional[str] = None):
    """Return a Gemini client or None if unavailable."""
    if not _GEMINI_AVAILABLE:
        return None
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return None
    return genai.Client(api_key=key)


def ask_assistant(
    question: str,
    api_key: Optional[str] = None,
    transactions: Optional[list] = None,
    budgets: Optional[dict[str, float]] = None,
    period_label: str = "selected period",
) -> str:
    """
    Send a finance question to Gemini with filtered transaction context injected.
    """
    return ask_assistant_with_metadata(
        question=question,
        api_key=api_key,
        transactions=transactions,
        budgets=budgets,
        period_label=period_label,
    )["content"]


def ask_assistant_with_metadata(
    question: str,
    api_key: Optional[str] = None,
    transactions: Optional[list] = None,
    budgets: Optional[dict[str, float]] = None,
    period_label: str = "selected period",
) -> AssistantResponse:
    """
    Send a finance question and report whether the final answer used AI.
    """
    transactions = transactions if transactions is not None else get_transactions()
    budgets = budgets if budgets is not None else get_budgets()
    summary = compute_summary(transactions)
    budget_map = budget_status(transactions, budgets)
    recent = recent_transactions_text(transactions=transactions, n=12)

    if not transactions:
        return _response(
            "I do not have any transactions for this period yet. Add income and expense "
            "entries first, then I can help with budgeting, overspending, or savings.",
            uses_ai=False,
            source=_RULES_SOURCE,
        )

    context = _build_context(summary, budget_map, recent, period_label)
    user_message = f"{context}\n\n--- User Question ---\n{question.strip()}"
    provider_errors = []

    client = get_gemini_client(api_key)
    if client is not None:
        try:
            return _response(
                _call_gemini(client, user_message),
                uses_ai=True,
                source=_GEMINI_SOURCE,
            )
        except Exception as exc:
            provider_errors.append(("Gemini", exc))
    else:
        provider_errors.append(("Gemini", RuntimeError("Gemini API key or google-genai package is missing")))

    try:
        return _response(_call_gemma(user_message), uses_ai=True, source=_GEMMA_SOURCE)
    except Exception as exc:
        provider_errors.append(("Gemma", exc))

    fallback = _fallback_response(question, summary, budget_map, period_label)
    return _response(
        f"{_provider_error_message(provider_errors)}\n\n{fallback}",
        uses_ai=False,
        source=_RULES_SOURCE,
    )


def _response(content: str, uses_ai: bool, source: str) -> AssistantResponse:
    return {
        "content": content,
        "uses_ai": uses_ai,
        "source": source,
    }


def _call_gemini(client, user_message: str) -> str:
    model = os.environ.get("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)
    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        temperature=0.3,
        max_output_tokens=320,
    )
    response = client.models.generate_content(
        model=model,
        contents=user_message,
        config=config,
    )
    text = getattr(response, "text", "")
    if not text:
        raise RuntimeError("Gemini returned an empty response")
    return text.strip()


def _call_gemma(user_message: str) -> str:
    host = os.environ.get("OLLAMA_HOST", _DEFAULT_OLLAMA_HOST).rstrip("/")
    model = os.environ.get("GEMMA_MODEL", _DEFAULT_GEMMA_MODEL)
    payload = json.dumps(
        {
            "model": model,
            "prompt": f"{_SYSTEM_PROMPT}\n\n{user_message}",
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 320},
        }
    ).encode("utf-8")
    http_request = request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=20) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = str(body.get("response", "")).strip()
    if not text:
        raise RuntimeError("Gemma returned an empty response")
    return text


def _provider_error_message(provider_errors: list[tuple[str, Exception]]) -> str:
    details = []
    for provider, exc in provider_errors:
        details.append(f"{provider}: {_short_error(exc)}")
    return f"AI providers failed ({'; '.join(details)}). Showing local FinBuddy advice instead."


def _short_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    if len(message) > 180:
        message = f"{message[:177]}..."
    return message


def _build_context(summary: dict, budget_map: dict, recent: str, period_label: str) -> str:
    budget_lines = []
    for category, details in budget_map.items():
        if details["has_budget"]:
            budget_lines.append(
                f"- {category}: spent Rs {details['spent']:,.2f}, budget Rs {details['budget']:,.2f}, "
                f"remaining Rs {details['remaining']:,.2f}"
            )
        else:
            budget_lines.append(
                f"- {category}: spent Rs {details['spent']:,.2f}, no budget set"
            )

    budget_text = "\n".join(budget_lines) if budget_lines else "- No budget data available"

    return f"""
=== FinBuddy Context ({period_label}) ===
Income: Rs {summary['total_income']:,.2f}
Expenses: Rs {summary['total_expenses']:,.2f}
Balance: Rs {summary['balance']:,.2f}
Category totals: {summary['category_totals']}

=== Budget Status ===
{budget_text}

=== Recent Transactions (newest first) ===
{recent}
""".strip()


def _fallback_response(question: str, summary: dict, budget_map: dict, period_label: str) -> str:
    """Return grounded, rule-based advice using actual FinBuddy data."""
    income = summary["total_income"]
    expenses = summary["total_expenses"]
    balance = summary["balance"]
    category_totals = summary["category_totals"]
    lower_question = question.lower()

    lines = [f"For {period_label}, here is the clearest read from your FinBuddy data:"]

    over_budget = [
        (category, details)
        for category, details in budget_map.items()
        if details["over_budget"]
    ]
    if over_budget:
        category, details = over_budget[0]
        lines.append(
            f"- {category} is over budget by Rs {abs(details['remaining']):,.2f}; that is the first place to act."
        )
    elif expenses == 0:
        lines.append("- I do not see any expenses in this period, so overspending analysis is limited.")

    if income <= 0:
        lines.append("- No income is recorded in this period, so I cannot judge savings quality yet.")
    else:
        savings_rate = balance / income * 100
        lines.append(f"- Savings rate: {savings_rate:.1f}% (balance Rs {balance:,.2f}).")

    if category_totals:
        top_category = max(category_totals, key=category_totals.get)
        top_amount = category_totals[top_category]
        lines.append(f"- Highest spending category: {top_category} at Rs {top_amount:,.2f}.")
    else:
        lines.append("- I need more spending history to identify a clear pattern by category.")

    if "budget" in lower_question:
        lines.append("- Set or tighten category budgets where spending is concentrated, then review weekly.")
    elif "save" in lower_question or "saving" in lower_question:
        lines.append("- Reduce the top discretionary category first; it gives the fastest savings impact.")
    elif "overspend" in lower_question or "spend" in lower_question:
        lines.append("- Compare your highest-spend category against its budget before cutting smaller categories.")
    else:
        lines.append("- Next step: review the Dashboard trend and budget cards together before making changes.")

    return "\n".join(lines)
