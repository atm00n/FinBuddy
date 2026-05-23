"""
model.py - AI-powered transaction categorizer with local fallback.

Primary:  Gemini (gemini-2.5-flash by default) via google-genai SDK.
Fallback: Gemma4 via local Ollama, then Naive Bayes trained on a built-in corpus.

Public interface is identical to the original model.py — all call sites in
app.py, utils.py, and tracker.py continue to work unchanged.

    predict(description)        -> str
    predict_proba(description)  -> dict[str, float]

New helpers:
    set_api_key(key)            -> None   call once from the sidebar
    categorize_batch(descs)     -> list[str]   efficient bulk import path
"""

import json
import os
import threading
from functools import lru_cache
from typing import Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    _SKLEARN_AVAILABLE = True
except ImportError:
    TfidfVectorizer = None
    MultinomialNB = None
    Pipeline = object
    _SKLEARN_AVAILABLE = False

# ── Try to import the Gemini SDK (optional dependency) ────────────────────────

try:
    from google import genai
    from google.genai import types as genai_types
    _GEMINI_SDK_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    _GEMINI_SDK_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

CATEGORIES = ["Food", "Transport", "Income", "Medicine", "Entertainment", "Misc"]

_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEFAULT_GEMMA_MODEL  = "gemma4"
_DEFAULT_OLLAMA_HOST  = "http://localhost:11434"
_TIMEOUT_GEMMA        = 20   # seconds (local model can be slower)
_BATCH_SIZE           = 25   # descriptions per Gemini call
_CACHE_SIZE           = 512  # LRU cache entries for predict()

# ── System prompt ─────────────────────────────────────────────────────────────
# Key rule: reason about the ITEM being purchased, not the delivery platform.
# "Dunzo run for cough syrup" → item = medicine → Misc
# "Swiggy dinner"             → item = food     → Food
# "Zepto vegetables"          → item = groceries → Food

_SYSTEM_PROMPT = """\
You are a transaction categorizer for an Indian personal finance app.

Categorize each transaction into exactly one of:
  Food          – food, groceries, restaurants, cafes, beverages, snacks
  Transport     – cabs, metro, bus, petrol, flights, train, parking, toll
  Income        – salary, freelance, bonus, dividends, interest, cashback, refunds
  Medicine      – pharmacy, medicines, doctor, hospital, clinic, dental, health tests,
                   medical bills, health insurance, eyecare, ayurveda
  Entertainment – movies, OTT (Netflix, Hotstar, Prime), concerts, games, sports events,
                   amusement parks, streaming, music subscriptions
  Misc          – everything else: utilities, rent, subscriptions (non-entertainment),
                   shopping, EMI, education, salon, gym, clothing, electronics, insurance

Critical rule: for delivery-platform transactions (Dunzo, Swiggy, Blinkit,
Zepto, Instamart, etc.) categorize by the ITEM being delivered, not the
platform name.
  "Dunzo run for cough syrup"  → Medicine   (not Food)
  "Blinkit order – vegetables" → Food       (groceries)
  "Swiggy Instamart – milk"    → Food       (groceries)
  "Swiggy dinner"              → Food       (restaurant food)
  "Dunzo grocery run"          → Food       (groceries)
  "Netmeds order"              → Medicine
  "BookMyShow movie"           → Entertainment
  "Netflix subscription"       → Entertainment

Respond ONLY with a JSON array of category strings, one per input, same order.
Use only the exact names: Food, Transport, Income, Medicine, Entertainment, Misc.
No explanation. No markdown. No extra text.\
"""

# ── Thread-safe API key state ─────────────────────────────────────────────────

_api_key: Optional[str] = None
_api_key_lock = threading.Lock()


def set_api_key(key: str) -> None:
    """Store the Gemini API key and clear the prediction cache."""
    global _api_key
    with _api_key_lock:
        _api_key = key.strip() if key else None
        predict.cache_clear()  # inside the lock to avoid race with concurrent predict()


def _get_api_key() -> Optional[str]:
    key = _api_key or os.environ.get("GEMINI_API_KEY", "")
    return key.strip() if key else None


def clear_cache() -> None:
    """Clear cached predictions after categories or learning examples change."""
    predict.cache_clear()


def _category_options() -> list[str]:
    try:
        import utils

        categories = utils.get_all_categories()
    except Exception:
        categories = CATEGORIES

    ordered = [category for category in CATEGORIES if category in categories]
    ordered.extend(category for category in categories if category not in ordered)
    return ordered


def _learning_examples(limit: int = 20) -> list[dict]:
    try:
        import utils

        return utils.get_learning_examples(limit=limit)
    except Exception:
        return []


def _build_system_prompt(
    categories: list[str] | None = None,
    examples: list[dict] | None = None,
) -> str:
    categories = categories or _category_options()
    examples = examples if examples is not None else _learning_examples()
    category_text = ", ".join(categories)
    example_lines = []
    for example in examples:
        original = str(example.get("original_description", "")).strip()
        description = str(example.get("description", "")).strip()
        category = str(example.get("category", "")).strip()
        if original and description and category in categories:
            example_lines.append(
                f'- "{original}" -> description "{description}", category "{category}"'
            )

    if not example_lines:
        example_lines.append("- No user examples yet.")

    return (
        f"{_SYSTEM_PROMPT}\n\n"
        "Authoritative category list for this user:\n"
        f"{category_text}\n\n"
        "Use this authoritative list even if earlier text lists fewer categories. "
        "Custom categories are valid when they clearly match the user's examples.\n\n"
        "User correction examples:\n"
        f"{chr(10).join(example_lines)}\n\n"
        "Respond ONLY with a JSON array of category strings, one per input, same order. "
        "Use only exact names from the authoritative category list. No explanation."
    )


# ── Gemini caller ─────────────────────────────────────────────────────────────

def _gemini_client():
    if not _GEMINI_SDK_AVAILABLE:
        raise RuntimeError("google-genai package is not installed.")
    key = _get_api_key()
    if not key:
        raise RuntimeError("No Gemini API key set.")
    return genai.Client(api_key=key)


def _parse_category_response(
    raw: str,
    expected_count: int,
    valid_categories: list[str] | None = None,
) -> list[str]:
    """Parse and validate the JSON array returned by the model."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    categories = json.loads(text)

    if not isinstance(categories, list) or len(categories) != expected_count:
        raise ValueError(
            f"Expected {expected_count} categories, got: {raw!r}"
        )

    valid = set(valid_categories or CATEGORIES)
    return [c if c in valid else "Misc" for c in (str(x).strip() for x in categories)]


def _call_gemini(descriptions: list[str]) -> list[str]:
    """Call Gemini with a batch of descriptions. Raises on failure."""
    client = _gemini_client()
    model_name = os.environ.get("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL)
    categories = _category_options()

    config = genai_types.GenerateContentConfig(
        system_instruction=_build_system_prompt(categories=categories),
        temperature=0.0,
        max_output_tokens=128,
    )

    user_message = json.dumps(descriptions, ensure_ascii=False)
    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
        config=config,
    )

    raw = getattr(response, "text", "") or ""
    if not raw.strip():
        raise RuntimeError("Gemini returned an empty response.")

    return _parse_category_response(raw, len(descriptions), valid_categories=categories)


# ── Gemma4 / Ollama caller ────────────────────────────────────────────────────

def _call_gemma(descriptions: list[str]) -> list[str]:
    """Call local Gemma4 via Ollama. Raises on failure."""
    import urllib.request

    host  = os.environ.get("OLLAMA_HOST", _DEFAULT_OLLAMA_HOST).rstrip("/")
    model = os.environ.get("GEMMA_MODEL", _DEFAULT_GEMMA_MODEL)
    categories = _category_options()

    user_message = json.dumps(descriptions, ensure_ascii=False)
    full_prompt  = f"{_build_system_prompt(categories=categories)}\n\n{user_message}"

    payload = json.dumps({
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 128},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=_TIMEOUT_GEMMA) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    raw = str(body.get("response", "")).strip()
    if not raw:
        raise RuntimeError("Gemma returned an empty response.")

    return _parse_category_response(raw, len(descriptions), valid_categories=categories)


# ── Local Naive Bayes fallback ────────────────────────────────────────────────

_TRAINING_DATA = [
    # Food
    ("grocery store", "Food"), ("supermarket", "Food"), ("restaurant", "Food"),
    ("pizza", "Food"), ("burger", "Food"), ("cafe", "Food"),
    ("coffee shop", "Food"), ("lunch", "Food"), ("dinner", "Food"),
    ("breakfast", "Food"), ("swiggy food", "Food"), ("zomato", "Food"),
    ("milk", "Food"), ("vegetables", "Food"), ("fruits", "Food"),
    ("bakery", "Food"), ("canteen", "Food"), ("snacks", "Food"),
    ("food delivery", "Food"), ("dominos", "Food"), ("kfc", "Food"),
    ("mcdonalds", "Food"), ("bigbasket groceries", "Food"),
    ("zepto vegetables", "Food"), ("blinkit milk", "Food"),
    ("dunzo grocery", "Food"), ("instamart", "Food"),
    # Transport
    ("uber", "Transport"), ("ola", "Transport"), ("rapido", "Transport"),
    ("metro", "Transport"), ("metro card recharge", "Transport"),
    ("bus ticket", "Transport"), ("train ticket", "Transport"),
    ("flight", "Transport"), ("petrol", "Transport"), ("diesel", "Transport"),
    ("fuel", "Transport"), ("cab", "Transport"), ("auto", "Transport"),
    ("toll", "Transport"), ("parking", "Transport"),
    ("bike rental", "Transport"), ("taxi", "Transport"),
    ("irctc", "Transport"), ("makemytrip flight", "Transport"),
    ("redbus", "Transport"), ("commute", "Transport"),
    ("indigo", "Transport"), ("air india", "Transport"),
    # Income
    ("salary", "Income"), ("salary credit", "Income"),
    ("paycheck", "Income"), ("freelance", "Income"),
    ("client payment", "Income"), ("bonus", "Income"),
    ("refund", "Income"), ("dividend", "Income"),
    ("interest earned", "Income"), ("interest credit", "Income"),
    ("rental income", "Income"), ("consulting fee", "Income"),
    ("side income", "Income"), ("cashback", "Income"),
    ("reward", "Income"), ("stipend", "Income"),
    ("reimbursement", "Income"), ("neft credit", "Income"),
    # Misc — general
    ("electricity bill", "Misc"),
    ("water bill", "Misc"), ("internet bill", "Misc"), ("rent", "Misc"),
    ("clothing", "Misc"), ("shopping", "Misc"), ("amazon purchase", "Misc"),
    ("flipkart", "Misc"), ("education", "Misc"), ("book purchase", "Misc"),
    ("mobile recharge", "Misc"), ("insurance", "Misc"), ("emi", "Misc"),
    ("loan", "Misc"), ("maintenance", "Misc"), ("haircut", "Misc"),
    ("salon", "Misc"), ("gym", "Misc"), ("subscription", "Misc"),
    # Medicine / health
    ("medicine", "Medicine"), ("pharmacy", "Medicine"), ("hospital", "Medicine"),
    ("doctor", "Medicine"), ("clinic", "Medicine"), ("dentist", "Medicine"),
    ("cough syrup", "Medicine"), ("tablets", "Medicine"), ("medical", "Medicine"),
    ("health", "Medicine"), ("apollo pharmacy", "Medicine"), ("netmeds", "Medicine"),
    ("1mg", "Medicine"), ("dunzo medicine", "Medicine"), ("dunzo pharmacy", "Medicine"),
    ("dunzo cough syrup", "Medicine"), ("chemist", "Medicine"), ("prescription", "Medicine"),
    ("eye care", "Medicine"), ("optician", "Medicine"), ("pathology", "Medicine"),
    ("diagnostic", "Medicine"), ("health insurance", "Medicine"),
    # Entertainment
    ("netflix", "Entertainment"), ("amazon prime", "Entertainment"),
    ("hotstar", "Entertainment"), ("spotify", "Entertainment"),
    ("prime video", "Entertainment"), ("zee5", "Entertainment"),
    ("jiocinema", "Entertainment"), ("bookmyshow", "Entertainment"),
    ("movie tickets", "Entertainment"), ("movie", "Entertainment"),
    ("concert", "Entertainment"), ("gaming", "Entertainment"),
    ("steam", "Entertainment"), ("playstation", "Entertainment"),
    ("xbox game pass", "Entertainment"), ("youtube premium", "Entertainment"),
    ("music subscription", "Entertainment"), ("amusement park", "Entertainment"),
    ("pvr", "Entertainment"), ("inox", "Entertainment"),
]


def _build_local_pipeline() -> Pipeline:
    if not _SKLEARN_AVAILABLE:
        return None
    texts, labels = zip(*_TRAINING_DATA)
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            lowercase=True, stop_words="english",
        )),
        ("clf", MultinomialNB(alpha=1.0)),
    ]).fit(list(texts), list(labels))


_local_model: Pipeline | None = _build_local_pipeline()


def _local_predict(description: str) -> str:
    if not description or not description.strip():
        return "Misc"
    if _local_model is None:
        return _keyword_predict(description)
    return _local_model.predict([description.lower().strip()])[0]


def _local_predict_proba(description: str) -> dict[str, float]:
    if not description or not description.strip():
        return {c: 0.0 for c in CATEGORIES}
    if _local_model is None:
        chosen = _keyword_predict(description)
        return {c: (1.0 if c == chosen else 0.0) for c in CATEGORIES}
    probs = _local_model.predict_proba([description.lower().strip()])[0]
    return {
        cls: round(float(p), 3)
        for cls, p in zip(_local_model.classes_, probs)
    }


def _keyword_predict(description: str) -> str:
    text = description.lower()
    keyword_groups = {
        "Food": ["swiggy", "zomato", "food", "dinner", "lunch", "grocery", "milk"],
        "Transport": ["uber", "ola", "metro", "cab", "bus", "train", "petrol"],
        "Income": ["salary", "credit", "interest", "bonus", "refund"],
        "Medicine": ["pharmacy", "medicine", "hospital", "doctor", "clinic"],
        "Entertainment": ["movie", "netflix", "spotify", "bookmyshow", "concert"],
    }
    for category, keywords in keyword_groups.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "Misc"


# ── Internal batch runner (Gemini → Gemma → local NB) ────────────────────────

def _categorize_batch_internal(descriptions: list[str]) -> list[str]:
    if not descriptions:
        return []

    if _get_api_key():
        try:
            return _call_gemini(descriptions)
        except Exception:
            pass

    try:
        return _call_gemma(descriptions)
    except Exception:
        pass

    return [_local_predict(d) for d in descriptions]


# ── Public API ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=_CACHE_SIZE)
def predict(description: str) -> str:
    """
    Predict the category for a single transaction description.

    Cascade: Gemini → Gemma4 (Ollama) → local Naive Bayes.

    Returns one of: "Food", "Transport", "Income", "Misc"
    """
    if not description or not description.strip():
        return "Misc"
    return _categorize_batch_internal([description])[0]


def predict_proba(description: str) -> dict[str, float]:
    """
    Return per-category confidence scores.

    When Gemini or Gemma is active: one-hot dict (1.0 for predicted, 0.0 for others).
    When local NB is active: raw Naive Bayes probabilities (approximate).
    """
    if not description or not description.strip():
        return {c: 0.0 for c in _category_options()}

    if _get_api_key():
        try:
            chosen = _call_gemini([description])[0]
            return {c: (1.0 if c == chosen else 0.0) for c in _category_options()}
        except Exception:
            pass

    try:
        chosen = _call_gemma([description])[0]
        return {c: (1.0 if c == chosen else 0.0) for c in _category_options()}
    except Exception:
        pass

    return _local_predict_proba(description)


def categorize_batch(descriptions: list[str]) -> list[str]:
    """
    Categorize multiple descriptions in one or a few API calls.

    Use this instead of calling predict() in a loop when importing transactions.
    Gemini batches up to 25 items per call; Gemma handles the full list in one call.
    Falls back to per-item local NB if both AI providers are unavailable.
    """
    if not descriptions:
        return []

    results: list[str] = []
    for i in range(0, len(descriptions), _BATCH_SIZE):
        chunk = descriptions[i : i + _BATCH_SIZE]
        results.extend(_categorize_batch_internal(chunk))
    return results
