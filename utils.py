"""
utils.py - Data handling utilities for the Finance Tracker.
Manages JSON persistence, transaction CRUD, budgets, and CSV export.
"""

import csv
import json
import os
import re
import shutil
from difflib import SequenceMatcher
from datetime import date, datetime
from io import BytesIO, StringIO
from math import isfinite

DATA_FILE = os.path.join(os.path.dirname(__file__), "data.json")
# Base categories that are always present and cannot be deleted.
BASE_CATEGORIES = {"Food", "Transport", "Income", "Misc", "Medicine", "Entertainment"}
# Runtime-resolved set (base + user-defined). Use get_valid_categories() everywhere.
_VALID_CATEGORIES_CACHE: set[str] | None = None
BUDGET_CATEGORIES = ("Food", "Transport", "Misc", "Medicine", "Entertainment")
DEFAULT_BUDGETS: dict[str, float] = {cat: 0.0 for cat in BUDGET_CATEGORIES}
CSV_COLUMNS = ["Date", "Details", "Amount", "Currency", "Debit/Credit", "Status"]
LEARNING_EXAMPLE_LIMIT = 100
LEARNING_SIMILARITY_THRESHOLD = 0.88
_LOAD_WARNINGS: list[str] = []


def get_valid_categories() -> set[str]:
    """Return all valid category names (base + user-defined)."""
    data = load_data()
    custom = set(data.get("custom_categories", []))
    return BASE_CATEGORIES | custom


def add_custom_category(name: str) -> None:
    """Persist a new user-defined category."""
    name = name.strip()
    if not name:
        raise ValueError("Category name cannot be empty.")
    if name in BASE_CATEGORIES:
        return  # already exists
    data = load_data()
    cats = data.get("custom_categories", [])
    if name not in cats:
        cats.append(name)
    data["custom_categories"] = cats
    save_data(data)


def get_all_categories() -> list[str]:
    """Return sorted list of all categories (base + user-defined)."""
    return sorted(get_valid_categories())


def get_learning_examples(limit: int | None = None) -> list[dict]:
    """Return recent user correction examples for AI prompting."""
    examples = load_data().get("learning_examples", [])
    return examples[:limit] if limit is not None else examples.copy()


def record_learning_example(
    original_description: str,
    description: str,
    category: str,
) -> dict | None:
    """Persist one user correction example for future AI prompts."""
    data = load_data()
    valid = BASE_CATEGORIES | set(data.get("custom_categories", []))
    example = _upsert_learning_example(
        data,
        original_description=original_description,
        description=description,
        category=category,
        valid_categories=valid,
    )
    if example is not None:
        save_data(data)
    return example


def apply_learned_description(description: str) -> str:
    """Return the user's preferred description when a prior correction matches."""
    examples = load_data().get("learning_examples", [])
    return _apply_learned_description(description, examples)


# Legacy alias kept for compatibility
VALID_CATEGORIES = BASE_CATEGORIES


def load_data() -> dict:
    """Load transaction data from JSON file. Auto-creates file if missing."""
    global _LOAD_WARNINGS
    _LOAD_WARNINGS = []

    if not os.path.exists(DATA_FILE):
        default = _default_data()
        save_data(default)
        return default

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        backup_path = _backup_corrupt_data_file()
        _LOAD_WARNINGS.append(
            "FinBuddy found a corrupted data file and reset it safely."
            f" Backup saved as `{os.path.basename(backup_path)}`."
        )
        default = _default_data()
        save_data(default)
        return default
    except OSError:
        _LOAD_WARNINGS.append(
            "FinBuddy could not read the data file and reset to a clean local store."
        )
        default = _default_data()
        save_data(default)
        return default

    return _normalize_data(data)


def save_data(data: dict) -> None:
    """Persist data dictionary to JSON file."""
    normalized = _normalize_data(data)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)


def consume_data_warnings() -> list[str]:
    """Return and clear load warnings so the UI can surface them once."""
    global _LOAD_WARNINGS
    warnings = _LOAD_WARNINGS[:]
    _LOAD_WARNINGS = []
    return warnings


def get_budgets() -> dict[str, float]:
    """Return persisted budgets for budgetable categories."""
    data = load_data()
    return data["budgets"].copy()


def save_budgets(budgets: dict[str, float]) -> dict[str, float]:
    """Persist validated budgets and return the normalized budget map."""
    normalized = _validate_budgets(budgets)
    data = load_data()
    data["budgets"] = normalized
    save_data(data)
    return normalized.copy()


def add_transaction(description: str, amount: float, category: str, date: str) -> dict:
    """Add a new transaction and persist it."""
    data = load_data()
    valid = BASE_CATEGORIES | set(data.get("custom_categories", []))
    transaction = _build_transaction(
        transaction_id=_next_id(data["transactions"]),
        description=description,
        amount=amount,
        category=category,
        date=date,
        created_at=datetime.now().isoformat(),
        valid_categories=valid,
    )
    data["transactions"].append(transaction)
    save_data(data)
    return transaction


def update_transaction(
    txn_id: int,
    description: str,
    amount: float,
    category: str,
    date: str,
) -> dict:
    """Update an existing transaction and persist the changes."""
    data = load_data()
    valid = BASE_CATEGORIES | set(data.get("custom_categories", []))
    for index, existing in enumerate(data["transactions"]):
        if existing["id"] == txn_id:
            updated = _build_transaction(
                transaction_id=txn_id,
                description=description,
                amount=amount,
                category=category,
                date=date,
                created_at=existing.get("created_at") or datetime.now().isoformat(),
                valid_categories=valid,
            )
            data["transactions"][index] = updated
            save_data(data)
            return updated
    raise ValueError(f"Transaction with id {txn_id} was not found.")


def get_transactions() -> list:
    """Return all transactions sorted newest-first."""
    data = load_data()
    txns = data.get("transactions", [])
    return sorted(
        txns,
        key=lambda txn: (txn.get("date", ""), txn.get("created_at", "")),
        reverse=True,
    )


def delete_transaction(txn_id: int) -> bool:
    """Remove a transaction by id. Returns True if found and deleted."""
    data = load_data()
    before = len(data["transactions"])
    data["transactions"] = [t for t in data["transactions"] if t["id"] != txn_id]
    if len(data["transactions"]) < before:
        save_data(data)
        return True
    return False


def export_csv(transactions: list[dict] | None = None) -> str:
    """Return transactions as a public CSV string for download."""
    txns = transactions if transactions is not None else get_transactions()
    if not txns:
        return ""
    return _transactions_to_csv(txns)


def _transactions_to_csv(transactions: list[dict]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for txn in transactions:
        amount = float(txn["amount"])
        writer.writerow(
            {
                "Date": txn["date"],
                "Details": txn["description"],
                "Amount": abs(amount),
                "Currency": "INR",
                "Debit/Credit": "Credit" if amount >= 0 else "Debit",
                "Status": "Completed",
            }
        )
    return output.getvalue()


def import_csv(csv_content: str | bytes, categorize=None) -> dict:
    """
    Import transactions from CSV content.

    Required columns: Date, Details, Amount, Currency, Debit/Credit, Status.
    Details is categorized by categorize(details) when provided, otherwise Misc.
    """
    return _persist_preview_result(preview_csv_import(csv_content, categorize=categorize))


def preview_csv_import(csv_content: str | bytes, categorize=None) -> dict:
    """
    Parse CSV content into editable import rows without persisting transactions.
    """
    text = _decode_csv_content(csv_content)
    reader = csv.DictReader(StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV file is empty or missing a header row.")

    normalized_headers = {
        header.strip().lower(): header
        for header in reader.fieldnames
        if header and header.strip()
    }
    required = {column.lower(): column for column in CSV_COLUMNS}
    missing = sorted(set(required) - set(normalized_headers))
    if missing:
        missing_labels = [required[column] for column in missing]
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(missing_labels)}."
        )

    rows = []
    errors = []
    data = load_data()
    valid = BASE_CATEGORIES | set(data.get("custom_categories", []))
    learning_examples = data.get("learning_examples", [])
    for row_number, row in enumerate(reader, start=2):
        try:
            rows.append(
                _transaction_from_csv_row(
                    row,
                    normalized_headers,
                    categorize=categorize,
                    learning_examples=learning_examples,
                    valid_categories=valid,
                )
            )
        except ValueError as exc:
            errors.append(f"Row {row_number}: {exc}")

    return {"rows": rows, "failed": len(errors), "errors": errors}


def build_import_preview_transaction(
    description: str,
    amount: float,
    date: str,
    categorize=None,
    learning_examples: list[dict] | None = None,
    valid_categories: set[str] | None = None,
) -> dict:
    """Build one editable import draft row without assigning a persisted id."""
    original_description = str(description).strip()
    learned_description = _apply_learned_description(
        original_description,
        learning_examples if learning_examples is not None else get_learning_examples(),
    )
    category = categorize(learned_description) if categorize else "Misc"
    transaction = _build_transaction(
        transaction_id=0,
        description=learned_description,
        amount=amount,
        category=category,
        date=date,
        created_at=datetime.now().isoformat(),
        valid_categories=valid_categories,
    )
    transaction["_import_original_description"] = original_description
    transaction["_import_suggested_description"] = transaction["description"]
    transaction["_import_suggested_category"] = transaction["category"]
    transaction["import"] = True
    transaction["custom_category"] = ""
    return transaction


def _transaction_from_csv_row(
    row: dict,
    normalized_headers: dict,
    categorize=None,
    learning_examples: list[dict] | None = None,
    valid_categories: set[str] | None = None,
) -> dict:
    description = row.get(normalized_headers["details"], "")
    amount = row.get(normalized_headers["amount"], "")
    txn_date = row.get(normalized_headers["date"], "")
    currency = str(row.get(normalized_headers["currency"], "")).strip()
    status = str(row.get(normalized_headers["status"], "")).strip()
    debit_credit = str(row.get(normalized_headers["debit/credit"], "")).strip()

    if not currency:
        raise ValueError("Currency cannot be empty.")
    if not status:
        raise ValueError("Status cannot be empty.")

    signed_amount = _apply_debit_credit(amount, debit_credit)
    return build_import_preview_transaction(
        description=description,
        amount=signed_amount,
        date=txn_date,
        categorize=categorize,
        learning_examples=learning_examples,
        valid_categories=valid_categories,
    )


def import_pdf(pdf_content: bytes, categorize=None) -> dict:
    """
    Import transactions from a text-based PDF statement.

    FinBuddy looks for one transaction per line with a date, description, and
    amount. Supported dates include YYYY-MM-DD, DD/MM/YYYY, and DD-MM-YYYY.
    Categories are optional and are filled by categorize(description) when
    provided, otherwise Misc.
    """
    return _persist_preview_result(preview_pdf_import(pdf_content, categorize=categorize))


def preview_pdf_import(pdf_content: bytes, categorize=None) -> dict:
    """Parse a text-based PDF statement into editable rows without saving."""
    text = _extract_pdf_text(pdf_content)
    return _preview_transaction_text(text, categorize=categorize)


def preview_pdf_text(pdf_content: bytes, max_lines: int = 20) -> str:
    """Return a short text preview from a PDF import file."""
    text = _extract_pdf_text(pdf_content)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[:max_lines])


def _default_data() -> dict:
    return {
        "transactions": [],
        "budgets": DEFAULT_BUDGETS.copy(),
        "custom_categories": [],
        "learning_examples": [],
    }


def _decode_csv_content(csv_content: str | bytes) -> str:
    if isinstance(csv_content, bytes):
        try:
            return csv_content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV file must be UTF-8 encoded.") from exc
    return str(csv_content)


def _extract_pdf_text(pdf_content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError(
            "PDF import requires the pypdf package. Run `pip install -r requirements.txt`."
        ) from exc

    try:
        reader = PdfReader(BytesIO(pdf_content))
    except Exception as exc:
        raise ValueError("Could not read that PDF file.") from exc

    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")

    text = "\n".join(pages).strip()
    if not text:
        raise ValueError(
            "No readable text was found in that PDF. Scanned image PDFs are not supported yet."
        )
    return text


def _parse_transaction_text(text: str, categorize=None) -> dict:
    return _persist_preview_result(_preview_transaction_text(text, categorize=categorize))


def _preview_transaction_text(text: str, categorize=None) -> dict:
    rows = []
    errors = []
    data = load_data()
    valid = BASE_CATEGORIES | set(data.get("custom_categories", []))
    learning_examples = data.get("learning_examples", [])
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = " ".join(raw_line.split())
        if not line or not re.search(r"\d", line):
            continue

        try:
            rows.append(
                _transaction_from_text_line(
                    line,
                    categorize=categorize,
                    learning_examples=learning_examples,
                    valid_categories=valid,
                )
            )
        except ValueError as exc:
            if _looks_like_transaction_line(line):
                errors.append(f"Line {line_number}: {exc}")

    return {"rows": rows, "failed": len(errors), "errors": errors}


def _transaction_from_text_line(
    line: str,
    categorize=None,
    learning_examples: list[dict] | None = None,
    valid_categories: set[str] | None = None,
) -> dict:
    date_match = re.search(r"\b(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{4})\b", line)
    if not date_match:
        raise ValueError("No supported date was found.")

    amount_matches = list(
        re.finditer(r"(?<!\w)(?:Rs\.?|INR|\u20b9)?\s*\(?[+-]?\d[\d,]*(?:\.\d{1,2})?\)?(?!\w)", line)
    )
    amount_matches = [
        match
        for match in amount_matches
        if not (date_match.start() <= match.start() < date_match.end())
    ]
    if not amount_matches:
        raise ValueError("No amount was found.")

    amount_match = amount_matches[-1]
    txn_date = _normalize_import_date(date_match.group(1))
    amount = _parse_import_amount(amount_match.group(0), line)

    description_parts = [
        line[: date_match.start()],
        line[date_match.end() : amount_match.start()],
        line[amount_match.end() :],
    ]
    description = " ".join(part.strip(" ,|-") for part in description_parts if part.strip(" ,|-"))
    description = re.sub(
        r"\b(debit|withdrawal|spent|paid|dr|credit|deposit|cr)\b",
        "",
        description,
        flags=re.IGNORECASE,
    )
    description = " ".join(description.split())
    category = ""
    valid_for_suffix = valid_categories if valid_categories is not None else VALID_CATEGORIES
    for valid_category in sorted(valid_for_suffix, key=len, reverse=True):
        suffix = f" {valid_category}".lower()
        if description.lower().endswith(suffix):
            description = description[: -len(valid_category)].strip(" ,|-")
            category = valid_category
            break
    original_description = description
    description = _apply_learned_description(
        description,
        learning_examples if learning_examples is not None else get_learning_examples(),
    )
    if not category:
        category = categorize(description) if categorize else "Misc"

    transaction = _build_transaction(
        transaction_id=0,
        description=description,
        amount=amount,
        category=category,
        date=txn_date,
        created_at=datetime.now().isoformat(),
        valid_categories=valid_categories,
    )
    transaction["_import_original_description"] = original_description
    transaction["_import_suggested_description"] = transaction["description"]
    transaction["_import_suggested_category"] = transaction["category"]
    transaction["import"] = True
    transaction["custom_category"] = ""
    return transaction


def _looks_like_transaction_line(line: str) -> bool:
    has_date = re.search(r"\b(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{4})\b", line)
    has_alpha = re.search(r"[A-Za-z]", line)
    return bool(has_date and has_alpha)


def _normalize_import_date(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", value):
        year, month, day = value.split("-")
    else:
        day, month, year = re.split(r"[/-]", value)
    return date(int(year), int(month), int(day)).isoformat()


def _parse_import_amount(raw_amount: str, line: str) -> float:
    text = raw_amount.strip()
    negative = (
        text.startswith("-")
        or (text.startswith("(") and text.endswith(")"))
        or bool(re.search(r"\b(debit|withdrawal|spent|paid|dr)\b", line, re.IGNORECASE))
    )
    cleaned = re.sub(r"[^\d.]", "", text)
    if not cleaned:
        raise ValueError("Amount must be a valid number.")
    amount = round(float(cleaned), 2)
    if negative:
        amount = -amount
    return amount


def _apply_debit_credit(raw_amount: str, debit_credit: str) -> float:
    value = _validate_amount(raw_amount)
    direction = debit_credit.strip().lower()
    if direction not in {"debit", "credit"}:
        raise ValueError("Debit/Credit must be either Debit or Credit.")
    amount = abs(value)
    return -amount if direction == "debit" else amount


def persist_reviewed_import_rows(rows: list[dict]) -> dict:
    """Persist user-reviewed import rows after validating the selected set."""
    selected = [
        (index, row)
        for index, row in enumerate(rows, start=1)
        if _row_bool(row, "import", "Import", default=True)
    ]
    if not selected:
        return {"imported": 0, "failed": 0, "errors": []}

    data = load_data()
    existing_custom = list(data.get("custom_categories", []))
    new_custom: list[str] = []
    errors: list[str] = []
    prepared: list[dict] = []

    for row_number, row in selected:
        category = _row_text(row, "category", "Category") or "Misc"
        custom_category = _row_text(row, "custom_category", "Custom category", "Custom Category")
        if category == "Custom":
            if not custom_category:
                errors.append(f"Row {row_number}: Custom category name is required.")
                continue
            category = custom_category
            if (
                category not in BASE_CATEGORIES
                and category not in existing_custom
                and category not in new_custom
            ):
                new_custom.append(category)

        valid = BASE_CATEGORIES | set(existing_custom) | set(new_custom)
        if category not in valid:
            errors.append(f"Row {row_number}: Unknown category '{category}'.")
            continue

        try:
            amount = _validate_amount(_row_value(row, "amount", "Amount (Rs)", "Amount"))
            if amount == 0:
                raise ValueError("Amount cannot be zero.")
            transaction = _build_transaction(
                transaction_id=0,
                description=_row_text(row, "description", "Description"),
                amount=amount,
                category=category,
                date=_row_text(row, "date", "Date"),
                created_at=datetime.now().isoformat(),
                valid_categories=valid,
            )
        except ValueError as exc:
            errors.append(f"Row {row_number}: {exc}")
            continue

        prepared.append(
            {
                "transaction": transaction,
                "original_description": _row_text(
                    row,
                    "_import_original_description",
                    "original_description",
                    "Original description",
                ) or transaction["description"],
                "suggested_description": _row_text(
                    row,
                    "_import_suggested_description",
                    "suggested_description",
                    "Suggested description",
                ) or transaction["description"],
                "suggested_category": _row_text(
                    row,
                    "_import_suggested_category",
                    "suggested_category",
                    "Suggested category",
                ) or transaction["category"],
                "used_custom": _row_text(row, "category", "Category") == "Custom",
            }
        )

    if errors:
        return {"imported": 0, "failed": len(errors), "errors": errors}

    data["custom_categories"] = existing_custom + new_custom
    valid = BASE_CATEGORIES | set(data["custom_categories"])
    next_id = _next_id(data["transactions"])
    created_at = datetime.now().isoformat()
    learned = 0

    for offset, item in enumerate(prepared):
        transaction = _build_transaction(
            transaction_id=next_id + offset,
            description=item["transaction"]["description"],
            amount=item["transaction"]["amount"],
            category=item["transaction"]["category"],
            date=item["transaction"]["date"],
            created_at=created_at,
            valid_categories=valid,
        )
        data["transactions"].append(transaction)

        if _row_was_overridden(item):
            if _upsert_learning_example(
                data,
                original_description=item["original_description"],
                description=transaction["description"],
                category=transaction["category"],
                valid_categories=valid,
            ) is not None:
                learned += 1

    save_data(data)
    return {"imported": len(prepared), "failed": 0, "errors": [], "learned": learned}


def _persist_preview_result(preview: dict) -> dict:
    rows = preview.get("rows", [])
    if not rows:
        return {
            "imported": 0,
            "failed": int(preview.get("failed", 0)),
            "errors": list(preview.get("errors", [])),
        }
    result = persist_reviewed_import_rows(rows)
    errors = list(preview.get("errors", [])) + list(result.get("errors", []))
    return {
        "imported": result["imported"],
        "failed": int(preview.get("failed", 0)) + int(result.get("failed", 0)),
        "errors": errors,
        "learned": result.get("learned", 0),
    }


def _row_was_overridden(item: dict) -> bool:
    transaction = item["transaction"]
    return (
        item["used_custom"]
        or _normalize_learning_key(transaction["description"])
        != _normalize_learning_key(item["suggested_description"])
        or transaction["category"] != item["suggested_category"]
    )


def _row_value(row: dict, *keys: str, default=None):
    for key in keys:
        if key in row:
            return row[key]
    return default


def _row_text(row: dict, *keys: str) -> str:
    value = _row_value(row, *keys, default="")
    if value is None:
        return ""
    return str(value).strip()


def _row_bool(row: dict, *keys: str, default: bool = False) -> bool:
    value = _row_value(row, *keys, default=default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if value is None:
        return default
    return str(value).strip().lower() not in {"", "0", "false", "no", "n"}


def _persist_imported_transactions(result: dict) -> dict:
    imported = result["imported"]
    if not imported:
        return {"imported": 0, "failed": result["failed"], "errors": result["errors"]}

    data = load_data()
    next_id = _next_id(data["transactions"])
    created_at = datetime.now().isoformat()
    for offset, transaction in enumerate(imported):
        transaction["id"] = next_id + offset
        transaction["created_at"] = created_at
        data["transactions"].append(transaction)
    save_data(data)

    return {
        "imported": len(imported),
        "failed": result["failed"],
        "errors": result["errors"],
    }


def _normalize_data(data: dict) -> dict:
    if not isinstance(data, dict):
        data = {}

    transactions = data.get("transactions", [])
    if not isinstance(transactions, list):
        transactions = []

    budgets = _validate_budgets(data.get("budgets", {}))

    custom_categories = data.get("custom_categories", [])
    if not isinstance(custom_categories, list):
        custom_categories = []
    custom_categories = [str(c).strip() for c in custom_categories if c and str(c).strip()]

    # Build the full valid-category set from the raw data (no load_data call needed)
    valid = BASE_CATEGORIES | set(custom_categories)
    learning_examples = _normalize_learning_examples(
        data.get("learning_examples", []),
        valid,
    )

    return {
        "transactions": [_normalize_transaction(txn, valid) for txn in transactions],
        "budgets": budgets,
        "custom_categories": custom_categories,
        "learning_examples": learning_examples,
    }


def _normalize_learning_examples(raw_examples, valid: set[str]) -> list[dict]:
    if not isinstance(raw_examples, list):
        return []

    by_key: dict[str, dict] = {}
    for raw in raw_examples:
        if not isinstance(raw, dict):
            continue
        original = str(raw.get("original_description", "")).strip()
        description = str(raw.get("description", "")).strip()
        if not original or not description:
            continue
        key = _normalize_learning_key(original)
        if not key:
            continue
        category = _validate_category(raw.get("category", "Misc"), valid)
        try:
            count = max(1, int(raw.get("count", 1)))
        except (TypeError, ValueError):
            count = 1
        example = {
            "original_description": original,
            "description": description,
            "category": category,
            "count": count,
            "updated_at": str(raw.get("updated_at") or datetime.now().isoformat()),
        }
        existing = by_key.get(key)
        if existing is None or example["updated_at"] >= existing["updated_at"]:
            by_key[key] = example

    examples = sorted(
        by_key.values(),
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )
    return examples[:LEARNING_EXAMPLE_LIMIT]


def _upsert_learning_example(
    data: dict,
    original_description: str,
    description: str,
    category: str,
    valid_categories: set[str],
) -> dict | None:
    original = str(original_description).strip()
    final_description = str(description).strip()
    key = _normalize_learning_key(original)
    if not key or not final_description:
        return None

    final_category = _validate_category(category, valid_categories)
    examples = list(data.get("learning_examples", []))
    updated_at = datetime.now().isoformat()
    updated = None
    remaining = []
    for example in examples:
        if _normalize_learning_key(example.get("original_description", "")) == key:
            try:
                count = max(1, int(example.get("count", 1))) + 1
            except (TypeError, ValueError):
                count = 2
            updated = {
                "original_description": original,
                "description": final_description,
                "category": final_category,
                "count": count,
                "updated_at": updated_at,
            }
        else:
            remaining.append(example)

    if updated is None:
        updated = {
            "original_description": original,
            "description": final_description,
            "category": final_category,
            "count": 1,
            "updated_at": updated_at,
        }

    data["learning_examples"] = _normalize_learning_examples(
        [updated, *remaining],
        valid_categories,
    )
    return updated


def _apply_learned_description(description: str, examples: list[dict]) -> str:
    value = str(description).strip()
    key = _normalize_learning_key(value)
    if not key:
        return value

    best_match = None
    best_score = 0.0
    for example in examples:
        example_key = _normalize_learning_key(example.get("original_description", ""))
        if not example_key:
            continue
        if example_key == key:
            return str(example.get("description", value)).strip() or value
        score = SequenceMatcher(None, key, example_key).ratio()
        if score > best_score:
            best_score = score
            best_match = example

    if best_match and best_score >= LEARNING_SIMILARITY_THRESHOLD:
        return str(best_match.get("description", value)).strip() or value
    return value


def _normalize_learning_key(description: str) -> str:
    value = re.sub(r"[^a-z0-9]+", " ", str(description).lower())
    return " ".join(value.split())


def _normalize_transaction(transaction: dict, valid: set[str] | None = None) -> dict:
    if not isinstance(transaction, dict):
        raise ValueError("Stored transaction must be a JSON object.")

    return _build_transaction(
        transaction_id=int(transaction["id"]),
        description=transaction["description"],
        amount=transaction["amount"],
        category=transaction["category"],
        date=transaction["date"],
        created_at=transaction.get("created_at") or datetime.now().isoformat(),
        valid_categories=valid,
    )


def _build_transaction(
    transaction_id: int,
    description: str,
    amount: float,
    category: str,
    date: str,
    created_at: str,
    valid_categories: set[str] | None = None,
) -> dict:
    return {
        "id": int(transaction_id),
        "description": _validate_description(description),
        "amount": _validate_amount(amount),
        "category": _validate_category(category, valid_categories),
        "date": _validate_date(date),
        "created_at": str(created_at),
    }


def _validate_description(description: str) -> str:
    value = str(description).strip()
    if not value:
        raise ValueError("Description cannot be empty.")
    return value


def _validate_amount(amount: float) -> float:
    try:
        value = round(float(amount), 2)
    except (TypeError, ValueError) as exc:
        raise ValueError("Amount must be a valid number.") from exc
    if not isfinite(value):
        raise ValueError("Amount must be a valid number.")
    return value


def _validate_category(category: str, valid: set[str] | None = None) -> str:
    value = str(category).strip()
    if valid is None:
        valid = BASE_CATEGORIES  # safe fallback that doesn't call load_data
    if value not in valid:
        return "Misc"
    return value


def _validate_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError("Date must be in YYYY-MM-DD format.") from exc
    return parsed.isoformat()


def _validate_budgets(budgets: dict) -> dict[str, float]:
    """Validate and normalise budgets. Preserves all provided non-Income categories."""
    normalized = DEFAULT_BUDGETS.copy()
    if not isinstance(budgets, dict):
        return normalized

    for category, raw_value in budgets.items():
        category = str(category).strip()
        if not category or category == "Income":
            continue  # Income is never budgeted
        try:
            normalized[category] = max(0.0, round(float(raw_value), 2))
        except (TypeError, ValueError):
            normalized[category] = 0.0
    return normalized


def _backup_corrupt_data_file() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(
        os.path.dirname(DATA_FILE),
        f"data.corrupt-{timestamp}.json",
    )
    shutil.copy2(DATA_FILE, backup_path)
    return backup_path


def _next_id(transactions: list) -> int:
    """Generate a monotonically increasing integer ID."""
    if not transactions:
        return 1
    return max(t["id"] for t in transactions) + 1
