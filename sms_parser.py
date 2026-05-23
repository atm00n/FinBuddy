"""
sms_parser.py - UPI / bank SMS parser for Indian transaction SMS formats.

Parses the standard SMS templates sent by:
  HDFC, SBI, ICICI, Axis, Kotak, Yes Bank, IndusInd, IDFC, BOI, PNB,
  Paytm Payments Bank, PhonePe, Google Pay, and generic UPI formats.

Public API
----------
    parse_sms(text: str)       -> SMSTransaction | None
    parse_sms_bulk(texts: str) -> list[SMSTransaction]   (newline-separated block)

SMSTransaction fields
---------------------
    amount      float   positive = credit, negative = debit
    description str     payee / UPI ID / merchant name
    date        str     ISO "YYYY-MM-DD", today if not found in SMS
    raw         str     original SMS text
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SMSTransaction:
    amount: float
    description: str
    date: str
    raw: str


# ---------------------------------------------------------------------------
# Regex building blocks
# ---------------------------------------------------------------------------

# Amount: "Rs 1,500.00" / "INR1500" / "Rs. 12,345" / "₹500"
_AMT = r"(?:Rs\.?\s*|INR\s*|₹\s*)([\d,]+(?:\.\d{1,2})?)"
# Reversed-amount variant: "500.00 INR debited" (some bank formats)
_AMT_REV = r"([\d,]+(?:\.\d{1,2})?)\s*(?:Rs\.?|INR|₹)"

# Date patterns found in Indian bank SMS
_DATE_PATTERNS = [
    # 15-Jan-25 / 15-Jan-2025
    (r"\b(\d{1,2})[- ]([A-Za-z]{3})[- ](\d{2,4})\b",
     lambda m: _parse_dmy_mon(m.group(1), m.group(2), m.group(3))),
    # 15/01/2025 or 15-01-2025
    (r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b",
     lambda m: _parse_dmy(m.group(1), m.group(2), m.group(3))),
    # 2025-01-15
    (r"\b(\d{4})-(\d{2})-(\d{2})\b",
     lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
    # 15/01/25
    (r"\b(\d{1,2})/(\d{1,2})/(\d{2})\b",
     lambda m: _parse_dmy(m.group(1), m.group(2), "20" + m.group(3))),
]

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _parse_dmy_mon(d: str, mon: str, y: str) -> Optional[str]:
    m = _MONTH_ABBR.get(mon.lower()[:3])
    if not m:
        return None
    year = int(y) if len(y) == 4 else 2000 + int(y)
    try:
        return date(year, m, int(d)).isoformat()
    except ValueError:
        return None


def _parse_dmy(d: str, m: str, y: str) -> Optional[str]:
    try:
        return date(int(y), int(m), int(d)).isoformat()
    except ValueError:
        return None


def _extract_date(text: str) -> str:
    for pattern, handler in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            result = handler(m)
            if result:
                return result
    return date.today().isoformat()


def _clean_amount(raw: str) -> float:
    return float(raw.replace(",", ""))


# ---------------------------------------------------------------------------
# SMS pattern library
# Each entry: (compiled_regex, handler_fn -> SMSTransaction | None)
# Handlers receive the match object and the full SMS text.
# ---------------------------------------------------------------------------

def _sign(text: str, amount: float) -> float:
    """
    Determine debit/credit from context words.
    Returns negative for debits, positive for credits.
    """
    tl = text.lower()
    credit_keywords = [
        "credited", "credit", "received", "deposited",
        "refund", "cashback", "reward", "salary", "added",
        "payment received", "upi cr", "neft cr", "imps cr",
    ]
    debit_keywords = [
        "debited", "debit", "paid", "payment of", "transferred",
        "spent", "withdrawn", "upi dr", "neft dr", "imps dr",
        "sent", "purchase",
    ]
    # Check explicit debit keywords first (they tend to appear close to amount)
    for kw in debit_keywords:
        if kw in tl:
            return -abs(amount)
    for kw in credit_keywords:
        if kw in tl:
            return abs(amount)
    # Default: treat as debit (most SMS are expense notifications)
    return -abs(amount)


def _extract_upi_id_or_vpa(text: str) -> Optional[str]:
    """Pull out a UPI VPA like name@upi, name@okaxis, etc."""
    m = re.search(r"[\w.\-+]+@[\w]+", text)
    return m.group(0) if m else None


def _extract_merchant(text: str) -> str:
    """
    Try several merchant-extraction strategies, return best guess.
    """
    # 1. Labelled fields: "Merchant: X", "Info: X", "Remarks: X", "Narration: X"
    for pat in [
        r"(?:Merchant|Info|Remarks?|Narration|Description|Details?)[:\s]+([A-Za-z0-9 @._\-&]{3,50}?)(?:\s+(?:Ref|UPI|NEFT|IMPS|on\s+\d)|\.|$|\n)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().rstrip(".")
            if len(candidate) >= 3:
                return candidate

    # 2. Explicit directional patterns: "to X", "at X", "from X"
    for pat in [
        r"(?:paid\s+to|sent\s+to|transferred\s+to|to\s+upi\s+id|payment\s+to)\s+([A-Za-z0-9 @._\-]{3,40}?)(?:\s+on|\s+via|\s+upi|\s+ref|\.|\n|$)",
        r"(?:at|from|by)\s+([A-Za-z][A-Za-z0-9 &._\-]{2,35}?)(?:\s+on|\s+via|\s+ref|\.|\n|$)",
        r"(?:txn|txnid|ref(?:no)?|rrn)[:\s#]+[\w\d]+\s+(?:at|to|by)\s+([A-Za-z0-9 @._\-]{3,40}?)(?:\s|\.|\n|$)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if len(candidate) >= 3:
                return candidate

    # 3. UPI VPA
    vpa = _extract_upi_id_or_vpa(text)
    if vpa:
        return vpa

    return "UPI Transaction"


# ── Pattern definitions ───────────────────────────────────────────────────────

# Pattern 1: Generic UPI debit/credit
# "Your a/c XX1234 is debited for Rs 500.00 on 15-Jan-25 via UPI to merchant@upi Ref 12345"
_P_UPI_GENERIC = re.compile(
    r"(?:a/?c|acct?|account)[^\d]*(?:XX|xx|\*{2,})?\d{4,}"
    r".{0,60}" + _AMT + r".{0,120}",
    re.IGNORECASE | re.DOTALL,
)

# Pattern 2: HDFC — "Rs.X.XX debited from a/c **1234 on dd-Mon-yy"
_P_HDFC_DEBIT = re.compile(
    _AMT + r"\s+(?:debited|credited)\s+(?:from|to)\s+(?:a/?c|acct?)[\s*X]+(\d{4})",
    re.IGNORECASE,
)

# Pattern 3: SBI — "Your a/c no. XX1234 is debited by Rs.X"
_P_SBI = re.compile(
    r"a/?c\s+(?:no\.?\s+)?[Xx*]+(\d{4})\s+is\s+(debited|credited)\s+(?:by|with)\s+" + _AMT,
    re.IGNORECASE,
)

# Pattern 4: ICICI — "ICICI Bank Acct XX1234 debited Rs 500"
_P_ICICI = re.compile(
    r"ICICI\s+Bank\s+Acct\s+[Xx*]+(\d{4})\s+(debited|credited)\s+(?:with\s+)?" + _AMT,
    re.IGNORECASE,
)

# Pattern 5: Axis — "Rs 500.00 debited from Axis Bank Acct XX1234"
_P_AXIS = re.compile(
    _AMT + r"\s+(debited|credited)\s+(?:from|to)\s+Axis\s+Bank",
    re.IGNORECASE,
)

# Pattern 6: Kotak — "Kotak Bank: Rs 500.00 debited"
_P_KOTAK = re.compile(
    r"Kotak\s+(?:Mahindra\s+)?Bank[:\s]+" + _AMT + r"\s+(debited|credited)",
    re.IGNORECASE,
)

# Pattern 7: Paytm / PhonePe / GPay — "Paid Rs 500 to merchant@upi"
_P_UPI_APPS = re.compile(
    r"(?:Paid|Sent|Received|Payment\s+of)\s+" + _AMT +
    r"\s+(?:to|from|by)\s+([A-Za-z0-9@._\- ]{3,40}?)(?:\s+on|\s+via|\s+ref|\.|$)",
    re.IGNORECASE,
)

# Pattern 8: Generic amount-only with debit/credit word (last resort)
_P_AMOUNT_ONLY = re.compile(
    r"(?:"
    # Keyword before amount: "debited Rs 500" / "paid Rs 500"
    r"(?:debited|credited|debit|credit|paid|received|sent)\D{0,20}" + _AMT +
    r"|"
    # Amount before keyword: "Rs 500 debited" / "INR 500 debited" / "₹500 paid"
    + _AMT + r"\D{0,30}(?:debited|credited|debit|credit|paid|received|sent)"
    r")",
    re.IGNORECASE,
)

# ── Parser functions per pattern ──────────────────────────────────────────────

def _try_upi_apps(text: str) -> Optional[SMSTransaction]:
    m = _P_UPI_APPS.search(text)
    if not m:
        return None
    amt_raw = m.group(1)
    merchant = m.group(2).strip()
    amount = _clean_amount(amt_raw)
    amount = _sign(text, amount)
    return SMSTransaction(
        amount=amount,
        description=merchant or "UPI Transaction",
        date=_extract_date(text),
        raw=text,
    )


def _try_sbi(text: str) -> Optional[SMSTransaction]:
    m = _P_SBI.search(text)
    if not m:
        return None
    direction = m.group(2).lower()
    amount = _clean_amount(m.group(3))
    amount = -amount if "debit" in direction else amount
    return SMSTransaction(
        amount=amount,
        description=_extract_merchant(text),
        date=_extract_date(text),
        raw=text,
    )


def _try_icici(text: str) -> Optional[SMSTransaction]:
    m = _P_ICICI.search(text)
    if not m:
        return None
    direction = m.group(2).lower()
    amount = _clean_amount(m.group(3))
    amount = -amount if "debit" in direction else amount
    return SMSTransaction(
        amount=amount,
        description=_extract_merchant(text),
        date=_extract_date(text),
        raw=text,
    )


def _try_hdfc(text: str) -> Optional[SMSTransaction]:
    m = _P_HDFC_DEBIT.search(text)
    if not m:
        return None
    amount = _clean_amount(m.group(1))
    direction = text[m.start():m.start()+200].lower()
    amount = _sign(direction, amount)
    return SMSTransaction(
        amount=amount,
        description=_extract_merchant(text),
        date=_extract_date(text),
        raw=text,
    )


def _try_axis(text: str) -> Optional[SMSTransaction]:
    m = _P_AXIS.search(text)
    if not m:
        return None
    amount = _clean_amount(m.group(1))
    direction = m.group(2).lower()
    amount = -amount if "debit" in direction else amount
    return SMSTransaction(
        amount=amount,
        description=_extract_merchant(text),
        date=_extract_date(text),
        raw=text,
    )


def _try_kotak(text: str) -> Optional[SMSTransaction]:
    m = _P_KOTAK.search(text)
    if not m:
        return None
    amount = _clean_amount(m.group(1))
    direction = m.group(2).lower()
    amount = -amount if "debit" in direction else amount
    return SMSTransaction(
        amount=amount,
        description=_extract_merchant(text),
        date=_extract_date(text),
        raw=text,
    )


def _try_generic_upi(text: str) -> Optional[SMSTransaction]:
    """Broad fallback: any SMS mentioning an amount with debit/credit language."""
    m = _P_AMOUNT_ONLY.search(text)
    if not m:
        return None
    # The regex has two alternations each with one capture group; pick the one that matched
    raw_amount = m.group(1) or m.group(2)
    if not raw_amount:
        return None
    amount = _clean_amount(raw_amount)
    amount = _sign(text, amount)
    return SMSTransaction(
        amount=amount,
        description=_extract_merchant(text),
        date=_extract_date(text),
        raw=text,
    )


def _try_reversed_amount(text: str) -> Optional[SMSTransaction]:
    """Fallback for 'amount Rs debited' style (reversed currency-marker) SMS."""
    m = re.search(
        _AMT_REV + r".{0,30}(?:debited|credited|debit|credit|paid|received|sent)",
        text,
        re.IGNORECASE,
    )
    if not m:
        # Also try: "debited ... 500 Rs"
        m = re.search(
            r"(?:debited|credited|debit|credit|paid|received|sent).{0,30}" + _AMT_REV,
            text,
            re.IGNORECASE,
        )
    if not m:
        return None
    amount = _clean_amount(m.group(1))
    amount = _sign(text, amount)
    return SMSTransaction(
        amount=amount,
        description=_extract_merchant(text),
        date=_extract_date(text),
        raw=text,
    )


# Ordered cascade — specific banks first, generic last
_PARSERS = [
    _try_upi_apps,
    _try_hdfc,
    _try_sbi,
    _try_icici,
    _try_axis,
    _try_kotak,
    _try_generic_upi,
    _try_reversed_amount,
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_sms(text: str) -> Optional[SMSTransaction]:
    """
    Parse a single UPI/bank SMS and return an SMSTransaction, or None if
    the text doesn't look like a transaction SMS.

    Args:
        text: Raw SMS body text.

    Returns:
        SMSTransaction with amount (negative=debit, positive=credit),
        description (payee/merchant), date (ISO), and the raw text.
        Returns None if no amount can be extracted.
    """
    if not text or not text.strip():
        return None

    # Quick pre-filter: must contain a currency marker (with flexible spacing/punctuation)
    # and at least one digit after it.
    if not re.search(
        r"(?:Rs\.?\s*|INR\s*|₹\s*)[\d,]+|[\d,]+(?:\.\d{1,2})?\s*(?:Rs|INR|₹)",
        text,
        re.IGNORECASE,
    ):
        return None

    # Skip OTP / alert / promo messages — but only when no explicit debit/credit word present
    skip_keywords = [
        "otp", "password", "login", "offer", "discount", "cashback promo",
        "click here", "dear customer, your", "kyc", "expired",
    ]
    tl = text.lower()
    transaction_words = ("debited", "credited", "debit", "credit", "paid", "received", "sent")
    has_transaction_word = any(w in tl for w in transaction_words)
    if any(kw in tl for kw in skip_keywords) and not has_transaction_word:
        return None

    for parser in _PARSERS:
        result = parser(text)
        if result:
            return result

    return None


def parse_sms_bulk(block: str) -> list[SMSTransaction]:
    """
    Parse a block of text containing multiple SMS messages separated by
    blank lines or "---" dividers.

    Args:
        block: Multi-SMS text block pasted by the user.

    Returns:
        List of successfully parsed SMSTransaction objects.
    """
    # Split on blank lines or explicit dividers
    segments = re.split(r"\n\s*\n|^---+$", block.strip(), flags=re.MULTILINE)
    results = []
    for seg in segments:
        seg = seg.strip()
        if seg:
            parsed = parse_sms(seg)
            if parsed:
                results.append(parsed)
    return results
