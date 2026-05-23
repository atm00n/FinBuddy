"""
tracker.py - Core financial analytics and recommendations engine.

All computation is pure-Python; no external services required.
"""

from collections import defaultdict
from datetime import date, timedelta
from typing import Optional

from utils import BUDGET_CATEGORIES, get_budgets, get_transactions, get_all_categories


def filter_transactions(
    transactions: Optional[list] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[dict]:
    """Return transactions inside the provided inclusive date range."""
    if transactions is None:
        transactions = get_transactions()

    filtered = []
    for txn in transactions:
        txn_date = date.fromisoformat(txn["date"])
        if start_date and txn_date < start_date:
            continue
        if end_date and txn_date > end_date:
            continue
        filtered.append(txn)
    return filtered


def preset_date_range(preset: str, today: Optional[date] = None) -> tuple[Optional[date], Optional[date]]:
    """Map a named filter preset to an inclusive date range."""
    today = today or date.today()
    if preset == "All time":
        return None, None
    if preset == "This month":
        return today.replace(day=1), today
    if preset == "Last 30 days":
        return today - timedelta(days=29), today
    raise ValueError(f"Unknown preset: {preset}")


def compute_summary(transactions: Optional[list] = None) -> dict:
    """
    Compute high-level financial summary.

    Returns:
        {
          total_income:    float,
          total_expenses:  float,
          balance:         float,
          category_totals: {category: float},
        }
    """
    if transactions is None:
        transactions = get_transactions()

    total_income = 0.0
    total_expenses = 0.0
    category_totals: dict[str, float] = defaultdict(float)

    for txn in transactions:
        amount = txn.get("amount", 0.0)
        category = txn.get("category", "Misc")
        if amount >= 0:
            total_income += amount
        else:
            expense_value = abs(amount)
            total_expenses += expense_value
            category_totals[category] += expense_value

    return {
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "balance": round(total_income - total_expenses, 2),
        "category_totals": {k: round(v, 2) for k, v in category_totals.items()},
    }


def category_breakdown(transactions: Optional[list] = None) -> dict:
    """Return spending per category and percentage of total spending."""
    if transactions is None:
        transactions = get_transactions()

    summary = compute_summary(transactions)
    totals = summary["category_totals"]
    total_spend = summary["total_expenses"] or 1

    return {
        category: {
            "amount": amount,
            "percent": round(amount / total_spend * 100, 1),
        }
        for category, amount in totals.items()
    }


def budget_status(
    transactions: Optional[list] = None,
    budgets: Optional[dict[str, float]] = None,
) -> dict[str, dict]:
    """Return budget progress for each budgetable category."""
    if transactions is None:
        transactions = get_transactions()
    if budgets is None:
        budgets = get_budgets()

    category_totals = compute_summary(transactions)["category_totals"]
    status = {}
    # Cover all known categories except Income (income is not budgeted)
    all_budget_cats = [c for c in get_all_categories() if c != "Income"]
    for category in all_budget_cats:
        spent = round(category_totals.get(category, 0.0), 2)
        budget = round(float(budgets.get(category, 0.0)), 2)
        remaining = round(budget - spent, 2)
        usage_pct = round((spent / budget) * 100, 1) if budget > 0 else None
        status[category] = {
            "spent": spent,
            "budget": budget,
            "remaining": remaining,
            "usage_pct": usage_pct,
            "over_budget": budget > 0 and spent > budget,
            "has_budget": budget > 0,
        }
    return status


def monthly_trend(transactions: Optional[list] = None) -> list[dict]:
    """Aggregate income, expenses, and balance by month."""
    if transactions is None:
        transactions = get_transactions()

    buckets: dict[str, dict[str, float]] = defaultdict(
        lambda: {"income": 0.0, "expenses": 0.0}
    )
    for txn in transactions:
        month_key = txn["date"][:7]
        amount = float(txn["amount"])
        if amount >= 0:
            buckets[month_key]["income"] += amount
        else:
            buckets[month_key]["expenses"] += abs(amount)

    rows = []
    for month in sorted(buckets):
        income = round(buckets[month]["income"], 2)
        expenses = round(buckets[month]["expenses"], 2)
        rows.append(
            {
                "month": month,
                "income": income,
                "expenses": expenses,
                "balance": round(income - expenses, 2),
            }
        )
    return rows


def generate_recommendations(
    transactions: Optional[list] = None,
    budgets: Optional[dict[str, float]] = None,
) -> list[str]:
    """Analyze transactions and return a list of human-readable insights."""
    if transactions is None:
        transactions = get_transactions()
    if budgets is None:
        budgets = get_budgets()

    if not transactions:
        return ["No transactions yet. Add some to get personalized insights."]

    summary = compute_summary(transactions)
    income = summary["total_income"]
    expenses = summary["total_expenses"]
    balance = summary["balance"]
    cat_totals = summary["category_totals"]
    budget_map = budget_status(transactions, budgets)

    recommendations = []

    for category, details in budget_map.items():
        if details["over_budget"]:
            recommendations.append(
                f"Budget alert: {category} is over budget by "
                f"Rs {abs(details['remaining']):,.2f} this period."
            )
        elif details["has_budget"] and details["usage_pct"] is not None and details["usage_pct"] >= 85:
            recommendations.append(
                f"Budget watch: {category} has used {details['usage_pct']:.1f}% "
                f"of its budget."
            )

    if expenses > 0:
        for category, amount in cat_totals.items():
            pct = amount / expenses * 100
            if pct > 50:
                recommendations.append(
                    f"{category} accounts for {pct:.1f}% of your spending "
                    f"(Rs {amount:,.2f}). It deserves a closer look."
                )

        if cat_totals:
            top_category = max(cat_totals, key=cat_totals.get)
            top_amount = cat_totals[top_category]
            top_pct = top_amount / expenses * 100
            if top_pct <= 50:
                recommendations.append(
                    f"Your highest spend is {top_category} at Rs {top_amount:,.2f} "
                    f"({top_pct:.1f}% of expenses)."
                )

    if income > 0:
        savings_rate = balance / income * 100
        if savings_rate < 0:
            recommendations.append(
                f"You are spending more than you earn this period "
                f"(deficit: Rs {abs(balance):,.2f})."
            )
        elif savings_rate < 20:
            recommendations.append(
                f"Your savings rate is {savings_rate:.1f}%, below the 20% target."
            )
        else:
            recommendations.append(
                f"Your savings rate is {savings_rate:.1f}%, which is healthy."
            )
    else:
        recommendations.append(
            "No income is recorded in this date range, so savings analysis is limited."
        )

    food_spend = cat_totals.get("Food", 0)
    if income > 0 and food_spend / income > 0.30:
        recommendations.append(
            f"Food spend is Rs {food_spend:,.2f}, above 30% of income."
        )

    transport_spend = cat_totals.get("Transport", 0)
    if income > 0 and transport_spend / income > 0.15:
        recommendations.append(
            f"Transport spend is Rs {transport_spend:,.2f}, above 15% of income."
        )

    if not recommendations:
        recommendations.append("Your finances look stable in this period.")

    return recommendations


def recent_transactions_text(
    transactions: Optional[list] = None,
    n: int = 10,
) -> str:
    """Return the most recent transactions as a compact text block."""
    if transactions is None:
        transactions = get_transactions()

    recent = transactions[:n]
    if not recent:
        return "No transactions recorded."

    lines = [
        f"- [{txn['date']}] {txn['description']} | Rs {txn['amount']:+.2f} | {txn['category']}"
        for txn in recent
    ]
    return "\n".join(lines)
