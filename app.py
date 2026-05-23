"""
app.py - Main Streamlit entry point for the AI Personal Finance Tracker.

Run with:
    streamlit run app.py
"""

from datetime import date, datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import assistant
import model
import sms_parser
import tracker
import utils

FILTER_PRESETS = ["All time", "This month", "Last 30 days", "Custom range"]
CUSTOM_CATEGORY_OPTION = "Custom"

st.set_page_config(
    page_title="FinBuddy",
    page_icon="FB",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10231f 0%, #1b3b34 100%);
    border-right: 1px solid #294f46;
}

[data-testid="stSidebar"] * {
    color: #eef7f4 !important;
}

[data-testid="metric-container"] {
    background: #f5faf8;
    border: 1px solid #d8e8e1;
    border-radius: 12px;
    padding: 14px;
}

.stButton > button {
    background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stTextArea textarea,
.stTextInput input {
    font-family: 'JetBrains Mono', monospace;
}

h1, h2, h3 {
    letter-spacing: 0;
}
</style>
""",
    unsafe_allow_html=True,
)


def _init_state() -> None:
    st.session_state.setdefault("editing_txn_id", None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("filter_preset", "All time")
    st.session_state.setdefault("custom_start", date.today().replace(day=1))
    st.session_state.setdefault("custom_end", date.today())
    st.session_state.setdefault("chat_filter_signature", "")
    st.session_state.setdefault("import_result", None)


def _resolve_date_filter() -> tuple[str, date | None, date | None]:
    preset = st.session_state.get("filter_preset", "All time")
    if preset == "Custom range":
        start_date = st.session_state.get("custom_start")
        end_date = st.session_state.get("custom_end")
        if start_date and end_date and start_date > end_date:
            start_date, end_date = end_date, start_date
        label = (
            f"{start_date.isoformat()} to {end_date.isoformat()}"
            if start_date and end_date
            else "custom range"
        )
        return label, start_date, end_date

    start_date, end_date = tracker.preset_date_range(preset)
    return preset, start_date, end_date


def _render_budget_editor(budgets: dict[str, float]) -> None:
    with st.sidebar.expander("Category budgets", expanded=False):
        # Dynamic budget inputs for all budgetable categories (non-Income)
        budget_cats = [c for c in utils.get_all_categories() if c != "Income"]
        with st.form("budget_form"):
            new_values: dict[str, float] = {}
            for cat in budget_cats:
                new_values[cat] = st.number_input(
                    f"{cat} budget (Rs)",
                    min_value=0.0,
                    value=float(budgets.get(cat, 0.0)),
                    step=500.0,
                    format="%.2f",
                )
            submitted = st.form_submit_button("Save budgets", use_container_width=True)

        if submitted:
            utils.save_budgets(new_values)
            st.sidebar.success("Budgets saved.")
            st.rerun()

    # New category creation
    with st.sidebar.expander("Add custom category", expanded=False):
        new_cat_name = st.text_input("Category name", key="new_cat_input", max_chars=40)
        if st.button("Create category", key="create_cat_btn", use_container_width=True):
            if new_cat_name and new_cat_name.strip():
                try:
                    utils.add_custom_category(new_cat_name.strip())
                    st.sidebar.success(f"Category '{new_cat_name.strip()}' added.")
                    st.rerun()
                except ValueError as exc:
                    st.sidebar.error(str(exc))
            else:
                st.sidebar.warning("Enter a category name first.")


def _render_filter_summary(period_label: str, filtered_count: int, total_count: int) -> None:
    st.caption(
        f"Active period: {period_label} | {filtered_count} of {total_count} transactions in view"
    )


def _render_chat_turn(turn: dict) -> None:
    with st.chat_message(turn["role"]):
        st.markdown(turn.get("content", ""))
        if turn["role"] == "assistant" and turn.get("uses_ai") is False:
            source = str(turn.get("source") or "Local FinBuddy rules")
            if source == "Local FinBuddy rules":
                source = "local FinBuddy rules"
            st.caption(f"Not using AI - {source}")


def _assistant_turn(response: assistant.AssistantResponse) -> dict:
    return {
        "role": "assistant",
        "content": response["content"],
        "uses_ai": response["uses_ai"],
        "source": response["source"],
    }


def _find_transaction(transactions: list[dict], txn_id: int | None) -> dict | None:
    if txn_id is None:
        return None
    for txn in transactions:
        if txn["id"] == txn_id:
            return txn
    return None


def _resolve_form_category(selection: str, custom_name: str) -> str:
    if selection != CUSTOM_CATEGORY_OPTION:
        return selection
    custom_category = custom_name.strip()
    if not custom_category:
        raise ValueError("Custom category name is required.")
    utils.add_custom_category(custom_category)
    model.clear_cache()
    return custom_category


def _draft_rows_for_editor(rows: list[dict]) -> pd.DataFrame:
    visible_rows = []
    for row in rows:
        visible_rows.append(
            {
                "Import": bool(row.get("import", True)),
                "Date": row["date"],
                "Description": row["description"],
                "Amount (Rs)": float(row["amount"]),
                "Category": row["category"],
                "Custom category": row.get("custom_category", ""),
            }
        )
    return pd.DataFrame(visible_rows)


def _merge_review_rows(draft_rows: list[dict], edited_df: pd.DataFrame) -> list[dict]:
    reviewed = []
    edited_rows = edited_df.to_dict("records")
    for draft, edited in zip(draft_rows, edited_rows):
        reviewed.append(
            {
                "Import": edited.get("Import", True),
                "Date": edited.get("Date", draft["date"]),
                "Description": edited.get("Description", draft["description"]),
                "Amount (Rs)": edited.get("Amount (Rs)", draft["amount"]),
                "Category": edited.get("Category", draft["category"]),
                "Custom category": edited.get("Custom category", ""),
                "_import_original_description": draft.get(
                    "_import_original_description",
                    draft["description"],
                ),
                "_import_suggested_description": draft.get(
                    "_import_suggested_description",
                    draft["description"],
                ),
                "_import_suggested_category": draft.get(
                    "_import_suggested_category",
                    draft["category"],
                ),
            }
        )
    return reviewed


def _render_import_review(
    draft_rows: list[dict],
    source_label: str,
    key: str,
    skipped_errors: list[str] | None = None,
    skipped_failed: int = 0,
) -> None:
    if not draft_rows:
        st.info(f"No transactions were found in that {source_label}.")
        return

    category_options = [*utils.get_all_categories(), CUSTOM_CATEGORY_OPTION]
    edited_df = st.data_editor(
        _draft_rows_for_editor(draft_rows),
        key=f"{key}_editor",
        use_container_width=True,
        hide_index=True,
        disabled=["Date", "Amount (Rs)"],
        column_config={
            "Import": st.column_config.CheckboxColumn("Import"),
            "Description": st.column_config.TextColumn("Description", max_chars=120),
            "Amount (Rs)": st.column_config.NumberColumn("Amount (Rs)", format="Rs %.2f"),
            "Category": st.column_config.SelectboxColumn(
                "Category",
                options=category_options,
                required=True,
            ),
            "Custom category": st.column_config.TextColumn("Custom category", max_chars=40),
        },
    )

    selected_count = int(edited_df["Import"].sum()) if "Import" in edited_df else len(draft_rows)
    if st.button(
        f"Import selected {source_label} transaction(s)",
        key=f"{key}_import",
        use_container_width=True,
        disabled=selected_count == 0,
    ):
        result = utils.persist_reviewed_import_rows(
            _merge_review_rows(draft_rows, edited_df)
        )
        if result["errors"]:
            st.warning("Fix the highlighted import issue(s) and try again.")
            with st.expander("Import issue details", expanded=True):
                for error in result["errors"]:
                    st.write(error)
            return

        model.clear_cache()
        if skipped_errors:
            result = {
                **result,
                "failed": result.get("failed", 0) + skipped_failed,
                "errors": [*skipped_errors, *result.get("errors", [])],
            }
        st.session_state["import_result"] = result
        if key == "sms":
            st.session_state["sms_input_area"] = ""
        st.rerun()


def _render_transaction_form(all_transactions: list[dict]) -> None:
    editing_txn = _find_transaction(all_transactions, st.session_state.get("editing_txn_id"))
    editing = editing_txn is not None

    st.title("Transactions")
    st.markdown(
        "Capture income and expenses here. FinBuddy will categorize entries automatically unless you override them."
    )

    default_description = editing_txn["description"] if editing else ""
    default_amount = float(editing_txn["amount"]) if editing else 0.0
    default_date = (
        date.fromisoformat(editing_txn["date"]) if editing else date.today()
    )
    default_category = editing_txn["category"] if editing else "Auto (ML)"

    # Dynamic category list (loaded fresh each render to pick up new categories)
    all_cats = utils.get_all_categories()
    cat_options = ["Auto (ML)", *all_cats, CUSTOM_CATEGORY_OPTION]
    cat_index = cat_options.index(default_category) if default_category in cat_options else 0

    with st.form("transaction_form", clear_on_submit=not editing):
        col1, col2 = st.columns([3, 1])
        with col1:
            description = st.text_input(
                "Description",
                value=default_description,
                placeholder="Examples: Salary credit, Swiggy dinner, Uber ride",
                max_chars=120,
            )
            if description.strip():
                predicted = model.predict(description)
                confidence = model.predict_proba(description).get(predicted, 0.0)
                st.caption(
                    f"Suggested category: {predicted} ({confidence * 100:.0f}% confidence)"
                )
        with col2:
            amount = st.number_input(
                "Amount (Rs)",
                value=default_amount,
                step=1.0,
                format="%.2f",
                help="Positive for income, negative for expenses.",
            )

        col3, col4 = st.columns(2)
        with col3:
            txn_date = st.date_input("Date", value=default_date)
        with col4:
            category_override = st.selectbox(
                "Category",
                cat_options,
                index=cat_index,
            )
            custom_category_name = ""
            if category_override == CUSTOM_CATEGORY_OPTION:
                custom_category_name = st.text_input(
                    "Custom category",
                    max_chars=40,
                    placeholder="Enter category name",
                )

        action_col, cancel_col = st.columns([1, 1])
        submitted = action_col.form_submit_button(
            "Save changes" if editing else "Add transaction",
            use_container_width=True,
        )
        cancelled = cancel_col.form_submit_button(
            "Cancel edit" if editing else "Reset",
            use_container_width=True,
        )

    if cancelled and editing:
        st.session_state["editing_txn_id"] = None
        st.rerun()

    if submitted:
        if not description.strip():
            st.error("Description cannot be empty.")
            return
        if amount == 0:
            st.error("Amount cannot be zero.")
            return

        try:
            category = (
                model.predict(description)
                if category_override == "Auto (ML)"
                else _resolve_form_category(category_override, custom_category_name)
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        try:
            if editing:
                utils.update_transaction(
                    txn_id=editing_txn["id"],
                    description=description,
                    amount=amount,
                    category=category,
                    date=txn_date.isoformat(),
                )
                st.session_state["editing_txn_id"] = None
                st.success("Transaction updated.")
            else:
                utils.add_transaction(
                    description=description,
                    amount=amount,
                    category=category,
                    date=txn_date.isoformat(),
                )
                st.success("Transaction saved.")
            if category_override != "Auto (ML)":
                utils.record_learning_example(description, description, category)
                model.clear_cache()
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.markdown("---")
    st.markdown("## Recent transactions")

    if not all_transactions:
        st.info("No transactions yet. Add your first one above.")
        return

    preview_df = pd.DataFrame(all_transactions)[["date", "description", "amount", "category"]].copy()
    preview_df["amount"] = preview_df["amount"].map(lambda value: f"Rs {value:+,.2f}")
    st.dataframe(preview_df.head(10), use_container_width=True, hide_index=True)

    st.markdown("### Manage transactions")
    for txn in all_transactions[:12]:
        amount_text = f"Rs {txn['amount']:+,.2f}"
        row = st.columns([1.2, 2.8, 1.2, 1.2, 0.8, 0.8])
        row[0].markdown(txn["date"])
        row[1].markdown(txn["description"])
        row[2].markdown(amount_text)
        row[3].markdown(txn["category"])
        if row[4].button("Edit", key=f"edit_{txn['id']}"):
            st.session_state["editing_txn_id"] = txn["id"]
            st.rerun()
        if row[5].button("Delete", key=f"delete_{txn['id']}"):
            utils.delete_transaction(txn["id"])
            if st.session_state.get("editing_txn_id") == txn["id"]:
                st.session_state["editing_txn_id"] = None
            st.rerun()


def _render_budget_cards(filtered_transactions: list[dict], budgets: dict[str, float]) -> None:
    st.markdown("## Budget status")
    budget_map = tracker.budget_status(filtered_transactions, budgets)
    # Show all categories except Income that either have a budget or have spending
    display_cats = [
        cat for cat in utils.get_all_categories()
        if cat != "Income" and (budgets.get(cat, 0) > 0 or budget_map.get(cat, {}).get("spent", 0) > 0)
    ]
    if not display_cats:
        display_cats = [c for c in utils.get_all_categories() if c != "Income"]
    cols_per_row = 3
    for i in range(0, len(display_cats), cols_per_row):
        row_cats = display_cats[i:i + cols_per_row]
        columns = st.columns(len(row_cats))
        for column, category in zip(columns, row_cats):
            details = budget_map.get(category, {"spent": 0, "remaining": 0, "has_budget": False, "over_budget": False, "usage_pct": 0, "budget": 0})
            with column:
                st.metric(
                    category,
                    f"Rs {details['spent']:,.2f}",
                    delta=(
                        f"Remaining Rs {details['remaining']:,.2f}"
                        if details["has_budget"]
                        else "No budget set"
                    ),
                    delta_color="inverse" if details["over_budget"] else "normal",
                )
                if details["has_budget"]:
                    usage = min(details["usage_pct"] or 0.0, 100.0)
                    st.progress(int(usage))
                    st.caption(f"Budget Rs {details['budget']:,.2f} | Used {details['usage_pct']:.1f}%")
                else:
                    st.caption("Set a budget from the sidebar to track this category.")



def _render_dashboard(filtered_transactions: list[dict], budgets: dict[str, float], period_label: str) -> None:
    st.title("Dashboard")
    st.markdown("A focused view of income, expenses, budgets, and trends for the active date range.")

    if not filtered_transactions:
        st.info(f"No transactions found for {period_label}. Adjust the date filter or add new entries.")
        return

    summary = tracker.compute_summary(filtered_transactions)
    breakdown = tracker.category_breakdown(filtered_transactions)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total income", f"Rs {summary['total_income']:,.2f}")
    col2.metric("Total expenses", f"Rs {summary['total_expenses']:,.2f}")
    col3.metric("Balance", f"Rs {summary['balance']:,.2f}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("## Spending by category")
        if breakdown:
            bar_df = pd.DataFrame(
                [
                    {"Category": category, "Amount": details["amount"]}
                    for category, details in breakdown.items()
                ]
            )
            bar_fig = px.bar(
                bar_df,
                x="Category",
                y="Amount",
                color="Category",
                color_discrete_sequence=["#0f766e", "#1d4ed8", "#f59e0b", "#ef4444"],
                text_auto=".2s",
            )
            bar_fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.info("No expenses in this range yet, so the category chart is empty by design.")

    with right:
        st.markdown("## Spending share")
        if breakdown:
            pie_df = pd.DataFrame(
                [
                    {"Category": category, "Amount": details["amount"]}
                    for category, details in breakdown.items()
                ]
            )
            pie_fig = px.pie(
                pie_df,
                names="Category",
                values="Amount",
                hole=0.45,
                color_discrete_sequence=["#0f766e", "#1d4ed8", "#f59e0b", "#ef4444"],
            )
            st.plotly_chart(pie_fig, use_container_width=True)
        else:
            st.info("No expense share to show until at least one expense is recorded.")

    st.markdown("---")
    _render_budget_cards(filtered_transactions, budgets)

    st.markdown("---")
    st.markdown("## Monthly trend")
    trend_rows = tracker.monthly_trend(filtered_transactions)
    if trend_rows:
        trend_df = pd.DataFrame(trend_rows)
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Bar(name="Income", x=trend_df["month"], y=trend_df["income"], marker_color="#0f766e"))
        trend_fig.add_trace(go.Bar(name="Expenses", x=trend_df["month"], y=trend_df["expenses"], marker_color="#ef4444"))
        trend_fig.add_trace(go.Scatter(name="Balance", x=trend_df["month"], y=trend_df["balance"], mode="lines+markers", line=dict(color="#1d4ed8", width=3)))
        trend_fig.update_layout(barmode="group")
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("Not enough data in this range to plot a monthly trend yet.")


def _render_insights(filtered_transactions: list[dict], budgets: dict[str, float], period_label: str) -> None:
    st.title("Insights")
    st.markdown("Actionable recommendations based on your filtered transactions and current category budgets.")

    if not filtered_transactions:
        st.info(f"No transactions found for {period_label}. Add transactions or widen the filter.")
        return

    recommendations = tracker.generate_recommendations(filtered_transactions, budgets)
    for recommendation in recommendations:
        if "over budget" in recommendation.lower() or "spending more than you earn" in recommendation.lower():
            st.warning(recommendation)
        elif "healthy" in recommendation.lower():
            st.success(recommendation)
        else:
            st.info(recommendation)

    st.markdown("---")
    _render_budget_cards(filtered_transactions, budgets)

    st.markdown("---")
    st.markdown("## Savings rate")
    summary = tracker.compute_summary(filtered_transactions)
    if summary["total_income"] > 0:
        rate = max(0, summary["balance"] / summary["total_income"] * 100)
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=rate,
                number={"suffix": "%"},
                title={"text": "Savings rate"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1d4ed8"},
                    "steps": [
                        {"range": [0, 20], "color": "#fecaca"},
                        {"range": [20, 50], "color": "#fde68a"},
                        {"range": [50, 100], "color": "#bbf7d0"},
                    ],
                },
            )
        )
        st.plotly_chart(gauge, use_container_width=True)
    else:
        st.info("Savings rate is unavailable because there is no income in the active range.")


def _render_assistant(
    filtered_transactions: list[dict],
    budgets: dict[str, float],
    gemini_key: str,
    period_label: str,
) -> None:
    st.title("AI Assistant")
    st.markdown(
        "Ask questions about the active date range. FinBuddy will ground the answer in the visible transactions, totals, and budgets."
    )

    filter_signature = f"{period_label}|{len(filtered_transactions)}"
    if st.session_state.get("chat_filter_signature") != filter_signature:
        st.session_state["chat_history"] = []
        st.session_state["chat_filter_signature"] = filter_signature

    for turn in st.session_state["chat_history"]:
        _render_chat_turn(turn)

    if not st.session_state["chat_history"]:
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        suggestions = [
            "Where am I overspending in this period?",
            "What should I change first to save more?",
            "How do my budgets look right now?",
        ]
        for index, (col, prompt) in enumerate(zip(cols, suggestions)):
            if col.button(prompt, key=f"prompt_{index}"):
                st.session_state["chat_history"].append({"role": "user", "content": prompt})
                reply = assistant.ask_assistant_with_metadata(
                    prompt,
                    api_key=gemini_key or None,
                    transactions=filtered_transactions,
                    budgets=budgets,
                    period_label=period_label,
                )
                st.session_state["chat_history"].append(_assistant_turn(reply))
                st.rerun()

    question = st.chat_input("Ask about your finances for the active date range")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        reply = assistant.ask_assistant_with_metadata(
            question,
            api_key=gemini_key or None,
            transactions=filtered_transactions,
            budgets=budgets,
            period_label=period_label,
        )
        st.session_state["chat_history"].append(_assistant_turn(reply))
        st.rerun()

    if st.session_state["chat_history"] and st.button("Clear conversation"):
        st.session_state["chat_history"] = []
        st.rerun()


def _render_import_export(
    filtered_transactions: list[dict],
    all_transactions: list[dict],
    period_label: str,
) -> None:
    st.title("Import / Export")
    st.markdown("Upload transactions from a CSV or text-based PDF file, or download the current data.")

    st.markdown("## Import transactions")
    if st.session_state.get("import_result"):
        result = st.session_state["import_result"]
        if result["imported"]:
            st.success(f"Imported {result['imported']} transaction(s).")
        if not result["imported"] and not result["failed"]:
            st.info("No transactions were selected for import.")
        if result.get("learned"):
            st.caption(f"Learned from {result['learned']} correction(s).")
        if result["failed"]:
            st.warning(f"Skipped {result['failed']} row(s).")
            with st.expander("Skipped row details", expanded=True):
                for error in result["errors"]:
                    st.write(error)
        if st.button("Clear import summary"):
            st.session_state["import_result"] = None
            st.rerun()

    uploaded_file = st.file_uploader(
        "CSV or PDF file",
        type=["csv", "pdf"],
        help=(
            "CSV requires Date, Details, Amount, Currency, Debit/Credit, and Status columns. "
            "PDF import works with text-based statements that contain one transaction per line."
        ),
    )

    if uploaded_file is not None:
        is_pdf = uploaded_file.name.lower().endswith(".pdf")
        source_label = "PDF" if is_pdf else "CSV"
        file_content = uploaded_file.getvalue()
        try:
            preview = (
                utils.preview_pdf_import(file_content, categorize=model.predict)
                if is_pdf
                else utils.preview_csv_import(file_content, categorize=model.predict)
            )
        except ValueError as exc:
            st.error(str(exc))
        else:
            if preview["failed"]:
                st.warning(f"Skipped {preview['failed']} row(s) while reading the {source_label}.")
                with st.expander("Skipped row details", expanded=False):
                    for error in preview["errors"]:
                        st.write(error)
            st.markdown(f"**Review {len(preview['rows'])} {source_label} transaction(s) before importing:**")
            _render_import_review(
                preview["rows"],
                source_label=source_label,
                key=f"file_{source_label.lower()}_{uploaded_file.name}_{len(file_content)}",
                skipped_errors=preview["errors"],
                skipped_failed=preview["failed"],
            )

    st.markdown("---")
    st.markdown("## Import from UPI / bank SMS")
    st.markdown(
        "Paste one or more SMS messages below (separate multiple messages with a blank line). "
        "FinBuddy will extract the amount, payee, and date automatically."
    )

    sms_input = st.text_area(
        "SMS text",
        placeholder=(
            "Example:\n"
            "Rs 850.00 debited from a/c XX4321 on 12-May-25 via UPI to zomato@upi Ref 123456789\n\n"
            "Your a/c XX4321 is credited by Rs 50,000.00 on 01-May-25. Info: Salary May 2025"
        ),
        height=180,
        key="sms_input_area",
    )

    if sms_input and sms_input.strip():
        parsed_sms = sms_parser.parse_sms_bulk(sms_input)

        if not parsed_sms:
            st.warning("No transactions recognised in that text. Check that the SMS contains an amount (Rs / INR / ₹).")
        else:
            draft_rows = []
            valid_categories = set(utils.get_all_categories())
            learning_examples = utils.get_learning_examples()
            for txn in parsed_sms:
                draft_rows.append(
                    utils.build_import_preview_transaction(
                        description=txn.description,
                        amount=txn.amount,
                        date=txn.date,
                        categorize=model.predict,
                        learning_examples=learning_examples,
                        valid_categories=valid_categories,
                    )
                )
            st.markdown(f"**{len(parsed_sms)} transaction(s) found - review before importing:**")
            _render_import_review(draft_rows, source_label="SMS", key="sms")

            if st.button("Clear", use_container_width=True, key="sms_clear"):
                st.session_state["sms_input_area"] = ""
                st.rerun()

    st.markdown("---")
    st.markdown("## Export transactions")
    st.markdown("Download the current filtered view or the full transaction history.")

    if not all_transactions:
        st.info("No transactions to export yet.")
        return

    filtered_df = pd.DataFrame(filtered_transactions) if filtered_transactions else pd.DataFrame()
    all_df = pd.DataFrame(all_transactions)

    if not filtered_df.empty:
        st.markdown(f"**Filtered view ({period_label})**")
        st.dataframe(
            filtered_df[["date", "description", "amount", "category"]],
            use_container_width=True,
            hide_index=True,
        )
        filtered_csv = utils.export_csv(filtered_transactions)
        st.download_button(
            label="Download filtered CSV",
            data=filtered_csv,
            file_name=f"finbuddy_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("The active filter has no transactions to export.")

    st.markdown("---")
    st.markdown("**Full history**")
    st.dataframe(
        all_df[["date", "description", "amount", "category"]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        label="Download full CSV",
        data=utils.export_csv(),
        file_name=f"finbuddy_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


_init_state()

with st.sidebar:
    st.markdown("## FinBuddy")
    page = st.radio(
        "Navigate",
        ["Transactions", "Dashboard", "Insights", "AI Assistant", "Import / Export"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Date filter")
    st.selectbox("Period", FILTER_PRESETS, key="filter_preset")
    if st.session_state["filter_preset"] == "Custom range":
        st.date_input("Start date", key="custom_start")
        st.date_input("End date", key="custom_end")

    st.markdown("---")
    st.markdown("### Gemini API key")
    gemini_key = st.text_input(
        "Gemini API key (optional)",
        type="password",
        placeholder="AIza...",
        help=(
            "Powers both AI categorization and the AI Assistant chat. "
            "FinBuddy tries Gemini first, then local Gemma4, then built-in rules."
        ),
    )
    model.set_api_key(gemini_key)
    if gemini_key:
        st.sidebar.caption("AI categorization active (Gemini).")
    else:
        st.sidebar.caption("Using local categorization. Add a Gemini key to enable AI.")

all_transactions = utils.get_transactions()
warnings = utils.consume_data_warnings()
budgets = utils.get_budgets()
_render_budget_editor(budgets)

period_label, start_date, end_date = _resolve_date_filter()
filtered_transactions = tracker.filter_transactions(
    all_transactions,
    start_date=start_date,
    end_date=end_date,
)

for warning in warnings:
    st.warning(warning)

_render_filter_summary(period_label, len(filtered_transactions), len(all_transactions))

if page == "Transactions":
    _render_transaction_form(all_transactions)
elif page == "Dashboard":
    _render_dashboard(filtered_transactions, budgets, period_label)
elif page == "Insights":
    _render_insights(filtered_transactions, budgets, period_label)
elif page == "AI Assistant":
    _render_assistant(filtered_transactions, budgets, gemini_key, period_label)
elif page == "Import / Export":
    _render_import_export(filtered_transactions, all_transactions, period_label)
