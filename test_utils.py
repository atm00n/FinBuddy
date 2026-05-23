import unittest
from io import StringIO
import csv
import os
import tempfile

import utils


class CsvTransactionFormatTests(unittest.TestCase):
    def test_exports_only_public_csv_columns(self):
        csv_text = utils._transactions_to_csv(
            [
                {
                    "id": 9,
                    "date": "2026-05-01",
                    "description": "Salary",
                    "amount": 50000.0,
                    "category": "Income",
                    "created_at": "2026-05-06T10:00:00",
                },
                {
                    "id": 10,
                    "date": "2026-05-02",
                    "description": "Swiggy dinner",
                    "amount": -450.5,
                    "category": "Food",
                    "created_at": "2026-05-06T10:00:00",
                },
            ]
        )

        rows = list(csv.DictReader(StringIO(csv_text)))

        self.assertEqual(utils.CSV_COLUMNS, list(rows[0].keys()))
        self.assertEqual(rows[0]["Debit/Credit"], "Credit")
        self.assertEqual(rows[1]["Amount"], "450.5")
        self.assertEqual(rows[1]["Debit/Credit"], "Debit")

    def test_import_row_uses_required_csv_columns(self):
        headers = {column.lower(): column for column in utils.CSV_COLUMNS}
        transaction = utils._transaction_from_csv_row(
            {
                "Date": "2026-05-02",
                "Details": "Uber trip",
                "Amount": "320.00",
                "Currency": "INR",
                "Debit/Credit": "Debit",
                "Status": "Completed",
                "Ignored": "value",
            },
            headers,
            categorize=lambda description: "Transport",
        )

        self.assertEqual(transaction["date"], "2026-05-02")
        self.assertEqual(transaction["description"], "Uber trip")
        self.assertEqual(transaction["amount"], -320.0)
        self.assertEqual(transaction["category"], "Transport")


class PdfTransactionParsingTests(unittest.TestCase):
    def test_parses_iso_date_description_amount_and_category(self):
        transaction = utils._transaction_from_text_line(
            "2026-05-01 Swiggy dinner -450.50 Food"
        )

        self.assertEqual(transaction["date"], "2026-05-01")
        self.assertEqual(transaction["description"], "Swiggy dinner")
        self.assertEqual(transaction["amount"], -450.50)
        self.assertEqual(transaction["category"], "Food")

    def test_parses_indian_date_and_debit_as_expense(self):
        transaction = utils._transaction_from_text_line(
            "02/05/2026 UBER TRIP Rs 320.00 debit",
            categorize=lambda description: "Transport",
        )

        self.assertEqual(transaction["date"], "2026-05-02")
        self.assertEqual(transaction["description"], "UBER TRIP")
        self.assertEqual(transaction["amount"], -320.00)
        self.assertEqual(transaction["category"], "Transport")


class ImportPreviewAndLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_data_file = utils.DATA_FILE
        utils.DATA_FILE = os.path.join(self.temp_dir.name, "data.json")

    def tearDown(self):
        utils.DATA_FILE = self.original_data_file
        self.temp_dir.cleanup()

    def test_csv_preview_does_not_persist_transactions(self):
        csv_text = "\n".join(
            [
                ",".join(utils.CSV_COLUMNS),
                "2026-05-02,Uber trip,320.00,INR,Debit,Completed",
            ]
        )

        preview = utils.preview_csv_import(
            csv_text,
            categorize=lambda description: "Transport",
        )

        self.assertEqual(len(preview["rows"]), 1)
        self.assertEqual(preview["rows"][0]["description"], "Uber trip")
        self.assertEqual(preview["rows"][0]["category"], "Transport")
        self.assertEqual(utils.get_transactions(), [])

    def test_text_preview_does_not_persist_transactions(self):
        preview = utils._preview_transaction_text(
            "2026-05-02 Swiggy dinner -450.00",
            categorize=lambda description: "Food",
        )

        self.assertEqual(len(preview["rows"]), 1)
        self.assertEqual(preview["rows"][0]["description"], "Swiggy dinner")
        self.assertEqual(preview["rows"][0]["category"], "Food")
        self.assertEqual(utils.get_transactions(), [])

    def test_reviewed_import_persists_edited_description_and_category(self):
        result = utils.persist_reviewed_import_rows(
            [
                {
                    "Import": True,
                    "Date": "2026-05-03",
                    "Description": "Office cab",
                    "Amount (Rs)": -450,
                    "Category": "Transport",
                    "_import_original_description": "UPI Transaction",
                    "_import_suggested_description": "UPI Transaction",
                    "_import_suggested_category": "Misc",
                }
            ]
        )

        transactions = utils.get_transactions()
        self.assertEqual(result["imported"], 1)
        self.assertEqual(transactions[0]["description"], "Office cab")
        self.assertEqual(transactions[0]["category"], "Transport")
        self.assertEqual(utils.get_learning_examples()[0]["category"], "Transport")

    def test_reviewed_import_creates_custom_category_before_validation(self):
        result = utils.persist_reviewed_import_rows(
            [
                {
                    "Import": True,
                    "Date": "2026-05-04",
                    "Description": "nayanbhattad007@okaxis",
                    "Amount (Rs)": -213,
                    "Category": "Custom",
                    "Custom category": "anna",
                    "_import_original_description": "nayanbhattad007@okaxis",
                    "_import_suggested_description": "nayanbhattad007@okaxis",
                    "_import_suggested_category": "Misc",
                }
            ]
        )

        self.assertEqual(result["imported"], 1)
        self.assertIn("anna", utils.get_all_categories())
        self.assertEqual(utils.get_transactions()[0]["category"], "anna")

    def test_reviewed_import_is_all_or_nothing_on_validation_error(self):
        result = utils.persist_reviewed_import_rows(
            [
                {
                    "Import": True,
                    "Date": "2026-05-04",
                    "Description": "Good row",
                    "Amount (Rs)": -100,
                    "Category": "Food",
                },
                {
                    "Import": True,
                    "Date": "bad-date",
                    "Description": "Bad row",
                    "Amount (Rs)": -50,
                    "Category": "Food",
                },
            ]
        )

        self.assertEqual(result["imported"], 0)
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(utils.get_transactions(), [])

    def test_learning_examples_dedupe_cap_and_reuse_description(self):
        for index in range(110):
            utils.record_learning_example(f"raw merchant {index}", f"Merchant {index}", "Misc")
        utils.record_learning_example("PAYTM*RAW MERCHANT", "Clean Merchant", "Food")
        utils.record_learning_example("paytm raw merchant", "Cleaner Merchant", "Transport")

        examples = utils.get_learning_examples()
        self.assertLessEqual(len(examples), utils.LEARNING_EXAMPLE_LIMIT)
        matches = [
            example for example in examples
            if example["original_description"].lower() == "paytm raw merchant"
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["count"], 2)
        self.assertEqual(
            utils.apply_learned_description("PAYTM RAW MERCHANT"),
            "Cleaner Merchant",
        )


if __name__ == "__main__":
    unittest.main()
