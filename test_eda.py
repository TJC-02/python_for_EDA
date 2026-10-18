"""Regression checks for missing data and categorical-only CSV files."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from eda import analyze, profile


class AnalysisTests(unittest.TestCase):
    def test_profile_distinguishes_missing_from_constant(self):
        frame = pd.DataFrame({"empty": [None, None], "constant": [1, 1],
                              "varying": [1, 2]})
        result = profile(frame)
        self.assertEqual(result["all_missing_columns"], ["empty"])
        self.assertEqual(result["constant_columns"], ["constant"])
        self.assertEqual(result["missing_cells"], 2)

    def test_categorical_data_with_missing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "sample.csv"
            csv.write_text("label,group\nyes,a\nno,b\n,a\n")
            output = root / "results"
            result = analyze(csv, output, target="label")
            self.assertEqual(result["rows"], 3)
            self.assertTrue((output / "target_distribution.png").exists())
            self.assertEqual(pd.read_csv(output / "target_counts.csv")["count"].sum(), 3)
            self.assertFalse((output / "correlations.png").exists())

    def test_invalid_target_does_not_create_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "sample.csv"
            csv.write_text("value\n1\n")
            with self.assertRaisesRegex(ValueError, "absent"):
                analyze(csv, root / "results", target="missing")
            self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
