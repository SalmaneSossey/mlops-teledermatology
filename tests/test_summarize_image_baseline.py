import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.summarize_image_baseline import summarize_run


class SummarizeImageBaselineTest(unittest.TestCase):
    def test_summarizes_metrics_and_high_risk_mistakes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "test_metrics.json").write_text(
                json.dumps(
                    {
                        "test_macro_f1": 0.6636222473178995,
                        "test_balanced_accuracy": 0.6803457938658227,
                        "test_high_risk_recall": 0.8292682926829268,
                    }
                )
            )
            pd.DataFrame(
                {
                    "precision": {"BCC": 0.8, "MEL": 0.5, "SCC": 0.7},
                    "recall": {"BCC": 0.9, "MEL": 0.25, "SCC": 0.8},
                    "f1-score": {"BCC": 0.847, "MEL": 0.333, "SCC": 0.747},
                    "support": {"BCC": 127, "MEL": 8, "SCC": 29},
                }
            ).to_csv(run_dir / "classification_report.csv")
            pd.DataFrame(
                {
                    "ACK": {"BCC": 2, "MEL": 1, "SCC": 0},
                    "BCC": {"BCC": 114, "MEL": 1, "SCC": 3},
                    "MEL": {"BCC": 2, "MEL": 2, "SCC": 1},
                    "NEV": {"BCC": 0, "MEL": 3, "SCC": 0},
                    "SCC": {"BCC": 4, "MEL": 1, "SCC": 23},
                    "SEK": {"BCC": 5, "MEL": 0, "SCC": 2},
                }
            ).to_csv(run_dir / "confusion_matrix.csv")

            summary = summarize_run(run_dir)

        self.assertIn("macro_f1: 0.6636", summary)
        self.assertIn("BCC: precision=0.8000, recall=0.9000", summary)
        self.assertIn("MEL: 2/8 correct", summary)
        self.assertIn("SCC: 23/29 correct", summary)


if __name__ == "__main__":
    unittest.main()
