import unittest
from importlib.util import find_spec
from pathlib import Path

from src.training.train_metadata_baseline import (
    build_metadata_artifact_paths,
    evaluate_predictions,
)


class TrainMetadataBaselineTest(unittest.TestCase):
    def test_metadata_artifact_paths_use_expected_names(self):
        paths = build_metadata_artifact_paths(Path("/tmp/run"))

        self.assertEqual(paths.metrics_json, Path("/tmp/run/metadata_test_metrics.json"))
        self.assertEqual(
            paths.classification_report_csv,
            Path("/tmp/run/metadata_classification_report.csv"),
        )
        self.assertEqual(paths.confusion_matrix_csv, Path("/tmp/run/metadata_confusion_matrix.csv"))
        self.assertEqual(paths.feature_names_json, Path("/tmp/run/metadata_feature_names.json"))

    def test_evaluate_predictions_reports_high_risk_recall(self):
        if find_spec("sklearn") is None:
            self.skipTest("scikit-learn is not installed")

        metrics, report, matrix = evaluate_predictions(
            targets=[0, 1, 2, 3, 4],
            predictions=[0, 1, 2, 0, 3],
            labels=["ACK", "BCC", "MEL", "NEV", "SCC"],
        )

        self.assertAlmostEqual(metrics["test_high_risk_recall"], 2 / 3)
        self.assertIn("BCC", report.index)
        self.assertEqual(matrix.loc["BCC", "BCC"], 1)


if __name__ == "__main__":
    unittest.main()
