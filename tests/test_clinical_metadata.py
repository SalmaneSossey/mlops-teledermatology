import unittest

import pandas as pd

from src.features.clinical_metadata import (
    LEAKAGE_FIELDS,
    clinical_feature_fields,
    encode_clinical_metadata,
    fit_clinical_metadata_encoder,
    merge_clinical_metadata,
    metadata_missingness_summary,
)


class ClinicalMetadataTest(unittest.TestCase):
    def make_metadata(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "img_id": ["a.png", "b.png", "c.png"],
                "age": [50, 70, 60],
                "region": ["FACE", "ARM", "BACK"],
                "itch": ["True", "False", "UNK"],
                "grew": ["True", "False", "False"],
                "hurt": ["False", "False", "True"],
                "changed": ["True", "False", "UNK"],
                "bleed": ["False", "True", "False"],
                "elevation": ["True", "False", "UNK"],
                "gender": ["FEMALE", None, "MALE"],
                "fitspatrick": [3.0, None, 2.0],
                "diameter_1": [6.0, None, 10.0],
                "diameter_2": [5.0, None, 8.0],
                "skin_cancer_history": ["True", None, "False"],
                "cancer_history": ["False", None, "True"],
                "smoke": ["False", None, "True"],
                "drink": ["False", None, "False"],
                "pesticide": ["True", None, "False"],
                "diagnostic": ["BCC", "ACK", "MEL"],
                "biopsed": [True, False, True],
            }
        )

    def test_clinical_feature_fields_exclude_leakage_columns(self):
        fields = set(clinical_feature_fields(include_optional=True))

        self.assertTrue(set(LEAKAGE_FIELDS).isdisjoint(fields))

    def test_merge_clinical_metadata_joins_by_image_without_leakage_fields(self):
        split = pd.DataFrame({"img_id": ["a.png", "b.png"], "label_idx": [1, 0]})

        merged = merge_clinical_metadata(split, self.make_metadata())

        self.assertIn("age", merged.columns)
        self.assertIn("region", merged.columns)
        self.assertNotIn("biopsed", merged.columns)
        self.assertNotIn("diagnostic", merged.columns)

    def test_encoder_adds_missing_indicators_and_stable_columns(self):
        metadata = self.make_metadata()
        train = metadata.iloc[:2].copy()
        test = metadata.iloc[2:].copy()

        encoder, [train_encoded, test_encoded] = encode_clinical_metadata(train, [train, test])

        self.assertEqual(list(train_encoded.columns), encoder.feature_names)
        self.assertEqual(list(test_encoded.columns), encoder.feature_names)
        self.assertIn("diameter_1__missing", encoder.feature_names)
        self.assertIn("gender__missing", encoder.feature_names)
        self.assertIn("region__FACE", encoder.feature_names)
        self.assertIn("region____OTHER__", encoder.feature_names)
        self.assertEqual(float(train_encoded.loc[1, "diameter_1__missing"]), 1.0)
        self.assertEqual(float(test_encoded.iloc[0]["region____OTHER__"]), 1.0)

    def test_missingness_summary_reports_optional_gaps(self):
        summary = metadata_missingness_summary(self.make_metadata())
        missing_rate = summary.set_index("feature").loc["gender", "missing_rate"]

        self.assertAlmostEqual(missing_rate, 1 / 3)

    def test_complete_field_missing_values_fail_fast(self):
        metadata = self.make_metadata()
        metadata.loc[0, "age"] = None

        with self.assertRaises(ValueError):
            fit_clinical_metadata_encoder(metadata)


if __name__ == "__main__":
    unittest.main()
