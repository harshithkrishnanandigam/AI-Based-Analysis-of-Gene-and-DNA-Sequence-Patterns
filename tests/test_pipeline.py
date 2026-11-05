"""
Unit and Integration Tests for DNA Splice-Junction Prediction Pipeline
Implemented using Python's standard library 'unittest' (also 100% pytest compatible).
"""

import sys
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src/models is in Python path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir / "src" / "models"))

from predict import extract_sequence_features, predict_dna_sequence, DNAPatternPredictor


class TestDNAPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.models_dir = base_dir / "models_saved"
        cls.sample_donor = "CCAGCTGCATCACAGGAGGCCAGCGAGCAGGTCTGTTCCAAGGGCCTTCGAGCCAGTCTG"
        cls.sample_acceptor = "TCGATCGGGACCTCCTCGCGCCCCAGGCGCCCCTGCCCGCCCCCGCCCCGCGCCCCGCCG"

    def test_feature_extraction_dimension(self):
        """Verify that feature extraction yields exactly 91 features."""
        features_df = extract_sequence_features(self.sample_donor)
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertEqual(features_df.shape, (1, 91), f"Expected 91 features, got {features_df.shape[1]}")

    def test_gc_content_exact(self):
        """Verify that GC content calculation is mathematically exact."""
        # 50% GC content
        feats_50 = extract_sequence_features("ATGC")
        self.assertEqual(feats_50["GC_Content"].iloc[0], 50.0)

        # 100% GC content
        feats_100 = extract_sequence_features("GGCC")
        self.assertEqual(feats_100["GC_Content"].iloc[0], 100.0)

        # 0% GC content
        feats_0 = extract_sequence_features("AATT")
        self.assertEqual(feats_0["GC_Content"].iloc[0], 0.0)

    def test_predictor_output_keys(self):
        """Verify that inference engine produces standard schema dictionary."""
        result = predict_dna_sequence(self.sample_donor, models_dir=self.models_dir)

        expected_keys = {
            "sequence_length",
            "gc_content",
            "predicted_label",
            "predicted_name",
            "confidence_percent",
            "class_probabilities",
            "biological_role",
            "biological_consensus",
            "color",
        }
        self.assertTrue(expected_keys.issubset(result.keys()), f"Missing keys in result: {expected_keys - set(result.keys())}")
        self.assertIn(result["predicted_label"], {"EI", "IE", "N"})
        self.assertGreaterEqual(result["confidence_percent"], 0.0)
        self.assertLessEqual(result["confidence_percent"], 100.0)

    def test_class_probabilities_sum(self):
        """Verify that multi-class probabilities sum to approximately 100%."""
        result = predict_dna_sequence(self.sample_acceptor, models_dir=self.models_dir)
        probs = result["class_probabilities"]
        total = sum(probs.values())
        self.assertAlmostEqual(total, 100.0, delta=1.0, msg=f"Probabilities do not sum to 100%: {total}")

    def test_invalid_short_sequence(self):
        """Verify that input sequences under 3 base pairs return a validation error."""
        result = predict_dna_sequence("AT", models_dir=self.models_dir)
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
