"""
================================================================================
AI-BASED ANALYSIS OF GENE AND DNA SEQUENCE PATTERNS
Course: Introduction to Biological Data (Amrita Vishwa Vidyapeetham, Coimbatore)
Department: Artificial Intelligence and Data Science (1st Year)

PERSON 3 INFERENCE MODULE: REAL-TIME DNA PATTERN PREDICTION & BIOLOGICAL INTERPRETATION
Author: Harshith Krishna (CB.AI.U4AID26023)

This module provides a production-grade inference engine:
- Extracts identical 91 k-mer & compositional features from raw DNA sequences
- Uses the best-performing trained model (saved by model_pipeline.py)
- Returns class predictions (EI, IE, N), biological definitions, and probabilities
- Ready for seamless integration into Person 4's Streamlit Dashboard
================================================================================
"""

import sys
from pathlib import Path
from collections import Counter
from itertools import product
from typing import Dict, Any, Union

import numpy as np
import pandas as pd
import joblib

# Biological metadata and descriptions
SPLICE_SITE_METADATA = {
    "EI": {
        "name": "Exon-to-Intron Boundary (Donor Splice Site)",
        "role": "Marks the 5' end of an intron where RNA splicing begins.",
        "biological_consensus": "Typically characterized by a canonical 'GT' dinucleotide consensus in DNA (GU in pre-mRNA) at the 5' splice junction.",
        "color": "#e63946"
    },
    "IE": {
        "name": "Intron-to-Exon Boundary (Acceptor Splice Site)",
        "role": "Marks the 3' end of an intron where RNA splicing concludes.",
        "biological_consensus": "Preceded by a polypyrimidine tract (rich in C and T nucleotides) terminating with a canonical 'AG' dinucleotide at the 3' junction.",
        "color": "#457b9d"
    },
    "N": {
        "name": "Non-Splice Region (Neither)",
        "role": "Internal sequence within an exon or intron with no spliceosome cleavage activity.",
        "biological_consensus": "Lacks the characteristic paired spliceosomal consensus signals.",
        "color": "#2a9d8f"
    }
}

CANONICAL_BASES = "ATGC"
TWO_MERS = ["".join(x) for x in product(CANONICAL_BASES, repeat=2)]
THREE_MERS = ["".join(x) for x in product(CANONICAL_BASES, repeat=3)]


def extract_sequence_features(sequence: str) -> pd.DataFrame:
    """
    Extracts the exact 91 genomic and k-mer features required by the trained model:
    - 4 Mononucleotide Counts (A, T, G, C)
    - 4 Mononucleotide Frequencies (A, T, G, C)
    - GC Content (%)
    - Sequence Length
    - Ambiguous Base Count
    - 16 Dinucleotide (2-mer) Frequencies
    - 64 Trinucleotide (3-mer) Frequencies
    """
    seq = sequence.strip().upper()
    seq_len = len(seq)
    base_counts = Counter(seq)

    features = {}

    # 1. Mononucleotide Counts & Frequencies
    for base in CANONICAL_BASES:
        cnt = base_counts[base]
        features[f"{base}_Count"] = cnt
        features[f"{base}_Frequency"] = (cnt / seq_len) if seq_len > 0 else 0.0

    # 2. GC Content (%)
    gc_count = base_counts["G"] + base_counts["C"]
    features["GC_Content"] = (gc_count / seq_len * 100.0) if seq_len > 0 else 0.0

    # 3. Sequence Length & Ambiguity
    features["Sequence_Length"] = seq_len
    features["Ambiguous_Count"] = sum(1 for b in seq if b not in CANONICAL_BASES)

    # 4. Dinucleotide (2-mer) Frequencies
    valid_2mers = [
        seq[i:i+2] for i in range(seq_len - 1)
        if set(seq[i:i+2]).issubset(set(CANONICAL_BASES))
    ]
    cnt_2mers = Counter(valid_2mers)
    num_2mers = len(valid_2mers)
    for mer in TWO_MERS:
        features[f"2mer_{mer}"] = (cnt_2mers[mer] / num_2mers) if num_2mers > 0 else 0.0

    # 5. Trinucleotide (3-mer) Frequencies
    valid_3mers = [
        seq[i:i+3] for i in range(seq_len - 2)
        if set(seq[i:i+3]).issubset(set(CANONICAL_BASES))
    ]
    cnt_3mers = Counter(valid_3mers)
    num_3mers = len(valid_3mers)
    for mer in THREE_MERS:
        features[f"3mer_{mer}"] = (cnt_3mers[mer] / num_3mers) if num_3mers > 0 else 0.0

    return pd.DataFrame([features])


class DNAPatternPredictor:
    """
    Loads saved model artifacts and performs real-time DNA sequence classification.
    """
    def __init__(self, models_dir: Union[str, Path] = None):
        if models_dir is None:
            candidates = [
                Path(__file__).resolve().parent / "models_saved",
                Path(__file__).resolve().parent.parent.parent / "models_saved",
                Path.cwd() / "models_saved",
                Path(__file__).resolve().parent / ".." / "models_saved",
            ]
            for cand in candidates:
                cand = cand.resolve()
                if cand.exists() and (cand / "best_model.joblib").exists():
                    models_dir = cand
                    break
            if models_dir is None:
                models_dir = candidates[1].resolve()

        self.models_dir = Path(models_dir)

        model_path = self.models_dir / "best_model.joblib"
        encoder_path = self.models_dir / "label_encoder.joblib"

        if not model_path.exists() or not encoder_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {self.models_dir}. "
                f"Please run 'src/models/model_pipeline.py' first to train and save models."
            )

        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        self.classes = list(self.label_encoder.classes_)

    def predict(self, sequence: str) -> Dict[str, Any]:
        """
        Takes a raw DNA sequence string, extracts features, and runs inference.
        Returns prediction, probability distribution, and biological insights.
        """
        clean_seq = sequence.strip().upper()
        
        # Validation
        if len(clean_seq) < 3:
            return {"error": "DNA sequence must be at least 3 nucleotides long."}

        features_df = extract_sequence_features(clean_seq)

        # Run model inference
        pred_idx = self.model.predict(features_df)[0]
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]

        # Probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features_df)[0]
            prob_dict = {cls_name: round(float(prob) * 100, 2) for cls_name, prob in zip(self.classes, probs)}
            confidence = round(float(probs[pred_idx]) * 100, 2)
        else:
            prob_dict = {cls_name: ("100.0" if cls_name == pred_label else "0.0") for cls_name in self.classes}
            confidence = 100.0

        meta = SPLICE_SITE_METADATA.get(pred_label, {})

        return {
            "sequence_length": len(clean_seq),
            "gc_content": round(float(features_df["GC_Content"].iloc[0]), 2),
            "predicted_label": pred_label,
            "predicted_name": meta.get("name", pred_label),
            "confidence_percent": confidence,
            "class_probabilities": prob_dict,
            "biological_role": meta.get("role", ""),
            "biological_consensus": meta.get("biological_consensus", ""),
            "color": meta.get("color", "#333333")
        }


def predict_dna_sequence(sequence: str, models_dir: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Convenience function for external scripts or Streamlit apps to predict directly.
    """
    predictor = DNAPatternPredictor(models_dir=models_dir)
    return predictor.predict(sequence)


if __name__ == "__main__":
    print("=" * 70)
    print("DNA SPLICE-JUNCTION PREDICTION INFERENCE ENGINE")
    print("=" * 70)

    # Example test sequences (from splice.data)
    # Test EI (Exon-Intron) sample:
    ei_example = "CCAGCTGCATCACAGGAGGCCAGCGAGCAGGTCTGTTCCAAGGGCCTTCGAGCCAGTCTG"
    # Test IE (Intron-Exon) sample:
    ie_example = "TCGATCGGGACCTCCTCGCGCCCCAGGCGCCCCTGCCCGCCCCCGCCCCGCGCCCCGCCG"
    # Test N (Neither) sample:
    n_example = "AGCCCGGGATTACAGGCGTGAGCCACCGCGCCCGGCCCTAATTTTTGTATTTTTAGTAGA"

    predictor = DNAPatternPredictor()

    print("\n--- Testing Known Donor (EI) Example ---")
    res_ei = predictor.predict(ei_example)
    print(f"Prediction: {res_ei['predicted_label']} ({res_ei['predicted_name']})")
    print(f"Confidence: {res_ei['confidence_percent']}% | Probabilities: {res_ei['class_probabilities']}")

    print("\n--- Testing Known Acceptor (IE) Example ---")
    res_ie = predictor.predict(ie_example)
    print(f"Prediction: {res_ie['predicted_label']} ({res_ie['predicted_name']})")
    print(f"Confidence: {res_ie['confidence_percent']}% | Probabilities: {res_ie['class_probabilities']}")

    print("\n--- Testing Known Non-Splice (N) Example ---")
    res_n = predictor.predict(n_example)
    print(f"Prediction: {res_n['predicted_label']} ({res_n['predicted_name']})")
    print(f"Confidence: {res_n['confidence_percent']}% | Probabilities: {res_n['class_probabilities']}")
