
# DNA SEQUENCE PREPROCESSING PROJECT
# Person 2 - Data Cleaning and Feature Engineering

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import product
from sklearn.model_selection import train_test_split

# ==========================================
# STAGE 1: LOAD DATASET
# ==========================================

file_path = r"C:\Users\Mowliya\Videos\ibd project\PYTHON codes for ibd final project\molecular+biology+splice+junction+gene+sequences\splice.data"

output_folder = Path("output")
output_folder.mkdir(exist_ok=True)

df = pd.read_csv(
    file_path,
    header=None,
    names=["Label", "ID", "Sequence"],
    skipinitialspace=True
)

print("\nDATASET LOADED")
print("--------------------------------")
print("Original dataset shape:", df.shape)
print(df.head())

# ==========================================
# STAGE 2: INITIAL DATA CLEANING
# ==========================================

df["Label"] = df["Label"].str.strip()
df["ID"] = df["ID"].str.strip()
df["Sequence"] = df["Sequence"].str.strip().str.upper()

print("\nMISSING VALUES")
print("--------------------------------")
print(df.isnull().sum())

# Check duplicate records
duplicate_count = df.duplicated().sum()

print("\nDUPLICATE RECORDS")
print("--------------------------------")
print("Duplicates found:", duplicate_count)

# Remove exact duplicate records
df = df.drop_duplicates().copy()

print("Records after removing duplicates:", len(df))

# Check valid IUPAC DNA characters
valid_iupac = set("ATGCRYSWKMBDHVN")

invalid_sequences = df[
    ~df["Sequence"].apply(
        lambda seq: set(seq).issubset(valid_iupac)
    )
]

print("\nDNA CHARACTER VALIDATION")
print("--------------------------------")
print("Sequences with unrecognized characters:",
      len(invalid_sequences))

# Check sequence lengths
df["Sequence_Length"] = df["Sequence"].str.len()

print("\nSEQUENCE LENGTH")
print("--------------------------------")
print(df["Sequence_Length"].describe())

# ==========================================
# STAGE 3: FEATURE ENGINEERING
# ==========================================

print("\nFEATURE ENGINEERING")
print("--------------------------------")

# Generate all possible canonical DNA k-mers
bases = "ATGC"

two_mers = [
    "".join(x) for x in product(bases, repeat=2)
]

three_mers = [
    "".join(x) for x in product(bases, repeat=3)
]

# Function to extract features from one DNA sequence
def extract_features(sequence):

    features = {}

    sequence_length = len(sequence)

    # Count canonical bases
    base_counts = Counter(sequence)

    # Nucleotide frequencies
    for base in bases:
        count = base_counts[base]

        features[base + "_Count"] = count

        features[base + "_Frequency"] = (
            count / sequence_length
            if sequence_length > 0 else 0
        )

    # GC content
    gc_count = base_counts["G"] + base_counts["C"]

    features["GC_Content"] = (
        gc_count / sequence_length * 100
        if sequence_length > 0 else 0
    )

    # Sequence length
    features["Sequence_Length"] = sequence_length

    # Number of ambiguous bases
    ambiguous_count = sum(
        1 for base in sequence if base not in bases
    )

    features["Ambiguous_Count"] = ambiguous_count

    # 2-mer frequencies
    valid_2mers = [
        sequence[i:i+2]
        for i in range(len(sequence) - 1)
        if set(sequence[i:i+2]).issubset(set(bases))
    ]

    count_2mers = Counter(valid_2mers)

    for mer in two_mers:
        features["2mer_" + mer] = (
            count_2mers[mer] / len(valid_2mers)
            if valid_2mers else 0
        )

    # 3-mer frequencies
    valid_3mers = [
        sequence[i:i+3]
        for i in range(len(sequence) - 2)
        if set(sequence[i:i+3]).issubset(set(bases))
    ]

    count_3mers = Counter(valid_3mers)

    for mer in three_mers:
        features["3mer_" + mer] = (
            count_3mers[mer] / len(valid_3mers)
            if valid_3mers else 0
        )

    return features


# Extract features for every sequence
feature_list = []

for sequence in df["Sequence"]:
    feature_list.append(extract_features(sequence))

features_df = pd.DataFrame(feature_list)

# Keep ID and Label with the extracted features
features_df.insert(0, "ID", df["ID"].values)
features_df.insert(1, "Label", df["Label"].values)

print("\nFEATURE EXTRACTION COMPLETED")
print("--------------------------------")
print("Number of records:", len(features_df))
print("Number of columns:", len(features_df.columns))

print("\nFIRST FIVE FEATURE RECORDS")
print("--------------------------------")
print(features_df.head())

# ==========================================
# STAGE 4: FEATURE ANALYSIS
# ==========================================

print("\nGC CONTENT STATISTICS")
print("--------------------------------")
print(features_df["GC_Content"].describe())

print("\nAVERAGE NUCLEOTIDE FREQUENCIES")
print("--------------------------------")

for base in bases:
    print(
        base,
        ":",
        round(features_df[base + "_Frequency"].mean(), 4)
    )

print("\nCLASS DISTRIBUTION AFTER CLEANING")
print("--------------------------------")
print(df["Label"].value_counts())

# ==========================================
# STAGE 5: PREPARE TRAINING AND TESTING DATA
# ==========================================

print("\nPREPARING TRAINING AND TESTING DATA")
print("--------------------------------")

X = features_df.drop(columns=["ID", "Label"])
y = features_df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTRAINING CLASS DISTRIBUTION")
print("--------------------------------")
print(y_train.value_counts())

print("\nTESTING CLASS DISTRIBUTION")
print("--------------------------------")
print(y_test.value_counts())

# ==========================================
# STAGE 6: SAVE OUTPUT FILES
# ==========================================

df.to_csv(
    output_folder / "cleaned_splice_data.csv",
    index=False
)

features_df.to_csv(
    output_folder / "engineered_features.csv",
    index=False
)

X_train.to_csv(
    output_folder / "X_train.csv",
    index=False
)

X_test.to_csv(
    output_folder / "X_test.csv",
    index=False
)

y_train.to_csv(
    output_folder / "y_train.csv",
    index=False
)

y_test.to_csv(
    output_folder / "y_test.csv",
    index=False
)

print("\nFILES SAVED SUCCESSFULLY")
print("--------------------------------")

for file in output_folder.iterdir():
    print(file.name)

print("\n==========================================")
print("DATA CLEANING AND FEATURE ENGINEERING DONE")
print("==========================================")
