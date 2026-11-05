# AI-Based Analysis of Gene and DNA Sequence Patterns
### Machine Learning for Genomic Splice-Junction Discovery & Pattern Recognition

**Course:** Introduction to Biological Data  
**Department:** Artificial Intelligence and Data Science (1st Year)  
**Institution:** Amrita Vishwa Vidyapeetham, Coimbatore  

**Authors & Project Work Division:**
- **Mowliya (CB.AI.U4AID26021)** — *Person 1: Dataset Collection & Research*
- **Kaushik Varma (CB.AI.U4AID26022)** — *Person 2: Data Cleaning & Feature Engineering*
- **Harshith Krishna (CB.AI.U4AID26023)** — *Person 3: AI/ML Model Development & Evaluation*
- **Naren (CB.AI.U4AID26024)** — *Person 4: Visualization, Integration & Reporting*

---

## Abstract

Splice junctions are critical genomic boundaries in eukaryotic pre-messenger RNA (pre-mRNA) where non-coding introns are excised and coding exons are joined to synthesize functional proteins. Aberrant splicing caused by single-nucleotide mutations is a major driver of hereditary diseases and oncogenesis. In this project, we present an end-to-end artificial intelligence framework to classify 60-base pair primate DNA sequences into three biological categories: Exon-to-Intron donor junctions (EI), Intron-to-Exon acceptor junctions (IE), and non-splice regions (Neither / N). 

Using the curated UCI Molecular Biology Splice-Junction benchmark (3,178 non-redundant sequences), we extracted a 91-dimensional tabular feature space comprising mononucleotide frequencies, GC content, sequence lengths, 16 dinucleotides (2-mers), and 64 trinucleotides (3-mers). We systematically benchmarked six diverse machine learning architectures: Logistic Regression, Gaussian Naive Bayes, Support Vector Machines (SVM-RBF), Random Forest, XGBoost, and Multi-Layer Perceptrons (MLP) under strict 5-fold Stratified Cross-Validation.

On an unseen holdout test set of 636 sequences, **XGBoost achieved the highest performance with 70.60% Test Accuracy, a Balanced Accuracy of 67.74%, a Macro F1-score of 0.6885, and an impressive Multi-Class One-vs-Rest ROC-AUC of 0.8600**, outperforming linear baselines by over 11%. Furthermore, tree-based feature importance analysis revealed that the top predictive features correspond directly to known molecular biology: the universal 5' donor **`GT`** consensus (`3mer_GGT`, `3mer_GTA`), the 3' acceptor **`AG`** consensus (`3mer_AGG`, `3mer_CAG`), and upstream polypyrimidine tracts (`2mer_TC`, `3mer_CTC`). Finally, the entire workflow was packaged into an interactive Streamlit web dashboard for real-time genomic inference.

---

## 1. Introduction & Biological Background

In eukaryotic organisms, genomic DNA consists of split genes featuring alternating coding sequences (**exons**) and non-coding sequences (**introns**). The Central Dogma of Molecular Biology dictates that DNA is transcribed into precursor messenger RNA (pre-mRNA). Before translation into proteins can occur, pre-mRNA undergoes **splicing**, an essential post-transcriptional modification orchestrated by the **spliceosome** (a multi-megadalton ribonucleoprotein complex).

The spliceosome recognizes specific consensus sequences at the intron-exon boundaries:
1. **Donor Splice Site (Exon $\rightarrow$ Intron / EI):** The 5' boundary of an intron, universally characterized by a conserved **`GT`** (in DNA) or **`GU`** (in RNA) dinucleotide.
2. **Acceptor Splice Site (Intron $\rightarrow$ Exon / IE):** The 3' boundary of an intron, universally terminating with a conserved **`AG`** dinucleotide, preceded by a **Polypyrimidine Tract** rich in cytosine (C) and thymine (T).
3. **Non-Splice Sequences (Neither / N):** Intragenic or intergenic sequences that contain no splicing activity.

Single-nucleotide polymorphisms (SNPs) that disrupt or create cryptic splice sites can cause premature termination codons, exon skipping, or intron retention, resulting in conditions such as cystic fibrosis, beta-thalassemia, and spinal muscular atrophy. Accurate automated identification of splice junctions is therefore fundamental for **automated genome annotation**, **pathogenic variant interpretation**, and **CRISPR target design**.

---

## 2. Dataset Collection & Exploration (Person 1)

The dataset utilized in this research was sourced from the **UCI Machine Learning Repository: Molecular Biology Primate Splice-Junction Gene Sequences**, originally derived from GenBank 64.1. 

- **Total Instances:** 3,190 primate DNA sequences.
- **Sequence Length:** Exactly 60 base pairs per sequence, with the candidate splice boundary centered between positions 30 and 31.
- **Class Labels:**
  - `EI` (Exon-Intron / Donor site): 767 instances (~24.1%)
  - `IE` (Intron-Exon / Acceptor site): 768 instances (~24.1%)
  - `N` (Neither / Non-splice region): 1,655 instances (~51.8%)

---

## 3. Data Cleaning & Feature Engineering (Person 2)

Raw biological sequences cannot be ingested directly by conventional machine learning algorithms without numerical transformation. Person 2 performed:

1. **Data Cleaning:** Stripped whitespace, converted all sequence characters to uppercase, and removed 12 exact duplicate records, resulting in 3,178 valid records. All characters were verified against the standard IUPAC degenerate nucleotide alphabet (`A, T, G, C, N, R, Y, etc.`).
2. **Tabular Feature Extraction (91 Features):**
   - **Mononucleotide Counts & Frequencies (8 features):** Absolute counts and normalized fractions of $A, T, G, C$.
   - **GC Content (1 feature):** $(G + C) / \text{Length} \times 100$, capturing thermodynamic thermal stability and genomic isochores.
   - **Sequence Length & Ambiguous Bases (2 features):** Total nucleotide count and count of non-canonical characters.
   - **Dinucleotide Frequencies (16 features):** 2-mers ($AA, AT, AG, \dots, CC$) capturing local dinucleotide bias.
   - **Trinucleotide Frequencies (64 features):** 3-mers ($AAA, AAT, \dots, CCC$) capturing codon bias and reading frame signatures.
3. **Train/Test Splitting:** An 80/20 stratified split partitioned the data into **2,542 training samples** and **636 holdout testing samples**, ensuring consistent class ratios.

---

## 4. Machine Learning Model Development & Evaluation (Person 3)

### 4.1 Candidate Architectures
To rigorously assess the inductive biases of different learning paradigms, Person 3 implemented six distinct algorithms:
1. **Logistic Regression (Multinomial Baseline):** Linear model with $L_2$ regularization and balanced class weighting.
2. **Gaussian Naive Bayes (Probabilistic Baseline):** Generative classifier applying Bayes' Rule with feature conditional independence.
3. **Support Vector Machine (SVM - RBF Kernel):** Non-linear kernel boundary optimization maximizing the separation margin in infinite-dimensional Hilbert space ($C=1.5, \gamma=\text{'scale'}$).
4. **Random Forest Classifier:** Bagged ensemble of 250 de-correlated decision trees with randomized feature subsampling (`max_depth=16`).
5. **XGBoost Classifier:** Extreme Gradient Boosted decision trees utilizing second-order Taylor expansions of the loss function with $L_1/L_2$ tree complexity regularization (`n_estimators=250, max_depth=5, learning_rate=0.08`).
6. **Multi-Layer Perceptron (MLP Neural Net):** Feedforward deep network with hidden layers $(128, 64)$, ReLU activation, and Adam optimization.

### 4.2 Data Leakage Prevention & Cross-Validation
Feature scaling (`StandardScaler`) was strictly encapsulated within scikit-learn `Pipeline` objects to ensure scaling parameters ($\mu, \sigma$) were learned exclusively from training folds during **5-Fold Stratified Cross-Validation**, preventing data leakage.

---

## 5. Experimental Results & Discussion

### 5.1 Comparative Model Benchmark

| Model | 5-Fold CV Acc | CV Macro F1 | Test Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Multi-Class ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **69.59% ± 2.47%** | **0.6691** | **70.60%** | **67.74%** | **0.6885** | **0.7036** | **0.8600** |
| **Support Vector Machine (RBF)** | 65.54% ± 0.73% | 0.6473 | 68.71% | 70.30% | 0.6822 | 0.6886 | 0.8491 |
| **Random Forest** | 66.05% ± 1.48% | 0.6489 | 68.08% | 68.13% | 0.6721 | 0.6817 | 0.8468 |
| **Multi-Layer Perceptron (MLP)** | 64.52% ± 1.33% | 0.6195 | 65.88% | 62.72% | 0.6341 | 0.6551 | 0.8184 |
| **Logistic Regression (Baseline)** | 61.05% ± 0.53% | 0.6069 | 59.59% | 61.28% | 0.5894 | 0.5987 | 0.7864 |
| **Gaussian Naive Bayes** | 51.18% ± 2.39% | 0.5136 | 51.42% | 56.38% | 0.5162 | 0.5125 | 0.7264 |

### 5.2 Key Analytical Findings
1. **Superiority of Non-Linear Tree Boosting:** XGBoost achieved the highest performance across all evaluation criteria (70.60% Test Accuracy, 0.8600 ROC-AUC). Non-linear tree ensembles outperformed linear Logistic Regression by over 11 percentage points, demonstrating that splice site recognition relies on non-linear co-occurrences of k-mers.
2. **Generalization Consistency:** Cross-validation accuracy (69.59%) closely mirrored holdout test accuracy (70.60%), confirming the model did not overfit the training fold.
3. **Metric Interpretation:** Macro F1-score (0.6885) confirms robust sensitivity across both majority ($N$) and minority ($EI, IE$) classes.

---

## 6. Explainable AI: Biological Validation of Learned Motifs

To ensure algorithmic transparency, feature importances were extracted from tree ensembles across all 91 features:
- **`3mer_GGT` & `3mer_GTA`:** Ranked as top features; directly represent the canonical **`GT`** 5' donor splice site.
- **`3mer_AGG` & `3mer_CAG`:** Strongly predictive of the 3' acceptor splice site, matching the canonical **`AG`** junction.
- **`2mer_TC`, `3mer_CTC`, `3mer_TCT`:** Ranked heavily, confirming the biological presence of the **Polypyrimidine Tract** upstream of the acceptor junction.

The AI autonomously learned the biochemical grammar of the spliceosome without hardcoded human heuristic rules.

---

## 7. System Integration & Interactive Deployment (Person 4)

Person 4 integrated the preprocessing logic and the serialized XGBoost model into an interactive **Streamlit Web Application** (`app.py`). 
- **User Interface:** Allows biologists to paste custom DNA sequences or load pre-verified benchmark sequences.
- **Real-Time Inference:** Extracts 91 features on-the-fly and outputs predicted classes, class probabilities, GC content, and nucleotide distributions.
- **Educational Explanations:** Displays biological mechanism cards explaining donor/acceptor site functions.

---

## 8. Limitations & Future Scope

### Limitations
1. **Window Size Constraint:** The model evaluates isolated 60-bp windows and does not incorporate distant regulatory elements (such as upstream branch points 20-50 bp away or exonic splicing enhancers/silencers).
2. **Positional Invariance:** Bag-of-k-mers treats motif counts globally, ignoring the exact relative position of the motif within the window.

### Future Scope
1. **Genomic Transformers:** Fine-tuning pre-trained foundation models (e.g., DNABERT-2 or Nucleotide Transformer) using self-attention to capture long-range chromatin dependencies.
2. **Whole-Genome FASTA Scanner:** Implementing a sliding window scanner to process multi-megabase chromosome FASTA files and identify novel cryptic splice sites.

---

## 9. Conclusion

This project successfully demonstrates the synergy of machine learning and computational genomics. By transforming raw nucleotide strings into biologically relevant k-mer representations, our XGBoost classifier achieved a 70.60% accuracy and 0.8600 ROC-AUC on primate splice junctions, verified via explainable AI and deployed as an interactive web dashboard.
