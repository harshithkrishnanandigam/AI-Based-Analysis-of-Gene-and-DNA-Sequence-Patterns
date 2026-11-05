# Master Handover & Action Guide for Person 4 (Visualization, Integration & Reporting)

**Recipient:** Naren (`CB.AI.U4AID26024`) — Person 4  
**From:** Harshith Krishna (`CB.AI.U4AID26023`) — Person 3 (AI/ML Lead)  
**Project:** AI-Based Analysis of Gene and DNA Sequence Patterns  
**Course:** Introduction to Biological Data, Amrita Vishwa Vidyapeetham, Coimbatore  

---

## 👋 Welcome Naren! Everything is Prepared for You

Person 1 (Mowliya) collected the dataset, Person 2 (Kaushik Varma) cleaned and engineered the 91 k-mer features, and Person 3 (Harshith Krishna) trained 6 machine learning models, performed 5-fold cross-validation, optimized hyperparameters, extracted biological feature importances, and built the prediction engine.

Your role as **Person 4** is **Visualization, Integration & Reporting**. To ensure our team gets the **highest marks** in the course and impresses our lecturer, I have created a **turnkey package** for you, including a fully functioning **interactive Streamlit Web Application**, 4 publication-quality 300 DPI figures, benchmark tables, and presentation materials.

---

## 📦 1. What is Inside this Handover Package?

| File / Folder | What It Is | How You Use It |
| :--- | :--- | :--- |
| **`app.py`** | Interactive Streamlit Web Application | Run it live during the project presentation to show our AI in action! |
| **`predict.py`** | Real-time DNA sequence prediction engine | The core backend engine powering `app.py`. |
| **`models_saved/`** | Serialized trained models (`best_model.joblib`, `all_models.joblib`, `label_encoder.joblib`) | The trained AI brains loaded by `app.py`. |
| **`results/model_comparison.csv`** | Official 6-model benchmark table | Embed directly into your final report and PowerPoint slides. |
| **`results/metrics_summary.json`** | Consolidated results JSON | For programmatic reporting or data download in the app. |
| **`results/feature_importances.csv`** | Ranked list of all 91 features | Table of the top biological motifs. |
| **`results/figures/model_metrics_comparison.png`** | Grouped bar chart comparing Test Acc, Balanced Acc, and Macro F1 | Slide / Report Figure 1. |
| **`results/figures/confusion_matrices.png`** | 2x3 grid of confusion matrices across all 6 models | Slide / Report Figure 2. |
| **`results/figures/roc_curves_best_model.png`** | Multi-class One-vs-Rest ROC curves (AUC = 0.860) | Slide / Report Figure 3. |
| **`results/figures/top_features_importance.png`** | Top 20 biological sequence motifs bar chart | Slide / Report Figure 4. |

---

## 🚀 2. How to Run the Interactive Streamlit Web App

Open your command prompt or terminal inside this folder and run:

```bash
streamlit run app.py
```

This will automatically open the dashboard in your web browser at `http://localhost:8501`.

### 💡 What the App Does (Your Presentation Demo):
1. **Interactive Sequence Analysis:** The user can paste any 60-bp DNA sequence, or click one of the 3 pre-loaded sidebar test buttons:
   - *Load Known Donor (EI) Sample*
   - *Load Known Acceptor (IE) Sample*
   - *Load Known Non-Splice (N) Sample*
2. **Instant Biological Feedback:** Displays the predicted class, confidence percentage (e.g. 82.9%), nucleotide counts, GC content, and probability distribution.
3. **Biological Consensus Card:** Explains the molecular mechanism (e.g., canonical `GT` donor site, `AG` acceptor site, or polypyrimidine tract).
4. **Model Benchmark Tab:** Displays the full 6-model comparison table and high-res performance plots.
5. **Explainable AI (XAI) Tab:** Shows the top 20 biological k-mers driving predictions.
6. **Download Buttons:** Allows anyone to download the CSV benchmark and JSON results directly.

---

## 🎯 3. Live Presentation Script (How to Impress the Teacher in 3 Minutes)

When it is your turn to speak during the project viva/presentation, follow this exact structure:

### Step 1: Introduce Integration & The Streamlit Dashboard (30 Seconds)
> *"Respected Professor, as Person 4, my responsibility was to integrate Person 1's dataset, Person 2's preprocessed features, and Person 3's trained AI models into a unified, user-facing application. Rather than just showing static code, we built an end-to-end interactive web application using Streamlit."*

### Step 2: Demonstrate Real-Time Classification (60 Seconds)
*(Switch to browser showing `app.py`)*
> *"Here, a geneticist or researcher can input any raw 60-base pair DNA sequence. Let's test a candidate donor splice site by clicking 'Load Known Donor Sample'. In real-time, our backend extracts the 91 mononucleotide, dinucleotide, and trinucleotide frequencies and feeds them into our optimized XGBoost model.*  
> *As you can see, the model classifies this sequence as **EI (Exon-to-Intron Boundary / Donor Splice Site)** with high confidence. The dashboard also breaks down the nucleotide composition, GC content, and explains the biological consensus."*

### Step 3: Present the Comparative Benchmark (45 Seconds)
*(Click on Tab 2: Model Benchmark & Evaluation)*
> *"We benchmarked 6 diverse algorithms. Our baseline Logistic Regression achieved 59.59% accuracy. However, by capturing non-linear k-mer interactions, our **XGBoost Classifier achieved 70.60% Test Accuracy, a Macro F1-score of 0.6885, and a Multi-Class ROC-AUC of 0.8600**, significantly outperforming both linear baselines and deep neural networks."*

### Step 4: Highlight Explainable AI & Biological Discovery (45 Seconds)
*(Click on Tab 3: Biological Interpretability)*
> *"Most importantly, we didn't treat AI as a black box. Our feature importance analysis revealed that the top predictive features are trinucleotides containing `GT` (the universal 5' donor consensus), `AG` (the universal 3' acceptor consensus), and pyrimidine-rich dinucleotides like `TC`. The AI autonomously discovered the textbook rules of pre-mRNA splicing!"*

---

## 📊 4. How to Organize the Final Presentation Slides (PPT)

Here is the exact slide-by-slide structure you should prepare for the final presentation:

- **Slide 1: Title Slide**
  - Project Title: *AI-Based Analysis of Gene and DNA Sequence Patterns*
  - Subtitle: *Primate Splice-Junction Discovery using Machine Learning & Explainable AI*
  - Team Members: Mowliya (Person 1), Kaushik Varma (Person 2), Harshith Krishna (Person 3), Naren (Person 4).
  - Department: AI & Data Science, Amrita Vishwa Vidyapeetham, Coimbatore.
- **Slide 2: Biological Motivation & Problem Statement**
  - Explain pre-mRNA splicing: Exons vs Introns.
  - Three target classes: `EI` (Donor), `IE` (Acceptor), `N` (Neither).
  - Importance: Preventing frameshift mutations, genome annotation, disease understanding.
- **Slide 3: End-to-End Pipeline Architecture**
  - Dataset Acquisition (UCI Splice Junction, 3,178 sequences) $\rightarrow$ Cleaning & 91 k-mer features $\rightarrow$ Model Suite $\rightarrow$ Web App.
- **Slide 4: Machine Learning Model Comparison**
  - Insert Table from `results/model_comparison.csv`.
  - Embed `results/figures/model_metrics_comparison.png`.
  - Highlight XGBoost as the top-performing model.
- **Slide 5: Error Analysis & Generalization**
  - Embed `results/figures/confusion_matrices.png` and `results/figures/roc_curves_best_model.png`.
  - Explain that Stratified 5-Fold CV ensured no overfitting (CV: 69.59% vs Test: 70.60%).
- **Slide 6: Explainable AI & Biological Discovery**
  - Embed `results/figures/top_features_importance.png`.
  - Explain the biological discovery of `GT`, `AG`, and polypyrimidine `TC` tracts.
- **Slide 7: Live Demonstration & Streamlit App**
  - Screenshot of the Streamlit dashboard (`app.py`).
  - Announce live demo.
- **Slide 8: Limitations & Future Scope**
  - Current limitation: Fixed 60-bp window; ignores absolute positional indices and upstream branch point sites.
  - Future scope: Fine-tuning Genomic Transformers (DNABERT, Nucleotide Transformer) on whole-genome FASTA data.
- **Slide 9: Conclusion & Summary**
  - Summary of contributions by all 4 members.
  - Questions & Answers.

---

## 📝 5. How to Structure the Final Written Report

For your written report submission (Word / PDF), follow this structure:

1. **Title & Abstract (200 words):** Summary of the biological problem, dataset, 6 ML models, best model results (70.60% accuracy, 0.8600 ROC-AUC), and the Streamlit deployment.
2. **Introduction:** Biology of splicing, introns, exons, spliceosomes, and the role of computational biology.
3. **Dataset & Preprocessing (Person 1 & 2):** UCI Primate Splice-Junction database; extraction of 91 features (mononucleotides, GC content, 16 dinucleotides, 64 trinucleotides).
4. **Machine Learning Methodology (Person 3):**
   - 6 candidate models (Logistic Regression, Naive Bayes, SVM-RBF, Random Forest, XGBoost, MLP).
   - Zero-leakage scikit-learn Pipelines.
   - Stratified 5-Fold Cross-Validation.
   - Hyperparameter optimization.
5. **Results & Benchmark Evaluation (Person 3 & 4):**
   - Insert benchmark table.
   - Insert and discuss Figures: `model_metrics_comparison.png`, `confusion_matrices.png`, `roc_curves_best_model.png`.
6. **Explainable AI & Biological Validation (Person 3 & 4):**
   - Insert Figure: `top_features_importance.png`.
   - Detailed discussion connecting `GT`, `AG`, and pyrimidine motifs to literature.
7. **System Integration & Web Interface (Person 4):**
   - Architecture of the Streamlit application (`app.py`).
   - Screenshots of live prediction on sample sequences.
8. **Limitations & Future Scope:**
   - Bag-of-k-mers vs positional transformers.
9. **Conclusion & References:**
   - Formal bibliographic references from the proposal PPT.

---

## 🐙 6. GitHub Repository Setup Guide

To make our GitHub repository look like a professional AI project:

1. **Repository Name:** `dna-splice-junction-pattern-ai`
2. **Directory Layout:**
   ```
   ├── data/
   │   ├── raw/
   │   └── processed/
   ├── src/
   │   ├── preprocessing/
   │   │   └── DNA_Preprocessing_Project.py
   │   └── models/
   │       ├── model_pipeline.py
   │       └── predict.py
   ├── models_saved/
   ├── results/
   │   ├── figures/
   │   └── model_comparison.csv
   ├── notebooks/
   │   └── Person_3_Model_Development.ipynb
   ├── app.py
   ├── requirements.txt
   └── README.md
   ```
3. **`requirements.txt`:**
   ```text
   numpy>=1.24.0
   pandas>=1.5.0
   scikit-learn>=1.3.0
   scipy>=1.10.0
   xgboost>=2.0.0
   matplotlib>=3.7.0
   seaborn>=0.12.0
   joblib>=1.3.0
   streamlit>=1.28.0
   Pillow>=9.0.0
   ```

---

*If you have any questions or need help running the app, coordinate with Harshith (Person 3).*
