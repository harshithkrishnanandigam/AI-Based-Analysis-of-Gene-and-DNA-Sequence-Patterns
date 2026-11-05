"""
================================================================================
AI-BASED ANALYSIS OF GENE AND DNA SEQUENCE PATTERNS
Interactive Genomic Pattern Discovery Dashboard
Course: Introduction to Biological Data (Amrita Vishwa Vidyapeetham, Coimbatore)
Department: Artificial Intelligence & Data Science (1st Year)

PERSON 4 DELIVERABLE: STREAMLIT WEB APPLICATION & DEMONSTRATION ENGINE
Developed in collaboration with Person 3 (AI/ML Lead: Harshith Krishna)
================================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Add current directory to path so predict.py is accessible
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from predict import predict_dna_sequence, SPLICE_SITE_METADATA

# Page configuration
st.set_page_config(
    page_title="Genomic AI: Splice Junction Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        margin-bottom: 15px;
    }
    .prediction-box {
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/dna-helix.png", width=80)
st.sidebar.title("🧬 DNA Pattern AI")
st.sidebar.markdown("**Course:** Intro to Biological Data  \n**Dept:** AI & Data Science (1st Year)  \n**Amrita Coimbatore**")
st.sidebar.markdown("---")

st.sidebar.subheader("👥 Project Team")
st.sidebar.markdown("""
- **Person 1 (Mowliya):** Dataset Acquisition
- **Person 2 (Kaushik Varma):** Preprocessing & Features
- **Person 3 (Harshith Krishna):** AI/ML Model Lead
- **Person 4 (Naren):** Visualization & Integration
""")
st.sidebar.markdown("---")

st.sidebar.subheader("🧪 Quick Test Sequences")
st.sidebar.caption("Click to load pre-verified benchmark sequences:")

SAMPLE_EI = "CCAGCTGCATCACAGGAGGCCAGCGAGCAGGTCTGTTCCAAGGGCCTTCGAGCCAGTCTG"
SAMPLE_IE = "TCGATCGGGACCTCCTCGCGCCCCAGGCGCCCCTGCCCGCCCCCGCCCCGCGCCCCGCCG"
SAMPLE_N  = "AGCCCGGGATTACAGGCGTGAGCCACCGCGCCCGGCCCTAATTTTTGTATTTTTAGTAGA"

if st.sidebar.button("Load Known Donor (EI) Sample"):
    st.session_state["dna_input"] = SAMPLE_EI

if st.sidebar.button("Load Known Acceptor (IE) Sample"):
    st.session_state["dna_input"] = SAMPLE_IE

if st.sidebar.button("Load Known Non-Splice (N) Sample"):
    st.session_state["dna_input"] = SAMPLE_N

# Main Header
st.markdown('<div class="main-title">🧬 AI-Based Analysis of Gene & DNA Sequence Patterns</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Primate Splice-Junction Discovery using Machine Learning & Explainable AI (XAI)</div>', unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Interactive Sequence Classifier",
    "📊 Model Benchmark & Evaluation",
    "🧠 Biological Interpretability (XAI)",
    "📖 Project Overview & Report"
])

# =============================================================================
# TAB 1: INTERACTIVE CLASSIFIER
# =============================================================================
with tab1:
    st.markdown("### Enter a 60-Base Pair DNA Sequence")
    st.write("The trained **XGBoost Classifier** evaluates 91 k-mer & compositional features to predict if the sequence contains an **Exon-to-Intron (EI) donor junction**, an **Intron-to-Exon (IE) acceptor junction**, or **Neither (N)**.")

    default_seq = st.session_state.get("dna_input", SAMPLE_EI)
    user_sequence = st.text_area(
        "DNA Nucleotide Sequence (A, T, G, C):",
        value=default_seq,
        height=100,
        help="Paste a raw FASTA sequence or uppercase/lowercase nucleotides."
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Sequence", type="primary", use_container_width=True)

    if analyze_clicked or user_sequence:
        clean_seq = user_sequence.strip().upper().replace("\n", "").replace(" ", "")
        
        # Validation
        valid_bases = set("ATGCN")
        if not set(clean_seq).issubset(valid_bases):
            st.error("⚠️ Invalid characters detected. DNA sequence must only contain A, T, G, C, or N.")
        elif len(clean_seq) < 3:
            st.warning("⚠️ DNA sequence must be at least 3 nucleotides long.")
        else:
            try:
                models_dir = current_dir / "models_saved"
                result = predict_dna_sequence(clean_seq, models_dir=models_dir)

                pred_label = result["predicted_label"]
                pred_name = result["predicted_name"]
                conf = result["confidence_percent"]
                probs = result["class_probabilities"]

                # Color definition
                badge_color = "#E11D48" if pred_label == "EI" else ("#2563EB" if pred_label == "IE" else "#059669")

                st.markdown("---")
                
                # Results Banner
                st.markdown(f"""
                <div style="background-color: {badge_color}; border-radius: 10px; padding: 18px; color: white; text-align: center; margin-bottom: 20px;">
                    <h2 style="margin: 0; color: white;">Predicted Class: {pred_label} — {pred_name}</h2>
                    <h4 style="margin: 5px 0 0 0; color: #F3F4F6;">Model Confidence: {conf:.1f}%</h4>
                </div>
                """, unsafe_allow_html=True)

                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sequence Length", f"{len(clean_seq)} bp")
                m2.metric("GC Content", f"{result['gc_content']}%")
                m3.metric("Top Probability", f"{conf}%")
                m4.metric("Model Architecture", "XGBoost (Optimized)")

                # Probability Breakdown & Nucleotide Composition
                col_probs, col_comp = st.columns(2)

                with col_probs:
                    st.subheader("🎯 Class Probabilities")
                    prob_df = pd.DataFrame({
                        "Biological Class": [
                            f"EI (Donor Splice Site): {probs.get('EI', 0)}%",
                            f"IE (Acceptor Splice Site): {probs.get('IE', 0)}%",
                            f"N (Non-Splice Region): {probs.get('N', 0)}%"
                        ],
                        "Probability (%)": [probs.get("EI", 0), probs.get("IE", 0), probs.get("N", 0)]
                    })
                    st.bar_chart(prob_df.set_index("Biological Class"), height=220)

                with col_comp:
                    st.subheader("🧬 Nucleotide Frequency")
                    counts = {b: clean_seq.count(b) for b in "ATGC"}
                    freq_df = pd.DataFrame({
                        "Base": list(counts.keys()),
                        "Count": list(counts.values()),
                        "Frequency (%)": [round(c / len(clean_seq) * 100, 1) for c in counts.values()]
                    })
                    st.dataframe(freq_df, use_container_width=True, hide_index=True)

                # Biological Explanation Card
                st.subheader("💡 Biological Consensus & Mechanism")
                st.info(f"""
                **Biological Role:**  
                {result['biological_role']}

                **Molecular Consensus:**  
                {result['biological_consensus']}
                """)

            except Exception as e:
                st.error(f"Error during inference: {str(e)}")

# =============================================================================
# TAB 2: MODEL BENCHMARKS
# =============================================================================
with tab2:
    st.header("📊 AI Model Performance & Benchmark Suite")
    st.write("We evaluated 6 diverse machine learning architectures using **Stratified 5-Fold Cross-Validation** and tested on an **unseen 20% holdout test set (636 samples)**.")

    benchmark_path = current_dir / "results" / "model_comparison.csv"
    if benchmark_path.exists():
        bench_df = pd.read_csv(benchmark_path)
        st.dataframe(bench_df.style.highlight_max(subset=["Test Accuracy (%)", "Macro F1-Score", "ROC-AUC (Macro OVR)"], color="#D1FAE5"), use_container_width=True)
    else:
        st.warning("Benchmark results file not found.")

    st.markdown("---")

    col_fig1, col_fig2 = st.columns(2)

    with col_fig1:
        st.subheader("🏆 Comparative Model Performance")
        img_path = current_dir / "results" / "figures" / "model_metrics_comparison.png"
        if img_path.exists():
            st.image(Image.open(img_path), caption="Benchmark: Test Accuracy, Balanced Accuracy, and Macro F1", use_container_width=True)

    with col_fig2:
        st.subheader("📈 Multi-Class ROC Curves (XGBoost)")
        roc_path = current_dir / "results" / "figures" / "roc_curves_best_model.png"
        if roc_path.exists():
            st.image(Image.open(roc_path), caption="One-vs-Rest ROC Curves: EI (Donor), IE (Acceptor), N (Neither)", use_container_width=True)

    st.subheader("🔍 Confusion Matrices Across All 6 Models")
    cm_path = current_dir / "results" / "figures" / "confusion_matrices.png"
    if cm_path.exists():
        st.image(Image.open(cm_path), caption="Normalized Confusion Matrices showing predicted vs actual biological classes", use_container_width=True)

# =============================================================================
# TAB 3: EXPLAINABLE AI (XAI)
# =============================================================================
with tab3:
    st.header("🧠 Explainable AI: Molecular Biological Interpretability")
    st.write("In biomedical AI, black-box predictions are not enough. We extracted tree feature importances across all 91 sequence features to determine which biological motifs drive the model's decisions.")

    feat_img_path = current_dir / "results" / "figures" / "top_features_importance.png"
    if feat_img_path.exists():
        st.image(Image.open(feat_img_path), caption="Top 20 DNA Features Governing Splice-Junction Discrimination", use_container_width=True)

    st.markdown("---")
    st.subheader("🔬 Biological Validation of Discovered Motifs")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        #### 1. Donor Site Motifs (EI)
        - **Discovered Features:** `3mer_GGT`, `3mer_GTA`
        - **Biological Reality:** Universally matches the canonical **`GT`** dinucleotide consensus at the 5' splice junction.
        """)

    with c2:
        st.markdown("""
        #### 2. Acceptor Site Motifs (IE)
        - **Discovered Features:** `3mer_AGG`, `3mer_CAG`
        - **Biological Reality:** Universally matches the canonical **`AG`** dinucleotide consensus (`NCAG|G`) at the 3' junction.
        """)

    with c3:
        st.markdown("""
        #### 3. Polypyrimidine Tract
        - **Discovered Features:** `2mer_TC`, `3mer_CTC`, `3mer_TCT`
        - **Biological Reality:** Matches the pyrimidine-rich tract ($C$ and $T$) upstream of the 3' splice site recognized by U2AF65.
        """)

# =============================================================================
# TAB 4: PROJECT OVERVIEW & DOCUMENTATION
# =============================================================================
with tab4:
    st.header("📖 Project Overview & Work Division")
    st.markdown("""
    ### **Project:** AI-Based Analysis of Gene and DNA Sequence Patterns
    **Course:** Introduction to Biological Data  
    **Institution:** Amrita Vishwa Vidyapeetham, Coimbatore  
    **Department:** Artificial Intelligence and Data Science (1st Year)  

    ---

    ### 👥 Team Work Division
    - **Person 1 (Mowliya - CB.AI.U4AID26021):** Dataset acquisition from UCI Machine Learning Repository and NCBI GenBank; documentation of raw FASTA/primate sequence formats.
    - **Person 2 (Kaushik Varma - CB.AI.U4AID26022):** Data cleaning, removal of duplicates/invalid IUPAC bases, extraction of 91 tabular features (nucleotide counts, frequencies, GC content, 16 dinucleotides, 64 trinucleotides), and train/test splitting.
    - **Person 3 (Harshith Krishna - CB.AI.U4AID26023):** AI/ML Model development: trained 6 algorithms, conducted 5-fold Stratified Cross-Validation, hyperparameter optimization, test holdout benchmarking, and biological motif interpretability.
    - **Person 4 (Naren - CB.AI.U4AID26024):** Visualizations, integration of preprocessing & model pipelines into this interactive Streamlit application, final project report, and slide presentation.

    ---
    ### 📥 Downloadable Deliverables
    """)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        if benchmark_path.exists():
            with open(benchmark_path, "rb") as f:
                st.download_button(
                    label="📄 Download Model Benchmark Table (CSV)",
                    data=f,
                    file_name="model_comparison.csv",
                    mime="text/csv"
                )

    with col_dl2:
        summary_json = current_dir / "results" / "metrics_summary.json"
        if summary_json.exists():
            with open(summary_json, "rb") as f:
                st.download_button(
                    label="📊 Download Metrics Summary (JSON)",
                    data=f,
                    file_name="metrics_summary.json",
                    mime="application/json"
                )
