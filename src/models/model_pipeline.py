"""
================================================================================
AI-BASED ANALYSIS OF GENE AND DNA SEQUENCE PATTERNS
Course: Introduction to Biological Data (Amrita Vishwa Vidyapeetham, Coimbatore)
Department: Artificial Intelligence and Data Science (1st Year)

PERSON 3 DELIVERABLE: AI/ML MODEL DEVELOPMENT & EVALUATION PIPELINE
Author: Harshith Krishna (CB.AI.U4AID26023)
Team Members:
  - Person 1: Mowliya (CB.AI.U4AID26021) - Dataset Collection & Research
  - Person 2: Kaushik Varma (CB.AI.U4AID26022) - Data Cleaning & Feature Engineering
  - Person 3: Harshith Krishna (CB.AI.U4AID26023) - AI/ML Model Development
  - Person 4: Naren (CB.AI.U4AID26024) - Visualization, Integration & Reporting
================================================================================
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("DNAModelingPipeline")

# Set random seed for complete reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class DNAPatternModelingPipeline:
    """
    End-to-end Machine Learning pipeline for DNA splice-junction pattern classification.
    Implements 6 distinct algorithms, stratified cross-validation, hyperparameter tuning,
    model evaluation, feature importance extraction, and artifact serialization.
    """

    def __init__(self, data_dir: Path, output_dir: Path, models_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        self.figures_dir = self.output_dir / "figures"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.label_encoder = LabelEncoder()
        self.models: Dict[str, Any] = {}
        self.results: List[Dict[str, Any]] = []
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.feature_names: List[str] = []

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Loads preprocessed feature sets and labels produced by Person 2.
        Applies consistent LabelEncoding to multi-class target labels (EI, IE, N).
        """
        logger.info("Loading training and testing datasets...")
        x_train_path = self.data_dir / "X_train.csv"
        x_test_path = self.data_dir / "X_test.csv"
        y_train_path = self.data_dir / "y_train.csv"
        y_test_path = self.data_dir / "y_test.csv"

        X_train = pd.read_csv(x_train_path)
        X_test = pd.read_csv(x_test_path)
        y_train_raw = pd.read_csv(y_train_path).iloc[:, 0].str.strip()
        y_test_raw = pd.read_csv(y_test_path).iloc[:, 0].str.strip()

        self.feature_names = list(X_train.columns)

        # Fit LabelEncoder on training labels and transform both train & test
        y_train = self.label_encoder.fit_transform(y_train_raw)
        y_test = self.label_encoder.transform(y_test_raw)

        logger.info(f"Dataset successfully loaded.")
        logger.info(f"Train Shape: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"Test Shape:  X={X_test.shape}, y={y_test.shape}")
        logger.info(f"Classes: {list(self.label_encoder.classes_)} -> {list(range(len(self.label_encoder.classes_)))}")

        # Save LabelEncoder
        joblib.dump(self.label_encoder, self.models_dir / "label_encoder.joblib")
        return X_train, X_test, y_train, y_test

    def get_candidate_models(self) -> Dict[str, Any]:
        """
        Defines a comprehensive suite of 6 machine learning models:
        - 2 Baselines (Logistic Regression, Gaussian Naive Bayes)
        - 1 Kernel Method (Support Vector Classifier - RBF)
        - 2 Ensemble Methods (Random Forest, XGBoost)
        - 1 Neural Architecture (Multi-Layer Perceptron)
        
        Pipelines with StandardScaler are used for distance/gradient-sensitive models
        to avoid data leakage during cross-validation.
        """
        models = {
            "Logistic Regression (Baseline)": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED
                ))
            ]),
            "Gaussian Naive Bayes": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB())
            ]),
            "Support Vector Machine (RBF)": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(
                    kernel="rbf",
                    C=1.5,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_SEED
                ))
            ]),
            "Random Forest": RandomForestClassifier(
                n_estimators=250,
                max_depth=16,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            "XGBoost Classifier": XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="mlogloss",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            "Multi-Layer Perceptron (MLP)": Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    max_iter=400,
                    early_stopping=True,
                    random_state=RANDOM_SEED
                ))
            ])
        }
        return models

    def run_cross_validation(self, models: Dict[str, Any], X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Performs 5-Fold Stratified Cross-Validation on the training set.
        Guarantees that each fold maintains identical class proportions (EI, IE, N),
        verifying generalization capability and guard against overfitting.
        """
        logger.info("Executing 5-Fold Stratified Cross-Validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_summary = {}

        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "f1_weighted": "f1_weighted"
        }

        for name, model in models.items():
            start_time = time.time()
            cv_scores = cross_validate(
                model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
            )
            elapsed = time.time() - start_time

            mean_acc = cv_scores["test_accuracy"].mean()
            std_acc = cv_scores["test_accuracy"].std()
            mean_f1 = cv_scores["test_f1_macro"].mean()
            std_f1 = cv_scores["test_f1_macro"].std()

            cv_summary[name] = {
                "cv_acc_mean": float(mean_acc),
                "cv_acc_std": float(std_acc),
                "cv_f1_mean": float(mean_f1),
                "cv_f1_std": float(std_f1),
                "cv_time_sec": float(elapsed)
            }
            logger.info(f"[{name}] 5-Fold CV Acc: {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%) | Macro F1: {mean_f1:.4f} ({elapsed:.2f}s)")

        return cv_summary

    def train_and_evaluate_holdout(
        self,
        models: Dict[str, Any],
        cv_summary: Dict[str, Dict[str, float]],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Fits each model on full X_train and evaluates rigorously on unseen holdout test set (X_test).
        Computes Accuracy, Balanced Accuracy, Precision, Recall, F1, and Multi-Class ROC-AUC.
        """
        logger.info("Training models on full training set and evaluating on unseen test set...")
        results_list = []
        predictions_dict = {}
        probabilities_dict = {}

        classes = list(self.label_encoder.classes_)

        for name, model in models.items():
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train

            # Predictions
            y_pred = model.predict(X_test)
            predictions_dict[name] = y_pred

            # Probabilities for ROC-AUC
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)
            else:
                y_prob = None
            probabilities_dict[name] = y_prob

            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
            prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # Multi-class ROC AUC (One-vs-Rest)
            if y_prob is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
                except Exception:
                    roc_auc = np.nan
            else:
                roc_auc = np.nan

            res = {
                "Model": name,
                "CV Accuracy Mean (%)": round(cv_summary[name]["cv_acc_mean"] * 100, 2),
                "CV Accuracy Std (%)": round(cv_summary[name]["cv_acc_std"] * 100, 2),
                "CV Macro F1": round(cv_summary[name]["cv_f1_mean"], 4),
                "Test Accuracy (%)": round(acc * 100, 2),
                "Balanced Accuracy (%)": round(bal_acc * 100, 2),
                "Macro Precision": round(prec_macro, 4),
                "Macro Recall": round(rec_macro, 4),
                "Macro F1-Score": round(f1_macro, 4),
                "Weighted F1-Score": round(f1_weighted, 4),
                "ROC-AUC (Macro OVR)": round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A",
                "Training Time (s)": round(train_time, 2)
            }
            results_list.append(res)
            self.models[name] = model

            # Log per-class breakdown
            report = classification_report(y_test, y_pred, target_names=classes)
            logger.info(f"\n--- {name} Test Classification Report ---\n{report}")

        results_df = pd.DataFrame(results_list).sort_values(by="Macro F1-Score", ascending=False).reset_index(drop=True)
        logger.info(f"\nFinal Comparative Benchmark Table:\n{results_df.to_string()}")

        # Save comparison table
        results_df.to_csv(self.output_dir / "model_comparison.csv", index=False)

        # Determine best model based on Test Macro F1-Score
        self.best_model_name = results_df.iloc[0]["Model"]
        self.best_model = self.models[self.best_model_name]
        logger.info(f">>> BEST PERFORMING MODEL: {self.best_model_name} (Test Macro F1 = {results_df.iloc[0]['Macro F1-Score']})")

        # Save all trained models and the best model
        joblib.dump(self.best_model, self.models_dir / "best_model.joblib")
        joblib.dump(self.models, self.models_dir / "all_models.joblib")
        logger.info(f"Saved trained models to {self.models_dir}")

        # Store predictions and probabilities for visualization
        self.predictions_dict = predictions_dict
        self.probabilities_dict = probabilities_dict
        self.results_df = results_df

        return results_df

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Any:
        """
        Performs Grid Search Cross-Validation to fine-tune the top ensemble model (Random Forest).
        Optimizes n_estimators, max_depth, min_samples_split, and criterion.
        """
        logger.info(f"Optimizing hyperparameters for Random Forest...")
        param_grid = {
            "n_estimators": [150, 250, 350],
            "max_depth": [12, 16, 20],
            "min_samples_split": [2, 4, 6],
            "criterion": ["gini", "entropy"]
        }

        rf_base = RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best Hyperparameters Found: {grid_search.best_params_}")
        logger.info(f"Best CV Macro F1: {grid_search.best_score_:.4f}")

        # Save tuned model
        joblib.dump(grid_search.best_estimator_, self.models_dir / "tuned_random_forest.joblib")
        return grid_search.best_estimator_

    def extract_feature_importance(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts feature importances from Random Forest and XGBoost.
        Identifies biological significance: e.g. donor (GT) and acceptor (AG) motif signals,
        dinucleotides, and codon patterns.
        """
        logger.info("Extracting feature importances for biological interpretability...")
        rf_model = self.models.get("Random Forest")
        xgb_model = self.models.get("XGBoost Classifier")

        importances_df = pd.DataFrame({"Feature": self.feature_names})

        if rf_model is not None:
            importances_df["RandomForest_Importance"] = rf_model.feature_importances_

        if xgb_model is not None:
            importances_df["XGBoost_Importance"] = xgb_model.feature_importances_

        # Calculate average importance
        cols = [c for c in importances_df.columns if "Importance" in c]
        importances_df["Mean_Importance"] = importances_df[cols].mean(axis=1)
        importances_df = importances_df.sort_values(by="Mean_Importance", ascending=False).reset_index(drop=True)

        # Categorize feature type (GC_Content, Count, Frequency, 2-mer, 3-mer)
        def categorize(name: str) -> str:
            if "3mer" in name:
                return "Trinucleotide (3-mer)"
            elif "2mer" in name:
                return "Dinucleotide (2-mer)"
            elif "GC" in name:
                return "GC Content"
            elif "Frequency" in name:
                return "Mononucleotide Freq"
            elif "Count" in name:
                return "Mononucleotide Count"
            return "Other"

        importances_df["Category"] = importances_df["Feature"].apply(categorize)

        logger.info(f"\nTop 15 Most Important DNA Sequence Features:\n{importances_df.head(15).to_string()}")
        importances_df.to_csv(self.output_dir / "feature_importances.csv", index=False)
        return importances_df

    def generate_visualizations(self, X_test: pd.DataFrame, y_test: np.ndarray, importances_df: pd.DataFrame):
        """
        Produces 4 high-resolution, publication-ready figures for Person 4 and final reporting:
        1. Confusion Matrices Grid for all 6 models
        2. Benchmark Performance Comparison Bar Chart
        3. Multi-Class ROC Curves for the best model
        4. Top 20 Biological Feature Importances Chart
        """
        logger.info("Generating publication-ready figures...")
        classes = list(self.label_encoder.classes_)
        class_full_names = {
            "EI": "EI (Donor Splice Site)",
            "IE": "IE (Acceptor Splice Site)",
            "N": "N (Non-Splice Sequence)"
        }
        display_labels = [f"{c}\n({class_full_names.get(c, c)})" for c in classes]

        sns.set_theme(style="whitegrid", font_scale=1.0)

        # -------------------------------------------------------------
        # Figure 1: Confusion Matrices Grid
        # -------------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (name, y_pred) in enumerate(self.predictions_dict.items()):
            cm = confusion_matrix(y_test, y_pred)
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

            annot = np.empty_like(cm).astype(str)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1f}%)"

            sns.heatmap(
                cm,
                annot=annot,
                fmt="",
                cmap="Blues",
                cbar=False,
                ax=axes[idx],
                xticklabels=classes,
                yticklabels=classes,
                annot_kws={"size": 11, "weight": "bold"}
            )
            axes[idx].set_title(f"{name}\nAcc: {accuracy_score(y_test, y_pred)*100:.1f}% | F1: {f1_score(y_test, y_pred, average='macro'):.3f}", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Predicted Label", fontweight="bold")
            axes[idx].set_ylabel("True Biological Label", fontweight="bold")

        plt.suptitle("DNA Splice Junction Classification: Confusion Matrices Across Models", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()
        cm_path = self.figures_dir / "confusion_matrices.png"
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {cm_path}")

        # -------------------------------------------------------------
        # Figure 2: Model Benchmark Comparison Chart
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(13, 7))
        df_plot = self.results_df.copy()
        
        # Melt metrics for grouped bar chart
        df_melt = pd.melt(
            df_plot,
            id_vars=["Model"],
            value_vars=["Test Accuracy (%)", "Balanced Accuracy (%)", "Macro F1-Score"],
            var_name="Metric",
            value_name="Score"
        )
        
        # Normalize Macro F1 to percentage scale for clean visual comparison
        df_melt.loc[df_melt["Metric"] == "Macro F1-Score", "Score"] = df_melt.loc[df_melt["Metric"] == "Macro F1-Score", "Score"] * 100
        df_melt.loc[df_melt["Metric"] == "Macro F1-Score", "Metric"] = "Macro F1-Score (x100)"

        palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette=palette, ax=ax)

        ax.set_title("Comparative Benchmark of AI Models on DNA Splice-Junction Prediction", fontsize=15, fontweight="bold", pad=15)
        ax.set_ylabel("Score (%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Machine Learning Model", fontsize=12, fontweight="bold")
        ax.set_ylim(60, 105)
        plt.xticks(rotation=20, ha="right", fontweight="bold")
        ax.legend(title="Evaluation Metric", title_fontsize="11", loc="upper left")

        # Annotate bar values
        for p in ax.patches:
            val = p.get_height()
            if not np.isnan(val) and val > 0:
                ax.annotate(f"{val:.1f}%", (p.get_x() + p.get_width() / 2., val),
                            ha="center", va="bottom", fontsize=8.5, xytext=(0, 3), textcoords="offset points")

        plt.tight_layout()
        metrics_plot_path = self.figures_dir / "model_metrics_comparison.png"
        plt.savefig(metrics_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {metrics_plot_path}")

        # -------------------------------------------------------------
        # Figure 3: Multi-Class ROC Curves for Best Model
        # -------------------------------------------------------------
        y_prob_best = self.probabilities_dict[self.best_model_name]
        if y_prob_best is not None:
            fig, ax = plt.subplots(figsize=(9, 7))
            
            # One-hot encode y_test for ROC calculation
            y_test_bin = np.zeros((len(y_test), len(classes)))
            for i, val in enumerate(y_test):
                y_test_bin[i, val] = 1

            colors = ["#d62728", "#1f77b4", "#2ca02c"]
            for i, class_label in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob_best[:, i])
                roc_auc_val = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[i], lw=2.5,
                        label=f"{class_label} ({class_full_names.get(class_label, class_label)}) (AUC = {roc_auc_val:.3f})")

            # Diagonal reference line
            ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
            ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold")
            ax.set_title(f"Multi-Class ROC Curves: {self.best_model_name}\n(One-vs-Rest Evaluation)", fontsize=14, fontweight="bold", pad=12)
            ax.legend(loc="lower right", fontsize=10)
            plt.tight_layout()

            roc_path = self.figures_dir / "roc_curves_best_model.png"
            plt.savefig(roc_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved: {roc_path}")

        # -------------------------------------------------------------
        # Figure 4: Top 20 Biological Feature Importances
        # -------------------------------------------------------------
        top20 = importances_df.head(20).copy()
        fig, ax = plt.subplots(figsize=(12, 8))

        palette_cat = {
            "Trinucleotide (3-mer)": "#2b5c8f",
            "Dinucleotide (2-mer)": "#d95f02",
            "GC Content": "#7570b3",
            "Mononucleotide Freq": "#1b9e77",
            "Mononucleotide Count": "#e7298a",
            "Other": "#66a61e"
        }

        sns.barplot(
            data=top20,
            y="Feature",
            x="Mean_Importance",
            hue="Category",
            dodge=False,
            palette=palette_cat,
            ax=ax
        )

        ax.set_title("Top 20 DNA Sequence Features Governing Splice-Junction Discrimination", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Ensemble Relative Feature Importance", fontsize=12, fontweight="bold")
        ax.set_ylabel("Genomic Feature / k-mer Motif", fontsize=12, fontweight="bold")
        ax.legend(title="Sequence Feature Type", loc="lower right", fontsize=10)

        # Highlight biological context
        for i, row in top20.reset_index().iterrows():
            ax.text(row["Mean_Importance"] + 0.0005, i, f"{row['Mean_Importance']:.4f}",
                    va="center", fontsize=9, fontweight="bold", color="#333333")

        plt.tight_layout()
        feat_plot_path = self.figures_dir / "top_features_importance.png"
        plt.savefig(feat_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {feat_plot_path}")

    def export_summary_json(self):
        """
        Exports a consolidated JSON summary file for Person 4 to integrate into
        the final project report and interactive dashboard.
        """
        summary = {
            "project_name": "AI-Based Analysis of Gene and DNA Sequence Patterns",
            "task": "Splice-Junction Classification (Donor EI vs Acceptor IE vs Non-Splice N)",
            "person_3_lead": "Harshith Krishna (CB.AI.U4AID26023)",
            "classes": list(self.label_encoder.classes_),
            "best_model": self.best_model_name,
            "models_evaluated": list(self.models.keys()),
            "benchmark_results": self.results_df.to_dict(orient="records")
        }

        with open(self.output_dir / "metrics_summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        logger.info("Exported summary JSON successfully.")

    def run_pipeline(self):
        """
        Executes the complete end-to-end Person 3 modeling workflow.
        """
        logger.info("==================================================================")
        logger.info("STARTING DNA SEQUENCE PATTERN MODELING PIPELINE (PERSON 3)")
        logger.info("==================================================================")

        # Step 1: Load data prepared by Person 2
        X_train, X_test, y_train, y_test = self.load_data()

        # Step 2: Define candidate model suite
        models = self.get_candidate_models()

        # Step 3: Run Stratified 5-Fold Cross-Validation
        cv_summary = self.run_cross_validation(models, X_train, y_train)

        # Step 4: Train on full train set and evaluate on unseen holdout test set
        results_df = self.train_and_evaluate_holdout(
            models, cv_summary, X_train, X_test, y_train, y_test
        )

        # Step 5: Hyperparameter optimization
        self.tune_hyperparameters(X_train, y_train)

        # Step 6: Feature importance and biological interpretability
        importances_df = self.extract_feature_importance(X_train)

        # Step 7: Generate publication-ready figures
        self.generate_visualizations(X_test, y_test, importances_df)

        # Step 8: Export structured summary for Person 4
        self.export_summary_json()

        logger.info("==================================================================")
        logger.info("PERSON 3 MODELING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("==================================================================")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data" / "processed"
    output_dir = base_dir / "results"
    models_dir = base_dir / "models_saved"

    pipeline = DNAPatternModelingPipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        models_dir=models_dir
    )
    pipeline.run_pipeline()
