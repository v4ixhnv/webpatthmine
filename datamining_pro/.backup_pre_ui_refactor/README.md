# WEBPATHMINE: Web Usage & Navigation Pattern Mining + User Behaviour Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![MLxtend](https://img.shields.io/badge/mlxtend-0.23+-green.svg)](http://rasbt.github.io/mlxtend/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple.svg)](https://plotly.com/)

**WEBPATHMINE** is an interactive, enterprise-grade data mining and machine learning dashboard built with Streamlit, Scikit-learn, MLxtend, and Plotly. It empowers e-commerce growth analysts and data scientists to analyze user session data, discover co-occurring browsing patterns with high-speed **FP-Growth**, map sequential navigation flows, segment shoppers through **K-Means clustering**, and predict purchase conversion intent using explainable **Decision Tree Classifiers**.

---

## 🌟 Key Capabilities

1. **Automated End-to-End Analysis Workflow (NEW)**
   - Unified upload of raw website activity event logs in **CSV or JSON**.
   - Automatic schema validation, timestamp verification, and quality audit.
   - Interactive chronological user activity timeline visualization.
   - Derives session-level features (`Administrative_Duration`, `ProductRelated_Duration`, `Cart_Interactions`, `Checkout_Activity`, `BounceRates`, `PageValues`) while keeping raw events separate.
   - Executes FP-Growth pattern mining, pre-trained K-Means behavioral clustering, and Decision Tree purchase intent prediction in one automated pipeline.
   - Exports processed events, session predictions, itemsets, rules, and executive analysis reports.

2. **Data Collection & Quality Auditing**
   - Ingest custom CSV/JSON datasets or the synthetic Online Shoppers Purchasing Intention sample dataset.
   - Comprehensive audit of data types, missing cells, and duplicate records.
   - Explicit data cleaning controls (duplicate purging, safe median/mode imputation) without silent data deletion.
   - Downloadable processed CSVs.

3. **Navigation Pattern Mining**
   - High-throughput frequent itemset discovery using **FP-Growth** with execution timing benchmarks.
   - Configurable minimum support, confidence, and lift thresholds.
   - Association rule generation with search, filtering, and scatter/bar visualizations.
   - **Unordered vs. Ordered Distinctions:** Unordered itemset co-occurrence vs. first-order Markovian chronological navigation sequence transition matrices.

4. **Behavioral Segmentation**
   - Unsupervised **K-Means** clustering on standardized engagement metrics (`BounceRates`, `ExitRates`, `ProductRelated_Duration`, etc.).
   - Model diagnostics with dual-axis Inertia (Elbow) and Silhouette Score curves.
   - 2D PCA cluster space projection and normalized radar profile comparisons.
   - Empirical, data-grounded cluster interpretations strictly derived from metric z-scores.

5. **Purchase Intent Prediction**
   - Leak-free `sklearn.pipeline.Pipeline` with `ColumnTransformer` (scaling, imputation, and one-hot encoding).
   - `DecisionTreeClassifier` with selectable Gini Impurity or Entropy (Information Gain) criteria, maximum depth, and class weighting.
   - Class imbalance analysis with Accuracy, Precision, Recall, F1-Score, ROC-AUC, and an interactive Confusion Matrix.
   - Feature importance rankings and persisted model export via **Joblib**.
   - Interactive live session simulator for real-time conversion probability estimation.

6. **Model Insights & Business Strategy**
   - Strategic synthesis translating pattern mining rules, cluster traits, and decision tree drivers into data-backed Conversion Rate Optimization (CRO) prescriptions.

---

## 📁 Project Architecture

```
d:\datamining_pro/
├── .streamlit/
│   └── config.toml                  # Dark navy (#0a1128) and cyan (#00e8c6) theme
├── data/
│   ├── sample_website_activity.json  # Synthetic sample website activity log in JSON format (450 events)
│   ├── online_shoppers_intention.csv # Synthetic Online Shoppers dataset sample (3,000 sessions generated via scripts/generate_sample_data.py)
│   └── clickstream_sample.csv        # Synthetic timestamped clickstream log (4,464 events)
├── models/
│   ├── kmeans_model.joblib           # Pre-trained K-Means clustering model & scaler
│   └── decision_tree_pipeline.joblib # Serialized trained scikit-learn pipeline (Schema v2.0-leakfree)
├── outputs/                         # Export directory for mined itemsets and rules
├── pages/
│   ├── 0_⚡_Automated_Workflow.py     # End-to-end automated activity ingestion & analysis pipeline
│   ├── 1_📊_Data_Explorer.py        # Schema preview, missing/duplicate audit, data cleaning
│   ├── 2_🕸️_Pattern_Mining.py       # FP-Growth itemsets, association rules & transition flows
│   ├── 3_👥_Behavior_Segmentation.py# K-Means clustering, silhouette curves & radar profiles
│   ├── 4_🎯_Purchase_Prediction.py   # Decision tree training, confusion matrix & live inference
│   └── 5_💡_Model_Insights.py       # Multi-module synthesis & CRO business recommendations
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Ingestion, schema audit, and safe cleaning
│   ├── feature_engineering.py        # Clickstream-to-transactions & sequence transition matrices
│   ├── pattern_mining.py             # FP-Growth wrapper, rule generation & filtering
│   ├── clustering.py                 # StandardScaler + KMeans, elbow/silhouette, cluster profiles
│   ├── prediction.py                 # Pipeline builder, stratified split, evaluation & inference
│   └── visualization.py              # Dark navy & cyan Plotly chart suite
├── tests/
│   ├── __init__.py
│   └── test_modules.py               # Unit tests covering all algorithms and transformations
├── scripts/
│   ├── generate_sample_data.py       # Deterministic generator for sample datasets
│   └── verify_pipeline.py            # End-to-end integration verification runner
├── app.py                            # Streamlit landing page & macro navigation architecture
├── requirements.txt                  # Pinned dependencies
├── README.md                         # Documentation
└── .gitignore                        # Git exclusions
```

---

## 🚀 Quick Start Guide

### 1. Installation

Ensure you have Python 3.10 or higher installed. Clone or navigate to the workspace directory and install dependencies:

```bash
cd d:\datamining_pro
pip install -r requirements.txt
```

### 2. Verify Tests & Pipeline

Run the automated unit test suite:

```bash
python tests/test_modules.py
```

Run the end-to-end integration check on the bundled datasets:

```bash
python scripts/verify_pipeline.py
```

### 3. Launch the Streamlit Application

Start the interactive dashboard locally:

```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 📊 Modules & Workflow Walkthrough

1. **Home / Overview (`app.py`)**
   - High-level KPIs (Total Sessions, Purchases, Conversion Rate, Data Completeness).
   - E-commerce macro navigation flow architecture (Sankey diagram).
   - 1-click dataset switcher (UCI Shoppers or Clickstream Log).

2. **Data Explorer (`pages/1_📊_Data_Explorer.py`)**
   - Upload custom CSVs or use bundled datasets.
   - Interactive preview of columns, data types, and non-null distributions.
   - Missing value percentage charts and duplicate record analysis.
   - Safe cleaning routine (drop duplicates, median/mode imputation) with transformation audit report.
   - Export processed CSVs.

3. **Pattern Mining (`pages/2_🕸️_Pattern_Mining.py`)**
   - Switch between timestamped Clickstream page navigation and shopper behavioral feature sets.
   - Adjust Min Support (0.01 to 0.60) and Metric thresholds (Confidence, Lift, Support).
   - Discovered itemsets and association rules displayed in interactive searchable tables.
   - Millisecond execution duration badge for FP-Growth benchmarking.
   - Chronological page-to-page Markov transition probability heatmap.
   - Export frequent itemsets and association rules as CSV.

4. **Behavior Segmentation (`pages/3_👥_Behavior_Segmentation.py`)**
   - Select behavioral features (`BounceRates`, `ExitRates`, `PageValues`, durations).
   - Automatic `StandardScaler` feature normalization.
   - Tunable cluster count $k \in [2, 8]$.
   - Elbow curve (Inertia) and Silhouette score analysis.
   - 2D PCA cluster projection and normalized radar profile comparisons.
   - Objective trait callouts computed purely from z-score metric differences.

5. **Purchase Prediction (`pages/4_🎯_Purchase_Prediction.py`)**
   - Configurable `DecisionTreeClassifier` (Gini vs. Entropy criteria, Max Depth, Balanced Class Weights).
   - Stratified train-test split (preventing target leakage).
   - Evaluation metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - Interactive confusion matrix heatmap and feature importance rankings.
   - Interactive single-session form for real-time shopper conversion probability prediction.
   - Download trained model pipeline as `decision_tree_pipeline.joblib`.

6. **Model Insights (`pages/5_💡_Model_Insights.py`)**
   - Executive synthesis linking pattern rules, cluster segments, and decision tree drivers into data-backed CRO actions.

---

## 🛡️ Engineering & Data Integrity Standards
- **No Silent Data Loss:** Explicit user confirmation required for any row dropping, with complete audit logs.
- **Leak-Free Pipelines:** All imputers and scalers are strictly fitted on the training split inside an atomic `sklearn.pipeline.Pipeline`.
- **Reproducible Seeds:** All stochastic operations utilize fixed seeds (`random_state=42`).
- **Offline Self-Contained:** Zero paid APIs or external network calls required; bundled with deterministic sample data generators.
