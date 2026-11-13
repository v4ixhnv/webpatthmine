# WebPathMine: Web Usage & Navigation Pattern Mining + User Behaviour Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-orange.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![MLxtend](https://img.shields.io/badge/mlxtend-0.23+-green.svg)](http://rasbt.github.io/mlxtend/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-purple.svg)](https://plotly.com/)

**WebPathMine** is an interactive, enterprise-grade data mining and machine learning application built with Streamlit, Scikit-learn, MLxtend, and Plotly. It empowers e-commerce growth analysts and data scientists to analyze user session data, discover co-occurring browsing patterns with high-speed **FP-Growth**, map sequential navigation flows, segment shoppers through **K-Means clustering**, rank visitors by explainable **0–100 Behavioural Engagement Scores**, and predict purchase conversion intent using leak-free **Decision Tree Classifiers**.

---

## Key Workspaces & Capabilities

WebPathMine is consolidated into **four focused workspaces** following the CRISP-DM data mining lifecycle:

### 1. Workspace 1: Data Hub & Explorer (`app.py`)
- Centralized data ingestion for custom CSV and JSON datasets.
- 1-click loading of benchmark and synthetic sample datasets (Shoppers Intent 3k, Clickstream CSV, Activity JSON).
- Transparent provenance disclosure (real vs. synthetic sample data, granularity, session/timestamp validation).
- Automated algorithm compatibility evaluation across all downstream tasks.
- High-level KPI summary, macro navigation architecture (Plotly Sankey diagram), and chronological user activity timelines.
- Data preview, schema specifications, missing-value percentage charts, and duplicate row detection.
- Safe, non-destructive data cleaning routines (median/mode imputation, duplicate removal) with target protection and cleaned CSV export.

### 2. Workspace 2: Pattern Mining (`pages/1_Pattern_Mining.py`)
- High-throughput frequent itemset mining using **FP-Growth** with execution duration benchmarks in milliseconds.
- Configurable minimum support, metric threshold (confidence, lift, support), and maximum itemset length.
- Association rule discovery with keyword search filtering and sortable metric tables.
- Interactive rule scatter plots (Support vs. Confidence colored by Lift) and top-rule bar charts.
- First-order Markov transition probability heatmap and ranked common direct navigation paths.
- One-click CSV exports for frequent itemsets and association rules.

### 3. Workspace 3: Segmentation & Engagement (`pages/2_Segmentation_and_Engagement.py`)
- Prominent **Behavioural Engagement Ranking Leaderboard (0–100)**: Transparent rule-based composite scoring of product catalog depth, duration, cart additions, and bounce penalties with zero target leakage.
- Filterable and searchable ranking table by cluster, intent tier (High, Moderate, Low), and session identifier.
- Individual session score inspector with granular sub-score breakdowns.
- Unsupervised **K-Means** clustering on standardized engagement features with tunable cluster count $k \in [2, 8]$.
- Model diagnostics with dual-axis Inertia (Elbow) and Silhouette Score curves for optimal $k$ validation.
- 2D PCA cluster space projections and normalized radar profile comparisons.
- Quantitative cluster feature averages and empirical z-score characteristic profiles.
- Leaderboard CSV export.

### 4. Workspace 4: Prediction & Insights (`pages/3_Prediction_and_Insights.py`)
- Leak-free `sklearn.pipeline.Pipeline` with `ColumnTransformer` (standard scaling, median imputation, one-hot encoding).
- Supervised `DecisionTreeClassifier` with selectable Gini Impurity or Entropy criteria, maximum tree depth, and balanced class weighting.
- Rigorous evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, interactive Confusion Matrix, and 5-Fold Stratified Cross-Validation stability intervals.
- Direct feature-importance rankings paired with empirical splitter interpretation.
- Streamlined live single-session prediction simulator with primary inputs and collapsible secondary attributes.
- Prominent conversion outcome verdict cards with confidence progress gauges and key decision drivers.
- Batch prediction master table with status filters (All, Predicted Purchases, Predicted Non-Purchases) and individual session verdict inspector.
- Dynamic cross-module conversion rate optimization (CRO) strategic recommendations.
- One-click export for serialized model pipelines (`.joblib`), predictions CSV, and executive Markdown analysis reports.

---

## Project Structure

```
d:\datamining_pro/
├── .streamlit/
│   └── config.toml                  # Clean white-and-orange theme configuration
├── data/
│   ├── sample_website_activity.json  # Synthetic sample website activity log in JSON format (450 events)
│   ├── online_shoppers_intention.csv # Synthetic Online Shoppers dataset sample (3,000 sessions)
│   └── clickstream_sample.csv        # Synthetic timestamped clickstream log (4,464 events)
├── models/
│   ├── kmeans_model.joblib           # Pre-trained K-Means clustering model & scaler
│   └── decision_tree_pipeline.joblib # Serialized trained scikit-learn pipeline (Schema v2.0-leakfree)
├── outputs/                         # Export directory for mined itemsets and rules
├── pages/
│   ├── 1_Pattern_Mining.py          # Workspace 2: FP-Growth itemsets, association rules & transition flows
│   ├── 2_Segmentation_and_Engagement.py # Workspace 3: K-Means clustering, diagnostics & 0-100 engagement ranking
│   └── 3_Prediction_and_Insights.py # Workspace 4: Decision Tree, 5-fold CV, inference & CRO insights
├── src/
│   ├── __init__.py
│   ├── preprocessing.py              # Ingestion, schema audit, capability validator, and safe cleaning
│   ├── feature_engineering.py        # Event aggregation, engagement scoring (0-100), and sequence transitions
│   ├── pattern_mining.py             # FP-Growth wrapper, rule generation & filtering
│   ├── clustering.py                 # StandardScaler + KMeans, elbow/silhouette, cluster profiles
│   ├── prediction.py                 # Leak-free pipeline, stratified CV, batch inference & persistence
│   └── visualization.py              # White-and-orange Plotly chart suite & clean card components
├── tests/
│   ├── __init__.py
│   ├── test_automated_workflow.py    # Pipeline integration, verdict cards, and schema tests
│   ├── test_modules.py               # Unit tests covering algorithms, leakage prevention, and metrics
│   └── test_ui_consolidation.py      # Automated UI consolidation, theme, and emoji-free verification
├── scripts/
│   ├── generate_sample_data.py       # Deterministic generator for sample datasets
│   ├── test_pages_render.py          # Smoke tests for workspace analytical logic
│   ├── test_workflow_render.py       # End-to-end execution runner on JSON activity data
│   └── verify_pipeline.py            # Integration verification runner
├── app.py                            # Workspace 1: Data Hub & Explorer (Primary entrypoint)
├── requirements.txt                  # Pinned dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Git exclusions
```

---

## Quick Start Guide

### 1. Installation

Ensure you have Python 3.10 or higher installed. Navigate to the project directory and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Verify Tests & Pipeline

Run the automated unit test suite (28 passing tests):

```bash
python -m unittest discover -v tests
```

Run the end-to-end integration check on bundled datasets:

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

## Engineering & Scientific Integrity Standards

- **Zero Target Leakage:** Conversion target attributes (`Revenue`, `has_purchase`, `PageValues`) are strictly excluded from prediction training matrices and from behavioural engagement score formulas.
- **Transparent Provenance:** Built-in synthetic datasets are clearly disclosed as synthetic samples; benchmark datasets are distinguished from user uploads.
- **No Calibrated Probability Confusion:** The Behavioural Engagement Score (0–100) is explicitly identified as an empirical ranking of browsing depth, not a calibrated financial purchase probability.
- **Cross-Validation Stability:** Evaluates predictive models using Stratified 5-Fold Cross-Validation alongside standard holdout test metrics.
- **Stale Model Invalidation:** Automatically detects and rejects legacy model artifacts lacking the current leak-free schema version.
- **Non-Destructive Operations:** Data cleaning operations require explicit user selection with complete transformation audit reports.
- **Offline Self-Contained:** Operates entirely locally with zero external network or paid API requirements.
