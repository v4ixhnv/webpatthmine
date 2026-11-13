"""
WEBPATHMINE: Web Usage and Navigation Pattern Mining + User Behaviour Prediction
Main Application Entrypoint & Overview Page.
"""
import os
import streamlit as st
import pandas as pd

from src.visualization import (
    inject_custom_css,
    render_hero,
    plot_navigation_sankey
)
from src.preprocessing import inspect_dataset, detect_dataset_type

st.set_page_config(
    page_title="WEBPATHMINE | Overview",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom Navy & Cyan styling
st.markdown(inject_custom_css(), unsafe_allow_html=True)

# Initialize Session State
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_source_name' not in st.session_state:
    st.session_state.df_source_name = None
if 'dataset_type' not in st.session_state:
    st.session_state.dataset_type = None
if 'clickstream_df' not in st.session_state:
    st.session_state.clickstream_df = None
if 'model_pipeline' not in st.session_state:
    st.session_state.model_pipeline = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = None
if 'rules_results' not in st.session_state:
    st.session_state.rules_results = None

# Auto-load default dataset if nothing loaded yet
DEFAULT_SHOPPERS_PATH = "data/online_shoppers_intention.csv"
DEFAULT_CLICKSTREAM_PATH = "data/clickstream_sample.csv"

if st.session_state.df is None and os.path.exists(DEFAULT_SHOPPERS_PATH):
    try:
        st.session_state.df = pd.read_csv(DEFAULT_SHOPPERS_PATH)
        st.session_state.df_source_name = "UCI Online Shoppers (Built-in)"
        st.session_state.dataset_type = detect_dataset_type(st.session_state.df)
    except Exception:
        pass

if st.session_state.clickstream_df is None and os.path.exists(DEFAULT_CLICKSTREAM_PATH):
    try:
        st.session_state.clickstream_df = pd.read_csv(DEFAULT_CLICKSTREAM_PATH)
    except Exception:
        pass

# Sidebar
with st.sidebar:
    st.markdown("### 🌐 WEBPATHMINE")
    st.caption("E-Commerce Web Usage Mining & Conversion Intelligence")
    st.markdown("---")
    
    st.markdown("#### 📂 Active Dataset Status")
    if st.session_state.df is not None:
        st.success(f"**Loaded:** {st.session_state.df_source_name}")
        st.write(f"- **Rows:** {len(st.session_state.df):,}")
        st.write(f"- **Columns:** {len(st.session_state.df.columns)}")
        st.write(f"- **Type:** `{st.session_state.dataset_type}`")
        if 'Revenue' in st.session_state.df.columns:
            rev_rate = (st.session_state.df['Revenue'].astype(bool).sum() / len(st.session_state.df) * 100)
            st.write(f"- **Purchase Rate:** `{rev_rate:.1f}%`")
    else:
        st.warning("No tabular dataset currently loaded.")
        
    if st.session_state.clickstream_df is not None:
        st.info(f"**Clickstream Events:** {len(st.session_state.clickstream_df):,} rows")
    
    st.markdown("---")
    st.markdown("#### ⚡ Quick Dataset Switcher")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Shoppers Data", use_container_width=True):
            if os.path.exists(DEFAULT_SHOPPERS_PATH):
                st.session_state.df = pd.read_csv(DEFAULT_SHOPPERS_PATH)
                st.session_state.df_source_name = "UCI Online Shoppers (Built-in)"
                st.session_state.dataset_type = detect_dataset_type(st.session_state.df)
                st.rerun()
    with col_btn2:
        if st.button("Clickstream Log", use_container_width=True):
            if os.path.exists(DEFAULT_CLICKSTREAM_PATH):
                st.session_state.df = pd.read_csv(DEFAULT_CLICKSTREAM_PATH)
                st.session_state.df_source_name = "Clickstream Events (Built-in)"
                st.session_state.dataset_type = detect_dataset_type(st.session_state.df)
                st.rerun()

# Hero Header
st.markdown(
    render_hero(
        title="WEBPATH<span>MINE</span>",
        subtitle="Web Usage & Navigation Pattern Mining + User Behaviour Prediction Dashboard",
        badge="ENTERPRISE E-COMMERCE INTELLIGENCE"
    ),
    unsafe_allow_html=True
)

# Project Introduction
st.markdown("""
<div class="navy-card">
    <p style="margin: 0; font-size: 1.05rem; line-height: 1.6; color: #cbd5e1;">
        <strong>WEBPATHMINE</strong> integrates pattern mining, unsupervised behavioral clustering, 
        and supervised decision tree modeling into a unified e-commerce analytics workspace. 
        Uncover frequent browsing co-occurrences with <strong>FP-Growth</strong>, map sequential navigation transitions, 
        segment shoppers via <strong>K-Means clustering</strong>, and predict purchase conversion intent 
        with explainable <strong>Decision Tree Classifiers</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

# Top KPI Metric Cards
st.markdown("### 📊 High-Level Session Summary")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

df_active = st.session_state.df
if df_active is not None:
    audit = inspect_dataset(df_active)
    total_sessions = audit['n_rows']
    total_cols = audit['n_cols']
    completeness = 100.0 - audit['missing_cell_pct']
    
    has_rev = 'Revenue' in df_active.columns
    if has_rev:
        purchases = int(df_active['Revenue'].astype(bool).sum())
        purchase_rate = round(purchases / total_sessions * 100, 2)
        rev_text = f"{purchases:,} ({purchase_rate}%)"
    else:
        rev_text = "N/A (No target)"
        
    with kpi1:
        st.metric(label="Total Sessions Analyzed", value=f"{total_sessions:,}")
    with kpi2:
        st.metric(label="Purchases (Conversion)", value=rev_text)
    with kpi3:
        st.metric(label="Attributes / Features", value=f"{total_cols}")
    with kpi4:
        st.metric(label="Data Completeness", value=f"{completeness:.1f}%")
else:
    with kpi1:
        st.metric(label="Total Sessions", value="--")
    with kpi2:
        st.metric(label="Conversion Rate", value="--")
    with kpi3:
        st.metric(label="Attributes", value="--")
    with kpi4:
        st.metric(label="Completeness", value="--")

st.markdown("<br>", unsafe_allow_html=True)

# Navigation Flow Architecture (Sankey)
st.markdown("### 🧭 E-Commerce User Navigation Architecture")
st.caption("Macro flow diagram illustrating typical shopper session progression from acquisition channels to checkout conversion.")
sankey_fig = plot_navigation_sankey()
st.plotly_chart(sankey_fig, use_container_width=True)

# Available Modules Summary
st.markdown("### 🛠️ Integrated Analysis Modules")

# Featured Automated Workflow Banner
st.markdown("""
<div class="navy-card" style="border-left: 4px solid #00f0ff;">
    <h3 style="color: #00f0ff; margin-top: 0;">⚡ Automated Analysis Pipeline (NEW)</h3>
    <p style="color: #cbd5e1; font-size: 0.95rem; margin-bottom: 0.5rem;">
        Upload raw activity logs in <strong>CSV or JSON</strong>. The automated pipeline ingests events, validates timestamps, 
        generates session behavioral features, discovers navigation patterns via <strong>FP-Growth</strong>, 
        assigns behavioral clusters via <strong>K-Means</strong>, predicts conversion intent using <strong>Decision Trees</strong>, 
        and produces full CSV and report downloads.
    </p>
    <a href="/Automated_Workflow" target="_self" style="color: #00e8c6; font-weight: 600; text-decoration: none;">➔ Launch Automated Workflow Dashboard</a>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="navy-card">
        <h4 style="color: #00e8c6; margin-top: 0;">📊 1. Data Explorer</h4>
        <p style="color: #94a3b8; font-size: 0.9rem;">
            Inspect schema, missing values, duplicates, and statistical distributions. 
            Clean data safely without silent record deletion.
        </p>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="navy-card">
        <h4 style="color: #00e8c6; margin-top: 0;">🕸️ 2. Pattern Mining</h4>
        <p style="color: #94a3b8; font-size: 0.9rem;">
            Discover frequent itemsets with high-speed FP-Growth and evaluate 
            association rules with support, confidence, and lift metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="navy-card">
        <h4 style="color: #00e8c6; margin-top: 0;">👥 3. Segmentation</h4>
        <p style="color: #94a3b8; font-size: 0.9rem;">
            Segment sessions with standardized K-Means, silhouette score optimization, 
            PCA projections, and radar profile charts.
        </p>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="navy-card">
        <h4 style="color: #00e8c6; margin-top: 0;">🎯 4. Intent Prediction</h4>
        <p style="color: #94a3b8; font-size: 0.9rem;">
            Train leak-free Decision Tree pipelines (Gini / Entropy), inspect feature 
            importance, and test real-time session inference.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.info("💡 Use the sidebar navigation on the left to launch the Automated Workflow or explore each analysis module individually.")
