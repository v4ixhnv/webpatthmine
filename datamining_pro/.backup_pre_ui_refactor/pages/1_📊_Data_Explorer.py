"""
Page 1: Data Explorer & Quality Audit
Upload datasets, preview columns, inspect missing values, duplicates, and distributions,
clean data safely, and export processed CSVs.
"""
import io
import os
import streamlit as st
import pandas as pd
import numpy as np

from src.visualization import (
    inject_custom_css,
    render_hero,
    plot_missing_values,
    plot_correlation_heatmap
)
from src.preprocessing import (
    load_dataset,
    inspect_dataset,
    detect_dataset_type,
    validate_dataset_capabilities,
    clean_dataset,
    map_clickstream_columns
)

st.set_page_config(
    page_title="Data Explorer | WEBPATHMINE",
    page_icon="📊",
    layout="wide"
)

st.markdown(inject_custom_css(), unsafe_allow_html=True)

st.markdown(
    render_hero(
        title="DATA <span>EXPLORER</span>",
        subtitle="Dataset Ingestion, Provenance Validation, Quality Auditing & Schema Diagnostics",
        badge="MODULE 1: DATA EXPLORATION"
    ),
    unsafe_allow_html=True
)

def _reset_downstream_cache():
    st.session_state.model_pipeline = None
    st.session_state.model_results = None
    st.session_state.clustering_results = None
    st.session_state.rules_results = None
    st.session_state.ranking_results = None

# Source Selection Controls
st.markdown("### 📥 Dataset Ingestion")
col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Custom Dataset (CSV or JSON):",
        type=["csv", "json"],
        help="Upload an e-commerce session dataset or timestamped clickstream log."
    )
    if uploaded_file is not None:
        try:
            df_loaded = load_dataset(uploaded_file)
            st.session_state.df = df_loaded.copy()
            st.session_state.raw_df = df_loaded.copy()
            st.session_state.df_source_name = uploaded_file.name
            st.session_state.dataset_type = detect_dataset_type(df_loaded)
            _reset_downstream_cache()
            
            # If clickstream format, also store in clickstream_df
            if st.session_state.dataset_type == 'clickstream':
                st.session_state.clickstream_df = df_loaded.copy()
                
            st.success(f"Successfully loaded '{uploaded_file.name}' ({len(df_loaded):,} rows)")
        except Exception as e:
            st.error(f"Failed to load dataset: {str(e)}")

with col_sample:
    st.markdown("#### Built-in Datasets")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Shoppers 3k", use_container_width=True, help="Synthetic 3,000-session dataset"):
            p = "data/online_shoppers_intention.csv"
            if os.path.exists(p):
                loaded = pd.read_csv(p)
                st.session_state.df = loaded.copy()
                st.session_state.raw_df = loaded.copy()
                st.session_state.df_source_name = "Online Shoppers Intent (Synthetic 3k Sample)"
                st.session_state.dataset_type = detect_dataset_type(loaded)
                _reset_downstream_cache()
                st.success("Loaded Synthetic Shoppers dataset!")
                st.rerun()
    with b2:
        if st.button("Clicks CSV", use_container_width=True, help="Synthetic 4,464 clickstream events"):
            p = "data/clickstream_sample.csv"
            if os.path.exists(p):
                df_c = pd.read_csv(p)
                st.session_state.df = df_c.copy()
                st.session_state.raw_df = df_c.copy()
                st.session_state.clickstream_df = df_c.copy()
                st.session_state.df_source_name = "Clickstream Events (Synthetic CSV Sample)"
                st.session_state.dataset_type = 'clickstream'
                _reset_downstream_cache()
                st.success("Loaded Clickstream CSV!")
                st.rerun()
    with b3:
        if st.button("Activity JSON", use_container_width=True, help="Synthetic 450 web activity events"):
            p = "data/sample_website_activity.json"
            if os.path.exists(p):
                df_j = load_dataset(p)
                st.session_state.df = df_j.copy()
                st.session_state.raw_df = df_j.copy()
                st.session_state.clickstream_df = df_j.copy()
                st.session_state.df_source_name = "Activity Log (Synthetic JSON Sample)"
                st.session_state.dataset_type = 'clickstream'
                _reset_downstream_cache()
                st.success("Loaded Activity JSON!")
                st.rerun()

if st.session_state.df is None:
    st.warning("⚠️ No dataset loaded. Please upload a CSV file or load one of the built-in datasets above.")
    st.stop()

df = st.session_state.df

# Dataset Provenance & Capabilities Inspection
meta = validate_dataset_capabilities(df, source_name=st.session_state.get('df_source_name', ''))
st.session_state.dataset_meta = meta

audit = inspect_dataset(df)

# Dataset Provenance & Capability Badges
st.markdown("---")
st.markdown("### 📋 Dataset Provenance & Analytical Capabilities")

prov_col1, prov_col2 = st.columns([1, 1])
with prov_col1:
    st.markdown(f"""
    <div class="navy-card">
        <h4 style="color: #00e8c6; margin: 0 0 0.5rem 0;">📁 Provenance & Data Granularity</h4>
        <p style="color: #cbd5e1; font-size: 0.95rem; margin: 0; line-height: 1.6;">
            <strong>Data Source:</strong> {meta['provenance']}<br>
            <strong>Status:</strong> {'⚠️ Synthetic Sample Data' if meta['is_synthetic'] else '✅ User Uploaded / Benchmark'}<br>
            <strong>Granularity:</strong> {meta['granularity']}<br>
            <strong>Session IDs Present:</strong> {'Yes' if meta['has_session_ids'] else 'No'}&nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Timestamps Present:</strong> {'Yes' if meta['has_timestamps'] else 'No'}&nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Target Column:</strong> {meta['target_col'] if meta['has_purchase_target'] else 'None detected'}
        </p>
    </div>
    """, unsafe_allow_html=True)

with prov_col2:
    caps = meta['capabilities']
    def _badge(status: bool, label: str, note: str):
        icon = "✅" if status else "⚠️"
        color = "#00e8c6" if status else "#f59e0b"
        return f"<div style='margin-bottom: 0.35rem; font-size: 0.9rem; color: #cbd5e1;'>{icon} <strong style='color:{color};'>{label}:</strong> {note}</div>"
        
    badge_html = f"""
    <div class="navy-card">
        <h4 style="color: #00f0ff; margin: 0 0 0.5rem 0;">⚡ Algorithm Compatibility</h4>
        {_badge(caps['clustering'][0], 'K-Means Clustering', caps['clustering'][1])}
        {_badge(caps['pattern_mining'][0], 'FP-Growth Association', caps['pattern_mining'][1])}
        {_badge(caps['sequence_analysis'][0], 'Navigation Sequences', caps['sequence_analysis'][1])}
        {_badge(caps['supervised_prediction'][0], 'Purchase Intent Prediction', caps['supervised_prediction'][1])}
        {_badge(caps['engagement_ranking'][0], 'Behavioural Ranking', caps['engagement_ranking'][1])}
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)

st.markdown("### 📈 Dataset Dimensions & Quality Metrics")
q1, q2, q3, q4, q5 = st.columns(5)
with q1:
    st.metric("Total Records", f"{audit['n_rows']:,}")
with q2:
    st.metric("Total Features", f"{audit['n_cols']}")
with q3:
    st.metric("Missing Cells", f"{audit['total_missing_cells']} ({audit['missing_cell_pct']}%)")
with q4:
    st.metric("Duplicate Rows", f"{audit['duplicate_count']} ({audit['duplicate_pct']}%)")
with q5:
    st.metric("Detected Type", audit.get('detected_type', st.session_state.dataset_type).replace('_', ' ').title())

# Tabs for detailed exploration
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Data Preview",
    "⚠️ Missing Values",
    "🔁 Duplicates & Cleaning",
    "📊 Statistical Summary",
    "💾 Export Processed CSV"
])

with tab1:
    st.markdown("#### Dataset Preview")
    preview_rows = st.slider("Rows to preview", min_value=5, max_value=50, value=10)
    st.dataframe(df.head(preview_rows), use_container_width=True)
    
    st.markdown("#### Column Information")
    col_info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Non-Null Count": df.notnull().sum().values,
        "Unique Values": df.nunique().values,
        "Sample Value": [str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else "N/A" for c in df.columns]
    })
    st.dataframe(col_info_df, use_container_width=True, height=280)

with tab2:
    st.markdown("#### Missing Value Analysis")
    if audit['total_missing_cells'] == 0:
        st.success("✅ Dataset has no missing values. Perfect data completeness!")
    else:
        st.warning(f"Found {audit['total_missing_cells']} missing values across the dataset.")
        
    m_fig = plot_missing_values(audit['missing_summary'])
    st.plotly_chart(m_fig, use_container_width=True)
    
    st.dataframe(audit['missing_summary'][audit['missing_summary']['Missing Values'] > 0], use_container_width=True)

with tab3:
    st.markdown("#### Duplicate Records & Data Cleaning")
    st.markdown("""
    > **Data Integrity Policy:** WEBPATHMINE does not silently discard data. 
    > Choose explicit cleaning actions below with immediate audit feedback.
    """)
    
    if audit['duplicate_count'] > 0:
        st.info(f"Detected **{audit['duplicate_count']} duplicate records** ({audit['duplicate_pct']}% of data).")
    else:
        st.success("No duplicate records detected.")
        
    col_clean1, col_clean2 = st.columns(2)
    with col_clean1:
        drop_dups = st.checkbox("Remove Exact Duplicate Rows", value=False)
        missing_action = st.selectbox(
            "Missing Value Strategy",
            options=["none", "impute", "drop_rows"],
            format_func=lambda x: {
                "none": "Do not modify missing values",
                "impute": "Impute safely (Median for numeric, Mode for categorical)",
                "drop_rows": "Drop rows containing missing values"
            }[x]
        )
        
    with col_clean2:
        target_protect = st.selectbox(
            "Target Column to Protect from Imputation",
            options=["None"] + df.columns.tolist(),
            index=df.columns.get_loc("Revenue") + 1 if "Revenue" in df.columns else 0
        )
        target_name = None if target_protect == "None" else target_protect
        
    if st.button("Apply Cleaning Routine", use_container_width=True):
        cleaned_df, report = clean_dataset(
            df,
            drop_duplicates=drop_dups,
            missing_strategy=missing_action,
            target_col=target_name
        )
        st.session_state.df = cleaned_df
        _reset_downstream_cache()
        st.success(f"Cleaning complete! Dataset row count: {len(cleaned_df):,}")
        st.json(report)
        st.rerun()

with tab4:
    st.markdown("#### Descriptive Statistics")
    if not audit['numeric_stats'].empty:
        st.markdown("##### Numerical Features")
        st.dataframe(audit['numeric_stats'], use_container_width=True)
        
        # Correlation heatmap
        if len(audit['numeric_columns']) >= 2:
            st.markdown("##### Feature Correlation Heatmap")
            corr_fig = plot_correlation_heatmap(df, audit['numeric_columns'])
            st.plotly_chart(corr_fig, use_container_width=True)
            
    if not audit['categorical_stats'].empty:
        st.markdown("##### Categorical Features")
        st.dataframe(audit['categorical_stats'], use_container_width=True)

with tab5:
    st.markdown("#### Download Processed Dataset")
    st.write(f"Active dataset: **{st.session_state.df_source_name}** ({len(df):,} rows, {len(df.columns)} columns)")
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv_bytes,
        file_name="webpathmine_processed_data.csv",
        mime="text/csv",
        use_container_width=True
    )
