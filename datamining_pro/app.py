"""
WebPathMine: Web Usage and Navigation Pattern Mining + User Behaviour Prediction
Workspace 1: Data Hub & Explorer (Primary Entrypoint & Central Ingestion Hub)
"""
import io
import os
import streamlit as st
import pandas as pd
import numpy as np

from src.visualization import (
    inject_custom_css,
    render_hero,
    plot_navigation_sankey,
    plot_missing_values,
    plot_correlation_heatmap,
    plot_session_timeline
)
from src.preprocessing import (
    load_dataset,
    inspect_dataset,
    detect_dataset_type,
    validate_dataset_capabilities,
    clean_dataset
)

st.set_page_config(
    page_title="Data Hub & Explorer | WebPathMine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom white-and-orange theme styling
st.markdown(inject_custom_css(), unsafe_allow_html=True)

# Initialize Session State
if 'df' not in st.session_state:
    st.session_state.df = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'df_source_name' not in st.session_state or st.session_state.df_source_name is None:
    st.session_state.df_source_name = ""
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
if 'ranking_results' not in st.session_state:
    st.session_state.ranking_results = None
if 'batch_prediction_df' not in st.session_state:
    st.session_state.batch_prediction_df = None
if 'prediction_summary' not in st.session_state:
    st.session_state.prediction_summary = None
if 'active_file_sig' not in st.session_state:
    st.session_state.active_file_sig = None
if 'cleaning_status' not in st.session_state:
    st.session_state.cleaning_status = None

def _reset_downstream_cache():
    """Invalidates downstream analysis results when a new dataset is loaded or cleaned."""
    st.session_state.model_pipeline = None
    st.session_state.model_results = None
    st.session_state.clustering_results = None
    st.session_state.rules_results = None
    st.session_state.ranking_results = None
    st.session_state.batch_prediction_df = None
    st.session_state.prediction_summary = None

# Auto-load default dataset if nothing loaded yet
DEFAULT_SHOPPERS_PATH = "data/online_shoppers_intention.csv"
DEFAULT_CLICKSTREAM_PATH = "data/clickstream_sample.csv"
DEFAULT_ACTIVITY_PATH = "data/sample_website_activity.json"

if st.session_state.df is None and os.path.exists(DEFAULT_SHOPPERS_PATH):
    try:
        loaded = pd.read_csv(DEFAULT_SHOPPERS_PATH)
        st.session_state.df = loaded.copy()
        st.session_state.raw_df = loaded.copy()
        st.session_state.df_source_name = "Online Shoppers Intent (Synthetic 3k Sample)"
        st.session_state.dataset_type = detect_dataset_type(loaded)
    except Exception:
        pass

if st.session_state.clickstream_df is None and os.path.exists(DEFAULT_CLICKSTREAM_PATH):
    try:
        st.session_state.clickstream_df = pd.read_csv(DEFAULT_CLICKSTREAM_PATH)
    except Exception:
        pass

# Render Sidebar Status Card
with st.sidebar:
    st.markdown("### WebPathMine")
    st.caption("Web Usage Mining & Conversion Intelligence")
    st.markdown("---")
    
    st.markdown("#### Active Dataset Status")
    if st.session_state.df is not None:
        display_source = str(st.session_state.df_source_name) if st.session_state.df_source_name else "In-Memory Dataset"
        st.success(f"Loaded: **{display_source}**")
        st.write(f"- **Records:** {len(st.session_state.df):,}")
        st.write(f"- **Features:** {len(st.session_state.df.columns)}")
        st.write(f"- **Detected Type:** `{st.session_state.dataset_type or 'tabular'}`")
        if 'Revenue' in st.session_state.df.columns:
            rev_rate = (st.session_state.df['Revenue'].astype(bool).sum() / len(st.session_state.df) * 100)
            st.write(f"- **Purchase Rate:** `{rev_rate:.1f}%`")
    else:
        st.warning("No dataset currently loaded.")
        
    if st.session_state.clickstream_df is not None:
        st.info(f"Clickstream Cache: {len(st.session_state.clickstream_df):,} events")
    st.markdown("---")

def render_data_hub():
    """Main rendering routine for Workspace 1: Data Hub & Explorer."""
    st.markdown(
        render_hero(
            title="Data Hub & Explorer",
            subtitle="Centralized dataset ingestion, provenance validation, data quality auditing, and exploratory analysis.",
            badge="WORKSPACE 1"
        ),
        unsafe_allow_html=True
    )
    
    # Canonical Ingestion Container
    st.markdown("### Dataset Ingestion")
    col_upload, col_sample = st.columns([1.6, 1.4])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload Custom Dataset (CSV or JSON):",
            type=["csv", "json"],
            help="Upload an e-commerce session dataset or timestamped clickstream log."
        )
        if uploaded_file is not None:
            file_id_attr = getattr(uploaded_file, 'file_id', '')
            file_sig = f"{uploaded_file.name}_{uploaded_file.size}_{file_id_attr}"
            if st.session_state.get('active_file_sig') != file_sig:
                try:
                    df_loaded = load_dataset(uploaded_file)
                    st.session_state.df = df_loaded.copy()
                    st.session_state.raw_df = df_loaded.copy()
                    upload_name = getattr(uploaded_file, 'name', 'Uploaded Dataset') or 'Uploaded Dataset'
                    st.session_state.df_source_name = upload_name
                    st.session_state.dataset_type = detect_dataset_type(df_loaded)
                    st.session_state.active_file_sig = file_sig
                    st.session_state.cleaning_status = None
                    _reset_downstream_cache()
                    
                    if st.session_state.dataset_type == 'clickstream':
                        st.session_state.clickstream_df = df_loaded.copy()
                        
                    st.success(f"Successfully loaded '{upload_name}' ({len(df_loaded):,} records)")
                except Exception as e:
                    st.error(f"Failed to load dataset: {str(e)}")
        else:
            if st.session_state.get('active_file_sig') is not None:
                st.session_state.active_file_sig = None
                
    with col_sample:
        st.markdown("#### Supported Benchmark & Sample Datasets")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Shoppers 3k", use_container_width=True, help="Synthetic 3,000-session dataset with behavioral metrics"):
                if os.path.exists(DEFAULT_SHOPPERS_PATH):
                    loaded = pd.read_csv(DEFAULT_SHOPPERS_PATH)
                    st.session_state.df = loaded.copy()
                    st.session_state.raw_df = loaded.copy()
                    st.session_state.df_source_name = "Online Shoppers Intent (Synthetic 3k Sample)"
                    st.session_state.dataset_type = detect_dataset_type(loaded)
                    st.session_state.active_file_sig = None
                    st.session_state.cleaning_status = None
                    _reset_downstream_cache()
                    st.rerun()
        with b2:
            if st.button("Clicks CSV", use_container_width=True, help="Synthetic 4,464 clickstream navigation events"):
                if os.path.exists(DEFAULT_CLICKSTREAM_PATH):
                    df_c = pd.read_csv(DEFAULT_CLICKSTREAM_PATH)
                    st.session_state.df = df_c.copy()
                    st.session_state.raw_df = df_c.copy()
                    st.session_state.clickstream_df = df_c.copy()
                    st.session_state.df_source_name = "Clickstream Events (Synthetic CSV Sample)"
                    st.session_state.dataset_type = 'clickstream'
                    st.session_state.active_file_sig = None
                    st.session_state.cleaning_status = None
                    _reset_downstream_cache()
                    st.rerun()
        with b3:
            if st.button("Activity JSON", use_container_width=True, help="Synthetic 450 web activity events"):
                if os.path.exists(DEFAULT_ACTIVITY_PATH):
                    df_j = load_dataset(DEFAULT_ACTIVITY_PATH)
                    st.session_state.df = df_j.copy()
                    st.session_state.raw_df = df_j.copy()
                    st.session_state.clickstream_df = df_j.copy()
                    st.session_state.df_source_name = "Activity Log (Synthetic JSON Sample)"
                    st.session_state.dataset_type = 'clickstream'
                    st.session_state.active_file_sig = None
                    st.session_state.cleaning_status = None
                    _reset_downstream_cache()
                    st.rerun()

    if st.session_state.df is None:
        st.info("No dataset is currently loaded. Please upload a CSV or JSON file or select one of the sample datasets above to begin exploration.")
        return

    df = st.session_state.df

    # Dataset Provenance & Capabilities Inspection
    active_source = str(st.session_state.df_source_name) if st.session_state.get('df_source_name') is not None else ""
    meta = validate_dataset_capabilities(df, source_name=active_source)
    st.session_state.dataset_meta = meta
    audit = inspect_dataset(df)

    # Provenance and Compatibility Panel
    st.markdown("---")
    st.markdown("### Dataset Provenance & Algorithm Compatibility")
    
    prov_col1, prov_col2 = st.columns([1, 1])
    with prov_col1:
        provenance_status = "Synthetic Sample Data" if meta['is_synthetic'] else "User Uploaded / Empirical Benchmark"
        st.markdown(f"""
        <div class="light-card">
            <h4 style="color: #F97316; margin: 0 0 0.5rem 0;">Provenance & Granularity</h4>
            <p style="color: #4B5563; font-size: 0.92rem; margin: 0; line-height: 1.6;">
                <strong>Source:</strong> {meta['provenance']}<br>
                <strong>Status:</strong> {provenance_status}<br>
                <strong>Granularity:</strong> {meta['granularity']}<br>
                <strong>Session Identifiers:</strong> {'Present' if meta['has_session_ids'] else 'Not Detected'}&nbsp;&nbsp;|&nbsp;&nbsp;
                <strong>Timestamps:</strong> {'Present' if meta['has_timestamps'] else 'Not Detected'}&nbsp;&nbsp;|&nbsp;&nbsp;
                <strong>Target Column:</strong> {meta['target_col'] if meta['has_purchase_target'] else 'None (Unlabelled)'}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with prov_col2:
        caps = meta['capabilities']
        def _cap_row(status: bool, label: str, note: str):
            badge_color = "#16A34A" if status else "#D97706"
            badge_bg = "#F0FDF4" if status else "#FFFBEB"
            badge_text = "Supported" if status else "Notice"
            return (
                f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.35rem; font-size: 0.88rem; color: #374151;'>"
                f"<span><strong>{label}:</strong> {note}</span>"
                f"<span style='background:{badge_bg}; color:{badge_color}; border:1px solid {badge_color}; border-radius:4px; padding:1px 6px; font-size:0.75rem; font-weight:700;'>{badge_text}</span>"
                f"</div>"
            )
            
        badge_html = f"""
        <div class="light-card">
            <h4 style="color: #1F2937; margin: 0 0 0.5rem 0;">Analytical Compatibility</h4>
            {_cap_row(caps['clustering'][0], 'K-Means Clustering', caps['clustering'][1])}
            {_cap_row(caps['pattern_mining'][0], 'FP-Growth Mining', caps['pattern_mining'][1])}
            {_cap_row(caps['sequence_analysis'][0], 'Navigation Sequences', caps['sequence_analysis'][1])}
            {_cap_row(caps['supervised_prediction'][0], 'Purchase Prediction', caps['supervised_prediction'][1])}
            {_cap_row(caps['engagement_ranking'][0], 'Engagement Ranking', caps['engagement_ranking'][1])}
        </div>
        """
        st.markdown(badge_html, unsafe_allow_html=True)

    # Organized Tabs
    tab_overview, tab_preview, tab_quality, tab_stats = st.tabs([
        "Overview",
        "Data Preview",
        "Data Quality & Cleaning",
        "Statistics & Relationships"
    ])

    # TAB 1: OVERVIEW
    with tab_overview:
        st.markdown("#### High-Level Dataset Metrics")
        k1, k2, k3, k4 = st.columns(4)
        total_records = audit['n_rows']
        total_features = audit['n_cols']
        completeness = 100.0 - audit['missing_cell_pct']
        
        has_rev = 'Revenue' in df.columns
        if has_rev:
            purchases = int(df['Revenue'].astype(bool).sum())
            purchase_rate = round(purchases / total_records * 100, 2)
            rev_text = f"{purchases:,} ({purchase_rate}%)"
        else:
            rev_text = "N/A (Unlabelled)"

        with k1:
            st.metric("Total Records / Sessions", f"{total_records:,}")
        with k2:
            st.metric("Attributes / Features", f"{total_features}")
        with k3:
            st.metric("Data Completeness", f"{completeness:.1f}%")
        with k4:
            st.metric("Purchase Conversion Rate", rev_text)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Conceptual E-Commerce Navigation Journey")
        st.caption("Standard visitor navigation progression from acquisition to conversion. For empirical transition probabilities, visit Workspace 2: Pattern Mining.")
        sankey_fig = plot_navigation_sankey()
        st.plotly_chart(sankey_fig, use_container_width=True)

        st.markdown("#### Consolidated Analytical Workspaces")
        w1, w2, w3 = st.columns(3)
        with w1:
            st.markdown("""
            <div class="light-card">
                <h4 style="color: #F97316; margin-top: 0;">Pattern Mining</h4>
                <p style="color: #6B7280; font-size: 0.88rem; margin: 0; line-height: 1.5;">
                    Discover frequent page co-occurrences with high-speed FP-Growth, 
                    filter association rules by support/confidence/lift, and inspect first-order Markov transition matrices.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with w2:
            st.markdown("""
            <div class="light-card">
                <h4 style="color: #F97316; margin-top: 0;">Segmentation & Engagement</h4>
                <p style="color: #6B7280; font-size: 0.88rem; margin: 0; line-height: 1.5;">
                    Segment user sessions via K-Means clustering, inspect PCA projections and radar profiles, 
                    and explore the 0–100 Behavioural Engagement Leaderboard.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with w3:
            st.markdown("""
            <div class="light-card">
                <h4 style="color: #F97316; margin-top: 0;">Prediction & Insights</h4>
                <p style="color: #6B7280; font-size: 0.88rem; margin: 0; line-height: 1.5;">
                    Train leak-free Decision Tree classifiers with 5-fold CV, run live single and batch session predictions, 
                    and review data-grounded CRO strategic recommendations.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # TAB 2: DATA PREVIEW
    with tab_preview:
        st.markdown("#### Dataset Preview")
        preview_rows = st.slider("Records to display:", min_value=5, max_value=50, value=10)
        st.dataframe(df.head(preview_rows), use_container_width=True)

        st.markdown("#### Schema & Attribute Specifications")
        col_info_df = pd.DataFrame({
            "Attribute": df.columns,
            "Data Type": [str(t) for t in df.dtypes],
            "Non-Null Count": df.notnull().sum().values,
            "Unique Values": df.nunique().values,
            "Sample Value": [str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else "N/A" for c in df.columns]
        })
        st.dataframe(col_info_df, use_container_width=True, height=300)

    # TAB 3: DATA QUALITY & CLEANING
    with tab_quality:
        st.markdown("#### Data Quality Audit")
        q1, q2, q3 = st.columns(3)
        with q1:
            st.metric("Missing Cells", f"{audit['total_missing_cells']} ({audit['missing_cell_pct']}%)")
        with q2:
            st.metric("Duplicate Rows", f"{audit['duplicate_count']} ({audit['duplicate_pct']}%)")
        with q3:
            st.metric("Data Completeness Rate", f"{100.0 - audit['missing_cell_pct']:.1f}%")

        st.markdown("##### Missing Value Breakdown")
        if audit['total_missing_cells'] == 0:
            st.success("Dataset has zero missing values (100% complete).")
        else:
            st.warning(f"Detected {audit['total_missing_cells']} missing values across dataset attributes.")
            m_fig = plot_missing_values(audit['missing_summary'])
            st.plotly_chart(m_fig, use_container_width=True)
            st.dataframe(audit['missing_summary'][audit['missing_summary']['Missing Values'] > 0], use_container_width=True)

        st.markdown("---")
        st.markdown("#### Safe Data Cleaning Routine")
        st.caption("Scientific Policy: Cleaning actions are explicit and transparent. No records are deleted without user confirmation.")
        
        c_col1, c_col2 = st.columns(2)
        with c_col1:
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
        with c_col2:
            target_protect = st.selectbox(
                "Target Column to Protect from Imputation",
                options=["None"] + df.columns.tolist(),
                index=df.columns.get_loc("Revenue") + 1 if "Revenue" in df.columns else 0
            )
            target_name = None if target_protect == "None" else target_protect

        if st.button("Apply Cleaning Routine", use_container_width=True):
            if not drop_dups and missing_action == 'none':
                st.session_state.cleaning_status = {
                    'type': 'info',
                    'message': "No cleaning action selected. Please enable 'Remove Exact Duplicate Rows' or choose a missing-value strategy to apply."
                }
                st.rerun()

            cleaned_df, report = clean_dataset(
                df,
                drop_duplicates=drop_dups,
                missing_strategy=missing_action,
                target_col=target_name
            )
            rows_removed = report['duplicates_removed'] + report['missing_rows_dropped']
            values_imputed = report['missing_values_imputed']

            if rows_removed == 0 and values_imputed == 0:
                st.session_state.cleaning_status = {
                    'type': 'info',
                    'message': "No cleaning needed: The active dataset contains no duplicate rows or eligible missing values matching the selected options. No changes were made."
                }
                st.rerun()
            else:
                st.session_state.df = cleaned_df
                if st.session_state.dataset_type == 'clickstream' and st.session_state.get('clickstream_df') is not None:
                    st.session_state.clickstream_df = cleaned_df.copy()
                _reset_downstream_cache()

                parts = []
                if report['duplicates_removed'] > 0:
                    parts.append(f"{report['duplicates_removed']:,} duplicate row(s) removed")
                if report['missing_rows_dropped'] > 0:
                    parts.append(f"{report['missing_rows_dropped']:,} row(s) with missing values dropped")
                if values_imputed > 0:
                    parts.append(f"{values_imputed:,} missing value(s) imputed across {len(report['imputed_columns'])} feature column(s)")

                details_str = "; ".join(parts)
                missing_handled = report['missing_rows_dropped'] + values_imputed
                success_msg = (
                    f"Cleaning routine applied successfully: Removed {rows_removed:,} row(s) and handled {missing_handled:,} missing value(s). "
                    f"Active dataset updated: {len(cleaned_df):,} records remaining ({details_str})."
                )
                st.session_state.cleaning_status = {
                    'type': 'success',
                    'message': success_msg,
                    'report': report
                }
                st.rerun()

        # Render Persistent Cleaning Status & Report
        if 'cleaning_status' in st.session_state and st.session_state.cleaning_status:
            status = st.session_state.cleaning_status
            if status.get('type') == 'success':
                st.success(status['message'])
                if status.get('report') and status['report'].get('transformations'):
                    with st.expander("Detailed Cleaning Transformations & Audit Report", expanded=False):
                        st.json(status['report'])
            elif status.get('type') == 'info':
                st.info(status['message'])
            elif status.get('type') == 'warning':
                st.warning(status['message'])

        if st.session_state.get('raw_df') is not None and not df.equals(st.session_state.raw_df):
            st.caption(f"Original uncleaned dataset preserved ({len(st.session_state.raw_df):,} raw records).")
            if st.button("Restore Original Dataset", help="Reverts active dataset to the originally loaded raw dataset"):
                st.session_state.df = st.session_state.raw_df.copy()
                if st.session_state.dataset_type == 'clickstream' and st.session_state.get('clickstream_df') is not None:
                    st.session_state.clickstream_df = st.session_state.raw_df.copy()
                st.session_state.cleaning_status = {
                    'type': 'info',
                    'message': f"Restored active dataset to original raw state ({len(st.session_state.raw_df):,} records)."
                }
                _reset_downstream_cache()
                st.rerun()

        st.markdown("---")
        st.markdown("#### Export Cleaned Dataset")
        st.caption(f"Cleaned active dataset ready for export: {len(df):,} records, {len(df.columns)} features.")
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        st.download_button(
            label="Download Cleaned CSV",
            data=csv_bytes,
            file_name="webpathmine_cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

    # TAB 4: STATISTICS & RELATIONSHIPS
    with tab_stats:
        st.markdown("#### Descriptive Statistics")
        if not audit['numeric_stats'].empty:
            st.markdown("##### Numerical Features")
            st.dataframe(audit['numeric_stats'], use_container_width=True)

            if len(audit['numeric_columns']) >= 2:
                st.markdown("##### Feature Correlation Heatmap")
                corr_fig = plot_correlation_heatmap(df, audit['numeric_columns'])
                st.plotly_chart(corr_fig, use_container_width=True)

        if not audit['categorical_stats'].empty:
            st.markdown("##### Categorical Features")
            st.dataframe(audit['categorical_stats'], use_container_width=True)

        # Chronological Activity Timeline (migrated from Automated Workflow)
        st.markdown("---")
        st.markdown("#### Chronological Activity Timeline")
        has_timeline_data = False
        
        # Check if active dataset or clickstream cache has session + event columns
        events_df = None
        if 'session_id' in df.columns and ('page_name' in df.columns or 'event_type' in df.columns):
            events_df = df
        elif st.session_state.clickstream_df is not None:
            c_df = st.session_state.clickstream_df
            if 'session_id' in c_df.columns and ('page_name' in c_df.columns or 'event_type' in c_df.columns):
                events_df = c_df

        if events_df is not None:
            st.caption("Visualizes sequential browsing steps and timestamps across top visitor sessions.")
            timeline_fig = plot_session_timeline(
                events_df,
                session_col='session_id',
                timestamp_col='timestamp' if 'timestamp' in events_df.columns else None,
                page_col='page_name' if 'page_name' in events_df.columns else events_df.columns[1],
                event_col='event_type' if 'event_type' in events_df.columns else None,
                max_sessions=8
            )
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.info("Chronological activity timelines are available for event-level clickstream or activity logs (e.g. Clicks CSV or Activity JSON).")

# Page Navigation Setup (4 Standard Workspaces)
p1 = st.Page(render_data_hub, title="Data Hub & Explorer", default=True)
p2 = st.Page("pages/1_Pattern_Mining.py", title="Pattern Mining")
p3 = st.Page("pages/2_Segmentation_and_Engagement.py", title="Segmentation & Engagement")
p4 = st.Page("pages/3_Prediction_and_Insights.py", title="Prediction & Insights")

nav = st.navigation([p1, p2, p3, p4])
nav.run()
