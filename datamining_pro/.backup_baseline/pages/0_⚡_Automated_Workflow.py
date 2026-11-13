"""
Page 0: Automated End-to-End Analysis Workflow & Instant Purchase Verdict
Ingests CSV/JSON activity event logs or tabular datasets, derives session features,
delivers the immediate final purchase intent verdict front-and-center, and provides
behavioral segmentation, navigation pattern mining, activity timeline, and downloads
all on a single unified page.
"""
import io
import os
import streamlit as st
import pandas as pd
import numpy as np

from src.visualization import (
    inject_custom_css,
    render_hero,
    render_purchase_verdict_card,
    plot_session_timeline,
    plot_rules_scatter,
    plot_top_rules_bar,
    plot_clusters_pca,
    plot_cluster_sizes,
    plot_confusion_matrix,
    plot_prediction_distribution,
    generate_analysis_report
)
from src.preprocessing import (
    load_dataset,
    validate_activity_dataset,
    detect_dataset_type
)
from src.feature_engineering import (
    prepare_unified_session_data
)
from src.pattern_mining import (
    run_fpgrowth,
    generate_rules
)
from src.clustering import (
    get_or_train_reference_clustering,
    assign_clusters,
    generate_cluster_profiles,
    DEFAULT_BEHAVIORAL_FEATURES
)
from src.prediction import (
    get_or_train_reference_pipeline,
    batch_predict_sessions
)

st.set_page_config(
    page_title="Instant Purchase Verdict | WEBPATHMINE",
    page_icon="⚡",
    layout="wide"
)

def render_html_block(html_content: str):
    """Renders HTML safely using st.html (or st.markdown with stripped indentation)."""
    clean_html = "\n".join(line.strip() for line in html_content.splitlines() if line.strip())
    if hasattr(st, 'html'):
        st.html(clean_html)
    else:
        st.markdown(clean_html, unsafe_allow_html=True)

render_html_block(inject_custom_css())

render_html_block(
    render_hero(
        title="INSTANT PURCHASE VERDICT & <span>AUTOMATED MINING</span>",
        subtitle="Upload website activity datasets to get an immediate conversion verdict, behavioral clusters, and navigation patterns — unified on a single page",
        badge="SINGLE-PAGE PIPELINE"
    )
)


# 1. Dataset Upload & Ingestion Controls
st.markdown("### 📥 Upload Website Activity Dataset (CSV / JSON)")
upload_col, sample_col = st.columns([1.6, 1.4])

uploaded_data = None
source_name = "None"

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload Activity / Clickstream Dataset (CSV or JSON):",
        type=["csv", "json"],
        help="Supports raw clickstream event logs (with session_id, page_name, timestamp) or tabular session records."
    )
    if uploaded_file is not None:
        try:
            uploaded_data = load_dataset(uploaded_file)
            source_name = uploaded_file.name
            st.session_state.auto_raw_df = uploaded_data
            st.session_state.auto_source_name = source_name
            st.success(f"Loaded '{source_name}' ({len(uploaded_data):,} records)")
        except Exception as e:
            st.error(f"Upload failed: {str(e)}")

with sample_col:
    st.markdown("#### Quick Load Sample Datasets")
    s_r1_1, s_r1_2, s_r1_3 = st.columns(3)
    with s_r1_1:
        if st.button("Sample Activity JSON", use_container_width=True):
            p = "data/sample_website_activity.json"
            if os.path.exists(p):
                st.session_state.auto_raw_df = load_dataset(p)
                st.session_state.auto_source_name = "sample_website_activity.json"
                st.rerun()
    with s_r1_2:
        if st.button("Clickstream CSV", use_container_width=True):
            p = "data/clickstream_sample.csv"
            if os.path.exists(p):
                st.session_state.auto_raw_df = load_dataset(p)
                st.session_state.auto_source_name = "clickstream_sample.csv"
                st.rerun()
    with s_r1_3:
        if st.button("Shoppers Tabular CSV", use_container_width=True):
            p = "data/online_shoppers_intention.csv"
            if os.path.exists(p):
                st.session_state.auto_raw_df = load_dataset(p)
                st.session_state.auto_source_name = "online_shoppers_intention.csv"
                st.rerun()

    s_r2_1, s_r2_2 = st.columns(2)
    with s_r2_1:
        if st.button("Single Buyer Test (JSON)", use_container_width=True):
            p = "data/sample_single_shopper.json"
            if os.path.exists(p):
                st.session_state.auto_raw_df = load_dataset(p)
                st.session_state.auto_source_name = "sample_single_shopper.json (1 Shopper)"
                st.rerun()
    with s_r2_2:
        if st.button("Single Non-Buyer Test (JSON)", use_container_width=True):
            p = "data/sample_single_nonbuyer.json"
            if os.path.exists(p):
                st.session_state.auto_raw_df = load_dataset(p)
                st.session_state.auto_source_name = "sample_single_nonbuyer.json (1 Shopper)"
                st.rerun()

# Resolve active raw dataset
raw_df = st.session_state.get('auto_raw_df')
if raw_df is None:
    # Auto fallback to JSON sample if available
    default_json = "data/sample_website_activity.json"
    if os.path.exists(default_json):
        raw_df = load_dataset(default_json)
        source_name = "sample_website_activity.json (Default)"
        st.session_state.auto_raw_df = raw_df
        st.session_state.auto_source_name = source_name
    else:
        st.warning("Please upload a CSV or JSON dataset above to begin.")
        st.stop()
else:
    source_name = st.session_state.get('auto_source_name', 'Uploaded Dataset')

# ==============================================================================
# UNIFIED PIPELINE EXECUTION (In Background Before Rendering)
# ==============================================================================
with st.spinner("Processing activity events, running clustering & predicting purchase intent..."):
    try:
        unified = prepare_unified_session_data(raw_df)
    except Exception as e:
        st.error(f"Failed to process dataset: {str(e)}")
        st.stop()

    session_features_df = unified['session_features_df']
    cleaned_events_df = unified['cleaned_events_df']
    one_hot_df = unified['one_hot_df']
    is_single_session = unified['is_single_session']
    total_sessions = unified['total_sessions']
    dtype_detected = unified['dataset_type']

    # 1. Behavioral Clustering (K-Means)
    clustering_features = [f for f in DEFAULT_BEHAVIORAL_FEATURES if f in session_features_df.columns]
    if not clustering_features:
        clustering_features = ['Administrative_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
    
    kmeans_model, kmeans_scaler, trained_feat_cols = get_or_train_reference_clustering(
        feature_cols=clustering_features,
        n_clusters=3
    )
    cluster_labels, sil_score, pca_coords, var_ratios = assign_clusters(
        kmeans_model,
        kmeans_scaler,
        session_features_df,
        feature_cols=trained_feat_cols
    )
    session_features_df['Cluster'] = [f"Cluster {l}" for l in cluster_labels]
    profile_df, size_df, descriptions = generate_cluster_profiles(
        session_features_df,
        cluster_labels,
        trained_feat_cols
    )

    # 2. Purchase Intent Prediction (Decision Tree)
    pred_pipeline, _ = get_or_train_reference_pipeline(criterion='gini', max_depth=5)
    predicted_sessions_df, pred_summary = batch_predict_sessions(
        pipeline=pred_pipeline,
        session_df=session_features_df,
        target_col='Revenue' if 'Revenue' in session_features_df.columns else None
    )

    # 3. FP-Growth Pattern Mining (Default parameters)
    min_supp_default = 0.08 if len(one_hot_df) > 10 else 0.01
    frequent_itemsets, exec_time = run_fpgrowth(one_hot_df, min_support=min_supp_default)
    rules_df = generate_rules(frequent_itemsets, metric='confidence', min_threshold=0.30)
    exec_ms = exec_time * 1000

# ==============================================================================
# SECTION 1: FINAL PURCHASE INTENT VERDICT (IMMEDIATE ANSWER)
# ==============================================================================
st.markdown("---")
st.markdown("## 🎯 1. Final Purchase Intent Verdict")
st.caption(
    f"Active Dataset: **{source_name}** | Format: `{dtype_detected.upper()}` | "
    f"Total Sessions: **{total_sessions:,}**"
)

def build_drivers_dict(row: pd.Series) -> dict:
    """Extracts informative behavioral decision drivers from a session row."""
    drivers = {}
    if 'PageValues' in row:
        drivers['Page Value Index'] = f"{float(row['PageValues']):.1f}"
    if 'ProductRelated_Duration' in row:
        drivers['Product Browse Time'] = f"{float(row['ProductRelated_Duration']):.0f}s"
    if 'BounceRates' in row:
        drivers['Bounce Rate'] = f"{float(row['BounceRates']) * 100:.1f}%"
    if 'ExitRates' in row:
        drivers['Exit Rate'] = f"{float(row['ExitRates']) * 100:.1f}%"
    if 'Cart_Interactions' in row and row['Cart_Interactions'] > 0:
        drivers['Cart Adds'] = int(row['Cart_Interactions'])
    if 'Checkout_Activity' in row and row['Checkout_Activity'] > 0:
        drivers['Checkout Steps'] = int(row['Checkout_Activity'])
    if 'Administrative_Duration' in row and row['Administrative_Duration'] > 0:
        drivers['Admin / Account Time'] = f"{float(row['Administrative_Duration']):.0f}s"
    return drivers

if is_single_session:
    # --------------------------------------------------------------------------
    # Single Session: Immediate Prominent Verdict Card
    # --------------------------------------------------------------------------
    s_row = predicted_sessions_df.iloc[0]
    s_id = str(s_row.get('session_id', 'Session_1'))
    s_pred = s_row.get('Predicted_Conversion', False)
    s_prob = s_row.get('Purchase_Probability', 0.0)
    s_cluster = s_row.get('Cluster', 'Cluster 0')
    drivers = build_drivers_dict(s_row)

    card_html = render_purchase_verdict_card(
        prediction=1 if s_pred else 0,
        probability=s_prob,
        session_id=s_id,
        cluster_name=s_cluster,
        key_drivers=drivers
    )
    render_html_block(card_html)

else:
    # --------------------------------------------------------------------------
    # Multi-Session Cohort: Summary KPIs + Shopper Inspector + Master Table
    # --------------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Sessions Evaluated", f"{total_sessions:,}")
    with k2:
        st.metric(
            "Predicted Purchases",
            f"{pred_summary['predicted_purchases']:,}",
            delta=f"{pred_summary['predicted_conversion_rate']:.1f}% Conversion"
        )
    with k3:
        st.metric("Predicted No Purchases", f"{pred_summary['predicted_no_purchases']:,}")
    with k4:
        gt_status = "Ground-Truth Evaluated" if pred_summary['has_ground_truth'] else "Unlabelled (Inference Only)"
        st.metric("Ground-Truth Status", gt_status)

    st.markdown("#### 🔍 Shopper Verdict Inspector")
    st.caption("Select any individual session below to inspect its detailed conversion verdict card, confidence meter, and decision drivers:")

    # Sort session options so likely buyers appear first for immediate inspection
    sorted_sessions = predicted_sessions_df.sort_values(
        by='Purchase_Probability',
        ascending=False
    )['session_id'].tolist()

    selected_sess_id = st.selectbox(
        "Select Session ID to Inspect Final Verdict:",
        options=sorted_sessions,
        index=0,
        format_func=lambda sid: (
            f"🛒 {sid} (LIKELY BUYER — {predicted_sessions_df.loc[predicted_sessions_df['session_id'] == sid, 'Purchase_Probability'].values[0] * 100:.1f}%)"
            if predicted_sessions_df.loc[predicted_sessions_df['session_id'] == sid, 'Predicted_Conversion'].values[0]
            else f"👀 {sid} (NON-BUYER — {predicted_sessions_df.loc[predicted_sessions_df['session_id'] == sid, 'Purchase_Probability'].values[0] * 100:.1f}%)"
        )
    )

    if selected_sess_id:
        inspected_row = predicted_sessions_df[predicted_sessions_df['session_id'] == selected_sess_id].iloc[0]
        inspected_pred = inspected_row.get('Predicted_Conversion', False)
        inspected_prob = inspected_row.get('Purchase_Probability', 0.0)
        inspected_cluster = inspected_row.get('Cluster', 'Cluster 0')
        inspected_drivers = build_drivers_dict(inspected_row)

        inspector_card = render_purchase_verdict_card(
            prediction=1 if inspected_pred else 0,
            probability=inspected_prob,
            session_id=str(selected_sess_id),
            cluster_name=inspected_cluster,
            key_drivers=inspected_drivers
        )
        render_html_block(inspector_card)

    # Master Session Table with filters
    st.markdown("#### 📋 Complete Session Verdict Master Table")
    f_col1, f_col2 = st.columns([1, 3])
    with f_col1:
        verdict_filter = st.radio(
            "Filter Sessions:",
            options=["Show All", "Predicted Purchases Only", "Predicted No Purchases Only"],
            horizontal=True
        )

    filtered_df = predicted_sessions_df.copy()
    if verdict_filter == "Predicted Purchases Only":
        filtered_df = filtered_df[filtered_df['Predicted_Conversion'] == True]
    elif verdict_filter == "Predicted No Purchases Only":
        filtered_df = filtered_df[filtered_df['Predicted_Conversion'] == False]

    # Format table for visual clarity
    display_df = filtered_df.copy()
    display_df['Final_Verdict'] = display_df['Predicted_Conversion'].apply(
        lambda p: "🟢 PURCHASE" if p else "🔴 NO PURCHASE"
    )
    display_df['Purchase_Probability'] = (display_df['Purchase_Probability'] * 100).round(1).astype(str) + "%"

    show_cols = [
        'session_id', 'Final_Verdict', 'Purchase_Probability', 'Cluster',
        'PageValues', 'ProductRelated_Duration', 'BounceRates', 'Cart_Interactions', 'Checkout_Activity'
    ]
    avail_cols = [c for c in show_cols if c in display_df.columns]
    
    st.dataframe(
        display_df[avail_cols].rename(columns={
            'session_id': 'Session ID',
            'Final_Verdict': 'Final Verdict',
            'Purchase_Probability': 'Likelihood',
            'Cluster': 'Behavior Segment',
            'PageValues': 'Page Values',
            'ProductRelated_Duration': 'Product Browse (s)',
            'BounceRates': 'Bounce Rate',
            'Cart_Interactions': 'Cart Adds',
            'Checkout_Activity': 'Checkout Steps'
        }),
        use_container_width=True,
        height=280
    )

    # Charts Row: Distribution & Confusion Matrix
    pv_col1, pv_col2 = st.columns(2)
    with pv_col1:
        st.plotly_chart(plot_prediction_distribution(pred_summary), use_container_width=True)
    with pv_col2:
        if pred_summary['has_ground_truth'] and pred_summary['evaluation'] is not None:
            ev = pred_summary['evaluation']
            st.markdown(f"""
            **Ground-Truth Evaluation Accuracy:** `{ev['accuracy'] * 100:.1f}%`  
            - Precision: `{ev['precision'] * 100:.1f}%` | Recall: `{ev['recall'] * 100:.1f}%` | F1-Score: `{ev['f1_score']:.3f}`
            """)
            st.plotly_chart(plot_confusion_matrix(ev['confusion_matrix']), use_container_width=True)
        else:
            st.info("ℹ️ Uploaded dataset does not contain ground-truth Revenue labels. Predictions are statistical model inferences without evaluation metrics fabrication.")

# ==============================================================================
# SECTION 2: BEHAVIORAL SEGMENTATION (K-MEANS)
# ==============================================================================
st.markdown("---")
st.markdown("## 👥 2. Behavioral Segmentation (K-Means)")
st.caption("Segments sessions into distinct browsing archetypes using the pre-trained K-Means clustering model.")

km1, km2, km3 = st.columns(3)
with km1:
    st.metric("Total Clustered Sessions", f"{len(session_features_df):,}")
with km2:
    st.metric("Behavioral Clusters", f"{len(np.unique(cluster_labels))}")
with km3:
    sil_str = f"{sil_score:.3f}" if sil_score is not None else ("N/A (Single Session)" if is_single_session else "N/A")
    st.metric("Silhouette Score", sil_str)

c_col1, c_col2 = st.columns(2)
with c_col1:
    st.plotly_chart(plot_clusters_pca(pca_coords, cluster_labels, var_ratios), use_container_width=True)
with c_col2:
    st.plotly_chart(plot_cluster_sizes(size_df), use_container_width=True)

with st.expander("📋 View Cluster Profile Characteristics", expanded=False):
    st.dataframe(profile_df, use_container_width=True)
    for d in descriptions:
        st.markdown(f"**{d['cluster']}** ({d['size']:,} sessions): {'; '.join(d['characteristics'])}")

# ==============================================================================
# SECTION 3: NAVIGATION PATTERN MINING (FP-GROWTH)
# ==============================================================================
st.markdown("---")
st.markdown("## 🕸️ 3. Navigation Pattern Mining (FP-Growth)")
st.caption("Identifies frequent navigation itemsets and sequential browsing association rules across sessions.")

fp_ctrl1, fp_ctrl2 = st.columns([1, 2])
with fp_ctrl1:
    min_supp = st.slider(
        "FP-Growth Minimum Support:",
        min_value=0.01,
        max_value=0.50,
        value=float(min_supp_default),
        step=0.01,
        help="Proportion of sessions that must contain the itemset."
    )
    min_conf = st.slider(
        "Rule Minimum Confidence:",
        min_value=0.10,
        max_value=1.00,
        value=0.30,
        step=0.05
    )

# Re-run FP-Growth if thresholds adjusted
frequent_itemsets, exec_time = run_fpgrowth(one_hot_df, min_support=min_supp)
rules_df = generate_rules(frequent_itemsets, metric='confidence', min_threshold=min_conf)
exec_ms = exec_time * 1000

with fp_ctrl2:
    f1, f2, f3 = st.columns(3)
    with f1:
        st.metric("Discovered Itemsets", f"{len(frequent_itemsets):,}")
    with f2:
        st.metric("Association Rules", f"{len(rules_df):,}")
    with f3:
        st.metric("Execution Time", f"{exec_ms:.2f} ms")

tab_fi, tab_ar, tab_vp = st.tabs(["📦 Frequent Itemsets", "🔗 Association Rules", "📈 Rule Visualizations"])
with tab_fi:
    if frequent_itemsets.empty:
        st.info("No itemsets meet the current support threshold. Try lowering the support slider.")
    else:
        st.dataframe(
            frequent_itemsets[['itemsets_str', 'support', 'length']].rename(
                columns={'itemsets_str': 'Frequent Pages / Itemsets', 'support': 'Support', 'length': 'Item Count'}
            ),
            use_container_width=True,
            height=250
        )
with tab_ar:
    if rules_df.empty:
        st.info("No association rules found meeting confidence threshold. Try lowering minimum confidence.")
    else:
        st.dataframe(
            rules_df[['rule_repr', 'support', 'confidence', 'lift']].rename(
                columns={'rule_repr': 'Association Rule (Antecedent ➔ Consequent)', 'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'}
            ),
            use_container_width=True,
            height=250
        )
with tab_vp:
    if not rules_df.empty:
        rv1, rv2 = st.columns(2)
        with rv1:
            st.plotly_chart(plot_rules_scatter(rules_df), use_container_width=True)
        with rv2:
            st.plotly_chart(plot_top_rules_bar(rules_df, top_n=8), use_container_width=True)

# ==============================================================================
# SECTION 4: ACTIVITY TIMELINE & DATA AUDIT
# ==============================================================================
st.markdown("---")
st.markdown("## 🕒 4. Activity Timeline & Data Audit")

if unified.get('has_event_stream', False):
    st.markdown("#### Chronological Session Activity Timeline")
    sess_col = unified['session_col']
    time_col = unified['timestamp_col']
    page_col = unified['page_col']
    event_col = unified['event_col']

    timeline_fig = plot_session_timeline(
        cleaned_events_df,
        session_col=sess_col,
        timestamp_col=time_col,
        page_col=page_col,
        event_col=event_col,
        max_sessions=8
    )
    st.plotly_chart(timeline_fig, use_container_width=True)

tab_raw, tab_derived = st.tabs(["📄 Raw Event Logs (Preserved)", "📊 Derived Session Features"])
with tab_raw:
    st.caption("Cleaned chronological activity events (preserved separately from derived features).")
    st.dataframe(cleaned_events_df.head(20), use_container_width=True)
with tab_derived:
    st.caption(f"Generated {len(session_features_df.columns)} behavioral metrics across {len(session_features_df):,} sessions.")
    st.dataframe(session_features_df.head(20), use_container_width=True)

# ==============================================================================
# SECTION 5: EXECUTIVE CRO INSIGHTS & EXPORTS
# ==============================================================================
st.markdown("---")
st.markdown("## 💡 5. Executive CRO Insights & Export Suite")

ins1, ins2 = st.columns(2)
with ins1:
    render_html_block(f"""
    <div class="navy-card">
        <h4 style="color: #00e8c6; margin-top: 0;">🎯 Intent Opportunity</h4>
        <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
            <strong>Predicted Conversion:</strong> {pred_summary['predicted_conversion_rate']:.1f}% of analyzed sessions exhibit high purchase intent. 
            Shoppers who browse product categories and interact with the Cart generate strong lift.
            Deploy targeted exit-intent overlays or instant free-shipping countdowns to capture on-the-fence visitors.
        </p>
    </div>
    """)

with ins2:
    render_html_block(f"""
    <div class="navy-card">
        <h4 style="color: #00e8c6; margin-top: 0;">🛡️ Segment Remediation</h4>
        <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
            <strong>High-Bounce Mitigation:</strong> Cluster analysis mapped sessions into {len(size_df)} distinct behavioral groups. 
            For sessions with high exit rates ({session_features_df['BounceRates'].mean() * 100:.1f}% average bounce indicator), 
            simplify navigation hierarchies from landing pages to accelerate path-to-product discovery.
        </p>
    </div>
    """)

st.markdown("#### 💾 One-Click Data & Report Exports")
d1, d2, d3, d4, d5 = st.columns(5)

with d1:
    sess_csv = predicted_sessions_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Intent Predictions CSV",
        data=sess_csv,
        file_name="session_behavioral_predictions.csv",
        mime="text/csv",
        use_container_width=True
    )

with d2:
    ev_csv = cleaned_events_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Processed Events CSV",
        data=ev_csv,
        file_name="processed_activity_events.csv",
        mime="text/csv",
        use_container_width=True
    )

with d3:
    if not frequent_itemsets.empty:
        fi_csv = frequent_itemsets[['itemsets_str', 'support', 'length']].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Frequent Itemsets CSV",
            data=fi_csv,
            file_name="frequent_itemsets.csv",
            mime="text/csv",
            use_container_width=True
        )

with d4:
    if not rules_df.empty:
        r_csv = rules_df[['rule_repr', 'support', 'confidence', 'lift']].to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Association Rules CSV",
            data=r_csv,
            file_name="association_rules.csv",
            mime="text/csv",
            use_container_width=True
        )

with d5:
    validation_info = {
        'total_events': len(cleaned_events_df),
        'total_sessions': total_sessions,
        'valid_timestamp_pct': 100.0 if unified['has_timestamps'] else 0.0,
        'errors': [],
        'warnings': []
    }
    report_text = generate_analysis_report(
        metadata={'file_name': source_name},
        validation=validation_info,
        rules_summary={
            'itemset_count': len(frequent_itemsets),
            'rule_count': len(rules_df),
            'execution_time_ms': exec_ms,
            'top_rule': rules_df.iloc[0]['rule_repr'] if not rules_df.empty else None
        },
        clustering_summary={
            'n_clusters': len(size_df),
            'features': trained_feat_cols,
            'silhouette': sil_score
        },
        prediction_summary=pred_summary
    )
    st.download_button(
        "📄 Executive Report (MD)",
        data=report_text.encode('utf-8'),
        file_name="webpathmine_analysis_report.md",
        mime="text/markdown",
        use_container_width=True
    )
