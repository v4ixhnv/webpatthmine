"""
Page 3: Behavioral Segmentation
Implements K-Means clustering with feature standardization, elbow/silhouette diagnostics,
PCA 2D projection, normalized radar profiles, and objective data-grounded characterization.
"""
import io
import os
import streamlit as st
import pandas as pd
import numpy as np

from src.visualization import (
    inject_custom_css,
    render_hero,
    plot_elbow_and_silhouette,
    plot_clusters_pca,
    plot_cluster_sizes,
    plot_cluster_radar
)
from src.clustering import (
    prepare_clustering_data,
    fit_kmeans,
    compute_elbow_and_silhouette,
    project_pca,
    generate_cluster_profiles,
    DEFAULT_BEHAVIORAL_FEATURES
)
from src.feature_engineering import compute_behavioural_engagement_score

st.set_page_config(
    page_title="Behavior Segmentation | WEBPATHMINE",
    page_icon="👥",
    layout="wide"
)

st.markdown(inject_custom_css(), unsafe_allow_html=True)

st.markdown(
    render_hero(
        title="BEHAVIORAL <span>SEGMENTATION</span>",
        subtitle="Unsupervised K-Means Session Clustering, Silhouette Validation & Empirical Profiles",
        badge="MODULE 3: USER SEGMENTATION"
    ),
    unsafe_allow_html=True
)

# Resolve dataset
if st.session_state.get('df') is None:
    if os.path.exists("data/online_shoppers_intention.csv"):
        st.session_state.df = pd.read_csv("data/online_shoppers_intention.csv")
    else:
        st.warning("No dataset loaded. Please upload or load data from the Data Explorer page.")
        st.stop()

df = st.session_state.df

# Find numerical candidates
numeric_all = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_all) < 2:
    st.warning("⚠️ The currently active dataset is an event-level clickstream log with insufficient numerical features for behavioral clustering.")
    if os.path.exists("data/online_shoppers_intention.csv"):
        if st.button("📊 Switch to UCI Online Shoppers Dataset for Clustering", use_container_width=True):
            st.session_state.df = pd.read_csv("data/online_shoppers_intention.csv")
            st.session_state.df_source_name = "UCI Online Shoppers"
            st.session_state.dataset_type = 'uci_online_shoppers'
            st.rerun()
    st.stop()

# Determine default features
default_features = [f for f in DEFAULT_BEHAVIORAL_FEATURES if f in numeric_all]
if not default_features:
    default_features = numeric_all[:min(5, len(numeric_all))]

# Controls
st.markdown("### ⚙️ Segmentation Parameters")
col_feat, col_k = st.columns([3, 1])

with col_feat:
    selected_features = st.multiselect(
        "Select Numerical Behavioral Features for Clustering:",
        options=numeric_all,
        default=default_features,
        help="Features will be automatically standardized using StandardScaler before clustering."
    )

with col_k:
    n_clusters = st.slider(
        "Number of Clusters (k):",
        min_value=2,
        max_value=8,
        value=3,
        help="Target number of shopper behavioral segments."
    )

if len(selected_features) < 2:
    st.warning("Please select at least 2 numerical features to perform clustering.")
    st.stop()

# Prepare Data & Standardize
scaled_data, scaler, valid_subset = prepare_clustering_data(df, selected_features)

# Run KMeans
kmeans_model, cluster_labels, sil_score = fit_kmeans(
    scaled_data,
    n_clusters=n_clusters,
    random_state=42
)

# Project to 2D PCA for visual mapping
pca_coords, var_ratios = project_pca(scaled_data, n_components=2, random_state=42)

# Generate Profiles
profile_df, size_df, descriptions = generate_cluster_profiles(df, cluster_labels, selected_features)

# Attach Cluster assignment to dataset copy for ranking and export
df_segmented = df.copy()
df_segmented['Cluster'] = [f"Cluster {l}" for l in cluster_labels]

# Compute Explainable Behavioural Engagement Ranking (Stage 4)
ranking_df, rank_meta = compute_behavioural_engagement_score(df_segmented)
st.session_state.ranking_results = {
    'ranking_df': ranking_df,
    'meta': rank_meta
}

# Save to session state for Model Insights page
st.session_state.clustering_results = {
    'features': selected_features,
    'n_clusters': n_clusters,
    'silhouette': sil_score,
    'profile_df': profile_df,
    'size_df': size_df,
    'descriptions': descriptions
}

# Top KPI Metric Cards
st.markdown("---")
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Total Clustered Sessions", f"{len(cluster_labels):,}")
with k2:
    st.metric("Active Clusters (k)", f"{n_clusters}")
with k3:
    sil_display = f"{sil_score:.3f}" if sil_score is not None else "N/A"
    st.metric("Silhouette Score", sil_display)
with k4:
    # Interpretation badge
    if sil_score is None:
        interp = "N/A"
    elif sil_score >= 0.50:
        interp = "Strong Structure (>0.50)"
    elif sil_score >= 0.25:
        interp = "Reasonable Structure (0.25-0.50)"
    else:
        interp = "Weak / Overlapping (<0.25)"
    st.metric("Structure Quality", interp)

# Tabs
tab_profiles, tab_viz, tab_eval, tab_inspect, tab_ranking = st.tabs([
    "📋 Cluster Profiles & Traits",
    "🗺️ PCA Cluster Map & Radar",
    "📈 Elbow & Silhouette Curves",
    "🔎 Individual Cluster Inspector",
    "🏆 Behavioural Engagement Ranking"
])

with tab_profiles:
    st.markdown("#### Quantitative Cluster Feature Means")
    st.caption("Empirical average values for each selected feature across the discovered clusters.")
    st.dataframe(profile_df, use_container_width=True)
    
    st.markdown("#### Segment Size Breakdown")
    s_col1, s_col2 = st.columns([1, 1])
    with s_col1:
        st.dataframe(size_df, use_container_width=True)
    with s_col2:
        donut_fig = plot_cluster_sizes(size_df)
        st.plotly_chart(donut_fig, use_container_width=True)

with tab_viz:
    st.markdown("#### Cluster Visualizations")
    v1, v2 = st.columns([1, 1])
    with v1:
        pca_fig = plot_clusters_pca(pca_coords, cluster_labels, var_ratios)
        st.plotly_chart(pca_fig, use_container_width=True)
    with v2:
        radar_fig = plot_cluster_radar(profile_df, selected_features)
        st.plotly_chart(radar_fig, use_container_width=True)

with tab_eval:
    st.markdown("#### Model Diagnostics: Elbow & Silhouette Curves")
    st.caption("Evaluate inertia (sum of squared distances) and silhouette scores to verify optimal k.")
    with st.spinner("Computing diagnostics for k in [2, 8]..."):
        diag = compute_elbow_and_silhouette(scaled_data, k_range=range(2, 9))
        diag_fig = plot_elbow_and_silhouette(diag)
        st.plotly_chart(diag_fig, use_container_width=True)
        
        valid_sil = [s for s in diag['silhouette_scores'] if s is not None]
        if valid_sil:
            max_sil = max(valid_sil)
            best_k = diag['k_values'][diag['silhouette_scores'].index(max_sil)]
            st.info(
                f"💡 **Clustering Guidance:** Among tested values $k \\in [2, 8]$, "
                f"**k = {best_k}** achieves the highest silhouette score (**{max_sil:.3f}**). "
                f"Note: Silhouette scores measure geometric compactness and separation; final segment selection should also "
                f"consider operational business interpretability and actionability."
            )

with tab_inspect:
    st.markdown("#### Objective Data-Grounded Behavioral Interpretations")
    st.caption(
        "Scientific Policy: Behavioral characteristics are strictly computed from metric z-scores against dataset baseline averages."
    )
    for desc in descriptions:
        with st.expander(f"📌 {desc['cluster']} — {desc['size']:,} Sessions ({desc['pct']}%)", expanded=True):
            st.markdown(f"**Identified Quantitative Characteristics:**")
            for trait in desc['characteristics']:
                st.markdown(f"- {trait}")
            
            # Show actual feature values for this cluster
            c_row = profile_df.loc[desc['cluster']].to_frame().T
            st.dataframe(c_row, use_container_width=True)

with tab_ranking:
    st.markdown("#### 🏆 Explainable Behavioural Engagement Leaderboard (0–100)")
    st.caption(
        f"Entity Evaluated: **{rank_meta['entity_type']}s** | "
        f"Formula: Rule-based composite of Product Depth, Browsing Duration, Cart/Checkout actions, and Bounce Penalties."
    )
    
    # KPI metrics for ranking
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        st.metric("Total Ranked", f"{rank_meta['total_records']:,}")
    with r_col2:
        st.metric("Mean Engagement Score", f"{rank_meta['mean_score']:.1f} / 100")
    with r_col3:
        st.metric("High Intent Share (≥70)", f"{rank_meta['high_intent_pct']}%", f"{rank_meta['high_intent_count']:,} {rank_meta['entity_type']}s")
    with r_col4:
        st.metric("Moderate Intent (40–69)", f"{rank_meta['moderate_intent_count']:,}")

    # Educational & Scientific Alert
    st.markdown(f"""
    <div class="navy-card">
        <h4 style="color: #00e8c6; margin: 0 0 0.5rem 0;">📐 Explainable Scoring Formula & Weights</h4>
        <p style="margin: 0; color: #cbd5e1; font-size: 0.92rem; line-height: 1.5;">
            <strong>Formula:</strong> $Score = w_1 \\cdot \\text{{ProductDepth}} + w_2 \\cdot \\text{{BrowseDuration}} + w_3 \\cdot \\text{{CartActivity}} + w_4 \\cdot \\text{{CheckoutSteps}} - \\text{{BouncePenalty}}$<br>
            <strong>Applied Normalized Weights:</strong> {', '.join([f'{k.capitalize()}: {w:.1%}' for k, w in rank_meta['weights_applied'].items()])}<br>
            <strong>Target Leakage Protection:</strong> <code>Revenue</code>, <code>has_purchase</code>, and <code>PageValues</code> are strictly <em>excluded</em> from scoring. 
            This metric reflects observed browsing intensity, <strong>not a guaranteed purchase conversion</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Filters and Search
    st.markdown("##### 🔍 Filter & Search Leaderboard")
    f_c1, f_c2, f_c3 = st.columns([1, 1, 1.5])
    
    all_clusters = ["All Clusters"] + sorted([c for c in ranking_df['Cluster'].dropna().unique()])
    with f_c1:
        selected_cluster_filter = st.selectbox("Filter by Cluster:", options=all_clusters, index=0)
    
    tier_options = ["All Tiers", "High Intent (Active Explorer)", "Moderate Intent (Evaluating)", "Low Intent (Casual Visitor)"]
    with f_c2:
        selected_tier_filter = st.selectbox("Filter by Intent Tier:", options=tier_options, index=0)
        
    with f_c3:
        search_query = st.text_input(f"Search by {rank_meta['entity_type']} ID:", placeholder="e.g. Session_0001 or ID")

    # Apply filters
    filtered_ranking = ranking_df.copy()
    if selected_cluster_filter != "All Clusters":
        filtered_ranking = filtered_ranking[filtered_ranking['Cluster'] == selected_cluster_filter]
    if selected_tier_filter != "All Tiers":
        filtered_ranking = filtered_ranking[filtered_ranking['Engagement_Tier'] == selected_tier_filter]
    if search_query.strip():
        filtered_ranking = filtered_ranking[filtered_ranking['Entity_ID'].astype(str).str.contains(search_query.strip(), case=False)]

    st.markdown(f"Displaying **{len(filtered_ranking):,}** of **{len(ranking_df):,}** records:")
    
    display_cols = [c for c in [
        'Rank', 'Entity_ID', 'Engagement_Score', 'Engagement_Tier', 'Cluster',
        'Score_Product', 'Score_Duration', 'Score_Cart', 'Score_Checkout', 'Bounce_Penalty'
    ] if c in filtered_ranking.columns]
    
    st.dataframe(filtered_ranking[display_cols], use_container_width=True, height=350)

    # Single Record Deep Dive Inspector
    st.markdown("##### 🔬 Individual Record Score Breakdown")
    if not filtered_ranking.empty:
        sample_ids = filtered_ranking['Entity_ID'].tolist()
        inspect_id = st.selectbox(f"Select a {rank_meta['entity_type']} to Inspect:", options=sample_ids, index=0)
        selected_row = ranking_df[ranking_df['Entity_ID'] == inspect_id].iloc[0]
        
        i_col1, i_col2, i_col3 = st.columns([1, 1, 1])
        with i_col1:
            st.metric("Entity ID", str(selected_row['Entity_ID']))
            st.metric("Engagement Score", f"{selected_row['Engagement_Score']} / 100")
            st.metric("Tier", str(selected_row['Engagement_Tier']))
        with i_col2:
            st.write(f"- **Product Catalog Depth:** `{selected_row.get('Score_Product', 0.0):.1f} / 100`")
            st.write(f"- **Browsing Duration:** `{selected_row.get('Score_Duration', 0.0):.1f} / 100`")
            st.write(f"- **Cart Additions:** `{selected_row.get('Score_Cart', 0.0):.1f} / 100`")
            st.write(f"- **Checkout Progression:** `{selected_row.get('Score_Checkout', 0.0):.1f} / 100`")
        with i_col3:
            st.write(f"- **Bounce / Exit Penalty:** `-{selected_row.get('Bounce_Penalty', 0.0):.1f} pts`")
            if 'Cluster' in selected_row:
                st.write(f"- **Assigned Segment:** `{selected_row['Cluster']}`")
            if 'Revenue' in selected_row:
                st.write(f"- **Ground-Truth Purchase Label:** `{selected_row['Revenue']}`")

    # CSV Download Button
    csv_buf = io.StringIO()
    ranking_df.to_csv(csv_buf, index=False)
    csv_bytes = csv_buf.getvalue().encode('utf-8')
    st.download_button(
        label=f"📥 Download Behavioural Engagement Ranking CSV ({len(ranking_df):,} records)",
        data=csv_bytes,
        file_name="webpathmine_engagement_ranking.csv",
        mime="text/csv",
        use_container_width=True
    )
