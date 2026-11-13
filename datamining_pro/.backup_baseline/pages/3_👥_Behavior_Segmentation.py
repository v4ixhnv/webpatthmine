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
tab_profiles, tab_viz, tab_eval, tab_inspect = st.tabs([
    "📋 Cluster Profiles & Traits",
    "🗺️ PCA Cluster Map & Radar",
    "📈 Elbow & Silhouette Curves",
    "🔎 Individual Cluster Inspector"
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
