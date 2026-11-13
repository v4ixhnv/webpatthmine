"""
WebPathMine: Web Usage and Navigation Pattern Mining + User Behaviour Prediction
Workspace 4: Prediction & Insights
Implements leak-free Decision Tree classification, 5-fold cross-validation,
single and batch session inference, and data-grounded conversion optimization insights.
"""
import io
import os
import streamlit as st
import pandas as pd
import numpy as np

from src.visualization import (
    inject_custom_css,
    render_hero,
    plot_confusion_matrix,
    plot_feature_importance,
    render_purchase_verdict_card,
    generate_analysis_report
)
from src.prediction import (
    train_and_evaluate_model,
    predict_single_session,
    batch_predict_sessions,
    save_model,
    load_model,
    is_model_stale,
    MODEL_SCHEMA_VERSION
)

st.set_page_config(
    page_title="Prediction & Insights | WebPathMine",
    layout="wide"
)

st.markdown(inject_custom_css(), unsafe_allow_html=True)

st.markdown(
    render_hero(
        title="Prediction & Insights",
        subtitle="Supervised Decision Tree modeling, 5-fold cross-validation, single & batch inference, and strategic conversion insights.",
        badge="WORKSPACE 4"
    ),
    unsafe_allow_html=True
)

# Resolve dataset from session state
if st.session_state.get('df') is None:
    if os.path.exists("data/online_shoppers_intention.csv"):
        st.session_state.df = pd.read_csv("data/online_shoppers_intention.csv")
    else:
        st.warning("No dataset loaded. Please upload or load data from the Data Hub & Explorer.")
        st.stop()

df = st.session_state.df

# Identify candidate target columns
binary_candidates = []
for c in df.columns:
    if c.lower() in ['revenue', 'purchased', 'converted', 'target', 'label', 'buy', 'purchase']:
        binary_candidates.append(c)
    elif df[c].nunique() == 2:
        binary_candidates.append(c)

if not binary_candidates:
    st.warning("The currently loaded dataset does not contain a binary conversion target attribute (e.g. 'Revenue').")
    if os.path.exists("data/online_shoppers_intention.csv"):
        if st.button("Switch to Online Shoppers Intent Dataset (Synthetic 3k Sample)", use_container_width=True):
            st.session_state.df = pd.read_csv("data/online_shoppers_intention.csv")
            st.session_state.df_source_name = "Online Shoppers Intent (Synthetic 3k Sample)"
            st.session_state.dataset_type = 'synthetic_online_shoppers'
            st.session_state.model_pipeline = None
            st.session_state.model_results = None
            st.rerun()
    st.stop()

candidate_targets = binary_candidates
default_target_idx = candidate_targets.index('Revenue') if 'Revenue' in candidate_targets else 0

# Sidebar controls for training hyperparameters
with st.sidebar:
    st.markdown("### Decision Tree Parameters")
    
    target_col = st.selectbox(
        "Target Column (Purchase Label):",
        options=candidate_targets,
        index=default_target_idx,
        help="Column representing binary purchase outcome."
    )
    
    criterion = st.selectbox(
        "Split Criterion:",
        options=["gini", "entropy"],
        index=0,
        help="Gini Impurity vs Information Gain (Entropy)."
    )
    
    max_depth_opt = st.selectbox(
        "Tree Maximum Depth:",
        options=[3, 4, 5, 6, 8, 10, None],
        index=2,  # default depth 5
        format_func=lambda x: "Unlimited (Deepest)" if x is None else f"Depth {x}"
    )
    
    min_samples_split = st.slider(
        "Min Samples to Split Node:",
        min_value=2,
        max_value=30,
        value=5
    )
    
    class_weight_choice = st.selectbox(
        "Class Imbalance Weighting:",
        options=["balanced", None],
        index=0,
        format_func=lambda x: "Balanced (Counter Imbalance)" if x == "balanced" else "Equal Weights (Standard)"
    )
    
    test_size = st.slider(
        "Test Split Size (%):",
        min_value=10,
        max_value=40,
        value=20,
        step=5
    ) / 100.0

# Training Trigger and Persistence Status
col_train_btn, col_train_info = st.columns([1, 2])
with col_train_btn:
    train_clicked = st.button("Train Decision Tree Pipeline", use_container_width=True)

with col_train_info:
    disk_path = "models/decision_tree_pipeline.joblib"
    disk_stale = is_model_stale(disk_path) if os.path.exists(disk_path) else False
    if disk_stale:
        st.warning("Legacy model detected on disk (pre-v2.0 schema). Re-training required.")
    elif st.session_state.get('model_pipeline') is not None:
        st.success(f"Trained leak-free pipeline active in memory (Schema: {MODEL_SCHEMA_VERSION}).")
    else:
        st.info("Click 'Train Decision Tree Pipeline' to fit model on dataset.")

# Run training if requested or if first initialization
if train_clicked or st.session_state.get('model_pipeline') is None:
    with st.spinner("Fitting leak-free Decision Tree Pipeline with stratified split and 5-fold CV..."):
        try:
            pipeline, results = train_and_evaluate_model(
                df,
                target_col=target_col,
                criterion=criterion,
                max_depth=max_depth_opt,
                min_samples_split=min_samples_split,
                class_weight=class_weight_choice,
                test_size=test_size,
                random_state=42
            )
            st.session_state.model_pipeline = pipeline
            st.session_state.model_results = results
            
            # Persist model file to models/
            model_saved_path = "models/decision_tree_pipeline.joblib"
            save_model(pipeline, model_saved_path)
            st.session_state.model_filepath = model_saved_path
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.stop()

results = st.session_state.model_results
pipeline = st.session_state.model_pipeline

# Top KPI Evaluation Metrics
st.markdown("---")
st.markdown("### Evaluation Metrics (Test Set & 5-Fold Stratified Cross-Validation)")
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Test Accuracy", f"{results['accuracy'] * 100:.1f}%")
with m2:
    st.metric("Precision (Buyers)", f"{results['precision'] * 100:.1f}%")
with m3:
    st.metric("Recall (Buyers)", f"{results['recall'] * 100:.1f}%")
with m4:
    st.metric("F1-Score", f"{results['f1_score']:.3f}")
with m5:
    auc_str = f"{results['roc_auc']:.3f}" if results['roc_auc'] else "N/A"
    st.metric("ROC-AUC", auc_str)

if 'cv_summary' in results and results['cv_summary']:
    cv = results['cv_summary']
    st.markdown("##### 5-Fold Stratified Cross-Validation (Stability)")
    cv1, cv2, cv3, cv4 = st.columns(4)
    with cv1:
        st.metric("Mean CV Accuracy", f"{cv['cv_accuracy_mean']*100:.1f}%", f"+/- {cv['cv_accuracy_std']*100:.1f}%")
    with cv2:
        st.metric("Mean CV Precision", f"{cv['cv_precision_mean']*100:.1f}%", f"+/- {cv['cv_precision_std']*100:.1f}%")
    with cv3:
        st.metric("Mean CV Recall", f"{cv['cv_recall_mean']*100:.1f}%", f"+/- {cv['cv_recall_std']*100:.1f}%")
    with cv4:
        st.metric("Mean CV F1-Score", f"{cv['cv_f1_mean']:.3f}", f"+/- {cv['cv_f1_std']:.3f}")

# Four Consolidated Tabs
tab_eval, tab_predict, tab_insights, tab_export = st.tabs([
    "Model & Evaluation",
    "Predictions",
    "Insights & Recommendations",
    "Export"
])

# TAB 1: MODEL & EVALUATION
with tab_eval:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("#### Confusion Matrix")
        cm_fig = plot_confusion_matrix(results['confusion_matrix'])
        st.plotly_chart(cm_fig, use_container_width=True)
    with c2:
        st.markdown("#### Tree Architecture Properties")
        st.markdown(f"""
        <div class="light-card">
            <p style="margin: 0; color: #4B5563; line-height: 1.8; font-size: 0.9rem;">
                <strong>Split Criterion:</strong> {criterion.upper()}<br>
                <strong>Max Depth Configuration:</strong> {max_depth_opt} (Realized: {results['tree_depth']})<br>
                <strong>Terminal Leaves:</strong> {results['tree_leaves']}<br>
                <strong>Training Set Size:</strong> {results['train_samples']:,} sessions<br>
                <strong>Evaluation Set Size:</strong> {results['test_samples']:,} sessions<br>
                <strong>Class Weighting:</strong> {class_weight_choice}<br>
                <strong>Imbalance Ratio:</strong> {results['imbalance_ratio']}:1 (Non-Buyers to Buyers)
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Feature Importances & Direct Model Interpretation")
    
    imp_col1, imp_col2 = st.columns([1.3, 1])
    with imp_col1:
        imp_fig = plot_feature_importance(results['feature_importance'], top_n=12)
        st.plotly_chart(imp_fig, use_container_width=True)
    with imp_col2:
        st.markdown("##### Splitter Interpretation")
        top_feats = results['feature_importance'].head(3)['Feature'].tolist() if 'Feature' in results['feature_importance'].columns else results['feature_importance'].head(3)['feature'].tolist()
        top_weights = results['feature_importance'].head(3)['Importance'].tolist() if 'Importance' in results['feature_importance'].columns else results['feature_importance'].head(3)['importance'].tolist()
        
        st.markdown(f"""
        <div class="light-card">
            <h4 style="color: #F97316; margin-top: 0;">Governing Decision Drivers</h4>
            <p style="color: #4B5563; font-size: 0.9rem; line-height: 1.6;">
                1. <strong>{top_feats[0]}:</strong> Primary decision splitter ({top_weights[0] * 100:.1f}% relative weight). Produces the largest initial reduction in tree node impurity.<br>
                2. <strong>{top_feats[1]}:</strong> Secondary splitter ({top_weights[1] * 100:.1f}% relative weight).<br>
                {f'3. <strong>{top_feats[2]}:</strong> Tertiary splitter ({top_weights[2] * 100:.1f}% relative weight).' if len(top_feats) > 2 else ''}
            </p>
            <p style="color: #6B7280; font-size: 0.85rem; margin-bottom: 0;">
                Sessions exhibiting high values on primary drivers separate decisively into high-conversion leaf nodes.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(results['feature_importance'], use_container_width=True, height=200)

# TAB 2: PREDICTIONS (Single-Session Simulator + Batch Predictions & Verdict Inspector)
with tab_predict:
    st.markdown("#### Real-Time Single-Session Prediction Simulator")
    st.caption("Configure shopper browsing attributes to evaluate real-time conversion probability and verdict.")
    
    feat_cols = results['feature_columns']
    
    # Split features into primary (first 6) and secondary (in expander)
    primary_feats = feat_cols[:min(6, len(feat_cols))]
    secondary_feats = feat_cols[min(6, len(feat_cols)):]
    
    input_data = {}
    with st.form("single_session_form"):
        st.markdown("##### Primary Session Attributes")
        p_col1, p_col2, p_col3 = st.columns(3)
        for idx, col in enumerate(primary_feats):
            target_container = [p_col1, p_col2, p_col3][idx % 3]
            with target_container:
                col_series = df[col]
                if pd.api.types.is_bool_dtype(col_series):
                    input_data[col] = st.selectbox(f"{col}", options=[False, True], index=0)
                elif pd.api.types.is_numeric_dtype(col_series):
                    min_v = float(col_series.min())
                    max_v = float(col_series.max())
                    med_v = float(col_series.median())
                    is_int = pd.api.types.is_integer_dtype(col_series)
                    step_v = 1.0 if is_int else 0.01
                    input_data[col] = st.number_input(
                        f"{col}",
                        min_value=min_v,
                        max_value=max_v,
                        value=med_v,
                        step=step_v
                    )
                else:
                    unique_vals = [str(x) for x in col_series.dropna().unique().tolist()]
                    input_data[col] = st.selectbox(f"{col}", options=unique_vals, index=0)
                    
        if secondary_feats:
            with st.expander("Secondary & Categorical Attributes", expanded=False):
                s_col1, s_col2, s_col3 = st.columns(3)
                for idx, col in enumerate(secondary_feats):
                    target_container = [s_col1, s_col2, s_col3][idx % 3]
                    with target_container:
                        col_series = df[col]
                        if pd.api.types.is_bool_dtype(col_series):
                            input_data[col] = st.selectbox(f"{col}", options=[False, True], index=0)
                        elif pd.api.types.is_numeric_dtype(col_series):
                            min_v = float(col_series.min())
                            max_v = float(col_series.max())
                            med_v = float(col_series.median())
                            is_int = pd.api.types.is_integer_dtype(col_series)
                            step_v = 1.0 if is_int else 0.01
                            input_data[col] = st.number_input(
                                f"{col}",
                                min_value=min_v,
                                max_value=max_v,
                                value=med_v,
                                step=step_v
                            )
                        else:
                            unique_vals = [str(x) for x in col_series.dropna().unique().tolist()]
                            input_data[col] = st.selectbox(f"{col}", options=unique_vals, index=0)
                            
        predict_submit = st.form_submit_button("Predict Purchase Intent", use_container_width=True)
        
    if predict_submit:
        pred_res = predict_single_session(pipeline, input_data)
        st.markdown("##### Prediction Result")
        
        # Render clean verdict card
        drivers_dict = {k: input_data[k] for k in primary_feats[:4] if k in input_data}
        verdict_card_html = render_purchase_verdict_card(
            prediction=pred_res['outcome_label'],
            probability=pred_res['probability_purchase'],
            session_id="Simulated_Session",
            key_drivers=drivers_dict
        )
        st.markdown(verdict_card_html, unsafe_allow_html=True)
        st.caption(f"Statistical Disclaimer: {pred_res['disclaimer']}")

    # Batch Predictions & Master Table (migrated from Automated Workflow)
    st.markdown("---")
    st.markdown("#### Batch Session Predictions & Master Verdict Inspector")
    st.caption("Executes batch inference across active dataset sessions and allows individual session inspection.")
    
    with st.spinner("Generating batch predictions..."):
        batch_pred_df, batch_summary = batch_predict_sessions(
            pipeline,
            df,
            target_col=target_col if target_col in df.columns else None
        )
        
    st.session_state.batch_prediction_df = batch_pred_df
    st.session_state.prediction_summary = batch_summary
    
    bp1, bp2, bp3, bp4 = st.columns(4)
    with bp1:
        st.metric("Total Sessions Evaluated", f"{batch_summary['total_sessions']:,}")
    with bp2:
        st.metric("Predicted Purchases", f"{batch_summary['predicted_purchases']:,}")
    with bp3:
        st.metric("Predicted No Purchases", f"{batch_summary['predicted_no_purchases']:,}")
    with bp4:
        st.metric("Predicted Conversion Rate", f"{batch_summary['predicted_conversion_rate']:.1f}%")

    # Radio filter for master table
    verdict_filter = st.radio(
        "Filter Verdict Master Table:",
        options=["All Sessions", "Predicted Purchases Only", "Predicted No Purchases Only"],
        horizontal=True
    )
    
    filtered_batch = batch_pred_df.copy()
    if verdict_filter == "Predicted Purchases Only":
        filtered_batch = filtered_batch[filtered_batch['Predicted_Verdict'] == 'Purchase']
    elif verdict_filter == "Predicted No Purchases Only":
        filtered_batch = filtered_batch[filtered_batch['Predicted_Verdict'] == 'No Purchase']
        
    display_batch_cols = [c for c in [
        'Session_ID', 'Predicted_Verdict', 'Purchase_Probability', 'No_Purchase_Probability',
        'Actual_Ground_Truth', 'Prediction_Accurate'
    ] if c in filtered_batch.columns]
    
    st.dataframe(filtered_batch[display_batch_cols], use_container_width=True, height=280)

    # Individual Shopper Verdict Inspector
    st.markdown("##### Individual Session Verdict Inspector")
    if not filtered_batch.empty and 'Session_ID' in filtered_batch.columns:
        inspector_session_id = st.selectbox(
            "Select Session ID to Inspect Detailed Verdict:",
            options=filtered_batch['Session_ID'].tolist()[:100],
            index=0
        )
        selected_session = filtered_batch[filtered_batch['Session_ID'] == inspector_session_id].iloc[0]
        
        # Drivers from row
        sess_drivers = {}
        for feat in primary_feats[:4]:
            if feat in selected_session:
                sess_drivers[feat] = selected_session[feat]
                
        session_card_html = render_purchase_verdict_card(
            prediction=selected_session['Predicted_Verdict'],
            probability=float(selected_session['Purchase_Probability']),
            session_id=str(inspector_session_id),
            key_drivers=sess_drivers
        )
        st.markdown(session_card_html, unsafe_allow_html=True)

# TAB 3: INSIGHTS & RECOMMENDATIONS (Dynamic multi-module synthesis)
with tab_insights:
    st.markdown("#### Strategic Insights & Conversion Rate Optimization (CRO) Recommendations")
    st.caption("Derived dynamically from empirical pattern mining, clustering, ranking, and prediction models.")
    
    rules_res = st.session_state.get('rules_results')
    clust_res = st.session_state.get('clustering_results')
    rank_res = st.session_state.get('ranking_results')
    
    obs_cards = []
    
    # 1. Prediction Driver
    if results is not None and 'feature_importance' in results:
        imp_df = results['feature_importance']
        top_f = imp_df.head(1)['Feature'].iloc[0] if 'Feature' in imp_df.columns else imp_df.head(1)['feature'].iloc[0]
        top_w = float(imp_df.head(1)['Importance'].iloc[0]) if 'Importance' in imp_df.columns else float(imp_df.head(1)['importance'].iloc[0])
        obs_cards.append({
            'title': f"Intent Driver: Optimize Around '{top_f}'",
            'finding': f"Decision tree analysis isolates {top_f} as the primary split driver ({top_w:.1%} relative importance).",
            'action': f"Prioritize conversion rate optimization efforts around {top_f}. Ensure clear navigational paths and prominent value signals for this behavior."
        })
    else:
        obs_cards.append({
            'title': "Conversion Modeling Recommendation",
            'finding': "Supervised classification has not yet been executed.",
            'action': "Train the Decision Tree Pipeline in Tab 1 to isolate statistical predictors of purchase conversion."
        })

    # 2. Association Flow
    if rules_res is not None and 'rules' in rules_res and not rules_res['rules'].empty:
        top_r = rules_res['rules'].sort_values(by='lift', ascending=False).iloc[0]
        ant_str = ', '.join(list(top_r['antecedents']))
        con_str = ', '.join(list(top_r['consequents']))
        obs_cards.append({
            'title': f"Navigation Flow: Link '{ant_str}' to '{con_str}'",
            'finding': f"High association lift ({float(top_r['lift']):.2f}x) observed between {ant_str} and {con_str} (confidence: {float(top_r['confidence']):.1%}).",
            'action': f"Implement direct in-page shortcuts or cross-recommendation widgets connecting {ant_str} directly to {con_str} to minimize journey friction."
        })
    else:
        obs_cards.append({
            'title': "Navigation Mining Recommendation",
            'finding': "No high-lift association rules currently active in session cache.",
            'action': "Run FP-Growth in Workspace 2 (Pattern Mining) with tuned support thresholds to discover co-occurring navigation paths."
        })

    # 3. Behavioral Segment Prioritization
    if clust_res is not None and 'descriptions' in clust_res and len(clust_res['descriptions']) > 0:
        largest_cluster = max(clust_res['descriptions'], key=lambda x: x['size'])
        obs_cards.append({
            'title': f"Audience Focus: '{largest_cluster['cluster']}' ({largest_cluster['pct']}%)",
            'finding': f"The largest discovered segment represents {largest_cluster['size']:,} sessions with dominant traits: {'; '.join(largest_cluster['characteristics'][:2])}.",
            'action': "Tailor landing page messaging and search filters to address the dominant behavioral pattern observed in this majority segment."
        })
    else:
        obs_cards.append({
            'title': "Segmentation Recommendation",
            'finding': "No cluster segments currently active in session cache.",
            'action': "Run K-Means clustering in Workspace 3 (Segmentation & Engagement) to profile user behavioral segments."
        })

    # 4. Engagement Ranking Strategy
    if rank_res is not None and 'meta' in rank_res:
        r_meta = rank_res['meta']
        obs_cards.append({
            'title': f"Engagement Targeting: {r_meta['high_intent_pct']}% High Intent",
            'finding': f"Out of {r_meta['total_records']:,} evaluated {r_meta['entity_type']}s, {r_meta['high_intent_count']:,} scored >= 70 on pre-outcome engagement.",
            'action': "Target high-intent sessions with immediate real-time assistance or checkout incentives, while providing educational orientation to low-intent casual visitors."
        })
    else:
        obs_cards.append({
            'title': "Engagement Scoring Recommendation",
            'finding': "Behavioural engagement leaderboard not yet computed.",
            'action': "View Workspace 3 (Segmentation & Engagement) to compute the 0–100 behavioural engagement leaderboard."
        })

    # Render 2x2 Grid of Dynamic Insights Cards
    for i in range(0, len(obs_cards), 2):
        c1, c2 = st.columns(2)
        with c1:
            card = obs_cards[i]
            st.markdown(f"""
            <div class="light-card">
                <h4 style="color: #F97316; margin-top: 0;">{card['title']}</h4>
                <p style="color: #4B5563; font-size: 0.9rem; line-height: 1.6;">
                    <strong>Empirical Finding:</strong> {card['finding']}<br>
                    <strong>Recommended Action:</strong> {card['action']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        if i + 1 < len(obs_cards):
            with c2:
                card = obs_cards[i + 1]
                st.markdown(f"""
                <div class="light-card">
                    <h4 style="color: #F97316; margin-top: 0;">{card['title']}</h4>
                    <p style="color: #4B5563; font-size: 0.9rem; line-height: 1.6;">
                        <strong>Empirical Finding:</strong> {card['finding']}<br>
                        <strong>Recommended Action:</strong> {card['action']}
                    </p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top: 1.25rem; padding: 0.75rem 1rem; border-left: 3px solid #F97316; background: #FFF7ED; border-radius: 4px;">
        <p style="margin: 0; color: #6B7280; font-size: 0.85rem; line-height: 1.5;">
            <strong>Scientific Disclaimer:</strong> Discovered association rules and feature importances reflect empirical correlations 
            within the analyzed dataset. Association does not prove causality, nor does an engagement score guarantee conversion. 
            Proposed business actions should be validated through controlled A/B experimentation before widespread deployment.
        </p>
    </div>
    """, unsafe_allow_html=True)

# TAB 4: EXPORT
with tab_export:
    st.markdown("#### Export Model Artifacts & Comprehensive Analysis Report")
    
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    
    # 1. Download Persisted Model Pipeline
    with exp_col1:
        st.markdown("##### Trained Model Pipeline")
        st.caption("Serialized Scikit-learn Pipeline (preprocessing + DecisionTreeClassifier) using Joblib.")
        if os.path.exists("models/decision_tree_pipeline.joblib"):
            with open("models/decision_tree_pipeline.joblib", "rb") as f:
                model_bytes = f.read()
            st.download_button(
                label="Download Trained Model (.joblib)",
                data=model_bytes,
                file_name="decision_tree_pipeline.joblib",
                mime="application/octet-stream",
                use_container_width=True
            )
        else:
            st.info("Train the model above to generate the serializable joblib artifact.")
            
    # 2. Download Batch Predictions CSV
    with exp_col2:
        st.markdown("##### Batch Predictions CSV")
        st.caption("Session-level conversion predictions, probabilities, and evaluation tags.")
        if 'batch_prediction_df' in st.session_state and st.session_state.batch_prediction_df is not None:
            pred_csv = st.session_state.batch_prediction_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=pred_csv,
                file_name="webpathmine_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Run batch predictions in the Predictions tab to export.")
            
    # 3. Download Executive Markdown Report
    with exp_col3:
        st.markdown("##### Executive Analysis Report")
        st.caption("Comprehensive text report synthesizing quality, patterns, clusters, and predictions.")
        
        # Prepare validation and summaries
        val_info = {
            'total_events': len(df),
            'total_sessions': len(df),
            'valid_timestamp_pct': 100.0 if 'timestamp' in df.columns else 0.0,
            'errors': [],
            'warnings': []
        }
        pred_sum = st.session_state.get('prediction_summary')
        clust_sum = st.session_state.get('clustering_results')
        rules_sum = {
            'itemset_count': len(st.session_state.rules_results['itemsets']) if st.session_state.get('rules_results') else 0,
            'rule_count': len(st.session_state.rules_results['rules']) if st.session_state.get('rules_results') else 0,
            'execution_time_ms': (st.session_state.rules_results['exec_time'] * 1000) if st.session_state.get('rules_results') else 0.0
        } if st.session_state.get('rules_results') else None
        
        active_source_str = st.session_state.get('df_source_name') or 'Active Dataset'
        report_text = generate_analysis_report(
            metadata={'file_name': active_source_str},
            validation=val_info,
            rules_summary=rules_sum,
            clustering_summary=clust_sum,
            prediction_summary=pred_sum
        )
        
        st.download_button(
            label="Download Executive Report (Markdown)",
            data=report_text.encode('utf-8'),
            file_name="webpathmine_analysis_report.md",
            mime="text/markdown",
            use_container_width=True
        )
