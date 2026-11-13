"""
Page 4: Purchase Intent Prediction
Implements leak-free Scikit-learn Pipeline (preprocessing + DecisionTreeClassifier),
criteria selection (gini/entropy), class imbalance breakdown, evaluation metrics,
confusion matrix, feature importances, model export, and live single-session inference.
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
    plot_feature_importance
)
from src.prediction import (
    train_and_evaluate_model,
    predict_single_session,
    save_model,
    load_model
)

st.set_page_config(
    page_title="Purchase Prediction | WEBPATHMINE",
    page_icon="🎯",
    layout="wide"
)

st.markdown(inject_custom_css(), unsafe_allow_html=True)

st.markdown(
    render_hero(
        title="PURCHASE INTENT <span>PREDICTION</span>",
        subtitle="Supervised Decision Tree Classifier Pipeline with Gini/Entropy Optimization & Live Inference",
        badge="MODULE 4: CONVERSION PREDICTION"
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

# Identify candidate target columns
binary_candidates = []
for c in df.columns:
    if c.lower() in ['revenue', 'purchased', 'converted', 'target', 'label', 'buy', 'purchase']:
        binary_candidates.append(c)
    elif df[c].nunique() == 2:
        binary_candidates.append(c)

if not binary_candidates:
    st.warning("⚠️ The currently loaded dataset does not contain a binary purchase conversion target (e.g. 'Revenue').")
    if os.path.exists("data/online_shoppers_intention.csv"):
        if st.button("🎯 Switch to UCI Online Shoppers Dataset for Intent Prediction", use_container_width=True):
            st.session_state.df = pd.read_csv("data/online_shoppers_intention.csv")
            st.session_state.df_source_name = "UCI Online Shoppers"
            st.session_state.dataset_type = 'uci_online_shoppers'
            st.session_state.model_pipeline = None
            st.session_state.model_results = None
            st.rerun()
    st.stop()

candidate_targets = binary_candidates
default_target_idx = candidate_targets.index('Revenue') if 'Revenue' in candidate_targets else 0

# Sidebar controls for training hyperparameters
with st.sidebar:
    st.markdown("### 🌲 Decision Tree Parameters")
    
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
        index=2,  # default 5
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

# Model Training Trigger / Session persistence
col_train_btn, col_train_info = st.columns([1, 2])
with col_train_btn:
    train_clicked = st.button("🚀 Train Decision Tree Pipeline", use_container_width=True)

with col_train_info:
    if st.session_state.get('model_pipeline') is not None:
        st.success("Trained pipeline currently active in session state.")
    else:
        st.info("Click 'Train Decision Tree Pipeline' to fit model on dataset.")

# Run training if requested or if first time
if train_clicked or st.session_state.get('model_pipeline') is None:
    with st.spinner("Training leak-free Decision Tree Pipeline with stratified split..."):
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
st.markdown("### 📊 Pipeline Evaluation Metrics (Test Set)")
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Test Accuracy", f"{results['accuracy'] * 100:.1f}%")
with m2:
    st.metric("Precision (Purchase)", f"{results['precision'] * 100:.1f}%")
with m3:
    st.metric("Recall (Purchase)", f"{results['recall'] * 100:.1f}%")
with m4:
    st.metric("F1-Score", f"{results['f1_score']:.3f}")
with m5:
    auc_str = f"{results['roc_auc']:.3f}" if results['roc_auc'] else "N/A"
    st.metric("ROC-AUC", auc_str)

# Class Imbalance Educational Alert
st.markdown(f"""
<div class="navy-card">
    <h4 style="color: #00e8c6; margin: 0 0 0.5rem 0;">⚖️ Class Distribution & Imbalance Context</h4>
    <p style="margin: 0; color: #cbd5e1; font-size: 0.92rem; line-height: 1.5;">
        <strong>Dataset Distribution:</strong> {results['class_distribution']['No Purchase (0)']:,} No Purchase vs. 
        {results['class_distribution']['Purchase (1)']:,} Purchase sessions 
        (Imbalance Ratio: <strong>{results['imbalance_ratio']}:1</strong>).<br>
        In severe class imbalance, a naive model that predicts "No Purchase" for all visitors achieves high nominal accuracy 
        (~{100 - (results['class_distribution']['Purchase (1)'] / sum(results['class_distribution'].values()) * 100):.1f}%) 
        while detecting 0 real buyers. Therefore, <strong>Recall</strong>, <strong>Precision</strong>, and <strong>F1-Score</strong> 
        are the governing criteria for evaluating conversion predictability.
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_eval, tab_importance, tab_predict, tab_export = st.tabs([
    "📈 Confusion Matrix & Tree Architecture",
    "⭐ Feature Importance",
    "🧪 Live Single-Session Prediction",
    "💾 Download Persisted Model"
])

with tab_eval:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Confusion Matrix")
        cm_fig = plot_confusion_matrix(results['confusion_matrix'])
        st.plotly_chart(cm_fig, use_container_width=True)
    with c2:
        st.markdown("#### Tree Architecture Properties")
        st.markdown(f"""
        - **Split Criterion:** `{criterion.upper()}`
        - **Max Depth Limit:** `{max_depth_opt}` (Actual Realized Depth: `{results['tree_depth']}`)
        - **Total Terminal Leaves:** `{results['tree_leaves']}`
        - **Training Set Sessions:** `{results['train_samples']:,}`
        - **Evaluation Set Sessions:** `{results['test_samples']:,}`
        - **Class Weighting:** `{class_weight_choice}`
        """)

with tab_importance:
    st.markdown("#### Feature Importances")
    st.caption("Normalized total reduction of Gini impurity or entropy brought by each feature across all tree splits.")
    imp_fig = plot_feature_importance(results['feature_importance'], top_n=12)
    st.plotly_chart(imp_fig, use_container_width=True)
    
    st.dataframe(results['feature_importance'], use_container_width=True, height=250)

with tab_predict:
    st.markdown("#### Real-Time Single-Session Inference")
    st.caption("Configure browsing attributes for an incoming shopper session to evaluate predicted conversion intent.")
    
    # Generate form based on feature columns
    feat_cols = results['feature_columns']
    
    with st.form("single_session_form"):
        p_col1, p_col2, p_col3 = st.columns(3)
        
        input_data = {}
        for idx, col in enumerate(feat_cols):
            target_col_container = [p_col1, p_col2, p_col3][idx % 3]
            with target_col_container:
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
                    # Categorical input
                    unique_vals = [str(x) for x in col_series.dropna().unique().tolist()]
                    input_data[col] = st.selectbox(f"{col}", options=unique_vals, index=0)
                    
        predict_submit = st.form_submit_button("🔮 Predict Purchase Intent", use_container_width=True)
        
    if predict_submit:
        pred_res = predict_single_session(pipeline, input_data)
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            if pred_res['outcome_label'] == "Purchase":
                st.success("### 🎉 Prediction: PURCHASE")
            else:
                st.warning("### 🛑 Prediction: NO PURCHASE")
                
            st.write(f"- **Purchase Probability:** `{pred_res['probability_purchase'] * 100:.1f}%`")
            st.write(f"- **No-Purchase Probability:** `{pred_res['probability_no_purchase'] * 100:.1f}%`")
            st.progress(float(pred_res['probability_purchase']))
            
        with res_col2:
            st.info(f"ℹ️ **Statistical Disclaimer:**\n\n{pred_res['disclaimer']}")

with tab_export:
    st.markdown("#### Persisted Model Pipeline Download")
    st.write("The complete scikit-learn pipeline (including ColumnTransformer, Imputers, Scalers, OneHotEncoders, and DecisionTreeClassifier) is serialized using Joblib.")
    
    if os.path.exists("models/decision_tree_pipeline.joblib"):
        with open("models/decision_tree_pipeline.joblib", "rb") as f:
            model_bytes = f.read()
            
        st.download_button(
            label="📥 Download Trained Model Pipeline (.joblib)",
            data=model_bytes,
            file_name="decision_tree_pipeline.joblib",
            mime="application/octet-stream",
            use_container_width=True
        )
    else:
        st.info("Train the model above to generate the serializable joblib file.")
