"""
End-to-end integration verification script for WEBPATHMINE.
Verifies loading, preprocessing, pattern mining, clustering, and prediction pipeline
against both generated datasets.
"""
import os
import sys
import pandas as pd
import numpy as np

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_dataset, inspect_dataset, detect_dataset_type, clean_dataset
from src.feature_engineering import clickstream_to_transactions, shoppers_to_transactions, extract_navigation_sequences
from src.pattern_mining import run_fpgrowth, generate_rules
from src.clustering import prepare_clustering_data, fit_kmeans, compute_elbow_and_silhouette, generate_cluster_profiles
from src.prediction import train_and_evaluate_model, predict_single_session, save_model, load_model

def run_integration_check():
    print("=== 1. Testing UCI Online Shoppers Dataset ===")
    shoppers_path = "data/online_shoppers_intention.csv"
    assert os.path.exists(shoppers_path), f"File {shoppers_path} not found"
    
    df_shop = load_dataset(shoppers_path)
    print(f"Loaded shoppers dataset: {df_shop.shape}")
    
    audit_shop = inspect_dataset(df_shop)
    print(f"Audit: {audit_shop['n_rows']} rows, {audit_shop['n_cols']} cols, missing: {audit_shop['total_missing_cells']}")
    assert audit_shop['n_rows'] == 3000
    assert detect_dataset_type(df_shop) == 'uci_online_shoppers'
    
    cleaned_shop, report = clean_dataset(df_shop, drop_duplicates=True, missing_strategy='impute', target_col='Revenue')
    print(f"Cleaned shoppers: {cleaned_shop.shape}")
    
    print("\n=== 2. Testing Decision Tree Pipeline (Purchase Prediction) ===")
    pipeline_gini, res_gini = train_and_evaluate_model(
        cleaned_shop,
        target_col='Revenue',
        criterion='gini',
        max_depth=5,
        min_samples_split=5,
        class_weight='balanced',
        test_size=0.2
    )
    print(f"Gini Model Accuracy: {res_gini['accuracy']:.4f}, Recall: {res_gini['recall']:.4f}, F1: {res_gini['f1_score']:.4f}")
    
    pipeline_ent, res_ent = train_and_evaluate_model(
        cleaned_shop,
        target_col='Revenue',
        criterion='entropy',
        max_depth=5,
        min_samples_split=5,
        class_weight='balanced',
        test_size=0.2
    )
    print(f"Entropy Model Accuracy: {res_ent['accuracy']:.4f}, Recall: {res_ent['recall']:.4f}, F1: {res_ent['f1_score']:.4f}")
    
    # Test single-session inference
    sample_sess = cleaned_shop.iloc[0].to_dict()
    sample_sess.pop('Revenue', None)
    pred_res = predict_single_session(pipeline_gini, sample_sess)
    print(f"Single session prediction: {pred_res['outcome_label']} (Prob: {pred_res['probability_purchase']})")
    
    # Save & reload model
    save_model(pipeline_gini, "models/decision_tree_pipeline.joblib")
    loaded_pipe = load_model("models/decision_tree_pipeline.joblib")
    pred_res_loaded = predict_single_session(loaded_pipe, sample_sess)
    assert pred_res['outcome_label'] == pred_res_loaded['outcome_label']
    print("Pipeline model persistence verified successfully!")
    
    print("\n=== 3. Testing Behavioral Clustering (K-Means) ===")
    cluster_features = ['Administrative_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
    scaled_data, scaler, valid_subset = prepare_clustering_data(cleaned_shop, cluster_features)
    km_model, labels, sil = fit_kmeans(scaled_data, n_clusters=3)
    print(f"K-Means (k=3) fitted. Silhouette score: {sil:.4f}")
    
    prof_df, size_df, descs = generate_cluster_profiles(cleaned_shop, labels, cluster_features)
    print(f"Cluster sizes:\n{size_df}")
    
    diag = compute_elbow_and_silhouette(scaled_data, k_range=range(2, 6))
    print(f"Inertias for k in 2..5: {diag['inertias']}")
    
    print("\n=== 4. Testing Clickstream Dataset & Pattern Mining (FP-Growth) ===")
    click_path = "data/clickstream_sample.csv"
    assert os.path.exists(click_path), f"File {click_path} not found"
    
    df_click = load_dataset(click_path)
    print(f"Loaded clickstream dataset: {df_click.shape}")
    assert detect_dataset_type(df_click) == 'clickstream'
    
    tx, one_hot, items = clickstream_to_transactions(df_click, 'session_id', 'page_name', 'timestamp')
    print(f"Extracted {len(tx)} transactions, {len(items)} unique pages.")
    
    itemsets, exec_time = run_fpgrowth(one_hot, min_support=0.08)
    print(f"FP-Growth found {len(itemsets)} frequent itemsets in {exec_time * 1000:.2f} ms.")
    
    rules = generate_rules(itemsets, metric='confidence', min_threshold=0.3)
    print(f"Generated {len(rules)} association rules.")
    if not rules.empty:
        print(f"Top rule: {rules.iloc[0]['rule_repr']} (Conf: {rules.iloc[0]['confidence']:.2f}, Lift: {rules.iloc[0]['lift']:.2f})")
        
    seqs, prob_matrix = extract_navigation_sequences(df_click, 'session_id', 'page_name', 'timestamp')
    print(f"Extracted {len(seqs)} sequential paths. Transition matrix shape: {prob_matrix.shape}")
    
    print("\n=== ALL INTEGRATION CHECKS PASSED SUCCESSFULLY! ===")

if __name__ == '__main__':
    run_integration_check()
