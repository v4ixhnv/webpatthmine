"""
Verifies end-to-end execution of the automated workflow on JSON and CSV files.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_dataset, validate_activity_dataset
from src.feature_engineering import aggregate_events_to_sessions, clickstream_to_transactions
from src.pattern_mining import run_fpgrowth, generate_rules
from src.clustering import get_or_train_reference_clustering, assign_clusters, generate_cluster_profiles
from src.prediction import get_or_train_reference_pipeline, batch_predict_sessions
from src.visualization import plot_session_timeline, generate_analysis_report

def verify_automated_flow():
    print("Testing Automated Workflow on sample_website_activity.json...")
    df_events = load_dataset("data/sample_website_activity.json")
    print(f"Loaded {len(df_events)} events from JSON.")
    
    val = validate_activity_dataset(df_events)
    assert val['is_valid'], f"Validation failed: {val['errors']}"
    print(f"Validated: {val['total_sessions']} sessions, {val['valid_timestamp_pct']}% valid timestamps.")
    
    # Timeline
    timeline_fig = plot_session_timeline(df_events)
    assert timeline_fig is not None
    print("Timeline plot generated.")
    
    # Aggregation
    session_df, cleaned_events = aggregate_events_to_sessions(df_events)
    print(f"Aggregated into {len(session_df)} sessions with {len(session_df.columns)} features.")
    
    # FP-Growth
    tx, one_hot, items = clickstream_to_transactions(cleaned_events)
    itemsets, exec_time = run_fpgrowth(one_hot, min_support=0.05)
    rules = generate_rules(itemsets, metric='confidence', min_threshold=0.3)
    print(f"FP-Growth: {len(itemsets)} itemsets, {len(rules)} rules in {exec_time*1000:.2f} ms.")
    
    # Clustering
    km, scaler, feat_cols = get_or_train_reference_clustering(n_clusters=3)
    labels, sil, pca_coords, var_ratios = assign_clusters(km, scaler, session_df, feat_cols)
    print(f"Assigned {len(labels)} sessions to clusters. Silhouette: {sil}")
    
    # Prediction
    pipe, _ = get_or_train_reference_pipeline(criterion='gini', max_depth=5)
    pred_df, summary = batch_predict_sessions(pipe, session_df, target_col='Revenue')
    print(f"Predictions: {summary['predicted_purchases']} purchases, {summary['predicted_no_purchases']} no purchases ({summary['predicted_conversion_rate']}% rate).")
    
    # Report
    report = generate_analysis_report(
        metadata={'file_name': 'sample_website_activity.json'},
        validation=val,
        prediction_summary=summary
    )
    assert len(report) > 100
    print("Report generated successfully.")
    print("=== AUTOMATED WORKFLOW VERIFICATION PASSED! ===")

if __name__ == '__main__':
    verify_automated_flow()
